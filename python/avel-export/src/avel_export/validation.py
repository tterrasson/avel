"""Generates the browser validation set from PyTorch.

Each entry is a small exported model (`model.graph.json`, `model.safetensors`,
`model.testvec.safetensors`) that the headless-Chrome harness loads, compiles,
runs and compares against the exporter's test vectors. One case per supported
op, over several shapes, broadcasting rules and dtypes, then whole models up
to ResNet-18 and MobileNetV3. Kernels are f32 only for now, so non-f32 dtypes
are negative cases that must fail to compile. Models are also validated in f16
by the exporter, so the harness can run them with f16 activations.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from avel_export.export import export
from avel_export.generation import (
    cache_write,
    cached_attention,
    rotary,
    sample_candidates,
    sample_topk,
)
from avel_export.ops import SUPPORTED_OPS

IMAGE_SIZE = 224
TEXT_MAX_SEQ = 64


class Fn(nn.Module):
    """Wraps a function of the input tensors into a module."""

    def __init__(self, fn: Callable[..., Any]):
        super().__init__()
        self.fn = fn

    def forward(self, *xs: torch.Tensor) -> Any:
        return self.fn(*xs)


class Addmm(nn.Module):
    """`addmm(bias, x, weight)` with a broadcastable bias; weights are buffers."""

    def __init__(self, bias: torch.Tensor, beta: float = 1.0, alpha: float = 1.0):
        super().__init__()
        self.weight = nn.Buffer(torch.randn(4, 8))
        self.bias = nn.Buffer(bias)
        self.beta = beta
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.addmm(self.bias, x, self.weight, beta=self.beta, alpha=self.alpha)


class BothGelus(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return nn.functional.gelu(x), nn.functional.gelu(x, approximate="tanh")


class Mix(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (
            x.permute(2, 0, 1),
            x.sum(dim=1),
            x.sum(dim=-1),
            x.mean(dim=(0, 2), keepdim=True),
            x.mean(),
            nn.functional.relu(x),
        )


class Binary(nn.Module):
    """Tensor-tensor arithmetic against a broadcast buffer, and scalar arithmetic."""

    def __init__(self) -> None:
        super().__init__()
        self.other = nn.Buffer(torch.rand(3, 1) + 0.5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (
            x + self.other,
            torch.sub(x, self.other, alpha=2),
            x * self.other,
            x / self.other,
            self.other - x,
            x + 3,
            x - 1.5,
            x * 2,
            x / 6,
            torch.add(x, 2, alpha=3),
        )


class Views(nn.Module):
    """Every view op, read by kernels and returned as outputs."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (
            x.transpose(1, 2),
            torch.relu(x.transpose(0, 2)),
            x.reshape(2, -1),
            x.permute(2, 1, 0).reshape(-1),
            x.permute(2, 1, 0).contiguous() * 2,
            x[:, 1:3],
            x[..., ::2] + 1,
            x[:, -2:, 1],
            x[1],
            x[:, :1].expand(2, 3, 4) + x,
            x.unsqueeze(1),
            x[:, :1].squeeze(1),
            x.clone(),
        )


class LayerNormResidual(nn.Module):
    """`x + layer_norm(x)`: the add fuses into the layer norm epilogue."""

    def __init__(self) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ln(x)


class HeadAttention(nn.Module):
    """One attention head split out of a single input, read through views."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.heads = heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        h = self.heads
        qkv = self.qkv(x).reshape(b, s, 3, h, d // h).permute(2, 0, 3, 1, 4)
        return (
            F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
            .permute(0, 2, 1, 3)
            .reshape(b, s, d)
        )


class SdpaViews(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = HeadAttention(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SdpaResidual(nn.Module):
    """`x + attention(x)`: the add fuses into the attention epilogue."""

    def __init__(self) -> None:
        super().__init__()
        self.block = HeadAttention(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Embed(nn.Module):
    def __init__(self, padding_idx: int | None = None):
        super().__init__()
        self.emb = nn.Embedding(128, 64, padding_idx=padding_idx)
        if padding_idx is not None:
            with torch.no_grad():
                self.emb.weight[padding_idx].fill_(0.75)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.emb(ids)


class FoldedLinear(nn.Module):
    """A linear layer whose weight is computed from parameters only."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(6, 4))
        self.bias = nn.Parameter(torch.randn(6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight * 0.5 + 1, self.bias)


class AttentionBlock(nn.Module):
    """One pre-norm transformer block, written so that no reshape mixes a
    dynamic dimension with a static one (torch.export guards against that)."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)
        self.heads = heads

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        b, s, d = x.shape
        h = self.heads
        y = self.norm1(x)
        qkv = self.qkv(y).reshape(b, s, 3, h, d // h).permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=causal)
        x = x + self.proj(attn.permute(0, 2, 1, 3).reshape(b, s, d))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


def vit_small() -> nn.Module:
    """A small ViT: 32×32 images, 8×8 patches, 64-wide, 2 blocks, 4 heads."""
    dim, heads, layers, image, patch = 64, 4, 2, 32, 8

    class ViT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_embed = nn.Conv2d(3, dim, patch, stride=patch)
            self.cls = nn.Parameter(torch.randn(1, 1, dim))
            self.pos = nn.Parameter(torch.randn(1, (image // patch) ** 2 + 1, dim))
            self.blocks = nn.ModuleList(AttentionBlock(dim, heads) for _ in range(layers))
            self.norm = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b = x.shape[0]
            x = self.patch_embed(x).flatten(2).transpose(1, 2)
            x = torch.cat([self.cls.expand(b, 1, -1), x], dim=1) + self.pos
            for block in self.blocks:
                x = block(x)
            return self.head(self.norm(x[:, 0]))

    return ViT()


def text_encoder(max_seq: int = TEXT_MAX_SEQ) -> nn.Module:
    """A small causal text encoder: embedding plus position, 2 blocks, CLS pooling."""
    vocab, dim, heads, layers = 128, 64, 4, 2

    class TextEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab, dim)
            self.pos = nn.Parameter(torch.randn(1, max_seq, dim))
            self.blocks = nn.ModuleList(AttentionBlock(dim, heads) for _ in range(layers))
            self.norm = nn.LayerNorm(dim)

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            x = self.embed(ids) + self.pos[:, : ids.shape[1]]
            for block in self.blocks:
                x = block(x, causal=True)
            return self.norm(x)[:, 0]

    return TextEncoder()


def randomize_batch_norms(model: nn.Module) -> nn.Module:
    """Gives every batch norm non-trivial statistics, as a trained model has."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            channels = module.num_features
            module.running_mean = torch.randn(channels) * 0.1
            module.running_var = torch.rand(channels) + 0.5
            if module.affine:
                module.weight = nn.Parameter(torch.rand(channels) + 0.5)
                module.bias = nn.Parameter(torch.randn(channels) * 0.1)
    return model


def mlp() -> nn.Module:
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(8, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )


def cnn() -> nn.Module:
    """A small CNN chaining every convolution path through channels-last tensors."""
    return randomize_batch_norms(
        nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1, groups=8),
            nn.Hardswish(),
            nn.Conv2d(8, 20, 1),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(20, 5),
        )
    )


def cases() -> list[dict[str, Any]]:
    """One entry per supported op and shape family. `expect_error` marks cases
    that must be rejected by the compiler (non-f32 dtypes for now)."""
    conv = nn.Conv2d
    return [
        {"name": "relu-1d", "model": Fn(torch.relu), "inputs": (torch.randn(7),)},
        {"name": "relu-2d", "model": Fn(torch.relu), "inputs": (torch.randn(3, 4),)},
        {"name": "relu-3d", "model": Fn(torch.relu), "inputs": (torch.randn(2, 3, 5),)},
        {"name": "gelu-2d", "model": Fn(nn.functional.gelu), "inputs": (torch.randn(3, 4),)},
        {
            "name": "gelu-tanh-2d",
            "model": Fn(lambda x: nn.functional.gelu(x, approximate="tanh")),
            "inputs": (torch.randn(3, 4),),
        },
        {
            "name": "gelu-extreme",
            "model": BothGelus(),
            "inputs": (
                torch.tensor(
                    [
                        [
                            -1e20,
                            -1e4,
                            -100.0,
                            -16.0,
                            -7.0,
                            -6.0,
                            -5.0,
                            -1.0,
                            -1e-3,
                            0.0,
                            1e-3,
                            1.0,
                            5.0,
                            6.0,
                            7.0,
                            16.0,
                            100.0,
                            1e4,
                            1e20,
                        ]
                    ]
                ),
            ),
        },
        {
            "name": "sigmoid",
            "model": Fn(lambda x: (torch.sigmoid(x), nn.functional.silu(x))),
            "inputs": (torch.randn(4, 5) * 8,),
        },
        {
            "name": "clamp",
            "model": Fn(
                lambda x: (
                    x.clamp(min=-0.5),
                    x.clamp(max=0.5),
                    x.clamp(-1, 1),
                    nn.functional.relu6(x),
                    nn.functional.hardtanh(x, -2, 2),
                )
            ),
            "inputs": (torch.randn(3, 4) * 4,),
        },
        {
            "name": "hardswish",
            "model": Fn(lambda x: (nn.functional.hardswish(x), nn.functional.hardsigmoid(x))),
            "inputs": (torch.randn(2, 3, 4) * 4,),
        },
        {"name": "binary", "model": Binary(), "inputs": (torch.randn(2, 3, 4),)},
        {
            "name": "binary-two-inputs",
            "model": Fn(lambda x, y: (x + y, x * y, x - y, x / y)),
            "inputs": (torch.randn(2, 1, 4), torch.rand(3, 1) + 0.5),
        },
        {
            "name": "permute-2d",
            "model": Fn(lambda x: x.permute(1, 0)),
            "inputs": (torch.randn(2, 3),),
        },
        {
            "name": "permute-3d",
            "model": Fn(lambda x: x.permute(2, 0, 1)),
            "inputs": (torch.randn(2, 3, 4),),
        },
        {"name": "views", "model": Views(), "inputs": (torch.randn(2, 3, 4),)},
        {"name": "sum-dim", "model": Fn(lambda x: x.sum(dim=-1)), "inputs": (torch.randn(3, 4),)},
        {
            "name": "sum-dims-keepdim",
            "model": Fn(lambda x: x.sum(dim=(0, 2), keepdim=True)),
            "inputs": (torch.randn(2, 3, 5),),
        },
        {
            "name": "mean-dim",
            "model": Fn(lambda x: x.mean(dim=1)),
            "inputs": (torch.randn(2, 3, 4),),
        },
        {
            "name": "mean-dims-keepdim",
            "model": Fn(lambda x: x.mean(dim=(0, 2), keepdim=True)),
            "inputs": (torch.randn(2, 3, 5),),
        },
        {"name": "mean-all", "model": Fn(lambda x: x.mean()), "inputs": (torch.randn(2, 3),)},
        {"name": "addmm", "model": Addmm(torch.randn(8)), "inputs": (torch.randn(3, 4),)},
        {
            "name": "addmm-bias-broadcast",
            "model": Addmm(torch.randn(1, 8)),
            "inputs": (torch.randn(3, 4),),
        },
        {
            "name": "addmm-bias-row",
            "model": Addmm(torch.randn(3, 1)),
            "inputs": (torch.randn(3, 4),),
        },
        {
            "name": "addmm-bias-full",
            "model": Addmm(torch.randn(3, 8)),
            "inputs": (torch.randn(3, 4),),
        },
        {
            "name": "addmm-bias-scalar",
            "model": Addmm(torch.randn(())),
            "inputs": (torch.randn(3, 4),),
        },
        {
            "name": "addmm-beta-alpha",
            "model": Addmm(torch.randn(8), beta=0.5, alpha=2.0),
            "inputs": (torch.randn(3, 4),),
        },
        {"name": "folded-linear", "model": FoldedLinear(), "inputs": (torch.randn(3, 4),)},
        {"name": "mix", "model": Mix(), "inputs": (torch.randn(2, 3, 5) * 4,)},
        {"name": "conv2d", "model": conv(3, 8, 3, padding=1), "inputs": (torch.randn(2, 3, 9, 9),)},
        {
            "name": "conv2d-stem",
            "model": conv(3, 20, 7, stride=2, padding=3),
            "inputs": (torch.randn(1, 3, 32, 32),),
        },
        {
            "name": "conv2d-stride-dilation",
            "model": conv(4, 6, 3, stride=2, padding=2, dilation=2, bias=False),
            "inputs": (torch.randn(1, 4, 11, 13),),
        },
        {
            "name": "conv2d-groups",
            "model": conv(4, 6, 3, groups=2),
            "inputs": (torch.randn(1, 4, 7, 7),),
        },
        {
            "name": "conv2d-depthwise",
            "model": conv(8, 8, 3, padding=1, groups=8),
            "inputs": (torch.randn(1, 8, 10, 10),),
        },
        {
            "name": "conv2d-depthwise-multiplier",
            "model": conv(4, 8, 5, stride=2, padding=2, groups=4),
            "inputs": (torch.randn(2, 4, 9, 9),),
        },
        {
            "name": "conv2d-pointwise",
            "model": conv(16, 24, 1),
            "inputs": (torch.randn(1, 16, 7, 7),),
        },
        {
            "name": "conv2d-pointwise-strided",
            "model": conv(8, 16, 1, stride=2, bias=False),
            "inputs": (torch.randn(1, 8, 9, 9),),
        },
        {
            "name": "conv1d",
            "model": nn.Conv1d(3, 5, 3, padding=1),
            "inputs": (torch.randn(2, 3, 10),),
        },
        {
            "name": "conv3d",
            "model": nn.Conv3d(2, 3, 3, padding=1),
            "inputs": (torch.randn(1, 2, 5, 5, 5),),
        },
        {
            "name": "batch-norm",
            "model": randomize_batch_norms(nn.BatchNorm2d(5)),
            "inputs": (torch.randn(2, 5, 4, 4),),
        },
        {
            "name": "batch-norm-no-affine",
            "model": randomize_batch_norms(nn.BatchNorm2d(5, affine=False)),
            "inputs": (torch.randn(2, 5, 4, 4),),
        },
        {
            "name": "batch-norm-1d",
            "model": randomize_batch_norms(nn.BatchNorm1d(6)),
            "inputs": (torch.randn(4, 6),),
        },
        {"name": "max-pool", "model": nn.MaxPool2d(3, 2, 1), "inputs": (torch.randn(1, 4, 9, 9),)},
        {
            "name": "max-pool-ceil-dilation",
            "model": nn.MaxPool2d(3, 2, 1, dilation=2, ceil_mode=True),
            "inputs": (torch.randn(2, 3, 10, 11),),
        },
        {"name": "max-pool-unbatched", "model": nn.MaxPool2d(2), "inputs": (torch.randn(3, 6, 6),)},
        {"name": "avg-pool", "model": nn.AvgPool2d(3, 2, 1), "inputs": (torch.randn(1, 4, 9, 9),)},
        {
            "name": "avg-pool-exclude-padding",
            "model": nn.AvgPool2d(3, 2, 1, ceil_mode=True, count_include_pad=False),
            "inputs": (torch.randn(1, 2, 10, 10),),
        },
        {
            "name": "avg-pool-divisor",
            "model": nn.AvgPool2d(2, divisor_override=3),
            "inputs": (torch.randn(1, 2, 6, 6),),
        },
        {
            "name": "layer-norm",
            "model": nn.LayerNorm(64),
            "inputs": (torch.randn(2, 17, 64) * 3,),
        },
        {
            "name": "layer-norm-no-affine",
            "model": nn.LayerNorm(64, elementwise_affine=False),
            "inputs": (torch.randn(2, 17, 64) * 3,),
        },
        {
            "name": "layer-norm-two-axes",
            "model": Fn(lambda x: F.layer_norm(x, (4, 8), None, None, 1e-5)),
            "inputs": (torch.randn(2, 3, 4, 8) * 3,),
        },
        {
            "name": "sdpa",
            "model": Fn(lambda q, k, v: F.scaled_dot_product_attention(q, k, v)),
            "inputs": (
                torch.randn(2, 4, 17, 16),
                torch.randn(2, 4, 17, 16),
                torch.randn(2, 4, 17, 32),
            ),
        },
        {
            "name": "sdpa-causal",
            "model": Fn(lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)),
            "inputs": (
                torch.randn(1, 3, 23, 8),
                torch.randn(1, 3, 23, 8),
                torch.randn(1, 3, 23, 8),
            ),
        },
        {
            "name": "sdpa-causal-short-query",
            "model": Fn(lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)),
            "inputs": (torch.randn(1, 2, 3, 8), torch.randn(1, 2, 7, 8), torch.randn(1, 2, 7, 8)),
        },
        {
            "name": "sdpa-causal-long-query",
            "model": Fn(lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)),
            "inputs": (torch.randn(1, 2, 7, 8), torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8)),
        },
        {
            "name": "sdpa-mask-empty-rows",
            "model": Fn(lambda q, k, v, m: F.scaled_dot_product_attention(q, k, v, attn_mask=m)),
            "inputs": (
                torch.randn(1, 2, 3, 8),
                torch.randn(1, 2, 3, 8),
                torch.randn(1, 2, 3, 8),
                torch.tensor([[False, False, False], [True, False, True], [False, False, False]]),
            ),
        },
        {
            "name": "sdpa-mask-negative-infinity",
            "model": Fn(lambda q, k, v, m: F.scaled_dot_product_attention(q, k, v, attn_mask=m)),
            "inputs": (
                torch.randn(1, 2, 3, 8),
                torch.randn(1, 2, 3, 8),
                torch.randn(1, 2, 3, 8),
                torch.tensor([[float("-inf")] * 3, [0.0, float("-inf"), 0.0], [float("-inf")] * 3]),
            ),
        },
        {
            "name": "sdpa-scale",
            "model": Fn(lambda q, k, v: F.scaled_dot_product_attention(q, k, v, scale=0.5)),
            "inputs": (
                torch.randn(2, 2, 9, 16),
                torch.randn(2, 2, 9, 16),
                torch.randn(2, 2, 9, 16),
            ),
        },
        {
            "name": "sdpa-mask",
            "model": Fn(lambda q, k, v, m: F.scaled_dot_product_attention(q, k, v, attn_mask=m)),
            "inputs": (
                torch.randn(2, 4, 17, 16),
                torch.randn(2, 4, 17, 16),
                torch.randn(2, 4, 17, 16),
                torch.randn(2, 1, 17, 17),
            ),
        },
        {
            "name": "sdpa-mask-bool",
            "model": Fn(lambda q, k, v, m: F.scaled_dot_product_attention(q, k, v, attn_mask=m)),
            "inputs": (
                torch.randn(1, 2, 13, 8),
                torch.randn(1, 2, 13, 8),
                torch.randn(1, 2, 13, 8),
                torch.rand(1, 1, 1, 13) > 0.5,
            ),
        },
        {
            "name": "embedding",
            "model": Embed(),
            "inputs": (torch.randint(0, 128, (2, 17), dtype=torch.int32),),
        },
        {
            "name": "embedding-padding",
            "model": Embed(padding_idx=3),
            "inputs": (torch.full((4, 9), 3, dtype=torch.int32),),
        },
        {
            "name": "cat-single",
            "model": Fn(lambda x: torch.cat([x], dim=1)),
            "inputs": (torch.randn(2, 4, 8),),
        },
        {
            "name": "cat",
            "model": Fn(lambda a, b: torch.cat([a, b], dim=1)),
            "inputs": (torch.randn(2, 1, 8), torch.randn(2, 4, 8)),
        },
        {
            "name": "cat-three-dim0",
            "model": Fn(lambda a, b, c: torch.cat([a, b, c], dim=0)),
            "inputs": (torch.randn(2, 4), torch.randn(3, 4), torch.randn(1, 4)),
        },
        {
            "name": "cat-then-add",
            "model": Fn(lambda a, b, c: torch.cat([a, b], dim=1) + c),
            "inputs": (torch.randn(2, 1, 8), torch.randn(2, 4, 8), torch.randn(2, 5, 8)),
        },
        {
            "name": "layer-norm-residual",
            "model": LayerNormResidual(),
            "inputs": (torch.randn(2, 17, 64) * 3,),
        },
        {
            "name": "sdpa-views",
            "model": SdpaViews(),
            "inputs": (torch.randn(2, 13, 16),),
        },
        {
            "name": "sdpa-residual",
            "model": SdpaResidual(),
            "inputs": (torch.randn(2, 13, 16),),
        },
        {
            "name": "relu-f16",
            "model": Fn(torch.relu),
            "inputs": (torch.randn(3, 4).half(),),
            "expect_error": "which plans cannot hold",
        },
        {
            "name": "relu-i32",
            "model": Fn(torch.relu),
            "inputs": (torch.randint(-8, 8, (3, 4), dtype=torch.int32),),
            "expect_error": "does not accept",
        },
        *typed_cases(),
        *coverage_cases(),
        *int8_cases(),
        *generation_cases(),
    ]


class Accumulator(nn.Module):
    """Persistent state: a buffer the model adds its input to on every run."""

    def __init__(self) -> None:
        super().__init__()
        self.total = nn.Buffer(torch.zeros(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.total.mul_(0.5).add_(x)
        return self.total * 2


class Projection(nn.Module):
    """`x @ w` without a bias: `mm`, and `bmm` for batched operands."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Buffer(torch.randn(8, 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight


class StateSwap(nn.Module):
    """Both updates read the previous state, before either buffer is replaced."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Buffer(torch.tensor([1.0, 2.0, 3.0]))
        self.b = nn.Buffer(torch.tensor([4.0, 5.0, 6.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous = self.a.clone()
        self.a.copy_(self.b)
        self.b.copy_(previous)
        return self.a + self.b * 10 + x


class StateProjection(nn.Module):
    """Autotuning a matmul must leave the model's initial state untouched."""

    def __init__(self) -> None:
        super().__init__()
        self.total = nn.Buffer(torch.ones(2, 3))
        self.weight = nn.Buffer(torch.randn(4, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.total.mul_(0.5).add_(x @ self.weight)
        return self.total * 2


def typed_cases() -> list[dict[str, Any]]:
    """Integer and bool values, generators, multiple outputs, batched matmuls,
    cooperative reductions, preserved ops and state."""
    x = torch.randn(3, 4)
    ids = torch.randint(-50, 50, (2, 6))
    compare = {
        "eq": torch.eq,
        "ne": torch.ne,
        "lt": torch.lt,
        "le": torch.le,
        "gt": torch.gt,
        "ge": torch.ge,
    }
    cases: list[dict[str, Any]] = []

    def pooled_with_indices(a: torch.Tensor) -> torch.Tensor:
        values, indices = F.max_pool2d(a, 2, return_indices=True)
        return values + indices.float()

    for name, fn in compare.items():
        cases.append(
            {"name": f"{name}-scalar", "model": Fn(lambda a, fn=fn: fn(a, 0.25)), "inputs": (x,)}
        )
        cases.append(
            {
                "name": f"{name}-tensor",
                "model": Fn(lambda a, b, fn=fn: fn(a, b)),
                "inputs": (torch.randint(0, 3, (3, 4)).float(), torch.randint(0, 3, (4,)).float()),
            }
        )
    masks = (torch.rand(3, 4) > 0.5, torch.rand(4) > 0.5)
    cases += [
        {
            "name": "where-leaky",
            "model": Fn(lambda a: torch.where(a > 0, a, 0.1 * a)),
            "inputs": (x,),
        },
        {
            "name": "masked-fill",
            "model": Fn(lambda a, m: a.masked_fill(m, -1.0)),
            "inputs": (x, masks[0]),
        },
        {"name": "bitwise-and", "model": Fn(lambda a, b: a & b), "inputs": masks},
        {"name": "bitwise-or", "model": Fn(lambda a, b: a | b), "inputs": masks},
        {"name": "bitwise-xor", "model": Fn(lambda a, b: a ^ b), "inputs": masks},
        {"name": "bitwise-not", "model": Fn(lambda a: ~a), "inputs": (masks[0],)},
        {
            "name": "bitwise-int",
            "model": Fn(lambda a, b: (a & b) ^ (a | b)),
            "inputs": (ids, ids.flip(0)),
        },
        {"name": "int64-arithmetic", "model": Fn(lambda a: a * 3 + 1 - a), "inputs": (ids,)},
        {"name": "int64-divide", "model": Fn(lambda a: a / 4), "inputs": (ids,)},
        {
            "name": "cast",
            "model": Fn(lambda a: (a > 0).float() + a.long().float() + a.int().float()),
            "inputs": (x * 4,),
        },
        {"name": "cast-int64", "model": Fn(lambda a: a.to(torch.int64)), "inputs": (x * 4,)},
        {"name": "arange-int", "model": Fn(lambda a: a + torch.arange(6)), "inputs": (ids,)},
        {
            "name": "arange-float",
            "model": Fn(lambda a: a * torch.arange(0, 2, 0.5)),
            "inputs": (x,),
        },
        {
            "name": "causal-mask",
            "model": Fn(lambda a: a.masked_fill(torch.arange(4) > torch.arange(3)[:, None], 0.0)),
            "inputs": (x,),
        },
        {"name": "full", "model": Fn(lambda a: a + torch.full((3, 4), 2.5)), "inputs": (x,)},
        {"name": "full-like", "model": Fn(lambda a: torch.ones_like(a) - a), "inputs": (x,)},
        {"name": "split", "model": Fn(lambda a: a.split([1, 3], dim=1)), "inputs": (x,)},
        {"name": "chunk", "model": Fn(lambda a: a.chunk(3, dim=0)), "inputs": (x,)},
        {
            "name": "max-pool-indices",
            "model": Fn(lambda a: F.max_pool2d(a, 3, 2, 1, return_indices=True)),
            "inputs": (torch.randn(1, 3, 9, 9),),
        },
        {
            "name": "max-pool-indices-only",
            "model": Fn(lambda a: F.max_pool2d(a, 2, return_indices=True)[1]),
            "inputs": (torch.randn(1, 2, 4, 4),),
        },
        {
            "name": "max-pool-combined",
            "model": Fn(pooled_with_indices),
            "inputs": (torch.randn(1, 2, 4, 4),),
        },
        {
            "name": "bool-add-zero-alpha",
            "model": Fn(lambda a, b: torch.add(a, b, alpha=0)),
            "inputs": masks,
        },
        {
            "name": "full-int-min",
            "model": Fn(lambda a: a + torch.full((2, 6), -2147483648, dtype=torch.int64)),
            "inputs": (torch.zeros_like(ids),),
        },
        {
            "name": "arange-int-boundary",
            "model": Fn(lambda a: a + torch.arange(2147483647, 2147483648)),
            "inputs": (torch.zeros(1, dtype=torch.int64),),
        },
        {"name": "mm", "model": Projection(), "inputs": (torch.randn(6, 8),)},
        {
            "name": "bmm",
            "model": Fn(lambda a, b: a @ b),
            "inputs": (torch.randn(2, 3, 5, 7), torch.randn(3, 7, 4)),
        },
        {"name": "rms-norm", "model": nn.RMSNorm(64), "inputs": (torch.randn(2, 5, 64),)},
        {
            "name": "rms-norm-wide",
            "model": nn.RMSNorm(1024, eps=1e-5),
            "inputs": (torch.randn(3, 1024),),
        },
        {"name": "layer-norm-wide", "model": nn.LayerNorm(768), "inputs": (torch.randn(4, 768),)},
        {
            "name": "mean-long-row",
            "model": Fn(lambda a: a.mean(dim=-1)),
            "inputs": (torch.randn(2, 40000),),
        },
        {
            "name": "sum-long-row",
            "model": Fn(lambda a: a.relu().sum(dim=1)),
            "inputs": (torch.randn(5, 3000),),
        },
        {
            "name": "prologue-matmul",
            "model": Fn(lambda a, b: (a * 0.5) @ b),
            "inputs": (torch.randn(5, 8), torch.randn(8, 3)),
        },
        {
            "name": "state-accumulator",
            "model": Accumulator(),
            "inputs": (torch.randn(2, 3),),
            "extra_test_inputs": ((torch.randn(2, 3),), (torch.randn(2, 3),)),
        },
        {
            "name": "state-swap",
            "model": StateSwap(),
            "inputs": (torch.zeros(3),),
            "extra_test_inputs": ((torch.zeros(3),), (torch.zeros(3),)),
        },
        {
            "name": "state-projection",
            "model": StateProjection(),
            "inputs": (torch.randn(2, 4),),
            "extra_test_inputs": ((torch.randn(2, 4),),),
        },
    ]
    return cases


def int8_cases() -> list[dict[str, Any]]:
    modules = [
        ("int8-linear", nn.Linear(33, 17), torch.randn(4, 33)),
        ("int8-convolution", nn.Conv2d(3, 5, 3, padding=1), torch.randn(2, 3, 8, 7)),
        ("int8-depthwise", nn.Conv2d(3, 3, 3, groups=3), torch.randn(1, 3, 8, 7)),
    ]
    with torch.no_grad():
        for _, model, _ in modules:
            # Exactly representable weights isolate shader error from the
            # export's approximation error. Include zero and partial words.
            weight = model.weight
            integers = torch.randint(-127, 128, weight.shape)
            integers.reshape(weight.shape[0], -1)[:, 0] = 127
            integers[0].zero_()
            scales = 2.0 ** -(7 + torch.arange(weight.shape[0]) % 4)
            weight.copy_(integers * scales.reshape([-1] + [1] * (weight.ndim - 1)))
    return [
        {"name": name, "model": model, "inputs": (tensor,), "int8": True}
        for name, model, tensor in modules
    ]


class CachedDecoder(nn.Module):
    def __init__(self, capacity: int = 2048):
        super().__init__()
        self.key = nn.Buffer(torch.zeros(2, 3, capacity, 4))
        self.value = nn.Buffer(torch.zeros(2, 3, capacity, 4))

    def forward(self, x, position, seed):
        self.key.copy_(cache_write(self.key, x * 0.75, position))
        self.value.copy_(cache_write(self.value, x * 1.25, position))
        attn = cached_attention(x, self.key, self.value, position)
        return attn, sample_topk(attn, seed, 3)


def generation_cases() -> list[dict[str, Any]]:
    seq = torch.export.Dim("tokens", min=1, max=32)

    def args(length, position, seed):
        return (
            torch.randn(2, 3, length, 4),
            torch.tensor(position),
            torch.tensor(seed, dtype=torch.int32),
        )

    return [
        {
            "name": "cached-prefill-decode",
            "model": CachedDecoder(),
            "inputs": args(5, 0, 701),
            "dynamic_shapes": ({2: seq}, None, None),
            "extra_test_inputs": (args(1, 5, -13), args(1, 6, 17), args(3, 7, 101)),
        },
        {
            "name": "sample-degenerate",
            "model": Fn(sample_candidates),
            "inputs": (
                torch.tensor(
                    [[float("-inf")] * 3, [1.0, float("nan"), 1.0], [0.0, 1.0, float("-inf")]]
                ),
                torch.arange(9).reshape(3, 3),
                torch.tensor(729),
            ),
        },
        {
            "name": "sample-wide-topk",
            "model": Fn(lambda x, seed: sample_topk(x, seed, 32)),
            "inputs": (
                torch.stack([torch.randperm(10_501), torch.randperm(10_501)]).float() * 0.05,
                torch.tensor([-173], dtype=torch.int32),
            ),
        },
    ]


class ChannelAffine(nn.Module):
    def __init__(self, channels: int, spatial: bool = False):
        super().__init__()
        shape = (channels, 1, 1) if spatial else (channels,)
        self.scale = nn.Parameter(torch.randn(shape))
        self.shift = nn.Parameter(torch.randn(shape))

    def forward(self, x):
        return x * self.scale + self.shift


def coverage_cases() -> list[dict[str, Any]]:
    """ATen coverage against eager PyTorch, including strided and typed inputs."""
    aten = torch.ops.aten

    def case(name, fn, *inputs):
        return {"name": name, "model": Fn(fn), "inputs": inputs}

    x = torch.tensor(
        [-100.0, -88.0, -4.0, -1.0, -0.1, -1e-7, 0.0, 1e-7, 0.1, 3.0, 80.0, 89.0, 100.0]
    )
    ints = torch.tensor([[-3, -2, 0], [1, 2, 3]], dtype=torch.int32)
    rows = torch.randn(513, 3)
    # Equal extrema in different lanes, and a row with multiple NaNs.
    rows[[1, 64], 0] = 99.0
    rows[[2, 65], 1] = float("nan")
    rows[[3, 66], 2] = -99.0
    image = torch.randn(2, 4, 5, 7)
    i = torch.tensor([[-1], [0], [0]])
    j = torch.tensor([1, -2])
    return [
        case(
            "unary-math",
            lambda a: (
                a.tanh(),
                a.exp(),
                a.expm1(),
                a.log1p(),
                a.erf(),
                a.abs().log(),
                a.abs().sqrt(),
                a.abs().rsqrt(),
                a.reciprocal(),
                a.neg(),
                a.sign(),
                a.sin(),
                a.cos(),
                F.elu(a, 0.3),
                F.leaky_relu(a, 0.23),
                F.softplus(a),
                F.mish(a),
            ),
            x,
        ),
        case(
            "trig-range-reduction",
            lambda a: (a.sin(), a.cos()),
            torch.tensor(
                [-8192.0, -4096.0, -100.0, -3.14159265, -0.0, 0.0, 1.5707963, 4096.0, 8192.0]
            ),
        ),
        case("unary-int", lambda a: (a.abs(), a.neg(), a.sign(), a.exp()), ints),
        case(
            "rounding",
            lambda a, b: (
                a.floor(),
                a.ceil(),
                a.round(),
                a.trunc(),
                b.floor(),
                b.ceil(),
                b.round(),
                b.trunc(),
            ),
            torch.tensor([-3.5, -2.5, -1.5, -0.5, -0.25, 0.25, 0.5, 1.5, 2.5, 3.5]),
            ints * 123456789,
        ),
        case(
            "tensor-powers",
            lambda a, b, c: (a.pow(b), c.pow(b.int())),
            torch.tensor([[-2.0], [0.0], [4.0]]),
            torch.tensor([-3.0, -2.0, 0.0, 0.5, 2.0, 3.0]),
            torch.tensor([[-2], [-1], [0], [1], [2]], dtype=torch.int32),
        ),
        # C pow at infinite edges, and NaN through tanh's clamp.
        case(
            "power-edges",
            lambda a, b: (a.pow(b), a.tanh()),
            torch.tensor([float("-inf"), float("-inf"), -1.0, -1.0, float("-inf"), float("nan")]),
            torch.tensor([0.5, -0.5, float("inf"), float("-inf"), 3.0, 2.0]),
        ),
        case(
            "scalar-remainders",
            lambda a, b: (
                aten.remainder.Scalar(a, 2.5),
                aten.remainder.Scalar(a, -2.5),
                aten.fmod.Scalar(a, -2.5),
                aten.remainder.Scalar(b, -3),
                aten.fmod.Scalar(b, 3),
            ),
            torch.tensor([-8.5, -5, -1, -0.0, 0.0, 1, 5, 8.5]),
            ints,
        ),
        case(
            "product-and-any",
            lambda a, b, c: (
                a.prod(-1),
                b.prod(1, keepdim=True, dtype=torch.int32),
                c.prod(1),
                a.any(1),
                b.any(0, keepdim=True),
                c.any(-1),
            ),
            torch.full((2, 513), 1.0001),
            ints,
            ints > 0,
        ),
        case("elu-scaled", lambda a: aten.elu.default(a, 0.5, 1.25, 0.75), x),
        case(
            "scalar-overloads",
            lambda a: (
                aten.add.Scalar(a, 2, 3),
                aten.sub.Scalar(a, 2, 3),
                aten.mul.Scalar(a, 1.5),
                aten.div.Scalar(a, 3),
            ),
            ints,
        ),
        case(
            "scalar-powers",
            lambda a: (
                a.pow(2),
                a.pow(3),
                a.pow(4.0),
                a.pow(-3),
                a.pow(0.5),
                a.pow(-0.5),
                a.pow(-1),
                a.pow(1.5),
                torch.pow(-2.0, a),
            ),
            torch.tensor([-4.0, -1.0, -0.0, 0.0, 0.5, 1.0, 3.0, 4.0]),
        ),
        case("integer-powers", lambda a: (a.pow(3), torch.pow(-2, a)), ints),
        case(
            "binary-extrema",
            lambda a, b, c: (
                a.maximum(b),
                a.minimum(b),
                b.maximum(c),
                b.minimum(c),
                b.maximum(c.int()),
            ),
            torch.tensor([[float("nan"), -3.0], [1.0, 7.0]]),
            torch.tensor([6, -2], dtype=torch.int32),
            torch.tensor([True, False]),
        ),
        case(
            "extrema-scalar-promotion",
            lambda a, b: (a.maximum(b), a.minimum(b)),
            ints,
            torch.tensor(2),
        ),
        {
            "name": "affine-linear",
            "model": nn.Sequential(nn.Linear(8, 11), ChannelAffine(11)),
            "inputs": (torch.randn(4, 8),),
        },
        {
            "name": "affine-convolution",
            "model": nn.Sequential(nn.Conv2d(4, 7, 3, padding=1), ChannelAffine(7, spatial=True)),
            "inputs": (image,),
        },
        case(
            "decomposed-rms-norm",
            lambda a: a * (a.pow(2).mean(-1, keepdim=True) + 1e-5).rsqrt(),
            image,
        ),
        case("softmax-short", lambda a: (a.softmax(-1), a.log_softmax(1)), image * 100),
        case(
            "softmax-cooperative",
            lambda a: (a.T.softmax(-1), a.T.log_softmax(-1)),
            torch.randn(8193, 3) * 8,
        ),
        case(
            "softmax-nonfinite",
            lambda a: (a.softmax(-1), a.log_softmax(-1)),
            torch.tensor(
                [
                    [1.0, float("-inf"), 2.0],
                    [0.0, float("inf"), 2.0],
                    [float("-inf")] * 3,
                    [0.0, float("nan"), 2.0],
                ]
            ),
        ),
        case(
            "extrema-cooperative",
            lambda a: (
                a.T.amax(-1),
                a.T.amin(-1),
                *a.max(0),
                a.min(0)[1],
                a.T.argmax(),
                a.T.argmin(dim=-1, keepdim=True),
            ),
            rows,
        ),
        case(
            "extrema-typed",
            lambda a, b: (a.argmax(-1), *a.min(-1, keepdim=True), a.amax((1, 0)), b.amin(0)),
            ints,
            ints > 0,
        ),
        case(
            "variance-welford",
            lambda a: (a.var(0, correction=1), a.var((0, 1), correction=0.5, keepdim=True)),
            1024 + torch.randn(911, 3) * 0.25,
        ),
        case(
            "variance-small",
            lambda a: (a.var(1, correction=0), a.var(1, correction=3)),
            ints.float(),
        ),
        case(
            "scan-cooperative",
            lambda a, b: (a.T.cumsum(-1), a.cumsum(0, dtype=torch.int32), b.cumsum(0)),
            torch.randint(-4, 5, (1167, 3), dtype=torch.int32),
            torch.rand(1167, 3) > 0.5,
        ),
        case("scan-fused-bool", lambda a: a.cumsum(0) > 5, torch.randint(-3, 4, (513, 2))),
        case("scan-serial", lambda a: (a.cumsum(1), a.cumsum(0, dtype=torch.float32)), ints),
        case(
            "scalar-rows",
            lambda a: (
                a.softmax(0),
                a.log_softmax(-1),
                a.argmax(),
                a.cumsum(0),
                a.var(correction=0),
            ),
            torch.tensor(2.0),
        ),
        {"name": "group-norm", "model": nn.GroupNorm(2, 4), "inputs": (torch.randn(2, 4, 13, 17),)},
        case(
            "group-norm-bias-only",
            lambda a, b: aten.native_group_norm.default(a, None, b, 2, 4, 35, 2, 1e-5)[0],
            image,
            torch.randn(4),
        ),
        {"name": "instance-norm", "model": nn.InstanceNorm2d(4, affine=True), "inputs": (image,)},
        case("batch-norm-no-stats", lambda a: F.batch_norm(a, None, None, training=True), image),
        case(
            "adaptive-pool2d",
            lambda a: (F.adaptive_avg_pool2d(a, (3, 4)), F.adaptive_avg_pool2d(a[0], (8, 3))),
            image,
        ),
        case(
            "adaptive-max-pool2d",
            lambda a: (
                *F.adaptive_max_pool2d(a, (3, 4), return_indices=True),
                F.adaptive_max_pool2d(a[0].transpose(-2, -1), (8, 3), return_indices=True)[1],
            ),
            image,
        ),
        case(
            "rope-attention",
            lambda q, k, v, c, s: F.scaled_dot_product_attention(
                rotary(q.transpose(-1, -2), c, s),
                rotary(k, c, s),
                v,
                is_causal=True,
            ),
            torch.randn(2, 3, 8, 5),
            torch.randn(2, 3, 5, 8),
            torch.randn(2, 3, 5, 7),
            torch.randn(5, 8).cos(),
            torch.randn(5, 8).sin(),
        ),
        case(
            "rope-decode",
            lambda q, k, v, c, s: F.scaled_dot_product_attention(
                rotary(q, c[-1], s[-1]),
                rotary(k, c, s),
                v,
            ),
            torch.randn(2, 3, 1, 8),
            torch.randn(2, 3, 5, 8),
            torch.randn(2, 3, 5, 7),
            torch.randn(5, 8).cos(),
            torch.randn(5, 8).sin(),
        ),
        case(
            "order-tile",
            lambda a, b: (
                *a.sort(-1),
                *b.sort(dim=0, stable=True, descending=True),
                *(b > 0).sort(dim=0, stable=True),
                a.topk(2, largest=False)[1],
            ),
            torch.tensor([[4.0, 0.0, -2.0, 1.0, float("nan")], [4.0, -2.0, 1.0, 0.0, 3.0]]),
            torch.tensor([[2, -2], [2, 5], [1, -2], [2, -2]], dtype=torch.int32),
        ),
        case(
            "order-merge",
            lambda a: (
                *a.T.sort(dim=0, stable=True),
                *a.topk(41, dim=1, sorted=True),
            ),
            torch.stack([torch.randperm(2189), torch.randperm(2189)]).float(),
        ),
        case(
            "pool3d",
            lambda a: (
                F.avg_pool3d(a, (2, 2, 3), stride=(1, 2, 2), padding=(0, 1, 1), ceil_mode=True),
                *F.max_pool3d(
                    a[0],
                    (2, 2, 2),
                    stride=(1, 2, 2),
                    padding=1,
                    dilation=(2, 1, 1),
                    return_indices=True,
                ),
            ),
            torch.randn(2, 3, 4, 5, 6),
        ),
        case(
            "pool3d-ceil-edge",
            lambda a: (
                F.avg_pool3d(a, (2, 2, 3), stride=(2, 2, 2), ceil_mode=True),
                *F.max_pool3d(a, (2, 2, 3), stride=(2, 2, 2), ceil_mode=True, return_indices=True),
            ),
            torch.arange(2 * 3 * 4 * 4).float().reshape(2, 3, 4, 4),
        ),
        case(
            "resize-bicubic",
            lambda a: (
                F.interpolate(a, size=(1, 9), mode="bicubic", align_corners=True),
                F.interpolate(
                    a.transpose(-1, -2), size=(9, 3), mode="bicubic", align_corners=False
                ),
                F.interpolate(a, scale_factor=(1.5, 0.6), mode="bicubic", align_corners=False),
                F.interpolate(a, scale_factor=(1.5, 1.1), mode="bicubic", align_corners=False),
            ),
            image,
        ),
        case(
            "resize-nearest",
            lambda a: (
                F.interpolate(a, size=(9, 12)),
                F.interpolate(a, scale_factor=(2.1, 1.55)),
                # 2 -> 82 rounds the f32 scale: output 41 reads input 0, not 1.
                F.interpolate(a[..., :2], size=(5, 82)),
            ),
            image,
        ),
        case(
            "resize-bilinear",
            lambda a: (
                F.interpolate(a, size=(1, 9), mode="bilinear", align_corners=True),
                F.interpolate(a, scale_factor=(1.55, 0.65), mode="bilinear", align_corners=False),
                F.interpolate(a, scale_factor=(1.1, 1.55), mode="bilinear", align_corners=False),
            ),
            image,
        ),
        case(
            "padding-modes",
            lambda a: (
                F.pad(a, (-2, 3, 1, 2), "reflect"),
                F.pad(a, (3, -1, 1, 2), "replicate"),
                F.pad(a, (-2, 2, 1, -2), "circular"),
                F.pad(a, (0, -1, -2, 0)),
                aten.constant_pad_nd.default(a, (-2, 3, 1, 0), 2.5),
            ),
            image,
        ),
        case(
            "padding-typed",
            lambda a: (
                aten.constant_pad_nd.default(a, (1, -1), -7),
                F.pad(a > 0, (1, 1), value=1.0),
            ),
            ints,
        ),
        case(
            "gather-index-select",
            lambda a, idx, vec: (
                a.index_select(1, vec.int()),
                a.T.gather(0, idx),
            ),
            ints,
            torch.tensor([[2, 1], [0, 2], [1, 0]]),
            torch.tensor([2, 0, 2]),
        ),
        case(
            "advanced-index",
            lambda a, i, j: (
                aten.index.Tensor(a, [i, None, j]),
                aten.index.Tensor(a, [None, i, j]),
                aten.index.Tensor(a, [None, i, None, j]),
                aten.index.Tensor(a, [i[0, 0]]),
            ),
            image,
            i,
            j,
        ),
        case(
            "scatter",
            lambda a, ids, b: (
                a.scatter(1, ids, b),
                a.scatter_add(1, ids.flip((1,)) * 0 + 1, b),
                a.int().T.scatter_add(0, ids.T, b.int().T),
            ),
            torch.randn(3, 5),
            torch.tensor([[4, 0, 2], [1, 0, 3]]),
            torch.randn(3, 4) * 3,
        ),
        case(
            "index-put",
            lambda a, ids, j, b: (
                aten.index_put.default(a, [ids, None, j], b, False),
                aten.index_put.default(a.transpose(-1, -2), [ids, j], b[0, 0, 0, 0], True),
            ),
            image,
            torch.tensor([[-1], [0]]),
            torch.tensor([1, -2]),
            torch.randn(1, 1, 4, 7),
        ),
        case(
            "repeat-flip-roll",
            lambda a, b: (
                a.repeat(2, 1, 3),
                b.repeat(2, 5, 1),
                a.flip((-2, 0)),
                a.roll((1, -2, 11), (0, 2, 2)),
                a.transpose(1, 2).roll(-10),
            ),
            torch.randint(-9, 9, (2, 3, 4)),
            torch.randn(1, 1, 4),
        ),
        {
            "name": "transpose-conv-disjoint",
            "model": nn.ConvTranspose2d(4, 6, (2, 3), stride=(2, 3), groups=2),
            "inputs": (image,),
        },
        {
            "name": "transpose-conv-overlap",
            "model": nn.ConvTranspose2d(
                4,
                6,
                (3, 4),
                stride=(2, 3),
                padding=(2, 1),
                output_padding=(1, 2),
                dilation=(2, 1),
                groups=2,
            ),
            "inputs": (image,),
        },
        {
            "name": "transpose-conv1d",
            "model": nn.ConvTranspose1d(2, 3, 3, stride=3, output_padding=2, bias=False),
            "inputs": (torch.randn(2, 2, 9),),
        },
        {
            "name": "transpose-conv3d",
            "model": nn.ConvTranspose3d(2, 3, 2, stride=2, padding=1),
            "inputs": (torch.randn(2, 2, 3, 5, 4),),
        },
        *[
            {
                "name": f"conv-padding-{mode}",
                "model": nn.Sequential(pad, nn.Conv2d(4, 5, 3, padding=1)),
                "inputs": (image,),
            }
            for mode, pad in [
                ("reflect", nn.ReflectionPad2d((1, 2, 2, 1))),
                ("replicate", nn.ReplicationPad2d(1)),
                ("circular", nn.CircularPad2d(2)),
                ("asymmetric-zero", nn.ZeroPad2d((1, 0, 2, 1))),
                ("symmetric-zero", nn.ZeroPad2d(1)),
            ]
        ],
        {
            "name": "depthwise-reflect-padding",
            "model": nn.Conv2d(4, 8, 3, padding=1, groups=4, padding_mode="reflect"),
            "inputs": (image,),
        },
        {
            "name": "nearest-convolution",
            "model": nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(4, 5, 3, padding=1)),
            "inputs": (image,),
        },
    ]


def camera_input() -> torch.Tensor:
    """The browser paints the same opaque 8×6 RGB pattern; resize is nearest-center."""
    pixels = torch.arange(8 * 6 * 3).reshape(6, 8, 3).remainder(251).float() / 255
    # Center crop an 8×6 source to 6×6, then resize to 4×4.
    image = F.interpolate(
        pixels.permute(2, 0, 1).unsqueeze(0)[:, :, :, 1:7], size=(4, 4), mode="nearest-exact"
    )
    return (image * 2 - torch.tensor([0.1, 0.2, 0.3]).reshape(1, 3, 1, 1)) / torch.tensor(
        [0.5, 0.6, 0.7]
    ).reshape(1, 3, 1, 1)


def camera_cnn() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 4, 1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 2),
    )


@torch.library.custom_op("avel_validation::affine", mutates_args=())
def custom_affine(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x * scale + 0.25


@custom_affine.register_fake
def custom_affine_fake(x, scale):
    return torch.empty_like(x)


class CustomNetwork(torch.nn.Module):
    def forward(self, x):
        return torch.relu(custom_affine(x.transpose(0, 1), 1.75))


def models() -> list[dict[str, Any]]:
    """Whole networks. The vision models use random weights with realistic
    batch norm statistics, so no download is needed."""
    from torchvision import models as vision

    from .census import BertEncoder, LlamaBlock, TinyUNet

    image = (torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),)
    ids = (torch.randint(0, 128, (2, 17), dtype=torch.int32),)
    return [
        {
            "name": "custom-network",
            "model": CustomNetwork(),
            "inputs": (torch.randn(3, 5),),
            "custom_ops": ("avel_validation.affine.default",),
        },
        {"name": "mlp", "model": mlp(), "f16_compute": True, "inputs": (torch.randn(3, 4),)},
        {"name": "mlp-batch-17", "model": mlp(), "inputs": (torch.randn(17, 4),)},
        {"name": "camera-cnn", "model": camera_cnn(), "inputs": (camera_input(),)},
        {
            "name": "cnn",
            "model": cnn(),
            "f16_compute": True,
            "inputs": (torch.randn(2, 3, 16, 16),),
        },
        {"name": "resnet18", "model": randomize_batch_norms(vision.resnet18()), "inputs": image},
        {
            "name": "mobilenet-v3-small",
            "model": randomize_batch_norms(vision.mobilenet_v3_small()),
            "inputs": image,
        },
        {
            "name": "mobilenet-v3-large",
            "model": randomize_batch_norms(vision.mobilenet_v3_large()),
            "inputs": image,
        },
        {"name": "vit-small", "model": vit_small(), "inputs": (torch.randn(1, 3, 32, 32),)},
        {"name": "text-encoder", "model": text_encoder(), "inputs": ids},
        {
            "name": "census-llama",
            "model": LlamaBlock(),
            "f16": False,
            "inputs": (torch.randn(2, 16, 64),),
        },
        {
            "name": "census-bert",
            "model": BertEncoder(),
            "inputs": (ids[0].long(), torch.rand(2, 17) > 0.2),
        },
        {"name": "census-unet", "model": TinyUNet(), "inputs": (torch.randn(1, 3, 16, 16),)},
        {
            "name": "census-lraspp",
            "model": randomize_batch_norms(
                vision.segmentation.lraspp_mobilenet_v3_large(
                    weights=None,
                    weights_backbone=None,
                    num_classes=4,
                )
            ),
            "inputs": (torch.randn(1, 3, 48, 64),),
        },
    ]


def dynamic_models() -> list[dict[str, Any]]:
    """Models exported with a symbolic sequence length, with test vectors at
    several lengths: the browser harness compiles them per shape and through
    shape buckets."""
    seq = torch.export.Dim("seq", min=1, max=TEXT_MAX_SEQ)
    lengths = (17, 20, 40)
    return [
        {
            "name": "text-encoder-dynamic",
            "model": text_encoder(),
            "inputs": (torch.randint(0, 128, (2, lengths[0]), dtype=torch.int32),),
            "extra_test_inputs": tuple(
                (torch.randint(0, 128, (2, length), dtype=torch.int32),) for length in lengths[1:]
            ),
            "dynamic_shapes": ({1: seq},),
            "bucket": 16,
        }
    ]


def generate(out_dir: str | Path) -> dict[str, Any]:
    """Exports every case and model under `out_dir` and writes `manifest.json`.

    Shapes are static, except for the dynamic models, which carry one entry
    per test-vector length under `runs`.
    """
    out = Path(out_dir)
    for stale in ("cases", "models", "manifest.json"):
        path = out / stale
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    torch.manual_seed(0)
    entries: list[dict[str, Any]] = []
    groups = (("cases", cases()), ("models", models()), ("models", dynamic_models()))
    for kind, items in groups:
        for item in items:
            directory = out / kind / item["name"]
            export(
                item["model"],
                item["inputs"],
                directory,
                dynamic_shapes=item.get("dynamic_shapes"),
                extra_test_inputs=item.get("extra_test_inputs", ()),
                f16=item.get("f16", kind == "models"),
                int8=item.get("int8", False),
                custom_ops=item.get("custom_ops", ()),
            )
            graph = json.loads((directory / "model.graph.json").read_text())
            entry: dict[str, Any] = {
                "kind": kind.removesuffix("s"),
                "name": item["name"],
                "dir": f"{kind}/{item['name']}",
                "ops": sorted({node["op"] for node in graph["nodes"]}),
            }
            if item.get("f16_compute"):
                entry["f16Compute"] = True
            if item.get("f16") is False:
                entry["f16"] = False
            if expect_error := item.get("expect_error"):
                entry["expectError"] = expect_error
            if "dynamic_shapes" in item or "extra_test_inputs" in item:
                # One run per input set: the shapes to compile for, and the
                # test vectors file holding them.
                input_sets = (item["inputs"], *item.get("extra_test_inputs", ()))
                entry["runs"] = [
                    {
                        "shapes": {
                            value["name"]: list(tensor.shape)
                            for value, tensor in zip(graph["inputs"], inputs, strict=True)
                        },
                        "vectors": "model.testvec.safetensors"
                        if index == 0
                        else f"model.testvec-{index}.safetensors",
                    }
                    for index, inputs in enumerate(input_sets)
                ]
                if "bucket" in item:
                    entry["bucket"] = item["bucket"]
            entries.append(entry)

    covered = {op for entry in entries if entry["kind"] == "case" for op in entry["ops"]}
    missing = set(SUPPORTED_OPS) - covered
    if missing:
        raise ValueError(f"no validation case covers: {', '.join(sorted(missing))}")

    manifest = {"entries": entries}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m avel_export.validation <output-dir>", file=sys.stderr)
        return 2
    manifest = generate(args[0])
    for entry in manifest["entries"]:
        print(f"{entry['kind']} {entry['name']}: {', '.join(entry['ops'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
