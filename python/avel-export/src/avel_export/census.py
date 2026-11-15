"""Op census: exports a zoo of small models and ranks the ops the compiler
does not support yet, by how many models need them.

`python -m avel_export.census [--write census.json] [--check census.json]`
prints the ranking. With `--check`, it fails when a model supports a smaller
share of its ops than the recorded baseline, so coverage never regresses;
`--write` records a new baseline. Models use random weights: no download.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from avel_export.export import decompositions, op_nodes
from avel_export.ops import SUPPORTED_OPS


class LlamaBlock(nn.Module):
    """A LLaMA-style decoder block: RMS norm, rotary embeddings, causal
    attention without biases, SwiGLU."""

    def __init__(self, dim: int = 64, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.RMSNorm(dim), nn.RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.gate, self.up = (
            nn.Linear(dim, 4 * dim, bias=False),
            nn.Linear(dim, 4 * dim, bias=False),
        )
        self.down = nn.Linear(4 * dim, dim, bias=False)

    def rope(self, x: torch.Tensor) -> torch.Tensor:
        seq, depth = x.shape[-2], x.shape[-1]
        frequencies = 1.0 / 10000 ** (torch.arange(0, depth, 2).float() / depth)
        angles = torch.arange(seq).float()[:, None] * frequencies[None]
        cos, sin = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
        half = depth // 2
        rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return x * cos + rotated * sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, seq, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        attention = F.scaled_dot_product_attention(self.rope(q), self.rope(k), v, is_causal=True)
        x = x + self.out(attention.transpose(1, 2).reshape(batch, seq, dim))
        h = self.norm2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class BertEncoder(nn.Module):
    """A BERT-style encoder fed token ids and an attention mask, with position
    ids computed from the mask."""

    def __init__(self, vocab: int = 128, dim: int = 64) -> None:
        super().__init__()
        self.tokens, self.positions = nn.Embedding(vocab, dim), nn.Embedding(64, dim)
        self.layer = nn.TransformerEncoderLayer(dim, 4, 4 * dim, batch_first=True, dropout=0.0)
        self.classifier = nn.Linear(dim, 2)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        positions = mask.long().cumsum(-1) - 1
        x = self.tokens(ids) + self.positions(positions.clamp(min=0))
        x = self.layer(x, src_key_padding_mask=~mask)
        return self.classifier(x[:, 0]).softmax(-1)


class TinyUNet(nn.Module):
    """A U-Net with group norms, transposed convolutions and upsampling."""

    def __init__(self) -> None:
        super().__init__()
        self.down = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU())
        self.middle = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.SiLU()
        )
        self.up = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.out = nn.Conv2d(32, 3, 3, padding=1, padding_mode="reflect")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.down(x)
        y = self.up(self.middle(skip))
        y = F.interpolate(y, scale_factor=1.0, mode="nearest")
        return self.out(torch.cat([y, skip], dim=1))


def zoo() -> dict[str, Callable[[], tuple[nn.Module, tuple[torch.Tensor, ...]]]]:
    from torchvision import models as vision
    from torchvision.models import segmentation

    image = lambda size=224: (torch.randn(1, 3, size, size),)  # noqa: E731
    ids = torch.randint(0, 128, (2, 16))
    mask = torch.ones(2, 16, dtype=torch.bool)
    return {
        "resnet18": lambda: (vision.resnet18(), image()),
        "mobilenet_v3_small": lambda: (vision.mobilenet_v3_small(), image()),
        "efficientnet_b0": lambda: (vision.efficientnet_b0(), image()),
        "squeezenet1_1": lambda: (vision.squeezenet1_1(), image()),
        "shufflenet_v2_x0_5": lambda: (vision.shufflenet_v2_x0_5(), image()),
        "regnet_y_400mf": lambda: (vision.regnet_y_400mf(), image()),
        "convnext_tiny": lambda: (vision.convnext_tiny(), image()),
        "lraspp_mobilenet_v3_large": lambda: (
            segmentation.lraspp_mobilenet_v3_large(weights_backbone=None),
            image(128),
        ),
        "llama_block": lambda: (LlamaBlock(), (torch.randn(1, 16, 64),)),
        "bert_encoder": lambda: (BertEncoder(), (ids, mask)),
        "tiny_unet": lambda: (TinyUNet(), image(32)),
    }


def external_zoo() -> dict[str, Callable[[], tuple[nn.Module, tuple[torch.Tensor, ...]]]]:
    from timm import create_model
    from transformers import BertConfig, BertModel

    class HuggingFaceBert(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = BertModel(
                BertConfig(
                    vocab_size=128,
                    hidden_size=64,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    intermediate_size=128,
                    max_position_embeddings=64,
                    return_dict=False,
                )
            )

        def forward(self, ids):
            return self.encoder(input_ids=ids)[0]

    return {
        "hf_bert_tiny": lambda: (HuggingFaceBert(), (torch.randint(0, 128, (2, 16)),)),
        "timm_mobilenetv3_small_050": lambda: (
            create_model("mobilenetv3_small_050", pretrained=False),
            (torch.randn(1, 3, 64, 64),),
        ),
    }


def ops_of(model: nn.Module, inputs: tuple[torch.Tensor, ...]) -> list[str]:
    """The ATen ops the exporter would hand the compiler for `model`."""
    program = torch.export.export(model.eval(), inputs)
    program = program.run_decompositions(decompositions())
    return sorted({str(node.target) for node in op_nodes(program)})


def census(names: list[str] | None = None, *, external: bool = False) -> dict[str, Any]:
    """Per model: its ops, those unsupported, and the supported share."""
    models = zoo()
    if external:
        models.update(external_zoo())
    result: dict[str, Any] = {}
    for name in names or list(models):
        torch.manual_seed(0)
        model, inputs = models[name]()
        ops = ops_of(model, inputs)
        unsupported = [op for op in ops if op not in SUPPORTED_OPS]
        result[name] = {
            "ops": ops,
            "unsupported": unsupported,
            "supported": round(1 - len(unsupported) / len(ops), 4),
        }
    return result


def ranking(models: dict[str, Any]) -> list[tuple[str, int]]:
    """Unsupported ops, by how many models need them."""
    counts = Counter(op for model in models.values() for op in model["unsupported"])
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def regressions(models: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """Models whose supported share fell below the baseline."""
    return [
        f"{name}: {model['supported']:.2%} supported, {baseline[name]['supported']:.2%} before"
        for name, model in models.items()
        if name in baseline and model["supported"] + 1e-9 < baseline[name]["supported"]
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m avel_export.census")
    parser.add_argument(
        "--external", action="store_true", help="include random-weight Transformers and timm models"
    )
    parser.add_argument("--write", type=Path, help="record the census as a baseline")
    parser.add_argument("--check", type=Path, help="fail on a regression against a baseline")
    parser.add_argument("models", nargs="*", help="models to export, all by default")
    args = parser.parse_args(argv)

    models = census(args.models or None, external=args.external)
    for name, model in models.items():
        print(f"{name:28} {model['supported']:7.2%}  {', '.join(model['unsupported'])}")
    print("\nunsupported ops by model count:")
    for op, count in ranking(models):
        print(f"{count:3}  {op}")
    if args.write:
        args.write.write_text(json.dumps(models, indent=2) + "\n")
    if args.check:
        if failures := regressions(models, json.loads(args.check.read_text())):
            print("\ncoverage regressed:\n" + "\n".join(failures), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
