"""Symmetric weight-only quantization for constant GEMM operands."""

from collections import Counter, defaultdict
from dataclasses import replace
from functools import cache

import torch

from avel_export.format import Graph, Node, Parameter, ValueInfo


def quantize_weights(
    graph: Graph,
    weights: dict[str, torch.Tensor],
) -> tuple[Graph, dict[str, torch.Tensor]]:
    uses: dict[str, list[tuple[Node, int]]] = defaultdict(list)
    for node in graph.nodes:
        for slot, name in enumerate(node.inputs):
            if name is not None:
                uses[name].append((node, slot))
    outputs = {output.name for output in graph.outputs}
    # Aliased safetensors entries must retain a single storage format.
    tensors = Counter(param.tensor for param in graph.parameters)
    unique = set(tensors)
    aliases = {name for name, count in tensors.items() if count > 1}
    names = {item.name for item in (*graph.inputs, *graph.parameters)}
    names.update(out.name for node in graph.nodes for out in node.outputs if out is not None)

    @cache
    def channel_axis(name: str, rank: int) -> int | None:
        if name in outputs or not uses[name]:
            return None
        axes = set()
        for node, slot in uses[name]:
            if slot == 1 and node.op in ("aten.mm.default", "aten.bmm.default"):
                axes.add(rank - 1)
            elif slot == 2 and node.op == "aten.addmm.default":
                axes.add(rank - 1)
            elif (
                slot == 1
                and node.op == "aten.convolution.default"
                and not node.attributes["transposed"]
            ):
                axes.add(0)
            elif slot == 0 and node.op == "aten.permute.default":
                axis = channel_axis(node.outputs[0].name, rank)
                axes.add(None if axis is None else node.attributes["dims"][axis] % rank)
            else:
                return None
        return axes.pop() if len(axes) == 1 else None

    def fresh(name: str, used: set[str]) -> str:
        while name in used:
            name += "_"
        used.add(name)
        return name

    parameters = []
    decoders = []
    decoded = {}
    weights = weights.copy()
    for param in graph.parameters:
        parameters.append(param)
        if (
            param.update is not None
            or param.dtype != "f32"
            or len(param.shape) < 2
            or param.tensor in aliases
        ):
            continue
        axis = channel_axis(param.name, len(param.shape))
        if axis is None:
            continue
        weight = weights[param.tensor]
        if not weight.isfinite().all():
            continue
        reduced = tuple(i for i in range(weight.ndim) if i != axis)
        magnitude = weight.abs().amax(dim=reduced, keepdim=True)
        scale = (magnitude / 127).clamp_min(torch.finfo(torch.float32).tiny)
        scale = torch.where(magnitude == 0, 1.0, scale)
        packed = (weight / scale).round().clamp(-127, 127).to(torch.int8)
        scale_tensor = fresh(f"{param.tensor}.avel_scale", unique)
        scale_name = fresh(f"{param.name}$scale", names)
        result = fresh(f"{param.name}$dequant", names)
        weights[param.tensor] = packed
        weights[scale_tensor] = scale.reshape(-1).contiguous()
        parameters[-1] = replace(param, dtype="i8")
        parameters.append(Parameter(scale_name, scale_tensor, "f32", [param.shape[axis]]))
        decoders.append(
            Node(
                "avel.dequantize_per_channel.default",
                [param.name, scale_name],
                [ValueInfo(result, "f32", param.shape)],
                {"axis": axis},
            )
        )
        decoded[param.name] = result

    nodes = [
        replace(node, inputs=[decoded.get(name, name) for name in node.inputs])
        for node in graph.nodes
    ]
    return replace(graph, parameters=parameters, nodes=decoders + nodes), weights
