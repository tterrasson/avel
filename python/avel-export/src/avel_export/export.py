"""Entry point of the exporter."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import operator
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind, OutputKind
from torch.fx import Node as FxNode
from torch.utils import _pytree

from avel_export.format import Dim, Graph, Node, Parameter, SymbolicDim, ValueInfo
from avel_export.ops import PRESERVED_OPS, SUPPORTED_OPS
from avel_export.quantize import quantize_weights

GRAPH_FILE = "model.graph.json"
WEIGHTS_FILE = "model.safetensors"
TEST_VECTORS_FILE = "model.testvec.safetensors"

MAX_SAFE_INTEGER = 2**53 - 1

# f16 outputs may drift from f32 by this fraction of the largest output
# magnitude, plus an absolute slack for outputs close to zero.
F16_RTOL = 1e-2
F16_ATOL = 1e-3

DTYPES = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.int32: "i32",
    # WGSL has no 64-bit integers: `i64` values are stored as `i32`, which
    # `narrow` checks they fit in.
    torch.int64: "i64",
    torch.int8: "i8",
    torch.bool: "bool",
}

INT32 = torch.iinfo(torch.int32)


class ExportError(Exception):
    """A model cannot be represented in the avel format."""


def export(
    model: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    out_dir: str | Path,
    dynamic_shapes: Any | None = None,
    metadata: dict[str, Any] | None = None,
    media: dict[str, Any] | None = None,
    f16: bool = False,
    extra_test_inputs: tuple[tuple[torch.Tensor, ...], ...] = (),
    custom_ops: tuple[str, ...] = (),
    int8: bool = False,
    output_names: tuple[str, ...] | None = None,
) -> None:
    """Writes `model.graph.json`, `model.safetensors` and test vectors to `out_dir`.

    `model` is switched to eval mode. `dynamic_shapes` follows `torch.export.export`,
    `metadata` is stored as is in the graph for the application, and `media` maps
    input names to media input settings (see `MediaInput`). With `f16`, the model is also run
    in half precision and must stay within tolerance of f32 before the graph allows
    f16 activations. `extra_test_inputs` are additional input sets, written as
    `model.testvec-1.safetensors`, ... for models exported with dynamic shapes.
    `custom_ops` explicitly allows functional torch.library ops (namespace.op.overload),
    kept opaque for a user-registered WGSL kernel. With `int8`, eligible constant
    matmul and convolution weights are quantized per output channel. Test vectors
    still describe the original model. `output_names` names the graph outputs, in
    order, in place of the generated value names. Nothing is written if the export fails.
    """
    model = model.eval()
    program = torch.export.export(model, example_inputs, dynamic_shapes=dynamic_shapes)
    program = program.run_decompositions(decompositions(custom_ops))

    check_ops(program, custom_ops)
    graph, weights = convert(program, metadata or {})
    if output_names is not None:
        graph = rename_outputs(graph, output_names)
    if int8:
        graph, weights = quantize_weights(graph, weights)
    graph = dataclasses.replace(graph, media=media or {})

    # Running the model updates its state: the f16 check starts from the
    # same initial state as the first test vectors.
    initial = copy.deepcopy(model) if f16 else model
    with torch.no_grad():
        outputs = _pytree.tree_leaves(model(*example_inputs))
    if f16:
        check_f16(initial, example_inputs, outputs)
        graph = dataclasses.replace(graph, f16=True)

    # Serialize into a staging directory first. A failing extra example or
    # invalid metadata must not overwrite an existing, usable export.
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".avel-export-", dir=out_dir.parent) as temporary:
        staging = Path(temporary)
        (staging / GRAPH_FILE).write_text(
            json.dumps(graph.to_json(), indent=2, allow_nan=False) + "\n"
        )
        save_file(weights, staging / WEIGHTS_FILE)
        save_file(test_vectors(model, graph, example_inputs, outputs), staging / TEST_VECTORS_FILE)
        for index, inputs in enumerate(extra_test_inputs, start=1):
            save_file(
                test_vectors(model, graph, inputs), staging / f"model.testvec-{index}.safetensors"
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        for file in staging.iterdir():
            file.replace(out_dir / file.name)
        for file in out_dir.glob("model.testvec-*.safetensors"):
            suffix = file.name.removeprefix("model.testvec-").removesuffix(".safetensors")
            if suffix.isdigit() and int(suffix) > len(extra_test_inputs):
                file.unlink()


def decompositions(custom_ops: tuple[str, ...] = ()) -> dict[Any, Any]:
    """The default decompositions, but for the preserved ops and the explicitly
    allowed custom operators, kept whole in the graph."""
    kept = PRESERVED_OPS | set(custom_ops)
    return {
        op: decomposition
        for op, decomposition in torch.export.default_decompositions().items()
        if str(op) not in kept
    }


def test_vectors(
    model: torch.nn.Module,
    graph: Graph,
    inputs: tuple[torch.Tensor, ...],
    outputs: list[torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """The input and output tensors of one run of `model`, keyed by value name."""
    if outputs is None:
        with torch.no_grad():
            outputs = _pytree.tree_leaves(model(*inputs))
    vectors = {}
    for prefix, values, tensors in (
        ("input", graph.inputs, inputs),
        ("output", graph.outputs, outputs),
    ):
        for value, tensor in zip(values, tensors, strict=True):
            name = f"{prefix}.{value.name}"
            if tensor.dtype == torch.int64:
                narrow(tensor, f"test vector `{name}`")
            vectors[name] = tensor.detach().contiguous().clone()
    return vectors


def narrow(tensor: torch.Tensor, what: str) -> torch.Tensor:
    """`tensor`, an int64 tensor, as int32: plans store `i64` values in 32 bits."""
    if tensor.numel() and (tensor.min() < INT32.min or tensor.max() > INT32.max):
        raise ExportError(f"{what} holds int64 values outside the 32-bit range plans store")
    return tensor.to(torch.int32)


def check_f16(
    model: torch.nn.Module, inputs: tuple[torch.Tensor, ...], expected: list[torch.Tensor]
) -> None:
    """Runs `model` in half precision and compares it to the f32 `expected` outputs."""
    half = copy.deepcopy(model).half()
    with torch.no_grad():
        actual = _pytree.tree_leaves(
            half(*(x.half() if x.is_floating_point() else x for x in inputs))
        )
    for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
        if not b.is_floating_point():
            continue
        error = (a.float() - b).abs().max().item()
        limit = F16_ATOL + F16_RTOL * b.abs().max().item()
        if not error <= limit:
            raise ExportError(
                f"output {index} drifts by {error:.3g} in f16, over the {limit:.3g} tolerance"
            )


def check_ops(program: ExportedProgram, custom_ops: tuple[str, ...] = ()) -> None:
    ops = {str(n.target) for n in op_nodes(program)}
    if any(name.startswith(("aten.", "avel.")) for name in custom_ops):
        raise ExportError("custom_ops cannot override ATen or avel operators")
    for node in op_nodes(program):
        if str(node.target) in ("aten.index.Tensor", "aten.index_put.default") and any(
            isinstance(index, FxNode) and index.meta["val"].dtype in (torch.bool, torch.uint8)
            for index in (node.args[1] if len(node.args) > 1 else node.kwargs["indices"])
        ):
            raise ExportError("boolean-mask indexing has a data-dependent output shape")
        if str(node.target) in custom_ops:
            schema = node.target._schema
            if schema.is_mutable or any(arg.alias_info is not None for arg in schema.returns):
                raise ExportError(f"custom op `{node.target}` must be functional without aliases")
            value = node.meta["val"]
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
                raise ExportError(f"custom op `{node.target}` must return one f32 tensor")
            if any(
                isinstance(value, FxNode)
                and isinstance(value.meta.get("val"), torch.Tensor)
                and value.meta["val"].dtype != torch.float32
                for value in _pytree.tree_leaves((node.args, node.kwargs))
            ):
                raise ExportError(f"custom op `{node.target}` requires f32 inputs")
    if unsupported := ops - SUPPORTED_OPS - set(custom_ops):
        raise ExportError(f"unsupported ops: {', '.join(sorted(unsupported))}")


def convert(
    program: ExportedProgram, metadata: dict[str, Any]
) -> tuple[Graph, dict[str, torch.Tensor]]:
    values = {n.name: n for n in program.graph.nodes}
    signature = program.graph_signature

    # Buffers the model mutates are persistent state: the value each one ends
    # a run with, by buffer name.
    updates: dict[str, str] = {}
    outputs: list[ValueInfo] = []
    for spec in signature.output_specs:
        if spec.kind == OutputKind.BUFFER_MUTATION:
            updates[spec.target] = spec.arg.name
        elif spec.kind == OutputKind.USER_OUTPUT:
            outputs.append(value_info(program, values[spec.arg.name]))
        else:
            raise ExportError(f"unsupported output `{spec.arg.name}` of kind {spec.kind.name}")

    inputs: list[ValueInfo] = []
    parameters: list[Parameter] = []
    weights: dict[str, torch.Tensor] = {}
    for spec in signature.input_specs:
        node = values[spec.arg.name]
        if spec.kind == InputKind.USER_INPUT:
            inputs.append(value_info(program, node))
        elif spec.kind in (InputKind.PARAMETER, InputKind.BUFFER, InputKind.CONSTANT_TENSOR):
            update = updates.get(spec.target)
            # Buffers that inference never reads, such as `num_batches_tracked`.
            if not node.users and update is None:
                continue
            value = value_info(program, node)
            tensor = program.state_dict.get(spec.target, program.constants.get(spec.target))
            tensor = tensor.detach().contiguous().clone()
            if tensor.dtype == torch.int64:
                tensor = narrow(tensor, f"tensor `{spec.target}`")
            weights[spec.target] = tensor
            parameters.append(
                Parameter(value.name, spec.target, value.dtype, list(tensor.shape), update)
            )
        else:
            raise ExportError(f"unsupported input `{spec.arg.name}` of kind {spec.kind.name}")

    nodes = [convert_node(program, n) for n in op_nodes(program)]
    return Graph(inputs, outputs, nodes, parameters, metadata=metadata), weights


def rename_outputs(graph: Graph, names: tuple[str, ...]) -> Graph:
    """`graph` with its outputs renamed, in order, wherever their values appear."""
    names = tuple(names)
    if len(names) != len(graph.outputs):
        raise ExportError(f"{len(names)} output names for {len(graph.outputs)} outputs")
    if len(set(names)) != len(names) or not all(isinstance(n, str) and n for n in names):
        raise ExportError("output names must be distinct non-empty strings")
    old = [o.name for o in graph.outputs]
    if len(set(old)) != len(old):
        raise ExportError("cannot name an output returned more than once")
    fixed = {i.name for i in graph.inputs} | {p.name for p in graph.parameters}
    if fixed.intersection(old):
        raise ExportError("cannot rename an output that is a graph input or parameter")
    mapping = dict(zip(old, names, strict=True))
    produced = {o.name for n in graph.nodes for o in n.outputs if o is not None} | fixed
    if clash := (produced - set(old)).intersection(names):
        raise ExportError(f"output name `{sorted(clash)[0]}` is already a value of the graph")

    def value(v: ValueInfo | None) -> ValueInfo | None:
        return v if v is None else dataclasses.replace(v, name=mapping.get(v.name, v.name))

    def ref(name: str | None) -> str | None:
        return name if name is None else mapping.get(name, name)

    nodes = [
        dataclasses.replace(
            n, inputs=[ref(i) for i in n.inputs], outputs=[value(o) for o in n.outputs]
        )
        for n in graph.nodes
    ]
    parameters = [dataclasses.replace(p, update=ref(p.update)) for p in graph.parameters]
    outputs = [value(o) for o in graph.outputs]
    return dataclasses.replace(graph, nodes=nodes, parameters=parameters, outputs=outputs)  # type: ignore[arg-type]


def op_nodes(program: ExportedProgram) -> list[FxNode]:
    """ATen calls that compute tensors: without the `getitem` nodes that unpack
    multiple outputs, and without integer-only nodes such as `sym_size`, whose
    values reach other nodes as symbolic attributes."""
    return [
        n
        for n in program.graph.nodes
        if n.op == "call_function" and n.target is not operator.getitem and computes_tensors(n)
    ]


def computes_tensors(node: FxNode) -> bool:
    value = node.meta.get("val")
    if isinstance(value, torch.Tensor):
        return True
    return (
        isinstance(value, (tuple, list))
        and bool(value)
        and all(isinstance(item, torch.Tensor) for item in value)
    )


def convert_node(program: ExportedProgram, node: FxNode) -> Node:
    inputs: list[str | None] = []
    attributes: dict[str, Any] = {}
    for index, argument in enumerate(node.target._schema.arguments):
        if index < len(node.args):
            value = node.args[index]
        else:
            value = node.kwargs.get(argument.name, argument.default_value)

        if isinstance(value, FxNode) and isinstance(value.meta["val"], torch.Tensor):
            inputs.append(value.name)
        elif value is None and is_optional_tensor(argument):
            # An optional tensor left out keeps its slot.
            inputs.append(None)
        elif (
            isinstance(argument.type, torch.ListType)
            and isinstance(argument.type.getElementType(), torch.OptionalType)
            and isinstance(argument.type.getElementType().getElementType(), torch.TensorType)
        ):
            # Tensor?[] (advanced indexing): nulls are unindexed axes.
            if not isinstance(value, (list, tuple)) or not all(
                item is None
                or (isinstance(item, FxNode) and isinstance(item.meta["val"], torch.Tensor))
                for item in value
            ):
                raise ExportError(f"invalid optional tensor list in `{node.name}`")
            inputs.extend(None if item is None else item.name for item in value)
        elif (
            isinstance(value, (list, tuple))
            and value
            and all(
                isinstance(item, FxNode) and isinstance(item.meta["val"], torch.Tensor)
                for item in value
            )
        ):
            # A `Tensor[]` argument, such as the list of `cat`: one input each.
            inputs.extend(item.name for item in value)
        else:
            # Non-tensor arguments, including integer-valued nodes such as
            # `sym_size` feeding a size: symbolic attributes.
            attributes[argument.name] = attribute(node, argument.name, value)
    return Node(str(node.target), inputs, node_outputs(program, node), attributes)


def is_optional_tensor(argument: Any) -> bool:
    kind = argument.type
    return isinstance(kind, torch.OptionalType) and isinstance(
        kind.getElementType(), torch.TensorType
    )


def node_outputs(program: ExportedProgram, node: FxNode) -> list[ValueInfo | None]:
    """One entry per output. Outputs of multi-output ops are read through
    `getitem` nodes, whose names they take; unread ones are `None`."""
    value = node.meta["val"]
    if not isinstance(value, (tuple, list)):
        return [value_info(program, node)]
    outputs: list[ValueInfo | None] = [None] * len(value)
    for user in node.users:
        if user.target is not operator.getitem:
            raise ExportError(f"`{user.name}` reads the outputs of `{node.name}` as a whole")
        outputs[user.args[1]] = value_info(program, user)
    return outputs


def attribute(node: FxNode, name: str, value: Any) -> Any:
    """Converts a non-tensor argument to JSON."""
    if isinstance(value, FxNode):
        # An integer-valued node such as `sym_size` feeding a size argument:
        # a symbolic size, named after its expression, as the graph dims are.
        val = value.meta["val"]
        if isinstance(val, torch.SymInt):
            return {"$sym": str(val.node.expr)}
        raise ExportError(f"argument `{name}` of `{node.name}` reads the non-tensor value `{val}`")
    if isinstance(value, torch.SymInt):
        return {"$sym": str(value.node.expr)}
    if isinstance(value, (list, tuple)):
        return [attribute(node, name, item) for item in value]
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) > MAX_SAFE_INTEGER:
        # JavaScript reads JSON numbers as doubles. Such values only mean
        # "past the end", like the `end` of a slice to the last element.
        return MAX_SAFE_INTEGER if value > 0 else -MAX_SAFE_INTEGER
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            # A decomposed attention mask uses scalar_tensor(-inf). Encode
            # that literal as a string so the graph stays valid JSON.
            if str(node.target) == "aten.scalar_tensor.default" and name == "s":
                return str(value)
            raise ExportError(f"argument `{name}` of `{node.name}` is not finite: {value}")
        return value
    if isinstance(value, (torch.memory_format, torch.dtype, torch.device, torch.layout)):
        return str(value)
    raise ExportError(
        f"argument `{name}` of `{node.name}` has unsupported type {type(value).__name__}"
    )


def value_info(program: ExportedProgram, node: FxNode) -> ValueInfo:
    value = node.meta["val"]
    if value.dtype not in DTYPES:
        raise ExportError(f"value `{node.name}` has unsupported dtype {value.dtype}")
    return ValueInfo(node.name, DTYPES[value.dtype], [dim(program, d) for d in value.shape])


def dim(program: ExportedProgram, size: int | torch.SymInt) -> Dim:
    if isinstance(size, int):
        return size
    expr = size.node.expr
    bounds = program.range_constraints.get(expr)
    if bounds is None:
        return SymbolicDim(str(expr))
    return SymbolicDim(
        str(expr),
        min=int(bounds.lower) if math.isfinite(bounds.lower) else None,
        max=int(bounds.upper) if math.isfinite(bounds.upper) else None,
    )
