"""The `model.graph.json` format, mirrored from `avel-compiler`."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1
OPSET = "core-aten/1"


@dataclass(frozen=True)
class SymbolicDim:
    name: str
    min: int | None = None
    max: int | None = None

    def to_json(self) -> dict[str, Any]:
        bounds = {"min": self.min, "max": self.max}
        return {"name": self.name} | {k: v for k, v in bounds.items() if v is not None}


type Dim = int | SymbolicDim


@dataclass(frozen=True)
class ValueInfo:
    name: str
    dtype: str
    shape: list[Dim]

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": [d.to_json() if isinstance(d, SymbolicDim) else d for d in self.shape],
        }


@dataclass(frozen=True)
class Node:
    op: str
    inputs: list[str | None]
    """One slot per tensor argument in schema order, `None` for an optional tensor left out."""
    outputs: list[ValueInfo | None]
    """One entry per output, `None` for outputs the graph never reads."""
    attributes: dict[str, Any] = field(default_factory=dict)
    """Non-tensor arguments; a symbolic integer is `{"$sym": expr}`."""

    def to_json(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "inputs": self.inputs,
            "outputs": [o.to_json() if o is not None else None for o in self.outputs],
            "attributes": self.attributes,
        }


@dataclass(frozen=True)
class Parameter:
    name: str
    tensor: str
    """The safetensors tensor; `i64` tensors are stored as int32."""
    dtype: str
    shape: list[int]
    update: str | None = None
    """For persistent state, such as a buffer the model mutates: the value
    holding its new content. The tensor is the initial state."""

    def to_json(self) -> dict[str, Any]:
        parameter = {
            "name": self.name,
            "tensor": self.tensor,
            "dtype": self.dtype,
            "shape": self.shape,
        }
        if self.update is not None:
            parameter["update"] = self.update
        return parameter


@dataclass(frozen=True)
class Graph:
    inputs: list[ValueInfo]
    outputs: list[ValueInfo]
    nodes: list[Node] = field(default_factory=list)
    parameters: list[Parameter] = field(default_factory=list)
    media: dict[str, Any] = field(default_factory=dict)
    """Media input settings, by input name."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Free-form application data; the compiler skips it."""
    f16: bool = False
    """The exporter validated the model with f16 activations."""

    def to_json(self) -> dict[str, Any]:
        return {
            "formatVersion": FORMAT_VERSION,
            "opset": OPSET,
            "inputs": [i.to_json() for i in self.inputs],
            "outputs": [o.to_json() for o in self.outputs],
            "nodes": [n.to_json() for n in self.nodes],
            "parameters": [p.to_json() for p in self.parameters],
            "media": self.media,
            "metadata": self.metadata,
            "f16": self.f16,
        }

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n")
