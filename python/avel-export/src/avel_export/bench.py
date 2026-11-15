"""Generates the benchmark set: each model exported for avel and to ONNX.

Both exports share the same random weights, so the browser benchmark can run
avel and ONNX Runtime Web on the same network and check that they agree.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

from avel_export.export import export
from avel_export.validation import IMAGE_SIZE, randomize_batch_norms

ONNX_FILE = "model.onnx"


def models() -> list[dict[str, Any]]:
    """The step 2 models, at 224×224 with a batch of one."""
    from torchvision import models as vision

    return [
        {"name": "resnet18", "model": vision.resnet18()},
        {"name": "mobilenet-v3-small", "model": vision.mobilenet_v3_small()},
        {"name": "mobilenet-v3-large", "model": vision.mobilenet_v3_large()},
    ]


def export_onnx(model: nn.Module, inputs: tuple[torch.Tensor, ...], path: Path) -> None:
    """Writes a single-file ONNX model whose input and output names match the avel graph."""
    graph = json.loads((path.parent / "model.graph.json").read_text())
    torch.onnx.export(
        model,
        inputs,
        path,
        dynamo=True,
        external_data=False,
        input_names=[value["name"] for value in graph["inputs"]],
        output_names=[value["name"] for value in graph["outputs"]],
    )


def generate(out_dir: str | Path) -> dict[str, Any]:
    """Exports every benchmark model under `out_dir` and writes `manifest.json`."""
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)

    torch.manual_seed(0)
    entries: list[dict[str, Any]] = []
    for item in models():
        model = randomize_batch_norms(item["model"]).eval()
        inputs = (torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),)
        directory = out / item["name"]
        export(model, inputs, directory, f16=True)
        export_onnx(model, inputs, directory / ONNX_FILE)
        entries.append({"name": item["name"], "dir": item["name"]})

    manifest = {"entries": entries}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m avel_export.bench <output-dir>", file=sys.stderr)
        return 2
    for entry in generate(args[0])["entries"]:
        print(f"model {entry['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
