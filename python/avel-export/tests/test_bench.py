import json

import pytest
import torch

from avel_export import bench


@pytest.mark.integration
def test_exports_each_model_for_avel_and_onnx(tmp_path, monkeypatch):
    import onnx

    from avel_export.validation import cnn

    # One small model exercises the real ONNX toolchain.
    monkeypatch.setattr(bench, "models", lambda: [{"name": "cnn", "model": cnn()}])
    with torch.random.fork_rng(devices=[]):
        manifest = bench.generate(tmp_path)
    assert manifest == {"entries": [{"name": "cnn", "dir": "cnn"}]}
    directory = tmp_path / "cnn"
    graph = json.loads((directory / "model.graph.json").read_text())
    assert graph["f16"] is True

    model = onnx.load(directory / bench.ONNX_FILE, load_external_data=False)
    assert [value.name for value in model.graph.input] == [
        value["name"] for value in graph["inputs"]
    ]
    assert [value.name for value in model.graph.output] == [
        value["name"] for value in graph["outputs"]
    ]
    # Weights are inline, so the browser loads a single file.
    assert model.graph.initializer
    assert all(
        tensor.data_location != onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer
    )
