import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
from safetensors.torch import load_file

from avel_export import validation
from avel_export.ops import SUPPORTED_OPS


@pytest.fixture
def small_corpus(monkeypatch):
    monkeypatch.setattr(validation, "cases", list)
    monkeypatch.setattr(validation, "models", list)
    monkeypatch.setattr(validation, "dynamic_models", list)
    monkeypatch.setattr(validation, "SUPPORTED_OPS", {"aten.relu.default"})


def test_refuses_to_publish_a_manifest_with_missing_ops(tmp_path, small_corpus):
    with torch.random.fork_rng(devices=[]):
        with pytest.raises(ValueError, match="no validation case covers: aten.relu.default"):
            validation.generate(tmp_path)
    assert not (tmp_path / "manifest.json").exists()


def test_replaces_stale_artifacts_and_records_dynamic_test_vectors(
    tmp_path, monkeypatch, small_corpus
):
    class Relu(torch.nn.Module):
        def forward(self, x):
            return x.relu()

    first = torch.tensor([[-2.0, 1.0], [0.0, 3.0]])
    second = torch.tensor([[-1.0, 4.0, -3.0], [5.0, 0.0, 2.0]])
    monkeypatch.setattr(
        validation,
        "cases",
        lambda: [{"name": "relu", "model": Relu(), "inputs": (first,)}],
    )
    monkeypatch.setattr(
        validation,
        "dynamic_models",
        lambda: [
            {
                "name": "dynamic",
                "model": Relu(),
                "inputs": (first,),
                "extra_test_inputs": ((second,),),
                "dynamic_shapes": ({1: torch.export.Dim("width", min=1, max=8)},),
                "bucket": 4,
            }
        ],
    )
    for kind in ("cases", "models"):
        (tmp_path / kind).mkdir()
        (tmp_path / kind / "stale").touch()
    (tmp_path / "manifest.json").write_text("stale")

    with torch.random.fork_rng(devices=[]):
        manifest = validation.generate(tmp_path)
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
    assert all(not (tmp_path / kind / "stale").exists() for kind in ("cases", "models"))
    case, dynamic = manifest["entries"]
    assert case == {
        "kind": "case",
        "name": "relu",
        "dir": "cases/relu",
        "ops": ["aten.relu.default"],
    }
    assert dynamic["bucket"] == 4
    assert len(dynamic["runs"]) == 2
    assert len({run["vectors"] for run in dynamic["runs"]}) == 2
    for run, tensor in zip(dynamic["runs"], (first, second), strict=True):
        assert run["shapes"] == {"x": list(tensor.shape)}
        vectors = load_file(tmp_path / dynamic["dir"] / run["vectors"])
        torch.testing.assert_close(vectors["input.x"], tensor)
        torch.testing.assert_close(vectors["output.relu"], tensor.relu())


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    with TemporaryDirectory(dir=tmp_path_factory.getbasetemp()) as directory:
        out = Path(directory)
        with torch.random.fork_rng(devices=[]):
            manifest = validation.generate(out)
        yield out, manifest


@pytest.mark.integration
def test_generates_a_runnable_case_per_supported_op(artifacts):
    out, manifest = artifacts
    covered = set()
    for entry in manifest["entries"]:
        directory = out / entry["dir"]
        for file in ("model.graph.json", "model.safetensors", "model.testvec.safetensors"):
            assert (directory / file).is_file(), f"{entry['dir']}/{file}"
        graph = json.loads((directory / "model.graph.json").read_text())
        ops = {node["op"] for node in graph["nodes"]}
        assert sorted(ops) == entry["ops"], entry["dir"]
        if entry["kind"] == "case":
            covered.update(ops)
        else:
            assert ops <= SUPPORTED_OPS | {"avel_validation.affine.default"}
        if "runs" in entry:
            continue
        # Static shapes, so the browser harness can compile without symbolic dims.
        for value in graph["inputs"] + graph["outputs"]:
            assert all(isinstance(dim, int) for dim in value["shape"]), entry["dir"]
    assert covered == set(SUPPORTED_OPS)


@pytest.mark.integration
def test_marks_non_f32_dtypes_as_expected_failures(artifacts):
    out, manifest = artifacts
    negative = [entry for entry in manifest["entries"] if "expectError" in entry]
    assert {entry["name"] for entry in negative} == {"relu-f16", "relu-i32"}
    dtypes = set()
    for entry in negative:
        graph = json.loads((out / entry["dir"] / "model.graph.json").read_text())
        dtypes.add(graph["inputs"][0]["dtype"])
    assert dtypes == {"f16", "i32"}


@pytest.mark.integration
def test_models_carry_parameters_and_test_vectors(artifacts):
    out, manifest = artifacts
    models = [entry for entry in manifest["entries"] if entry["kind"] == "model"]
    assert models
    for entry in models:
        graph = json.loads((out / entry["dir"] / "model.graph.json").read_text())
        if entry["name"] != "custom-network":
            assert graph["parameters"]
        vectors = load_file(out / entry["dir"] / "model.testvec.safetensors")
        assert vectors.keys() == {
            f"{kind}.{value['name']}" for kind in ("input", "output") for value in graph[f"{kind}s"]
        }


@pytest.mark.integration
def test_dynamic_models_carry_one_run_per_length_and_a_bucket(artifacts):
    out, manifest = artifacts
    (entry,) = [entry for entry in manifest["entries"] if entry["name"] == "text-encoder-dynamic"]
    assert entry["bucket"] == 16
    assert [run["shapes"] for run in entry["runs"]] == [
        {"ids": [2, 17]},
        {"ids": [2, 20]},
        {"ids": [2, 40]},
    ]
    assert [run["vectors"] for run in entry["runs"]] == [
        "model.testvec.safetensors",
        "model.testvec-1.safetensors",
        "model.testvec-2.safetensors",
    ]
    for run in entry["runs"]:
        assert (out / entry["dir"] / run["vectors"]).is_file()
    graph = json.loads((out / entry["dir"] / "model.graph.json").read_text())
    (ids,) = graph["inputs"]
    assert (ids["name"], ids["dtype"]) == ("ids", "i32")
    batch, length = ids["shape"]
    assert batch == 2
    assert length["name"]
    assert (length["min"], length["max"]) == (1, 64)
