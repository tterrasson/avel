import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
from safetensors.torch import load_file
from torch import nn
from torch.export import Dim

from avel_export import PRESERVED_OPS, SUPPORTED_OPS, ExportError, export


def mlp() -> nn.Module:
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(8, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )


def read_graph(path):
    return json.loads((path / "model.graph.json").read_text())


@pytest.fixture(scope="module")
def exported_mlp(tmp_path_factory):
    with TemporaryDirectory(dir=tmp_path_factory.getbasetemp()) as directory:
        path = Path(directory)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = mlp()
            x = torch.randn(3, 4)
            export(model, (x,), path, metadata={"labels": ["a", "b"]})
        yield path, model, x


def test_exports_a_mlp(exported_mlp):
    path, _, _ = exported_mlp
    graph = read_graph(path)

    assert graph["inputs"] == [{"name": "input", "dtype": "f32", "shape": [3, 4]}]
    assert graph["outputs"][0]["shape"] == [3, 2]
    assert graph["metadata"] == {"labels": ["a", "b"]}
    assert graph["f16"] is False
    assert {node["op"] for node in graph["nodes"]} <= SUPPORTED_OPS


def test_records_inputs_and_attributes_by_schema(exported_mlp):
    path, _, _ = exported_mlp
    nodes = {node["outputs"][0]["name"]: node for node in read_graph(path)["nodes"]}

    assert nodes["addmm"]["inputs"] == ["p_0_bias", "input", "permute"]
    assert nodes["addmm"]["attributes"] == {"beta": 1, "alpha": 1}
    assert nodes["permute"]["attributes"] == {"dims": [1, 0]}
    assert nodes["gelu"]["attributes"] == {"approximate": "none"}


def test_parameters_point_at_the_weights(exported_mlp):
    path, model, _ = exported_mlp
    graph = read_graph(path)
    weights = load_file(path / "model.safetensors")
    state = model.state_dict()

    assert {p["tensor"] for p in graph["parameters"]} == weights.keys() == state.keys()
    for parameter in graph["parameters"]:
        expected = state[parameter["tensor"]]
        assert parameter["shape"] == list(expected.shape)
        assert torch.equal(weights[parameter["tensor"]], expected)


def test_test_vectors_match_the_eval_model(exported_mlp):
    path, model, x = exported_mlp
    vectors = load_file(path / "model.testvec.safetensors")
    output = read_graph(path)["outputs"][0]["name"]

    assert torch.equal(vectors["input.input"], x)
    assert not model.training
    with torch.no_grad():
        assert torch.equal(vectors[f"output.{output}"], model(x))


def test_saves_non_contiguous_outputs(tmp_path):
    class Permute(nn.Module):
        def forward(self, x):
            return x.permute(1, 0)

    x = torch.randn(2, 3)
    export(Permute(), (x,), tmp_path)
    vectors = load_file(tmp_path / "model.testvec.safetensors")

    assert torch.equal(vectors["output.permute"], x.permute(1, 0))


def test_keeps_dynamic_dimensions_and_their_bounds(tmp_path):
    dynamic_shapes = ({0: Dim("batch", min=1, max=64)},)
    export(mlp(), (torch.randn(3, 4),), tmp_path, dynamic_shapes=dynamic_shapes)
    graph = read_graph(tmp_path)

    batch, width = graph["inputs"][0]["shape"]
    assert (batch["min"], batch["max"]) == (1, 64)
    assert width == 4
    assert graph["outputs"][0]["shape"][0] == batch


def test_leaves_unbounded_dimensions_open(tmp_path):
    dynamic_shapes = ({0: Dim("batch")},)
    export(mlp(), (torch.randn(3, 4),), tmp_path, dynamic_shapes=dynamic_shapes)

    assert "max" not in read_graph(tmp_path)["inputs"][0]["shape"][0]


def test_unpacks_multiple_outputs_into_optional_values(tmp_path):
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.MaxPool2d(2))
    export(model, (torch.randn(1, 3, 10, 10),), tmp_path)
    graph = read_graph(tmp_path)
    nodes = {node["op"]: node for node in graph["nodes"]}

    assert {node["op"] for node in graph["nodes"]} <= SUPPORTED_OPS
    norm = nodes["aten._native_batch_norm_legit_no_training.default"]
    assert norm["outputs"][1:] == [None, None]
    assert norm["inputs"][0] == "convolution"
    pool = nodes["aten.max_pool2d_with_indices.default"]
    assert pool["outputs"][0]["shape"] == [1, 4, 4, 4]
    assert pool["outputs"][1] is None
    assert pool["inputs"] == [norm["outputs"][0]["name"]]
    assert graph["outputs"][0]["name"] == pool["outputs"][0]["name"]


def test_keeps_the_slot_of_missing_optional_tensors(tmp_path):
    export(nn.Conv2d(3, 4, 3, bias=False), (torch.randn(1, 3, 8, 8),), tmp_path)
    (conv,) = read_graph(tmp_path)["nodes"]

    assert conv["inputs"] == ["input", "p_weight", None]
    assert "bias" not in conv["attributes"]
    assert conv["attributes"]["groups"] == 1


def test_serializes_enum_arguments(tmp_path):
    class Contiguous(nn.Module):
        def forward(self, x):
            return x.permute(1, 0).contiguous()

    export(Contiguous(), (torch.randn(2, 3),), tmp_path)
    clone = read_graph(tmp_path)["nodes"][-1]
    assert clone["attributes"]["memory_format"] == "torch.contiguous_format"


def test_rejects_unsupported_ops_and_writes_nothing(tmp_path):
    class Atan(nn.Module):
        def forward(self, x):
            return x.atan()

    out_dir = tmp_path / "out"
    with pytest.raises(ExportError, match="unsupported ops: aten.atan.default"):
        export(Atan(), (torch.randn(3),), out_dir)
    assert not out_dir.exists()


def test_rejects_unsupported_dtypes_and_writes_nothing(tmp_path):
    out_dir = tmp_path / "out"
    with pytest.raises(ExportError, match="unsupported dtype torch.float64"):
        export(mlp().double(), (torch.randn(3, 4).double(),), out_dir)
    assert not out_dir.exists()


def test_keeps_integers_within_javascript_precision(tmp_path):
    class Tail(nn.Module):
        def forward(self, x):
            return x[:, 1:]

    export(Tail(), (torch.randn(2, 3),), tmp_path)
    (slice_,) = read_graph(tmp_path)["nodes"]
    assert slice_["attributes"]["end"] == 2**53 - 1


def test_allows_f16_after_validating_it(tmp_path):
    model = mlp()
    export(model, (torch.ones(3, 4),), tmp_path, f16=True)
    assert read_graph(tmp_path)["f16"] is True
    assert {weight.dtype for weight in model.parameters()} == {torch.float32}
    weights = load_file(tmp_path / "model.safetensors")
    assert {weight.dtype for weight in weights.values()} == {torch.float32}


class Overflow(nn.Module):
    """Exceeds the f16 range in its intermediate, though not in its output."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * 1e5) / 1e5


def test_rejects_f16_for_models_that_drift(tmp_path):
    with pytest.raises(ExportError, match="drifts by"):
        export(Overflow(), (torch.ones(3, 4),), tmp_path, f16=True)
    assert not (tmp_path / "model.graph.json").exists()


class Sdpa(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)


def test_keeps_scaled_dot_product_attention_fused(tmp_path):
    q, k, v = torch.randn(1, 2, 9, 8), torch.randn(1, 2, 9, 8), torch.randn(1, 2, 9, 8)
    export(Sdpa(), (q, k, v), tmp_path)
    nodes = {node["op"]: node for node in read_graph(tmp_path)["nodes"]}

    assert "aten.scaled_dot_product_attention.default" in nodes
    assert not any("softmax" in op or "bmm" in op for op in nodes)
    sdpa = nodes["aten.scaled_dot_product_attention.default"]
    assert sdpa["inputs"] == [value["name"] for value in read_graph(tmp_path)["inputs"]] + [None]
    assert sdpa["attributes"]["is_causal"] is True
    assert "attn_mask" not in sdpa["attributes"]


class Cat(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, b], dim=1)


def test_flattens_tensor_list_arguments_into_inputs(tmp_path):
    export(Cat(), (torch.randn(2, 1, 8), torch.randn(2, 4, 8)), tmp_path)
    (cat,) = read_graph(tmp_path)["nodes"]
    assert cat["op"] == "aten.cat.default"
    assert cat["inputs"] == ["a", "b"]
    assert cat["attributes"] == {"dim": 1}


def test_writes_symbolic_integer_attributes_as_named_expressions(tmp_path):
    class Slice(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(8, 4)

        def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
            return self.fc(x) + pos[:, : x.shape[1]]

    dynamic = ({1: Dim("seq", min=1, max=64)}, {})
    export(
        Slice(), (torch.randn(2, 17, 8), torch.randn(1, 64, 4)), tmp_path, dynamic_shapes=dynamic
    )
    graph = read_graph(tmp_path)

    # Integer-only nodes such as `sym_size` are not ops: their values reach
    # size arguments as symbolic attributes, named after their expression.
    assert all(node["op"].startswith("aten.") for node in graph["nodes"])
    slice_ = next(node for node in graph["nodes"] if node["op"] == "aten.slice.Tensor")
    assert slice_["inputs"] == ["pos"]
    (seq_dim,) = [dim for dim in graph["inputs"][0]["shape"] if isinstance(dim, dict)]
    assert slice_["attributes"]["end"] == {"$sym": seq_dim["name"]}
    # A reshape by the flattened batch*seq size is an expression too.
    view = next(node for node in graph["nodes"] if node["op"] == "aten.view.default")
    assert view["attributes"]["size"][0] == {"$sym": f"2*{seq_dim['name']}"}


def test_writes_extra_test_vectors_for_dynamic_models(tmp_path):
    class Identity(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    export(
        Identity(),
        (torch.randn(2, 17),),
        tmp_path,
        dynamic_shapes=({1: Dim("seq", min=1, max=64)},),
        extra_test_inputs=((torch.randn(2, 40),),),
    )
    assert (tmp_path / "model.testvec.safetensors").is_file()
    extra = load_file(tmp_path / "model.testvec-1.safetensors")
    assert extra["input.x"].shape == (2, 40)
    assert extra["output.x"].shape == (2, 40)


@torch.library.custom_op("avel_tests::affine", mutates_args=())
def custom_affine(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x * scale + 0.25


@custom_affine.register_fake
def custom_affine_fake(x, scale):
    return torch.empty_like(x)


def test_custom_op_is_opaque_and_requires_explicit_allowlist(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.relu(custom_affine(x, 2.0))

    x = torch.randn(3, 5)
    with pytest.raises(ExportError, match="avel_tests.affine.default"):
        export(Model(), (x,), tmp_path)
    export(Model(), (x,), tmp_path, custom_ops=("avel_tests.affine.default",))
    graph = json.loads((tmp_path / "model.graph.json").read_text())
    node = graph["nodes"][0]
    assert node["op"] == "avel_tests.affine.default"
    assert node["attributes"] == {"scale": 2.0}
    assert node["outputs"][0]["shape"] == [3, 5]


def test_failing_extra_example_preserves_existing_export(tmp_path):
    out = tmp_path / "model"
    export(mlp(), (torch.ones(3, 4),), out)
    before = {file.name: file.read_bytes() for file in out.iterdir()}
    with pytest.raises(RuntimeError, match="mat1 and mat2 shapes cannot be multiplied"):
        export(mlp(), (torch.zeros(3, 4),), out, extra_test_inputs=((torch.zeros(3, 7),),))
    assert {file.name: file.read_bytes() for file in out.iterdir()} == before
    assert list(tmp_path.iterdir()) == [out]


def test_reexport_removes_obsolete_extra_test_vectors(tmp_path):
    export(mlp(), (torch.randn(3, 4),), tmp_path, extra_test_inputs=((torch.randn(5, 4),),))
    assert (tmp_path / "model.testvec-1.safetensors").is_file()
    export(mlp(), (torch.randn(3, 4),), tmp_path)
    assert not (tmp_path / "model.testvec-1.safetensors").exists()


def test_writes_media_settings_apart_from_metadata(tmp_path):
    media = {"input": {"source": {"kind": "external"}}}
    export(mlp(), (torch.randn(3, 4),), tmp_path, metadata={"labels": ["a"]}, media=media)
    graph = read_graph(tmp_path)
    assert graph["media"] == media
    assert graph["metadata"] == {"labels": ["a"]}


def test_rejects_non_finite_attributes(tmp_path):
    class Clamp(nn.Module):
        def forward(self, x):
            return torch.clamp(x, min=0.0, max=float("inf"))

    with pytest.raises(ExportError, match="not finite"):
        export(Clamp(), (torch.randn(3),), tmp_path)


def test_invalid_metadata_writes_no_model_files(tmp_path):
    with pytest.raises(ValueError, match="Out of range float values"):
        export(mlp(), (torch.randn(3, 4),), tmp_path / "model", metadata={"bad": float("nan")})
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("op", sorted(PRESERVED_OPS))
def test_preserved_ops_survive_decomposition(tmp_path, op):
    # A PyTorch upgrade that decomposes a preserved op must fail loudly.
    from avel_export.generation import cache_write, cached_attention, rotary, sample_topk
    from avel_export.validation import Fn

    modules = {
        "aten.adaptive_max_pool2d.default": (
            nn.AdaptiveMaxPool2d((2, 3)),
            (torch.randn(1, 2, 4, 4),),
        ),
        "aten.avg_pool3d.default": (nn.AvgPool3d(2), (torch.randn(1, 2, 4, 4, 4),)),
        "aten.max_pool3d_with_indices.default": (
            nn.MaxPool3d(2, return_indices=True),
            (torch.randn(1, 2, 4, 4, 4),),
        ),
        "aten.index_put.default": (
            Fn(lambda x, i, y: x.index_put((i,), y)),
            (torch.zeros(3), torch.tensor([1]), torch.ones(1)),
        ),
        "aten.scatter.src": (
            Fn(lambda x, i, y: x.scatter(0, i, y)),
            (torch.zeros(3), torch.tensor([1]), torch.ones(1)),
        ),
        "aten.scatter_add.default": (
            Fn(lambda x, i, y: x.scatter_add(0, i, y)),
            (torch.zeros(3), torch.tensor([1]), torch.ones(1)),
        ),
        "aten.sort.default": (Fn(lambda x: x.sort()), (torch.randn(2, 8),)),
        "aten.sort.stable": (Fn(lambda x: x.sort(stable=True)), (torch.randn(2, 8),)),
        "aten.topk.default": (Fn(lambda x: x.topk(2)), (torch.randn(2, 8),)),
        "aten.upsample_bicubic2d.vec": (
            nn.Upsample(scale_factor=2, mode="bicubic"),
            (torch.randn(1, 2, 4, 4),),
        ),
        "avel.cache_write.default": (
            Fn(cache_write),
            (torch.randn(1, 2, 4, 4), torch.randn(1, 2, 2, 4), torch.tensor(1)),
        ),
        "avel.cached_attention.default": (
            Fn(cached_attention),
            (*((torch.randn(1, 2, 3, 4),) * 3), torch.tensor(0)),
        ),
        "avel.rotary.default": (Fn(rotary), (torch.randn(2, 8), torch.randn(8), torch.randn(8))),
        "avel.sample_candidates.default": (
            Fn(lambda x, seed: sample_topk(x, seed, 2)),
            (torch.randn(2, 8), torch.tensor(0)),
        ),
        "aten.pad.default": (nn.ReflectionPad2d(1), (torch.randn(1, 2, 4, 4),)),
        "aten.roll.default": (Fn(lambda x: x.roll(1, -1)), (torch.randn(2, 8),)),
        "aten.rms_norm.default": (nn.RMSNorm(8), (torch.randn(2, 8),)),
        "aten.scaled_dot_product_attention.default": (Sdpa(), (torch.randn(1, 2, 3, 4),) * 3),
    }
    model, inputs = modules[op]
    export(model, inputs, tmp_path)
    assert op in {node["op"] for node in read_graph(tmp_path)["nodes"]}


def test_narrows_int64_values_to_32_bits(tmp_path):
    class Positions(nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = nn.Buffer(torch.tensor([3]))

        def forward(self, ids):
            return ids + self.offset + torch.arange(ids.shape[1])

    ids = torch.randint(0, 100, (2, 5))
    export(Positions(), (ids,), tmp_path)
    graph = read_graph(tmp_path)
    assert graph["inputs"][0]["dtype"] == "i64"
    assert graph["parameters"][0]["dtype"] == "i64"
    assert load_file(tmp_path / "model.safetensors")["offset"].dtype == torch.int32
    assert load_file(tmp_path / "model.testvec.safetensors")["input.ids"].dtype == torch.int64

    with pytest.raises(ExportError, match="32-bit range"):
        export(Positions(), (ids + 2**40,), tmp_path / "large")


def test_exports_mutated_buffers_as_state(tmp_path):
    class Accumulator(nn.Module):
        def __init__(self):
            super().__init__()
            self.total = nn.Buffer(torch.zeros(3))

        def forward(self, x):
            self.total.add_(x)
            return self.total * 2

    model = Accumulator()
    x = torch.ones(3)
    export(model, (x,), tmp_path, extra_test_inputs=((x,),))
    graph = read_graph(tmp_path)
    (state,) = graph["parameters"]
    assert state["tensor"] == "total"
    assert state["update"] in {node["outputs"][0]["name"] for node in graph["nodes"]}
    assert len(graph["outputs"]) == 1
    # The initial state is saved, and test vectors follow successive runs.
    assert torch.equal(load_file(tmp_path / "model.safetensors")["total"], torch.zeros(3))
    first = load_file(tmp_path / "model.testvec.safetensors")
    second = load_file(tmp_path / "model.testvec-1.safetensors")
    output = graph["outputs"][0]["name"]
    assert torch.equal(first[f"output.{output}"], torch.full((3,), 2.0))
    assert torch.equal(second[f"output.{output}"], torch.full((3,), 4.0))


def test_preserves_optional_index_slots_and_rejects_masks(tmp_path):
    class Indexed(nn.Module):
        def forward(self, x, i, j):
            return torch.ops.aten.index.Tensor(x, [i, None, j])

    x = torch.randn(3, 4, 5)
    export(Indexed(), (x, torch.tensor([0, 1]), torch.tensor([-1, 2])), tmp_path / "integer")
    (index,) = read_graph(tmp_path / "integer")["nodes"]
    assert index["inputs"] == ["x", "i", None, "j"]

    class Masked(nn.Module):
        def forward(self, x, mask):
            return x[mask]

    with pytest.raises(
        ExportError, match="boolean-mask indexing has a data-dependent output shape"
    ):
        export(Masked(), (x, torch.tensor([True, False, True])), tmp_path / "mask")
    assert not (tmp_path / "mask").exists()


def test_int8_weights_are_per_channel_and_keep_original_test_vectors(tmp_path):
    layer = nn.Linear(17, 3, bias=False)
    with torch.no_grad():
        layer.weight[0].zero_()
        layer.weight[1].copy_(torch.linspace(-0.5, 0.5, 17))
        layer.weight[2].copy_(torch.linspace(-2, 2, 17))
    x = torch.randn(4, 17)
    export(layer, (x,), tmp_path, int8=True)
    graph = read_graph(tmp_path)
    (packed, scale) = graph["parameters"]
    assert (packed["dtype"], scale["dtype"], scale["shape"]) == ("i8", "f32", [3])
    assert graph["nodes"][0]["op"] == "avel.dequantize_per_channel.default"
    weights = load_file(tmp_path / "model.safetensors")
    q = weights[packed["tensor"]]
    scales = weights[scale["tensor"]]
    assert q.dtype == torch.int8
    assert q.numel() + 4 * scales.numel() < 4 * layer.weight.numel()
    assert q[0].count_nonzero() == 0 and scales[0] == 1
    torch.testing.assert_close(scales[1:], torch.tensor([0.5, 2.0]) / 127)
    assert ((q.float() * scales[:, None] - layer.weight).abs() <= scales[:, None] * 0.501).all()
    vectors = load_file(tmp_path / "model.testvec.safetensors")
    torch.testing.assert_close(vectors[f"output.{graph['outputs'][0]['name']}"], layer(x))


class TwoHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Linear(4, 8)
        self.a = nn.Linear(8, 3)
        self.b = nn.Linear(8, 1)

    def forward(self, x):
        h = torch.tanh(self.body(x))
        return self.a(h), self.b(h).reshape(-1)


def test_output_names_rename_outputs_everywhere(tmp_path):
    torch.manual_seed(0)
    model = TwoHeads()
    export(model, (torch.randn(2, 4),), tmp_path, output_names=("logits", "value"))
    graph = read_graph(tmp_path)
    assert [o["name"] for o in graph["outputs"]] == ["logits", "value"]
    produced = [o["name"] for n in graph["nodes"] for o in n["outputs"] if o is not None]
    assert "logits" in produced and "value" in produced
    vectors = load_file(tmp_path / "model.testvec.safetensors")
    assert set(vectors) == {"input.x", "output.logits", "output.value"}


@pytest.mark.parametrize(
    "names, message",
    [
        (("logits",), "1 output names for 2 outputs"),
        (("a", "a"), "distinct"),
        (("x", "value"), "already a value"),
    ],
)
def test_invalid_output_names_write_nothing(tmp_path, names, message):
    with pytest.raises(ExportError, match=message):
        export(TwoHeads(), (torch.randn(2, 4),), tmp_path / "out", output_names=names)
    assert not (tmp_path / "out").exists()
