import json
from pathlib import Path

from avel_export import Graph, Node, Parameter, SymbolicDim, ValueInfo

FIXTURES = Path(__file__).parents[3] / "tests" / "fixtures"


def test_graph_matches_the_shared_fixture(tmp_path):
    x = ValueInfo("x", "f32", [SymbolicDim("batch", min=1, max=64), 4])
    path = tmp_path / "model.graph.json"
    Graph(inputs=[x], outputs=[x]).write(path)
    expected = json.loads((FIXTURES / "identity.graph.json").read_text())
    assert json.loads(path.read_text()) == expected


def test_mlp_fixture_matches_format_py(tmp_path):
    """The rich shared fixture must round-trip through format.py unchanged."""
    s = SymbolicDim("s31", min=1, max=64)
    nodes = [
        Node(
            "aten.permute.default",
            ["p_0_weight"],
            [ValueInfo("permute", "f32", [4, 8])],
            {"dims": [1, 0]},
        ),
        Node(
            "aten.addmm.default",
            ["p_0_bias", "input", "permute"],
            [ValueInfo("addmm", "f32", [s, 8])],
            {"beta": 1, "alpha": 1},
        ),
        Node(
            "aten.gelu.default",
            ["addmm"],
            [ValueInfo("gelu", "f32", [s, 8])],
            {"approximate": "none"},
        ),
        Node(
            "aten.permute.default",
            ["p_2_weight"],
            [ValueInfo("permute_1", "f32", [8, 2])],
            {"dims": [1, 0]},
        ),
        Node(
            "aten.addmm.default",
            ["p_2_bias", "gelu", "permute_1"],
            [ValueInfo("addmm_1", "f32", [s, 2])],
            {"beta": 1, "alpha": 1},
        ),
        Node("aten.relu.default", ["addmm_1"], [ValueInfo("relu", "f32", [s, 2])], {}),
    ]
    parameters = [
        Parameter("p_0_weight", "0.weight", "f32", [8, 4]),
        Parameter("p_0_bias", "0.bias", "f32", [8]),
        Parameter("p_2_weight", "2.weight", "f32", [2, 8]),
        Parameter("p_2_bias", "2.bias", "f32", [2]),
    ]
    graph = Graph(
        inputs=[ValueInfo("input", "f32", [s, 4])],
        outputs=[ValueInfo("relu", "f32", [s, 2])],
        nodes=nodes,
        parameters=parameters,
    )
    path = tmp_path / "mlp.graph.json"
    graph.write(path)
    expected = json.loads((FIXTURES / "mlp.graph.json").read_text())
    assert json.loads(path.read_text()) == expected


def test_state_symbolic_attributes_and_media_fields(tmp_path):
    s = SymbolicDim("s", min=1, max=8)
    node = Node(
        "test.op.default",
        ["x", "state"],
        [ValueInfo("y", "f32", [s, 2])],
        {"rows": {"$sym": "2*s"}},
    )
    graph = Graph(
        inputs=[ValueInfo("x", "f32", [s, 2])],
        outputs=[ValueInfo("y", "f32", [s, 2])],
        nodes=[node],
        parameters=[Parameter("state", "state", "f32", [4], update="x")],
        media={"x": {"source": {"kind": "external"}}},
        metadata={"labels": ["a"]},
        f16=True,
    )
    path = tmp_path / "model.graph.json"
    graph.write(path)
    data = json.loads(path.read_text())
    assert data["parameters"] == [
        {"name": "state", "tensor": "state", "dtype": "f32", "shape": [4], "update": "x"}
    ]
    assert data["nodes"][0]["attributes"] == {"rows": {"$sym": "2*s"}}
    assert data["media"] == {"x": {"source": {"kind": "external"}}}
    assert data["metadata"] == {"labels": ["a"]}
    assert data["f16"] is True


def test_unread_outputs_are_null():
    x = ValueInfo("x", "f32", [2])
    node = Node("aten.max_pool2d_with_indices.default", ["x"], [x, None])
    assert node.to_json()["outputs"] == [x.to_json(), None]
