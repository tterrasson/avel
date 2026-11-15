use serde_json::{Value as Json, json};

use super::*;
use crate::plan::RequestedShapes;

// Direct paths into the shared fixtures: symlinks break on checkouts that
// lose them (Windows, source archives), and the CLI already reads these.
const IDENTITY: &str = include_str!("../../../../tests/fixtures/identity.graph.json");
const MLP: &str = include_str!("../../../../tests/fixtures/mlp.graph.json");

fn identity() -> Graph {
    serde_json::from_str(IDENTITY).unwrap()
}

fn mlp() -> Graph {
    serde_json::from_str(MLP).unwrap()
}

fn options(name: &str, shape: &[u64]) -> CompileOptions {
    CompileOptions {
        input_shapes: BTreeMap::from([(name.to_owned(), shape.to_vec())]),
        limits: Limits::default(),
        features: Features::default(),
        f16: false,
        f16_compute: false,
        bucket: None,
        media: BTreeMap::new(),
        custom_kernels: BTreeMap::new(),
        gpu_class: crate::GpuClass::default(),
        tiles: BTreeMap::new(),
    }
}

fn value(name: &str, shape: &[u64]) -> Json {
    json!({"name": name, "dtype": "f32", "shape": shape})
}

/// A node; `inputs` lists one slot per tensor argument, `null` when absent.
fn node(
    op: &str,
    inputs: impl serde::Serialize,
    output: &str,
    shape: &[u64],
    attributes: Json,
) -> Json {
    json!({"op": op, "inputs": inputs, "outputs": [value(output, shape)], "attributes": attributes})
}

/// A graph with input `x`, f32 parameters and the given nodes and outputs.
fn graph(
    x: &[u64],
    parameters: &[(&str, &[u64])],
    nodes: Vec<Json>,
    outputs: &[(&str, &[u64])],
) -> Graph {
    let parameters: Vec<Json> = parameters
        .iter()
        .map(|(name, shape)| json!({"name": name, "tensor": name, "dtype": "f32", "shape": shape}))
        .collect();
    let outputs: Vec<Json> = outputs
        .iter()
        .map(|(name, shape)| value(name, shape))
        .collect();
    serde_json::from_value(json!({
        "formatVersion": FORMAT_VERSION,
        "opset": "core-aten/1",
        "inputs": [value("x", x)],
        "outputs": outputs,
        "nodes": nodes,
        "parameters": parameters,
    }))
    .unwrap()
}

fn compile_x(graph: &Graph, x: &[u64]) -> Plan {
    compile(graph, &options("x", x)).unwrap()
}

fn typed(name: &str, dtype: &str, shape: &[u64]) -> Json {
    json!({"name": name, "dtype": dtype, "shape": shape})
}

/// A node whose outputs are given as values, `null` for those left unread.
fn node_with(op: &str, inputs: Json, outputs: Json, attributes: Json) -> Json {
    json!({"op": op, "inputs": inputs, "outputs": outputs, "attributes": attributes})
}

fn kernel_ops(plan: &Plan) -> Vec<Vec<String>> {
    plan.dispatches.iter().map(|d| d.ops.clone()).collect()
}

/// A graph with the single input `input`, no parameters, and `outputs`.
fn graph_with_input(input: Json, nodes: Vec<Json>, outputs: Vec<Json>) -> Graph {
    serde_json::from_value(json!({
        "formatVersion": FORMAT_VERSION,
        "opset": "core-aten/1",
        "inputs": [input],
        "outputs": outputs,
        "nodes": nodes,
        "parameters": [],
    }))
    .unwrap()
}

fn conv_bn_relu() -> Graph {
    let conv = json!({"stride": [1, 1], "padding": [1, 1], "dilation": [1, 1],
                          "transposed": false, "output_padding": [0, 0], "groups": 1});
    let bn = json!({"momentum": 0.1, "eps": 1e-5});
    graph(
        &[1, 3, 8, 8],
        &[
            ("w", &[4, 3, 3, 3]),
            ("gamma", &[4]),
            ("beta", &[4]),
            ("mean", &[4]),
            ("var", &[4]),
        ],
        vec![
            node(
                "aten.convolution.default",
                json!(["x", "w", null]),
                "conv",
                &[1, 4, 8, 8],
                conv,
            ),
            json!({
                "op": "aten._native_batch_norm_legit_no_training.default",
                "inputs": ["conv", "gamma", "beta", "mean", "var"],
                "outputs": [value("bn", &[1, 4, 8, 8]), null, null],
                "attributes": bn,
            }),
            node(
                "aten.relu.default",
                ["bn"],
                "relu",
                &[1, 4, 8, 8],
                json!({}),
            ),
        ],
        &[("relu", &[1, 4, 8, 8])],
    )
}

#[test]
fn compiles_a_forwarding_graph() {
    let plan = compile(&identity(), &options("x", &[2, 4])).unwrap();
    assert_eq!(plan.inputs, plan.outputs);
    assert_eq!(plan.outputs[0].shape, [2, 4]);
    assert!(plan.dispatches.is_empty());
}

#[test]
fn compiles_a_mlp_into_one_dispatch_per_layer() {
    let graph = mlp();
    let plan = compile(&graph, &options("input", &[3, 4])).unwrap();

    // Transposes are views, and activations are matmul epilogues.
    let addmms: Vec<&Node> = graph
        .nodes
        .iter()
        .filter(|node| node.op == "aten.addmm.default")
        .collect();
    assert_eq!(plan.dispatches.len(), addmms.len());
    for (dispatch, addmm) in plan.dispatches.iter().zip(&addmms) {
        assert_eq!(dispatch.ops.len(), 2);
        assert_eq!(dispatch.ops[0], addmm.outputs[0].as_ref().unwrap().name);
    }
    assert!(plan.prepare.is_empty());
    assert_eq!(plan.outputs[0].shape, [3, 2]);

    let output = &plan.buffers[plan.outputs[0].buffer as usize];
    assert_eq!(output.role, BufferRole::Output);
    assert_eq!(output.size, 3 * 2 * 4);
    // The hidden layer lives in the arena.
    let arena = plan.buffers.last().unwrap();
    assert_eq!(arena.role, BufferRole::Arena);
    assert_eq!(arena.size, 3 * 8 * 4);
    let parameters = plan
        .buffers
        .iter()
        .filter(|buffer| matches!(buffer.role, BufferRole::Parameter { .. }))
        .count();
    assert_eq!(parameters, graph.parameters.len());
}

#[test]
fn binds_inputs_before_the_output() {
    let plan = compile(&mlp(), &options("input", &[3, 4])).unwrap();
    for dispatch in &plan.dispatches {
        let output = dispatch.bindings.last().unwrap();
        assert_eq!(plan.buffers[output.buffer as usize].size, output.size);
        let role = &plan.buffers[output.buffer as usize].role;
        assert!(matches!(role, BufferRole::Arena | BufferRole::Output));
    }
}

#[test]
fn folds_weight_only_subgraphs() {
    let graph = graph(
        &[2, 3],
        &[("w", &[3])],
        vec![
            node(
                "aten.add.Tensor",
                ["w"],
                "shifted",
                &[3],
                json!({"other": 1, "alpha": 1}),
            ),
            node(
                "aten.mul.Tensor",
                ["x", "shifted"],
                "out",
                &[2, 3],
                json!({}),
            ),
        ],
        &[("out", &[2, 3])],
    );
    let plan = compile_x(&graph, &[2, 3]);

    assert_eq!(plan.prepare.len(), 1);
    assert_eq!(plan.dispatches.len(), 1);
    let folded = plan.prepare[0].bindings.last().unwrap().buffer as usize;
    assert_eq!(plan.buffers[folded].role, BufferRole::Constant);
    assert!(
        plan.dispatches[0]
            .bindings
            .iter()
            .any(|binding| binding.buffer as usize == folded)
    );
}

#[test]
fn folds_batch_norms_into_convolutions() {
    let plan = compile_x(&conv_bn_relu(), &[1, 3, 8, 8]);

    // Scale, shift and the scaled weight are computed once from the
    // parameters; the shift becomes the convolution bias.
    let prepared: Vec<&str> = plan
        .prepare
        .iter()
        .map(|dispatch| dispatch.ops[0].as_str())
        .collect();
    assert_eq!(prepared, ["bn.scale", "bn.shift", "bn.weight"]);
    // Convolution, normalization, relu and the transposition back to
    // row-major run as one dispatch, which writes the output directly.
    let [dispatch] = &plan.dispatches[..] else {
        panic!("{:?}", plan.dispatches)
    };
    assert_eq!(dispatch.ops, ["conv", "bn", "relu", "relu"]);
    assert_eq!(dispatch.bindings.len(), 4);
    let output = plan.outputs[0].buffer;
    assert_eq!(dispatch.bindings[3].buffer, output);
    assert_eq!(plan.buffers[output as usize].role, BufferRole::Output);
    assert!(
        plan.buffers
            .iter()
            .all(|buffer| buffer.role != BufferRole::Arena)
    );
}

#[test]
fn splits_long_reductions_and_fuses_the_epilogue_into_their_sum() {
    let conv = json!({"stride": [1, 1], "padding": [1, 1], "dilation": [1, 1],
                          "transposed": false, "output_padding": [0, 0], "groups": 1});
    let bn = json!({"momentum": 0.1, "eps": 1e-5});
    let shape = [1, 512, 7, 7];
    let graph = graph(
        &[1, 256, 7, 7],
        &[("w", &[512, 256, 3, 3]), ("mean", &[512]), ("var", &[512])],
        vec![
            node(
                "aten.convolution.default",
                json!(["x", "w", null]),
                "conv",
                &shape,
                conv,
            ),
            json!({
                "op": "aten._native_batch_norm_legit_no_training.default",
                "inputs": ["conv", null, null, "mean", "var"],
                "outputs": [value("bn", &shape), null, null],
                "attributes": bn,
            }),
            node("aten.relu.default", ["bn"], "relu", &shape, json!({})),
        ],
        &[("relu", &shape)],
    );
    let plan = compile_x(&graph, &[1, 256, 7, 7]);

    let ops: Vec<&[String]> = plan
        .dispatches
        .iter()
        .map(|dispatch| &dispatch.ops[..])
        .collect();
    assert_eq!(
        ops,
        [
            &["conv", "bn"][..],
            &["conv.sum", "conv.bias", "relu", "relu"]
        ]
    );
    // 8 slices of partial sums, in the arena, from 32×64 tiles.
    let partial = plan.dispatches[0].bindings.last().unwrap();
    assert_eq!(
        plan.buffers[partial.buffer as usize].role,
        BufferRole::Arena
    );
    assert_eq!(partial.size, 8 * 512 * 49 * 4);
    assert_eq!(plan.dispatches[0].workgroups, [2, 8 * 8, 1]);
}

#[test]
fn keeps_batch_norms_whose_input_is_read_elsewhere() {
    let mut graph = conv_bn_relu();
    graph
        .outputs
        .push(serde_json::from_value(value("conv", &[1, 4, 8, 8])).unwrap());
    let plan = compile_x(&graph, &[1, 3, 8, 8]);
    // Scale, shift, and the weight packed output channels innermost.
    let prepared: Vec<&str> = plan
        .prepare
        .iter()
        .map(|dispatch| dispatch.ops[0].as_str())
        .collect();
    assert_eq!(prepared, ["bn.scale", "bn.shift", "conv.packed"]);
    // The convolution, normalization and relu fused with the copy of
    // their output back to row-major, then the copy of the convolution.
    let ops: Vec<&[String]> = plan
        .dispatches
        .iter()
        .map(|dispatch| &dispatch.ops[..])
        .collect();
    assert_eq!(ops, [&["conv"][..], &["bn", "relu", "relu"], &["conv"]]);
}

#[test]
fn copies_only_the_reshapes_strides_cannot_express() {
    let graph = graph(
        &[2, 3, 4],
        &[],
        vec![
            node(
                "aten.permute.default",
                ["x"],
                "t",
                &[4, 3, 2],
                json!({"dims": [2, 1, 0]}),
            ),
            node(
                "aten.view.default",
                ["t"],
                "split",
                &[2, 2, 3, 2],
                json!({"size": [2, 2, 3, 2]}),
            ),
            node(
                "aten.view.default",
                ["t"],
                "flat",
                &[24],
                json!({"size": [-1]}),
            ),
            node(
                "aten.relu.default",
                ["split"],
                "relu",
                &[2, 2, 3, 2],
                json!({}),
            ),
        ],
        &[("flat", &[24]), ("relu", &[2, 2, 3, 2])],
    );
    let plan = compile_x(&graph, &[2, 3, 4]);

    // One copy for `flat`, one kernel for `relu`: `t` and `split` are views.
    assert_eq!(plan.dispatches.len(), 2);
    assert_eq!(plan.outputs[0].shape, [24]);
}

#[test]
fn copies_views_returned_as_outputs() {
    let graph = graph(
        &[4, 6],
        &[],
        vec![
            node(
                "aten.slice.Tensor",
                ["x"],
                "rows",
                &[2, 6],
                json!({"dim": 0, "start": 1, "end": 3, "step": 1}),
            ),
            node(
                "aten.select.int",
                ["x"],
                "first",
                &[6],
                json!({"dim": 0, "index": 0}),
            ),
            node("aten.clone.default", ["x"], "same", &[4, 6], json!({})),
        ],
        &[("rows", &[2, 6]), ("first", &[6]), ("same", &[4, 6])],
    );
    let plan = compile_x(&graph, &[4, 6]);

    assert_eq!(plan.dispatches.len(), 2);
    assert_eq!(plan.outputs[2].buffer, plan.inputs[0].buffer);
}

#[test]
fn shares_identical_shaders() {
    let graph = graph(
        &[2, 3],
        &[],
        vec![
            node("aten.relu.default", ["x"], "a", &[2, 3], json!({})),
            node("aten.relu.default", ["x"], "b", &[2, 3], json!({})),
        ],
        &[("a", &[2, 3]), ("b", &[2, 3])],
    );
    let plan = compile_x(&graph, &[2, 3]);
    assert_eq!(plan.dispatches.len(), 2);
    assert_eq!(plan.shaders.len(), 1);
}

#[test]
fn rejects_reading_unsupported_outputs() {
    // The statistics of a layer norm are not computed.
    let graph = graph(
        &[2, 8],
        &[],
        vec![json!({
            "op": "aten.native_layer_norm.default",
            "inputs": ["x", null, null],
            "outputs": [value("ln", &[2, 8]), value("mean", &[2, 1]), null],
            "attributes": {"normalized_shape": [8], "eps": 1e-5},
        })],
        &[("ln", &[2, 8])],
    );
    let error = compile(&graph, &options("x", &[2, 8])).unwrap_err();
    assert!(
        matches!(error, CompileError::InvalidOp { message, .. } if message == "output 1 is not supported")
    );
}

#[test]
fn places_intermediates_in_one_arena_by_lifetime() {
    // Each relu is read by two sums, so nothing fuses, and all three stay
    // alive until the sums at the end.
    let mut nodes = Vec::new();
    let mut outputs = Vec::new();
    for index in 0..3 {
        let name = format!("r{index}");
        nodes.push(node("aten.relu.default", ["x"], &name, &[100], json!({})));
        outputs.push(format!("s{index}"));
    }
    for (index, output) in outputs.iter().enumerate() {
        let (a, b) = (format!("r{index}"), format!("r{}", (index + 1) % 3));
        let attributes = json!({"alpha": 1});
        nodes.push(node(
            "aten.add.Tensor",
            [&a, &b],
            output,
            &[100],
            attributes,
        ));
    }
    let outputs: Vec<(&str, &[u64])> = outputs
        .iter()
        .map(|name| (name.as_str(), &[100][..]))
        .collect();
    let plan = compile_x(&graph(&[100], &[], nodes, &outputs), &[100]);

    let arena_index = plan.buffers.len() - 1;
    let arena = &plan.buffers[arena_index];
    assert_eq!(arena.role, BufferRole::Arena);
    let mut ranges: Vec<(u64, u64)> = plan
        .dispatches
        .iter()
        .flat_map(|dispatch| &dispatch.bindings)
        .filter(|binding| binding.buffer as usize == arena_index)
        .map(|binding| (binding.offset, binding.offset + binding.size))
        .collect();
    ranges.sort_unstable();
    ranges.dedup();
    assert_eq!(ranges, [(0, 400), (512, 912), (1024, 1424)]);
    assert_eq!(arena.size, 1424);
}

#[test]
fn reuses_arena_memory_between_dead_tensors() {
    let conv = json!({"stride": [1], "padding": [1], "dilation": [1],
                          "transposed": false, "output_padding": [0], "groups": 1});
    let layer = |input: &str, output: &str| {
        node(
            "aten.convolution.default",
            json!([input, "w", null]),
            output,
            &[1, 4, 16],
            conv.clone(),
        )
    };
    let graph = graph(
        &[1, 4, 16],
        &[("w", &[4, 4, 3])],
        vec![
            layer("x", "c0"),
            layer("c0", "c1"),
            layer("c1", "c2"),
            layer("c2", "c3"),
        ],
        &[("c3", &[1, 4, 16])],
    );
    let plan = compile_x(&graph, &[1, 4, 16]);
    assert_eq!(plan.dispatches.len(), 4);
    // At most two intermediates are alive at once: `c2` reuses `c0`'s memory.
    let arena = plan.buffers.last().unwrap();
    assert_eq!(arena.role, BufferRole::Arena);
    assert_eq!(arena.size, 256 + 4 * 16 * 4);
}

#[test]
fn stores_intermediates_in_f16_when_allowed() {
    let mut graph = conv_bn_relu();
    let mut options = options("x", &[1, 3, 8, 8]);
    options.f16 = true;
    assert!(matches!(
        compile(&graph, &options),
        Err(CompileError::F16Unavailable(_))
    ));
    options.features.shader_f16 = true;
    assert!(matches!(
        compile(&graph, &options),
        Err(CompileError::F16Unavailable(_))
    ));

    graph.f16 = true;
    let plan = compile(&graph, &options).unwrap();
    let wgsl = &plan.shaders[plan.dispatches[0].shader as usize].wgsl;
    // The folded weight and bias are f16; input and output stay f32.
    assert!(wgsl.starts_with("enable f16;"), "{wgsl}");
    assert!(wgsl.contains("weight: array<f16>"), "{wgsl}");
    assert!(wgsl.contains("src: array<f32>"), "{wgsl}");
    assert!(wgsl.contains("dst: array<f32>"), "{wgsl}");
    let weight = &plan.dispatches[0].bindings[1];
    assert_eq!(weight.size, 4 * 3 * 3 * 3 * 2);
}

#[test]
fn rejects_shapes_outside_the_declared_bounds() {
    let error = compile(&identity(), &options("x", &[65, 4])).unwrap_err();
    assert!(matches!(error, CompileError::DimMismatch { axis: 0, .. }));
}

#[test]
fn rejects_a_wrong_rank() {
    let error = compile(&identity(), &options("x", &[4])).unwrap_err();
    assert!(matches!(error, CompileError::RankMismatch { .. }));
}

#[test]
fn rejects_an_unknown_format_version() {
    let mut graph = identity();
    graph.format_version = FORMAT_VERSION + 1;
    let error = compile(&graph, &options("x", &[2, 4])).unwrap_err();
    assert!(matches!(error, CompileError::UnsupportedFormat { .. }));
}

#[test]
fn rejects_unsupported_ops() {
    let mut graph = mlp();
    graph.nodes[0].op = "aten.atan.default".to_owned();
    let error = compile(&graph, &options("input", &[3, 4])).unwrap_err();
    assert_eq!(
        error,
        CompileError::UnsupportedOp("aten.atan.default".to_owned())
    );
}

#[test]
fn rejects_shapes_that_disagree_with_the_export() {
    let mut graph = mlp();
    graph.nodes[0].outputs[0].as_mut().unwrap().shape.swap(0, 1);
    let error = compile(&graph, &options("input", &[3, 4])).unwrap_err();
    assert!(matches!(error, CompileError::DimMismatch { .. }));
}

#[test]
fn rejects_unknown_values() {
    let mut graph = mlp();
    graph.nodes[0].inputs[0] = Some("missing".to_owned());
    let error = compile(&graph, &options("input", &[3, 4])).unwrap_err();
    assert_eq!(error, CompileError::UnknownValue("missing".to_owned()));
}

#[test]
fn rejects_dtypes_an_op_does_not_accept() {
    // An i32 weight, permuted (views keep their dtype), then multiplied.
    let mut graph = mlp();
    graph.parameters[0].dtype = Dtype::I32;
    let weight = graph.parameters[0].name.clone();
    for node in &mut graph.nodes {
        if node.op == "aten.permute.default" && node.inputs[0].as_deref() == Some(&weight) {
            node.outputs[0].as_mut().unwrap().dtype = Dtype::I32;
        }
    }
    let error = compile(&graph, &options("input", &[3, 4])).unwrap_err();
    assert!(
        matches!(&error, CompileError::InvalidOp { message, .. } if message.contains("does not accept")),
        "{error}"
    );
}

#[test]
fn forwards_integer_and_bool_values() {
    for dtype in [Dtype::I32, Dtype::I64, Dtype::Bool] {
        let mut graph = identity();
        graph.inputs[0].dtype = dtype;
        graph.outputs[0].dtype = dtype;
        let plan = compile(&graph, &options("x", &[2, 4])).unwrap();
        assert_eq!(plan.outputs[0].dtype, dtype);
        assert_eq!(
            plan.buffers[plan.outputs[0].buffer as usize].size,
            2 * 4 * 4
        );
    }

    // A parameter returned as is.
    let mut graph = mlp();
    graph.parameters[0].dtype = Dtype::I64;
    graph.nodes.clear();
    graph.outputs[0].name = graph.parameters[0].name.clone();
    graph.outputs[0].dtype = Dtype::I64;
    graph.outputs[0].shape = graph.parameters[0]
        .shape
        .iter()
        .map(|&size| crate::Dim::Fixed(size))
        .collect();
    let plan = compile(&graph, &options("input", &[3, 4])).unwrap();
    assert_eq!(plan.outputs[0].dtype, Dtype::I64);
}

#[test]
fn rejects_storage_only_dtypes() {
    let mut graph = identity();
    graph.inputs[0].dtype = Dtype::F16;
    graph.outputs[0].dtype = Dtype::F16;
    let error = compile(&graph, &options("x", &[2, 4])).unwrap_err();
    assert!(matches!(error, CompileError::UnsupportedDtype { .. }));

    let mut graph = mlp();
    graph.parameters[0].dtype = Dtype::I8;
    let error = compile(&graph, &options("input", &[3, 4])).unwrap_err();
    assert!(matches!(error, CompileError::UnsupportedDtype { .. }));
}

#[test]
fn rejects_dtypes_that_disagree_with_the_export() {
    let mut graph = mlp();
    graph.nodes[0].outputs[0].as_mut().unwrap().dtype = Dtype::I64;
    let error = compile(&graph, &options("input", &[3, 4])).unwrap_err();
    assert!(matches!(error, CompileError::DtypeMismatch { .. }));
}

#[test]
fn rejects_empty_tensors() {
    let mut graph = identity();
    graph.inputs[0].shape[0] = crate::Dim::Fixed(0);
    let error = compile(&graph, &options("x", &[0, 4])).unwrap_err();
    assert_eq!(error, CompileError::EmptyTensor("x".to_owned()));
}

#[test]
fn rejects_individual_buffers_over_the_buffer_limit() {
    let mut options = options("input", &[3, 4]);
    options.limits.max_buffer_size = 64;
    let error = compile(&mlp(), &options).unwrap_err();
    assert_eq!(
        error,
        CompileError::BufferTooLarge {
            name: "p_0_weight".to_owned(),
            size: 8 * 4 * 4,
            limit: 64
        }
    );
}

#[test]
fn compiles_layer_norm_attention_and_cat() {
    let graph = graph(
        &[2, 17, 64],
        &[("gamma", &[64]), ("beta", &[64]), ("cls", &[1, 1, 64])],
        vec![
            json!({
                "op": "aten.native_layer_norm.default",
                "inputs": ["x", "gamma", "beta"],
                "outputs": [value("ln", &[2, 17, 64]), null, null],
                "attributes": {"normalized_shape": [64], "eps": 1e-5},
            }),
            node("aten.relu.default", ["ln"], "relu", &[2, 17, 64], json!({})),
            node(
                "aten.view.default",
                ["x"],
                "heads",
                &[2, 17, 1, 64],
                json!({"size": [2, 17, 1, 64]}),
            ),
            json!({
                "op": "aten.scaled_dot_product_attention.default",
                "inputs": ["heads", "heads", "heads", null],
                "outputs": [value("attn", &[2, 17, 1, 64])],
                "attributes": {"dropout_p": 0, "is_causal": true, "scale": null},
            }),
            node(
                "aten.expand.default",
                ["cls"],
                "cls_b",
                &[2, 1, 64],
                json!({"size": [2, 1, -1]}),
            ),
            json!({
                "op": "aten.cat.default",
                "inputs": ["cls_b", "relu"],
                "outputs": [value("cat", &[2, 18, 64])],
                "attributes": {"dim": 1},
            }),
        ],
        &[("cat", &[2, 18, 64]), ("attn", &[2, 17, 1, 64])],
    );
    let plan = compile_x(&graph, &[2, 17, 64]);

    let ops: Vec<&[String]> = plan
        .dispatches
        .iter()
        .map(|dispatch| &dispatch.ops[..])
        .collect();
    // The relu fuses into the layer norm epilogue; attention and cat stand alone.
    assert_eq!(ops, [&["ln", "relu"][..], &["attn"][..], &["cat"][..]]);
    let attention = &plan.shaders[plan.dispatches[1].shader as usize].wgsl;
    assert!(
        attention.contains("var acc: array<f32, 64>;"),
        "{attention}"
    );
    assert!(attention.contains("let key_bound = min("), "{attention}");
    let cat = &plan.shaders[plan.dispatches[2].shader as usize].wgsl;
    assert!(cat.contains("if (o1 < 1u)"), "{cat}");
    // The local coordinate is parenthesized before multiplying by the stride.
    assert!(cat.contains("(o1 - 1u) * 64u"), "{cat}");
    // Every generated shader passes naga.
    for shader in &plan.shaders {
        crate::codegen::tests::validate_wgsl(&shader.wgsl);
    }
}

#[test]
fn compiles_an_embedding_with_i32_indices() {
    let graph: Graph = serde_json::from_value(json!({
        "formatVersion": FORMAT_VERSION,
        "opset": "core-aten/1",
        "inputs": [{"name": "ids", "dtype": "i32", "shape": [2, 17]}],
        "outputs": [value("emb", &[2, 17, 64])],
        "nodes": [{
            "op": "aten.embedding.default",
            "inputs": ["weight", "ids"],
            "outputs": [value("emb", &[2, 17, 64])],
            "attributes": {"padding_idx": 0, "scale_grad_by_freq": false, "sparse": false},
        }],
        "parameters": [{"name": "weight", "tensor": "weight", "dtype": "f32", "shape": [128, 64]}],
    }))
    .unwrap();
    let plan = compile(&graph, &options("ids", &[2, 17])).unwrap();

    assert_eq!(plan.inputs[0].dtype, Dtype::I32);
    let input = &plan.buffers[plan.inputs[0].buffer as usize];
    assert_eq!(input.dtype, Dtype::I32);
    assert_eq!(input.size, 2 * 17 * 4);
    assert_eq!(plan.outputs[0].shape, [2, 17, 64]);
    let wgsl = &plan.shaders[plan.dispatches[0].shader as usize].wgsl;
    assert!(wgsl.contains("indices: array<u32>"), "{wgsl}");
    assert!(!wgsl.contains("select("), "{wgsl}");
}

#[test]
fn snaps_symbolic_inputs_to_buckets_and_records_the_requested_shapes() {
    let mut opts = options("x", &[17, 4]);
    opts.bucket = Some(16);
    let plan = compile(&identity(), &opts).unwrap();
    assert_eq!(plan.inputs[0].shape, [32, 4]);
    assert_eq!(plan.outputs[0].shape, [32, 4]);
    assert_eq!(
        plan.buffers[plan.inputs[0].buffer as usize].size,
        32 * 4 * 4
    );
    assert_eq!(
        plan.requested,
        Some(RequestedShapes {
            inputs: vec![vec![17, 4]],
            outputs: vec![vec![17, 4]],
        })
    );

    // A multiple of the bucket is not padded; no bucket means no `requested`.
    let mut exact = options("x", &[32, 4]);
    exact.bucket = Some(16);
    let plan = compile(&identity(), &exact).unwrap();
    assert_eq!(plan.inputs[0].shape, [32, 4]);
    assert_eq!(plan.requested.as_ref().unwrap().inputs, [vec![32, 4]]);
    let plan = compile(&identity(), &options("x", &[17, 4])).unwrap();
    assert_eq!(plan.requested, None);
    assert_eq!(plan.inputs[0].shape, [17, 4]);

    // The bucket is capped by the declared maximum.
    let mut opts = options("x", &[60, 4]);
    opts.bucket = Some(32);
    let plan = compile(&identity(), &opts).unwrap();
    assert_eq!(plan.inputs[0].shape, [64, 4]);

    // Out of bounds still fails, and a zero bucket is rejected.
    let mut opts = options("x", &[65, 4]);
    opts.bucket = Some(16);
    assert!(matches!(
        compile(&identity(), &opts),
        Err(CompileError::DimMismatch { .. })
    ));
    let mut opts = options("x", &[17, 4]);
    opts.bucket = Some(0);
    assert!(matches!(
        compile(&identity(), &opts),
        Err(CompileError::InvalidOp { .. })
    ));
}

#[test]
fn resolves_symbolic_attributes_from_the_compiled_input_shapes() {
    let graph: Graph = serde_json::from_value(json!({
        "formatVersion": FORMAT_VERSION,
        "opset": "core-aten/1",
        "inputs": [{"name": "x", "dtype": "f32", "shape": [{"name": "s", "min": 1, "max": 64}, 4]}],
        "outputs": [{"name": "flat", "dtype": "f32", "shape": [{"name": "s"}, 4]}],
        "nodes": [{
            "op": "aten.view.default",
            "inputs": ["x"],
            "outputs": [{"name": "flat", "dtype": "f32", "shape": [{"name": "s"}, 4]}],
            "attributes": {"size": [{"$sym": "s"}, {"$sym": "2*2"}]},
        }],
        "parameters": [],
    }))
    .unwrap();
    let plan = compile(&graph, &options("x", &[17, 4])).unwrap();
    assert_eq!(plan.outputs[0].shape, [17, 4]);

    // With a bucket, the symbols resolve against the snapped shape.
    let mut options = options("x", &[17, 4]);
    options.bucket = Some(16);
    let plan = compile(&graph, &options).unwrap();
    assert_eq!(plan.outputs[0].shape, [32, 4]);
    assert_eq!(
        plan.requested,
        Some(RequestedShapes {
            inputs: vec![vec![17, 4]],
            outputs: vec![vec![17, 4]],
        })
    );
}

#[test]
fn rejects_buffers_over_the_binding_limit() {
    let mut options = options("x", &[2, 4]);
    options.limits.max_storage_buffer_binding_size = 16;
    let error = compile(&identity(), &options).unwrap_err();
    assert!(matches!(
        error,
        CompileError::BufferTooLarge { size: 32, .. }
    ));
}
#[test]
fn media_prologues_read_textures_through_views_and_materialize_outputs() {
    for source in [
        json!({"kind": "external"}),
        json!({"kind": "image", "width": 8, "height": 6}),
    ] {
        for layout in ["nchw", "nhwc"] {
            let shape = if layout == "nchw" {
                vec![1, 3, 4, 5]
            } else {
                vec![1, 4, 5, 3]
            };
            let graph = graph(&shape, &[], vec![], &[("x", &shape)]);
            let mut opts = options("x", &shape);
            opts.media.insert("x".to_owned(), serde_json::from_value(json!({
                    "source": source, "layout": layout, "resize": "centerCrop", "mean": [0.1, 0.2, 0.3], "std": [0.5, 0.6, 0.7], "scale": 2.0
                })).unwrap());
            let plan = compile(&graph, &opts).unwrap();
            assert_eq!(plan.dispatches.len(), 1);
            assert_ne!(plan.outputs[0].buffer, plan.inputs[0].buffer);
            assert!(plan.shaders[0].wgsl.contains("fn load_"));
            crate::codegen::tests::validate_wgsl(&plan.shaders[0].wgsl);
        }
    }
}

#[test]
fn rejects_invalid_media_settings_and_reads_graph_media() {
    let shape = [1, 3, 4, 5];
    let mut graph = graph(&shape, &[], vec![], &[("x", &shape)]);
    let config = json!({"source": {"kind": "external"}});
    graph.media = serde_json::from_value(json!({"x": config})).unwrap();
    let opts = options("x", &shape);
    assert_eq!(compile(&graph, &opts).unwrap().media.len(), 1);
    for bad in [
        json!({"std": [1, 0, 1]}),
        json!({"layout": "nhwc"}),
        json!({"source": {"kind": "image", "width": 0, "height": 4}}),
    ] {
        let mut value = config.clone();
        value
            .as_object_mut()
            .unwrap()
            .extend(bad.as_object().unwrap().clone());
        graph.media = serde_json::from_value(json!({"x": value})).unwrap();
        assert!(matches!(
            compile(&graph, &opts),
            Err(CompileError::InvalidOp { .. })
        ));
    }
}
#[test]
fn custom_kernels_materialize_views_keep_f32_and_do_not_fuse() {
    let mut graph = graph(
        &[2, 3],
        &[],
        vec![
            node(
                "aten.permute.default",
                ["x"],
                "t",
                &[3, 2],
                json!({"dims": [1, 0]}),
            ),
            node("test.twice.default", ["t", "t"], "c", &[3, 2], json!({})),
            node("aten.relu.default", ["c"], "r", &[3, 2], json!({})),
        ],
        &[("r", &[3, 2])],
    );
    graph.f16 = true;
    let mut opts = options("x", &[2, 3]);
    opts.f16 = true;
    opts.features.shader_f16 = true;
    opts.custom_kernels.insert("c".into(), crate::CustomKernel {
            shape: vec![3, 2], requested_shape: None, workgroups: [1, 1, 1],
            wgsl: "@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) { if (id.x < 6u) { output[id.x] = a[id.x] + b[id.x]; } }".into(),
        });
    let plan = compile(&graph, &opts).unwrap();
    let custom = plan.dispatches.iter().find(|d| d.ops == ["c"]).unwrap();
    assert_eq!(custom.bindings.len(), 3);
    assert!(custom.bindings.iter().all(|b| b.size == 24));
    assert!(plan.dispatches.iter().any(|d| d.ops == ["r"]));
    for shader in &plan.shaders {
        crate::codegen::tests::validate_wgsl(&shader.wgsl);
    }
    opts.custom_kernels.clear();
    assert!(matches!(
        compile(&graph, &opts),
        Err(CompileError::UnsupportedOp(_))
    ));
}

#[test]
fn validates_every_matmul_tile_and_rejects_invalid_tiles() {
    let graph = mlp();
    let name = graph
        .nodes
        .iter()
        .find(|n| n.op == "aten.addmm.default")
        .unwrap()
        .outputs[0]
        .as_ref()
        .unwrap()
        .name
        .clone();
    for tile in [0, 8, 16] {
        let mut opts = options("input", &[3, 4]);
        opts.tiles.insert(
            name.clone(),
            crate::TileConfig {
                rows: 2,
                columns: 2,
                split: 1,
                matmul: tile,
            },
        );
        let plan = compile(&graph, &opts).unwrap();
        for shader in plan.shaders {
            crate::codegen::tests::validate_wgsl(&shader.wgsl);
        }
    }
    let mut opts = options("input", &[3, 4]);
    opts.tiles.insert(
        name,
        crate::TileConfig {
            rows: 99,
            columns: 2,
            split: 1,
            matmul: 16,
        },
    );
    assert!(matches!(
        compile(&graph, &opts),
        Err(CompileError::InvalidOp { .. })
    ));
}

#[test]
fn rejects_duplicate_graph_definitions_and_unknown_opsets() {
    let mut graph = identity();
    graph.inputs.push(graph.inputs[0].clone());
    assert!(matches!(
        compile(&graph, &options("x", &[2, 4])),
        Err(CompileError::InvalidOp { .. })
    ));
    let mut graph = identity();
    graph.opset = "unknown".to_owned();
    assert!(matches!(
        compile(&graph, &options("x", &[2, 4])),
        Err(CompileError::InvalidOp { .. })
    ));
}

#[test]
fn buckets_do_not_hide_invalid_requested_shapes() {
    let graph = identity();
    for shape in [&[65, 4][..], &[17][..], &[17, 5][..]] {
        let mut options = options("x", shape);
        options.bucket = Some(16);
        assert!(compile(&graph, &options).is_err(), "{shape:?}");
    }
}

#[test]
fn inputs_with_one_symbol_must_agree_before_rounding_to_the_same_bucket() {
    let mut graph = identity();
    let mut other = graph.inputs[0].clone();
    other.name = "other".into();
    graph.inputs.push(other);

    let mut opts = options("x", &[17, 4]);
    opts.bucket = Some(16);
    opts.input_shapes.insert("other".into(), vec![17, 4]);
    let plan = compile(&graph, &opts).unwrap();
    assert_eq!(plan.inputs[0].shape, [32, 4]);
    assert_eq!(plan.inputs[1].shape, [32, 4]);

    opts.input_shapes.insert("other".into(), vec![20, 4]);
    assert_eq!(
        compile(&graph, &opts).unwrap_err(),
        CompileError::InconsistentSymbol("batch".into())
    );
}

#[test]
fn bucketed_output_symbols_must_describe_the_inferred_shape() {
    let mut graph = identity();
    let mut opts = options("x", &[17, 4]);
    opts.bucket = Some(16);

    for (expression, expected) in [
        ("unbound", "unbound"),
        ("2*batch", "does not match inferred size 17"),
    ] {
        graph.outputs[0].shape[0] = Dim::Symbolic {
            name: expression.into(),
            min: None,
            max: None,
        };
        assert!(
            matches!(compile(&graph, &opts), Err(CompileError::InvalidOp { op, message })
                if op == "bucket output" && message.contains(expected)),
            "{expression}"
        );
    }
}

#[test]
fn attention_requires_a_boolean_key_mask_to_pad_noncausal_keys() {
    let key_len = json!({"name": "keys", "min": 1, "max": 32});
    let output = value("out", &[1, 2, 3, 4]);
    let mut graph: Graph = serde_json::from_value(json!({
        "formatVersion": FORMAT_VERSION,
        "opset": "core-aten/1",
        "parameters": [],
        "inputs": [
            value("q", &[1, 2, 3, 4]),
            {"name": "k", "dtype": "f32", "shape": [1, 2, key_len, 4]},
            {"name": "v", "dtype": "f32", "shape": [1, 2, key_len, 4]},
            {"name": "mask", "dtype": "bool", "shape": [key_len]},
        ],
        "nodes": [{
            "op": "aten.scaled_dot_product_attention.default",
            "inputs": ["q", "k", "v", "mask"],
            "outputs": [output],
            "attributes": {"is_causal": false},
        }],
        "outputs": [output],
    }))
    .unwrap();
    let mut opts = options("q", &[1, 2, 3, 4]);
    opts.input_shapes.extend([
        ("k".into(), vec![1, 2, 5, 4]),
        ("v".into(), vec![1, 2, 5, 4]),
        ("mask".into(), vec![5]),
    ]);
    opts.bucket = Some(8);
    let plan = compile(&graph, &opts).unwrap();
    assert_eq!(plan.inputs[1].shape, [1, 2, 8, 4]);
    assert_eq!(plan.inputs[3].shape, [8]);
    assert_eq!(plan.outputs[0].shape, [1, 2, 3, 4]);

    // Padding contributes false to a boolean mask, but zero to an additive
    // mask: the latter would leave the extra keys visible to softmax.
    graph.inputs[3].dtype = Dtype::F32;
    assert!(
        matches!(compile(&graph, &opts), Err(CompileError::InvalidOp { message, .. })
            if message.contains("bucket padding"))
    );
    opts.bucket = None;
    compile(&graph, &opts).unwrap();
}

#[test]
fn checks_binding_limits_before_returning_a_plan() {
    let mut options = options("input", &[3, 4]);
    options.limits.max_storage_buffers_per_shader_stage = 2;
    assert!(matches!(
        compile(&mlp(), &options),
        Err(CompileError::InvalidOp { .. })
    ));
}

#[test]
fn constants_read_in_prepare_use_the_same_writable_usage_as_their_producer() {
    let graph = graph(
        &[3],
        &[("w", &[3])],
        vec![
            node("aten.mul.Tensor", ["w"], "a", &[3], json!({"other": 2})),
            node("aten.add.Tensor", ["a"], "b", &[3], json!({"other": 1})),
            node(
                "aten.add.Tensor",
                ["x", "b"],
                "out",
                &[3],
                json!({"alpha": 1}),
            ),
        ],
        &[("out", &[3])],
    );
    let plan = compile(&graph, &options("x", &[3])).unwrap();
    assert_eq!(plan.prepare.len(), 2);
    let wgsl = &plan.shaders[plan.prepare[1].shader as usize].wgsl;
    assert!(wgsl.contains("var<storage, read_write>"));
    assert!(!wgsl.contains("var<storage, read>"));
}

#[test]
fn bucket_padding_rejects_reductions_and_reshapes_that_change_real_values() {
    let dynamic = |mut graph: Graph| {
        graph.inputs[0].shape[0] = Dim::Symbolic {
            name: "s".into(),
            min: Some(1),
            max: Some(64),
        };
        graph
    };
    let reduction = dynamic(graph(
        &[17, 4],
        &[],
        vec![node(
            "aten.mean.dim",
            ["x"],
            "out",
            &[4],
            json!({"dim": [0], "keepdim": false}),
        )],
        &[("out", &[4])],
    ));
    let mut reshape = dynamic(graph(
        &[17, 4],
        &[],
        vec![node(
            "aten.view.default",
            ["x"],
            "out",
            &[4, 17],
            json!({"size": [4, {"$sym": "s"}]}),
        )],
        &[("out", &[4, 17])],
    ));
    let dim = reshape.inputs[0].shape[0].clone();
    reshape.nodes[0].outputs[0].as_mut().unwrap().shape[1] = dim.clone();
    reshape.outputs[0].shape[1] = dim;
    // Test an exact bucket too: later runs may use any shorter length in it.
    for size in [17, 32] {
        let mut options = options("x", &[size, 4]);
        options.bucket = Some(16);
        for graph in [&reduction, &reshape] {
            assert!(
                matches!(compile(graph, &options), Err(CompileError::InvalidOp { message, .. }) if message.contains("bucket padding"))
            );
        }
    }
}

#[test]
fn bucket_padding_rejects_movements_and_statistics_that_read_padded_elements() {
    // Op, input shape, symbolic axis, output shape, attributes, and whether
    // the op stays exact under padding.
    type Case<'a> = (&'a str, &'a [u64], usize, &'a [u64], Json, bool);
    let cases: &[Case] = &[
        (
            "aten.flip.default",
            &[17, 4],
            0,
            &[17, 4],
            json!({"dims": [0]}),
            false,
        ),
        (
            "aten.flip.default",
            &[17, 4],
            0,
            &[17, 4],
            json!({"dims": [1]}),
            true,
        ),
        (
            "aten.roll.default",
            &[17, 4],
            0,
            &[17, 4],
            json!({"shifts": [1], "dims": [0]}),
            false,
        ),
        (
            "aten.roll.default",
            &[17, 4],
            0,
            &[17, 4],
            json!({"shifts": [1], "dims": [1]}),
            true,
        ),
        (
            "aten.repeat.default",
            &[17, 4],
            0,
            &[34, 4],
            json!({"repeats": [2, 1]}),
            false,
        ),
        (
            "aten.repeat.default",
            &[17, 4],
            0,
            &[17, 8],
            json!({"repeats": [1, 2]}),
            true,
        ),
        (
            "aten.constant_pad_nd.default",
            &[17, 4],
            0,
            &[18, 4],
            json!({"pad": [0, 0, 0, 1], "value": 1.0}),
            false,
        ),
        (
            "aten.constant_pad_nd.default",
            &[17, 4],
            0,
            &[18, 4],
            json!({"pad": [0, 0, 1, 0], "value": 1.0}),
            true,
        ),
        (
            "aten.upsample_bilinear2d.vec",
            &[1, 1, 4, 17],
            3,
            &[1, 1, 4, 34],
            json!({"scale_factors": [1.0, 2.0], "align_corners": false}),
            false,
        ),
        (
            "aten._adaptive_avg_pool2d.default",
            &[1, 1, 4, 17],
            3,
            &[1, 1, 2, 2],
            json!({"output_size": [2, 2]}),
            false,
        ),
        (
            "aten._native_batch_norm_legit.no_stats",
            &[17, 4],
            0,
            &[17, 4],
            json!({"training": true, "momentum": 0.1, "eps": 1e-5}),
            false,
        ),
    ];
    for (op, input, axis, output, attributes, safe) in cases {
        let inputs = if op.contains("batch_norm") {
            json!(["x", null, null])
        } else {
            json!(["x"])
        };
        let mut graph = graph(
            input,
            &[],
            vec![node(op, inputs, "out", output, attributes.clone())],
            &[("out", output)],
        );
        graph.inputs[0].shape[*axis] = Dim::Symbolic {
            name: "s".into(),
            min: Some(1),
            max: Some(64),
        };
        // The output follows the input along the symbolic axis.
        let (size, out) = (input[*axis], output[*axis]);
        let expression = match out {
            _ if out == size => Some("s"),
            _ if out == size + 1 => Some("s+1"),
            _ if out == 2 * size => Some("2*s"),
            _ => None,
        };
        if let Some(expression) = expression {
            let dim = Dim::Symbolic {
                name: expression.into(),
                min: None,
                max: None,
            };
            graph.nodes[0].outputs[0].as_mut().unwrap().shape[*axis] = dim.clone();
            graph.outputs[0].shape[*axis] = dim;
        }
        let mut options = options("x", input);
        options.bucket = Some(16);
        let result = compile(&graph, &options);
        let rejected = matches!(&result, Err(CompileError::InvalidOp { message, .. }) if message.contains("bucket padding"));
        assert_eq!(rejected, !safe, "{op} {attributes}: {:?}", result.err());
    }
}

#[test]
fn buckets_reject_symbolic_attributes_that_change_values_or_offsets() {
    let mut slice = graph(
        &[3, 4],
        &[("w", &[8])],
        vec![node(
            "aten.slice.Tensor",
            ["w"],
            "out",
            &[2],
            json!({"dim": 0, "start": {"$sym": "s"}, "end": {"$sym": "s+2"}, "step": 1}),
        )],
        &[("out", &[2])],
    );
    slice.inputs[0].shape[0] = Dim::Symbolic {
        name: "s".into(),
        min: Some(1),
        max: Some(4),
    };
    for length in [3, 4] {
        let mut exact = options("x", &[length, 4]);
        compile(&slice, &exact).unwrap();
        exact.bucket = Some(4);
        assert!(
            matches!(compile(&slice, &exact), Err(CompileError::InvalidOp { message, .. }) if message.contains("bucket padding")),
            "accepted a value-changing slice at length {length}"
        );
    }
}

#[test]
fn resolves_custom_op_attributes() {
    let attributes: BTreeMap<String, crate::Attr> = serde_json::from_value(
        json!({"size": {"$sym": "2*s"}, "config": {"k": [1, {"$sym": "s"}]}}),
    )
    .unwrap();
    let symbols = BTreeMap::from([("s".to_owned(), 3)]);
    let resolved = crate::resolve_attributes(&attributes, &symbols).unwrap();
    assert_eq!(
        serde_json::to_value(resolved).unwrap(),
        json!({"size": 6, "config": {"k": [1, 3]}})
    );
    assert!(crate::resolve_attributes(&attributes, &BTreeMap::new()).is_err());
}

#[test]
fn folds_generators_into_the_preparation() {
    // `x + arange(6)` and a causal mask `arange(4) <= arange(4)[:, None]`,
    // both from generators alone.
    let graph = graph(
        &[2, 6],
        &[],
        vec![
            node_with(
                "aten.arange.start_step",
                json!([]),
                json!([typed("range", "i64", &[6])]),
                json!({"start": 0, "end": 6}),
            ),
            node(
                "aten.add.Tensor",
                ["x", "range"],
                "shifted",
                &[2, 6],
                json!({}),
            ),
            node_with(
                "aten.arange.start_step",
                json!([]),
                json!([typed("positions", "i64", &[4])]),
                json!({"start": 0, "end": 4, "step": 1}),
            ),
            node_with(
                "aten.unsqueeze.default",
                json!(["positions"]),
                json!([typed("column", "i64", &[4, 1])]),
                json!({"dim": 1}),
            ),
            node_with(
                "aten.le.Tensor",
                json!(["positions", "column"]),
                json!([typed("mask", "bool", &[4, 4])]),
                json!({}),
            ),
            node_with(
                "aten.full.default",
                json!([]),
                json!([typed("ones", "f32", &[4, 4])]),
                json!({"size": [4, 4], "fill_value": 1.5}),
            ),
            node_with(
                "aten.scalar_tensor.default",
                json!([]),
                json!([typed("zero", "f32", &[])]),
                json!({"s": 0}),
            ),
            node(
                "aten.where.self",
                ["mask", "ones", "zero"],
                "masked",
                &[4, 4],
                json!({}),
            ),
        ],
        &[("shifted", &[2, 6]), ("masked", &[4, 4])],
    );
    let plan = compile_x(&graph, &[2, 6]);
    // Every generator and the mask fold; only the add runs per inference.
    assert_eq!(kernel_ops(&plan), [vec!["shifted".to_owned()]]);
    assert_eq!(plan.prepare.len(), 6);
    let masked = &plan.outputs[1];
    assert_eq!(
        plan.buffers[masked.buffer as usize].role,
        BufferRole::Constant
    );
    assert!(
        plan.shaders
            .iter()
            .any(|s| s.wgsl.contains("f32(bitcast<i32>("))
    );
}

#[test]
fn infers_dtypes_through_casts_comparisons_and_selects() {
    // `where(x > 0, x, 0.1 * x).long()` then back to float: one dispatch.
    let graph = graph(
        &[3, 4],
        &[],
        vec![
            node_with(
                "aten.gt.Scalar",
                json!(["x"]),
                json!([typed("positive", "bool", &[3, 4])]),
                json!({"other": 0}),
            ),
            node(
                "aten.mul.Tensor",
                ["x"],
                "scaled",
                &[3, 4],
                json!({"other": 0.1}),
            ),
            node(
                "aten.where.self",
                ["positive", "x", "scaled"],
                "leaky",
                &[3, 4],
                json!({}),
            ),
            node_with(
                "aten._to_copy.default",
                json!(["leaky"]),
                json!([typed("long", "i64", &[3, 4])]),
                json!({"dtype": "torch.int64"}),
            ),
            node_with(
                "aten._to_copy.default",
                json!(["long"]),
                json!([typed("int", "i32", &[3, 4])]),
                json!({"dtype": "torch.int32"}),
            ),
        ],
        &[],
    );
    let mut graph = graph;
    graph.outputs = vec![serde_json::from_value(typed("int", "i32", &[3, 4])).unwrap()];
    let plan = compile_x(&graph, &[3, 4]);
    // The i64 to i32 conversion is a view: one fused kernel.
    assert_eq!(plan.dispatches.len(), 1);
    assert_eq!(plan.outputs[0].dtype, Dtype::I32);
    let wgsl = &plan.shaders[0].wgsl;
    assert!(wgsl.contains("select(") && wgsl.contains("i32("), "{wgsl}");

    // Integer arithmetic: an i64 tensor plus an int scalar stays i64, and
    // divides as floats.
    let ids = graph_with_input(
        typed("x", "i64", &[5]),
        vec![
            node_with(
                "aten.add.Tensor",
                json!(["x"]),
                json!([typed("next", "i64", &[5])]),
                json!({"other": 1}),
            ),
            node(
                "aten.div.Tensor",
                ["next"],
                "half",
                &[5],
                json!({"other": 2}),
            ),
        ],
        vec![typed("half", "f32", &[5])],
    );
    compile_x(&ids, &[5]);
}

#[test]
fn splits_into_views() {
    let graph = graph(
        &[2, 6],
        &[],
        vec![node_with(
            "aten.split_with_sizes.default",
            json!(["x"]),
            json!([value("a", &[2, 2]), null, value("c", &[2, 1])]),
            json!({"split_sizes": [2, 3, 1], "dim": 1}),
        )],
        &[],
    );
    let mut graph = graph;
    graph.outputs = vec![
        serde_json::from_value(value("c", &[2, 1])).unwrap(),
        serde_json::from_value(value("a", &[2, 2])).unwrap(),
    ];
    let plan = compile_x(&graph, &[2, 6]);
    // Outputs are read back whole, so each slice is copied once, no more.
    assert_eq!(plan.dispatches.len(), 2);
    assert!(plan.shaders.iter().any(|s| s.wgsl.contains("+ 5u]")));
}

#[test]
fn returns_the_indices_of_a_max_pooling() {
    let pool = json!({"kernel_size": [2, 2], "stride": [2, 2]});
    let graph = graph(
        &[1, 1, 4, 4],
        &[],
        vec![json!({
            "op": "aten.max_pool2d_with_indices.default",
            "inputs": ["x"],
            "outputs": [value("max", &[1, 1, 2, 2]), typed("indices", "i64", &[1, 1, 2, 2])],
            "attributes": pool,
        })],
        &[("max", &[1, 1, 2, 2])],
    );
    let mut graph = graph;
    graph
        .outputs
        .push(serde_json::from_value(typed("indices", "i64", &[1, 1, 2, 2])).unwrap());
    let plan = compile_x(&graph, &[1, 1, 4, 4]);
    assert_eq!(plan.dispatches.len(), 1);
    assert_eq!(plan.dispatches[0].bindings.len(), 3);
    assert_eq!(plan.outputs[1].dtype, Dtype::I64);
    assert!(plan.shaders[0].wgsl.contains("dst1["));
}

#[test]
fn multiplies_batches_without_a_bias() {
    let graph = graph(
        &[2, 3, 4],
        &[("w", &[4, 5]), ("b", &[2, 4, 5])],
        vec![
            node(
                "aten.view.default",
                ["x"],
                "flat",
                &[6, 4],
                json!({"size": [6, 4]}),
            ),
            node("aten.mm.default", ["flat", "w"], "mm", &[6, 5], json!({})),
            node("aten.bmm.default", ["x", "b"], "bmm", &[2, 3, 5], json!({})),
        ],
        &[("mm", &[6, 5]), ("bmm", &[2, 3, 5])],
    );
    let plan = compile_x(&graph, &[2, 3, 4]);
    assert_eq!(plan.dispatches.len(), 2);
    // The batch is the third dispatch axis of the tiled kernel.
    assert_eq!(plan.dispatches[1].workgroups[2], 2);
    assert_eq!(plan.dispatches[1].bindings.len(), 3);
}

#[test]
fn reduces_long_rows_cooperatively() {
    // One row of 32k: a workgroup per slice, then a sum of the slices.
    let graph = graph(
        &[1, 32768],
        &[],
        vec![node(
            "aten.mean.dim",
            ["x"],
            "mean",
            &[1],
            json!({"dim": [1]}),
        )],
        &[("mean", &[1])],
    );
    let plan = compile_x(&graph, &[1, 32768]);
    assert_eq!(
        kernel_ops(&plan),
        [vec!["mean".to_owned()], vec!["mean.sum".to_owned()]]
    );
    assert!(plan.shaders[0].wgsl.contains("workgroup_sum"));
    assert_eq!(plan.dispatches[0].workgroups, [1, 1, 8]);

    // With subgroups of at least 16 invocations, one subgroup per row.
    let mut options = options("x", &[1, 32768]);
    options.features.subgroups = true;
    options.features.subgroup_min_size = 32;
    let plan = compile(&graph, &options).unwrap();
    assert!(plan.shaders[0].wgsl.contains("enable subgroups;"));
    assert!(plan.shaders[0].wgsl.contains("@workgroup_size(32)"));
}

#[test]
fn fuses_element_wise_producers_into_loads() {
    // relu(x) then mean: one dispatch, the relu inside the reduction loads.
    let graph = graph(
        &[4, 8],
        &[("w", &[8, 3])],
        vec![
            node("aten.relu.default", ["x"], "relu", &[4, 8], json!({})),
            node("aten.mean.dim", ["relu"], "mean", &[4], json!({"dim": [1]})),
            // A cheap scale before a matmul fuses, a GELU does not.
            node(
                "aten.mul.Tensor",
                ["x"],
                "scaled",
                &[4, 8],
                json!({"other": 0.5}),
            ),
            node("aten.mm.default", ["scaled", "w"], "mm", &[4, 3], json!({})),
            node(
                "aten.gelu.default",
                ["x"],
                "gelu",
                &[4, 8],
                json!({"approximate": "none"}),
            ),
            node("aten.mm.default", ["gelu", "w"], "mm2", &[4, 3], json!({})),
        ],
        &[("mean", &[4]), ("mm", &[4, 3]), ("mm2", &[4, 3])],
    );
    let plan = compile_x(&graph, &[4, 8]);
    let ops = kernel_ops(&plan);
    assert!(
        ops.contains(&vec!["relu".to_owned(), "mean".to_owned()]),
        "{ops:?}"
    );
    assert!(
        ops.contains(&vec!["scaled".to_owned(), "mm".to_owned()]),
        "{ops:?}"
    );
    assert!(ops.contains(&vec!["gelu".to_owned()]), "{ops:?}");
    assert_eq!(plan.dispatches.len(), 4);
}

#[test]
fn compiles_preserved_rms_norms() {
    let graph = graph(
        &[2, 5, 64],
        &[("weight", &[64])],
        vec![node(
            "aten.rms_norm.default",
            json!(["x", "weight"]),
            "rms",
            &[2, 5, 64],
            json!({"normalized_shape": [64], "eps": null}),
        )],
        &[("rms", &[2, 5, 64])],
    );
    let plan = compile_x(&graph, &[2, 5, 64]);
    assert_eq!(plan.dispatches.len(), 1);
    assert!(plan.shaders[0].wgsl.contains("let mean = 0.0;"));
    // Machine epsilon, as PyTorch defaults to.
    assert!(
        plan.shaders[0].wgsl.contains("1.1920929e-7"),
        "{}",
        plan.shaders[0].wgsl
    );
}

/// A running sum: `state += x`, returning `state * 2`.
fn accumulator() -> Graph {
    let mut graph = graph(
        &[3],
        &[("state", &[3])],
        vec![
            node(
                "aten.add.Tensor",
                ["state", "x"],
                "updated",
                &[3],
                json!({}),
            ),
            node(
                "aten.mul.Tensor",
                ["updated"],
                "doubled",
                &[3],
                json!({"other": 2}),
            ),
        ],
        &[("doubled", &[3])],
    );
    graph.parameters[0].update = Some("updated".to_owned());
    graph
}

#[test]
fn writes_state_back_after_each_run() {
    let plan = compile_x(&accumulator(), &[3]);
    let state = plan
        .buffers
        .iter()
        .position(|buffer| {
            buffer.role
                == BufferRole::State {
                    tensor: "state".to_owned(),
                }
        })
        .expect("a state buffer");
    // The update is written straight into the state buffer by the last
    // dispatch, which reads it through its output binding.
    let last = plan.dispatches.last().unwrap();
    assert_eq!(last.bindings.last().unwrap().buffer as usize, state);
    assert!(
        last.ops.contains(&"state.update".to_owned()),
        "{:?}",
        kernel_ops(&plan)
    );
    assert!(plan.dispatches.iter().all(|d| {
        d.bindings
            .iter()
            .filter(|b| b.buffer as usize == state)
            .count()
            <= 1
    }));
    assert!(plan.prepare.is_empty());

    // A state returned as is is copied before it is rewritten.
    let mut graph = accumulator();
    graph.outputs[0].name = "state".to_owned();
    let plan = compile_x(&graph, &[3]);
    let output = plan.outputs[0].buffer as usize;
    assert_ne!(output, state);
    let ops = kernel_ops(&plan);
    let copy = ops.iter().position(|ops| ops == &["state"]).unwrap();
    let update = ops
        .iter()
        .position(|ops| ops.contains(&"state.update".to_owned()));
    assert!(copy < update.unwrap(), "{ops:?}");

    // An update must match the state.
    let mut graph = accumulator();
    graph.parameters[0].update = Some("x".to_owned());
    graph.parameters[0].dtype = Dtype::I32;
    assert!(compile(&graph, &options("x", &[3])).is_err());
}

#[test]
fn never_writes_state_before_it_is_read() {
    // `state` is read by a matmul after the update is computed: the copy
    // into the state must not move up into the update's producer.
    let mut graph = graph(
        &[3, 3],
        &[("state", &[3, 3])],
        vec![
            node("aten.mm.default", ["x", "x"], "next", &[3, 3], json!({})),
            node("aten.mm.default", ["state", "x"], "out", &[3, 3], json!({})),
        ],
        &[("out", &[3, 3])],
    );
    graph.parameters[0].update = Some("next".to_owned());
    let plan = compile_x(&graph, &[3, 3]);
    let ops = kernel_ops(&plan);
    assert_eq!(
        ops.last().unwrap(),
        &vec!["state.update".to_owned()],
        "{ops:?}"
    );
}

#[test]
fn snapshots_cross_state_updates_before_writing_either_buffer() {
    let mut graph = graph(&[3], &[("a", &[3]), ("b", &[3])], vec![], &[("x", &[3])]);
    graph.parameters[0].update = Some("b".to_owned());
    graph.parameters[1].update = Some("a".to_owned());
    let plan = compile_x(&graph, &[3]);
    // No update may still read a state buffer after it was overwritten.
    for tensor in ["a", "b"] {
        let buffer = plan
            .buffers
            .iter()
            .position(|b| {
                b.role
                    == BufferRole::State {
                        tensor: tensor.to_owned(),
                    }
            })
            .unwrap() as u32;
        let write = plan
            .dispatches
            .iter()
            .position(|d| d.ops.contains(&format!("{tensor}.update")))
            .unwrap();
        assert!(
            !plan.dispatches[write + 1..].iter().any(|d| {
                d.bindings[..d.bindings.len() - 1]
                    .iter()
                    .any(|b| b.buffer == buffer)
            }),
            "{tensor}: {:?}",
            kernel_ops(&plan)
        );
    }
}

#[test]
fn returns_max_pool_indices_without_reading_the_values() {
    let graph = graph_with_input(
        value("x", &[1, 1, 4, 4]),
        vec![node_with(
            "aten.max_pool2d_with_indices.default",
            json!(["x"]),
            json!([null, typed("indices", "i64", &[1, 1, 2, 2])]),
            json!({"kernel_size": [2, 2], "stride": [2, 2]}),
        )],
        vec![typed("indices", "i64", &[1, 1, 2, 2])],
    );
    let plan = compile_x(&graph, &[1, 1, 4, 4]);
    assert_eq!(plan.dispatches.len(), 1);
    assert_eq!(plan.outputs[0].dtype, Dtype::I64);
}

#[test]
fn counts_both_pool_outputs_when_limiting_fusion_bindings() {
    let mut graph = graph(
        &[1, 1, 4, 4],
        &[("bias", &[1, 1, 2, 2])],
        vec![
            node_with(
                "aten.max_pool2d_with_indices.default",
                json!(["x"]),
                json!([
                    value("max", &[1, 1, 2, 2]),
                    typed("indices", "i64", &[1, 1, 2, 2])
                ]),
                json!({"kernel_size": [2, 2], "stride": [2, 2]}),
            ),
            node(
                "aten.add.Tensor",
                ["max", "bias"],
                "y",
                &[1, 1, 2, 2],
                json!({}),
            ),
        ],
        &[("y", &[1, 1, 2, 2])],
    );
    graph
        .outputs
        .push(serde_json::from_value(typed("indices", "i64", &[1, 1, 2, 2])).unwrap());
    let mut options = options("x", &[1, 1, 4, 4]);
    options.limits.max_storage_buffers_per_shader_stage = 3;
    let plan = compile(&graph, &options).unwrap();
    assert_eq!(plan.dispatches.len(), 2);
}

#[test]
fn keeps_a_consumer_of_both_pool_outputs_after_the_pool() {
    let graph = graph(
        &[1, 1, 4, 4],
        &[],
        vec![
            node_with(
                "aten.max_pool2d_with_indices.default",
                json!(["x"]),
                json!([
                    value("max", &[1, 1, 2, 2]),
                    typed("indices", "i64", &[1, 1, 2, 2])
                ]),
                json!({"kernel_size": [2, 2], "stride": [2, 2]}),
            ),
            node(
                "aten._to_copy.default",
                ["indices"],
                "float",
                &[1, 1, 2, 2],
                json!({"dtype": "torch.float32"}),
            ),
            node(
                "aten.add.Tensor",
                ["max", "float"],
                "y",
                &[1, 1, 2, 2],
                json!({}),
            ),
        ],
        &[("y", &[1, 1, 2, 2])],
    );
    let plan = compile_x(&graph, &[1, 1, 4, 4]);
    assert_eq!(plan.dispatches.len(), 2);
    assert_eq!(kernel_ops(&plan)[0], ["max"]);
}

#[test]
fn rejects_scalars_that_would_saturate_in_storage() {
    let graph = graph_with_input(
        typed("x", "i64", &[3]),
        vec![node_with(
            "aten.add.Tensor",
            json!(["x"]),
            json!([typed("y", "i64", &[3])]),
            json!({"other": 2147483648i64}),
        )],
        vec![typed("y", "i64", &[3])],
    );
    assert!(
        compile(&graph, &options("x", &[3]))
            .unwrap_err()
            .to_string()
            .contains("32 bits")
    );
}

#[test]
fn rejects_aliased_state_tensors() {
    let mut graph = accumulator();
    let mut alias = graph.parameters[0].clone();
    alias.name = "alias".to_owned();
    alias.update = None;
    graph.parameters.push(alias);
    assert!(
        validate(&graph)
            .unwrap_err()
            .to_string()
            .contains("duplicate state tensor")
    );
}

#[test]
fn coverage_math_helpers_survive_fusion() {
    let shape = &[2, 8];
    let graph = graph(
        shape,
        &[],
        vec![
            node("aten.exp.default", ["x"], "exp", shape, json!({})),
            node("aten.expm1.default", ["exp"], "expm1", shape, json!({})),
            node(
                "aten.pow.Tensor_Scalar",
                ["expm1"],
                "pow",
                shape,
                json!({"exponent": -3}),
            ),
            node(
                "aten.maximum.default",
                ["pow", "x"],
                "max",
                shape,
                json!({}),
            ),
        ],
        &[("max", shape)],
    );
    let plan = compile_x(&graph, shape);
    assert_eq!(plan.dispatches.len(), 1);
    crate::codegen::tests::validate_wgsl(&plan.shaders[0].wgsl);
}

#[test]
fn coverage_row_kernels_validate_on_both_strategies() {
    for depth in [7, 513] {
        let shape = &[3, depth];
        let graph = graph(
            shape,
            &[],
            vec![
                node(
                    "aten._softmax.default",
                    ["x"],
                    "softmax",
                    shape,
                    json!({"dim": -1}),
                ),
                node(
                    "aten._log_softmax.default",
                    ["x"],
                    "log_softmax",
                    shape,
                    json!({"dim": -1}),
                ),
                node(
                    "aten.cumsum.default",
                    ["x"],
                    "prefix",
                    shape,
                    json!({"dim": -1}),
                ),
                node(
                    "aten.var.correction",
                    ["x"],
                    "var",
                    &[3],
                    json!({"dim": [1], "correction": depth}),
                ),
                node(
                    "aten.amin.default",
                    ["x"],
                    "min",
                    &[3],
                    json!({"dim": [-1]}),
                ),
                node_with(
                    "aten.max.dim",
                    json!(["x"]),
                    json!([null, typed("idx", "i64", &[3])]),
                    json!({"dim": -1}),
                ),
            ],
            &[
                ("softmax", shape),
                ("log_softmax", shape),
                ("prefix", shape),
                ("var", &[3]),
                ("min", &[3]),
            ],
        );
        let mut graph = graph;
        graph.f16 = true;
        graph
            .outputs
            .push(serde_json::from_value(typed("idx", "i64", &[3])).unwrap());
        for f16 in [false, true] {
            let mut options = options("x", shape);
            options.f16 = f16;
            options.features.shader_f16 = f16;
            let plan = compile(&graph, &options).unwrap();
            assert_eq!(plan.dispatches.len(), 6);
            for (shader, dispatch) in plan.shaders.iter().zip(&plan.dispatches) {
                crate::codegen::tests::validate_wgsl(&shader.wgsl);
                if depth > 7 {
                    assert_eq!(dispatch.workgroups, [3, 1, 1]);
                }
            }
        }
    }
}

#[test]
fn inlines_nearest_resize_and_padding_into_convolution_loads() {
    for (op, attributes, intermediate) in [
        (
            "aten.upsample_nearest2d.vec",
            json!({"scale_factors": [2.0, 2.0]}),
            &[1, 3, 8, 12][..],
        ),
        (
            "aten.pad.default",
            json!({"mode": "reflect", "pad": [1, 2, 1, 1]}),
            &[1, 3, 6, 9],
        ),
        (
            "aten.constant_pad_nd.default",
            json!({"pad": [1, 1, 2, 2]}),
            &[1, 3, 8, 8],
        ),
    ] {
        let input = &[1, 3, 4, 6];
        let output = &[1, 5, intermediate[2] - 2, intermediate[3] - 2];
        let graph = graph(
            input,
            &[("w", &[5, 3, 3, 3])],
            vec![
                node(op, ["x"], "remapped", intermediate, attributes),
                node(
                    "aten.convolution.default",
                    json!(["remapped", "w", null]),
                    "out",
                    output,
                    json!({}),
                ),
            ],
            &[("out", output)],
        );
        let plan = compile_x(&graph, input);
        assert_eq!(plan.dispatches.len(), 1);
        assert_eq!(plan.dispatches[0].bindings.len(), 3);
        for shader in &plan.shaders {
            crate::codegen::tests::validate_wgsl(&shader.wgsl);
        }
        // A model that also returns the intermediate must still write it.
        let mut graph = graph;
        graph
            .outputs
            .push(serde_json::from_value(value("remapped", intermediate)).unwrap());
        let plan = compile_x(&graph, input);
        assert!(plan.dispatches.len() >= 2);
    }
}

#[test]
fn transpose_convolution_specializes_the_disjoint_case() {
    let input = &[1, 4, 8, 8];
    for (padding, output, disjoint) in [(0, &[1, 6, 16, 16], true), (1, &[1, 6, 14, 14], false)] {
        let graph = graph(
            input,
            &[("w", &[4, 3, 2, 2])],
            vec![node(
                "aten.convolution.default",
                json!(["x", "w", null]),
                "out",
                output,
                json!({"stride": [2, 2], "padding": [padding, padding], "groups": 2, "transposed": true}),
            )],
            &[("out", output)],
        );
        let plan = compile_x(&graph, input);
        let shader = &plan.shaders[plan.dispatches[0].shader as usize].wgsl;
        assert_eq!(shader.contains("numerator0"), !disjoint);
        assert_eq!(shader.contains("ra < 2u"), disjoint);
        crate::codegen::tests::validate_wgsl(shader);
    }
}

#[test]
fn f16_compute_is_opt_in_and_keeps_f32_accumulation() {
    let mut graph = mlp();
    let mut opts = options("input", &[3, 4]);
    opts.f16_compute = true;
    assert!(matches!(
        compile(&graph, &opts),
        Err(CompileError::F16Unavailable(_))
    ));
    opts.features.shader_f16 = true;
    assert!(matches!(
        compile(&graph, &opts),
        Err(CompileError::F16Unavailable(_))
    ));
    graph.f16 = true;
    let plan = compile(&graph, &opts).unwrap();
    let shader = &plan.shaders[plan.dispatches[0].shader as usize].wgsl;
    assert!(shader.contains("var<workgroup> a: array<f16"));
    assert!(shader.contains("var sum = 0.0") && shader.contains("sum += f32("));
    assert!(shader.contains("dst: array<f32>"));
    crate::codegen::tests::validate_wgsl(shader);
}

#[test]
fn cache_patches_keep_old_state_visible_until_its_last_reader() {
    let mut graph = graph(
        &[1, 1, 4],
        &[("state", &[1, 1024, 4])],
        vec![
            json!({
                "op": "aten.scalar_tensor.default", "inputs": [],
                "outputs": [{"name": "position", "dtype": "i32", "shape": []}],
                "attributes": {"s": 5, "dtype": "torch.int32"},
            }),
            node(
                "avel.cache_write.default",
                ["state", "x", "position"],
                "next",
                &[1, 1024, 4],
                json!({}),
            ),
            node(
                "aten.select.int",
                ["next"],
                "row",
                &[1, 4],
                json!({"dim": 1, "index": 5}),
            ),
        ],
        &[("row", &[1, 4])],
    );
    graph.parameters[0].update = Some("next".into());
    let fast = compile_x(&graph, &[1, 1, 4]);
    assert_eq!(fast.dispatches[0].workgroups, [1, 1, 1]);
    assert_eq!(fast.dispatches.len(), 2);
    assert!(
        !kernel_ops(&fast)
            .iter()
            .flatten()
            .any(|op| op == "state.update")
    );

    graph
        .outputs
        .push(serde_json::from_value(value("state", &[1, 1024, 4])).unwrap());
    let safe = compile_x(&graph, &[1, 1, 4]);
    assert_eq!(safe.dispatches[0].workgroups, [64, 1, 1]);
    assert_eq!(kernel_ops(&safe).last().unwrap(), &["state.update"]);
}

#[test]
fn splits_concatenations_wider_than_the_binding_limit() {
    // Eleven products of `x`, each with its own parameter, concatenated:
    // one dispatch would need twelve storage bindings.
    let names: Vec<String> = (0..11).map(|i| format!("p{i}")).collect();
    let parameters: Vec<(&str, &[u64])> =
        names.iter().map(|n| (n.as_str(), &[2u64, 4][..])).collect();
    let mut nodes: Vec<Json> = names
        .iter()
        .enumerate()
        .map(|(i, p)| {
            node(
                "aten.mul.Tensor",
                ["x", p.as_str()],
                &format!("m{i}"),
                &[2, 4],
                json!({}),
            )
        })
        .collect();
    let products: Vec<String> = (0..11).map(|i| format!("m{i}")).collect();
    nodes.push(json!({
        "op": "aten.cat.default",
        "inputs": products,
        "outputs": [value("cat", &[2, 44])],
        "attributes": {"dim": 1},
    }));
    let graph = graph(&[2, 4], &parameters, nodes, &[("cat", &[2, 44])]);
    for bindings in [4, 8, 10] {
        let mut options = options("x", &[2, 4]);
        options.limits.max_storage_buffers_per_shader_stage = bindings;
        let plan = compile(&graph, &options).unwrap();
        for dispatch in plan.prepare.iter().chain(&plan.dispatches) {
            assert!(
                dispatch.bindings.len() <= bindings as usize,
                "{:?}",
                dispatch.ops
            );
        }
        let cats = plan
            .dispatches
            .iter()
            .filter(|d| d.ops.iter().any(|op| op == "cat"))
            .count();
        assert!(cats > 1, "{bindings} bindings: {cats} cat dispatches");
        for shader in &plan.shaders {
            crate::codegen::tests::validate_wgsl(&shader.wgsl);
        }
    }
}

#[test]
fn reports_missing_and_unknown_input_shapes() {
    let mut opts = options("x", &[2, 4]);
    opts.input_shapes.clear();
    assert!(matches!(
        compile(&identity(), &opts),
        Err(CompileError::MissingInputShape(name)) if name == "x"
    ));

    let mut opts = options("x", &[2, 4]);
    opts.input_shapes.insert("y".to_owned(), vec![2]);
    assert!(matches!(
        compile(&identity(), &opts),
        Err(CompileError::UnknownInput(name)) if name == "y"
    ));
}

#[test]
fn validate_requires_an_output_and_the_core_opset() {
    assert!(validate(&identity()).is_ok());

    let mut graph = identity();
    graph.outputs.clear();
    assert!(matches!(
        validate(&graph),
        Err(CompileError::InvalidOp { message, .. }) if message.contains("needs an output")
    ));

    let mut graph = identity();
    graph.opset = "other/1".to_owned();
    assert!(matches!(
        validate(&graph),
        Err(CompileError::InvalidOp { message, .. }) if message.contains("opset")
    ));
}

#[test]
fn reports_an_arena_larger_than_the_buffer_limit() {
    // Every individual buffer fits, but the eight cumsums are all alive at the
    // cat dispatch, so the arena exceeds the limit.
    let mut nodes: Vec<Json> = (0..8)
        .map(|i| {
            node(
                "aten.cumsum.default",
                ["x"],
                &format!("r{i}"),
                &[256],
                json!({"dim": 0}),
            )
        })
        .collect();
    nodes.push(json!({
        "op": "aten.cat.default",
        "inputs": (0..8).map(|i| format!("r{i}")).collect::<Vec<_>>(),
        "outputs": [value("out", &[2048])],
        "attributes": {"dim": 0},
    }));
    let graph = graph(&[256], &[], nodes, &[("out", &[2048])]);
    let mut opts = options("x", &[256]);
    opts.limits.max_buffer_size = 9000;
    assert!(matches!(
        compile(&graph, &opts),
        Err(CompileError::ArenaTooLarge { .. })
    ));
}

#[test]
fn compiles_an_i8_parameter_through_dequantize_per_channel() {
    let dequant = |inputs: Json, outputs: Json| {
        json!({
            "op": "avel.dequantize_per_channel.default",
            "inputs": inputs,
            "outputs": outputs,
            "attributes": {"axis": 0},
        })
    };
    let parameters = json!([
        {"name": "q", "tensor": "q", "dtype": "i8", "shape": [8]},
        {"name": "scales", "tensor": "scales", "dtype": "f32", "shape": [8]},
    ]);
    let graph: Graph = serde_json::from_value(json!({
        "formatVersion": FORMAT_VERSION,
        "opset": "core-aten/1",
        "inputs": [value("x", &[8])],
        "outputs": [value("out", &[8])],
        "nodes": [
            dequant(json!(["q", "scales"]), json!([value("dequant", &[8])])),
            node("aten.relu.default", ["dequant"], "out", &[8], json!({})),
        ],
        "parameters": parameters,
    }))
    .unwrap();
    let plan = compile_x(&graph, &[8]);
    assert_eq!(plan.outputs[0].shape, [8]);
    for shader in &plan.shaders {
        crate::codegen::tests::validate_wgsl(&shader.wgsl);
    }

    // An i8 parameter read anywhere else is rejected.
    let bad: Graph = serde_json::from_value(json!({
        "formatVersion": FORMAT_VERSION,
        "opset": "core-aten/1",
        "inputs": [value("x", &[8])],
        "outputs": [value("out", &[8])],
        "nodes": [node("aten.relu.default", ["q"], "out", &[8], json!({}))],
        "parameters": parameters,
    }))
    .unwrap();
    assert!(matches!(
        compile(&bad, &options("x", &[8])),
        Err(CompileError::UnsupportedDtype {
            dtype: Dtype::I8,
            ..
        })
    ));
}
