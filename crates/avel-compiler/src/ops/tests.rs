use std::collections::BTreeMap;

use serde_json::{Value, json};

use super::*;

// Direct path into the exporter, not a symlink: symlinks break on checkouts
// that lose them (Windows, source archives).
const EXPORTER_OPS: &str = include_str!("../../../../python/avel-export/src/avel_export/ops.py");

/// A node with one present tensor argument per shape, unless `attributes`
/// holds an `inputs` list marking which slots are present.
fn node(op: &str, mut attributes: Value, tensors: usize) -> Node {
    let slots: Vec<bool> = match attributes.as_object_mut().unwrap().remove("inputs") {
        Some(slots) => serde_json::from_value(slots).unwrap(),
        None => vec![true; tensors],
    };
    let attributes: BTreeMap<String, Attr> = serde_json::from_value(attributes).unwrap();
    Node {
        op: op.to_owned(),
        inputs: (0..slots.len())
            .map(|i| slots[i].then(|| format!("t{i}")))
            .collect(),
        outputs: Vec::new(),
        attributes,
    }
}

fn resolved(
    op: &str,
    attributes: Value,
    shapes: &[&[u64]],
) -> Result<(Op, Vec<u64>), CompileError> {
    resolved_with(op, attributes, shapes, &[], &HashMap::new())
}

fn resolved_with(
    op: &str,
    attributes: Value,
    shapes: &[&[u64]],
    dtypes: &[Dtype],
    symbols: &HashMap<String, u64>,
) -> Result<(Op, Vec<u64>), CompileError> {
    let dtypes = if dtypes.is_empty() {
        vec![Dtype::F32; shapes.len()]
    } else {
        dtypes.to_vec()
    };
    resolve(
        &node(op, attributes, shapes.len()),
        shapes,
        &dtypes,
        symbols,
    )
    .map(|(op, outputs)| (op, outputs[0].shape.clone()))
}

fn shape_of(op: &str, attributes: Value, shapes: &[&[u64]]) -> Result<Vec<u64>, CompileError> {
    resolved(op, attributes, shapes).map(|(_, shape)| shape)
}

fn shape_of_with(
    op: &str,
    attributes: Value,
    shapes: &[&[u64]],
    symbols: &HashMap<String, u64>,
) -> Result<Vec<u64>, CompileError> {
    resolved_with(op, attributes, shapes, &[], symbols).map(|(_, shape)| shape)
}

fn invalid(result: Result<Vec<u64>, CompileError>) -> bool {
    matches!(result, Err(CompileError::InvalidOp { .. }))
}

#[test]
fn matches_the_exporter_op_list() {
    let (supported, preserved) = EXPORTER_OPS
        .split_once("PRESERVED_OPS =")
        .expect("the exporter lists preserved ops after supported ones");
    for (list, ops) in [(supported, SUPPORTED_OPS), (preserved, PRESERVED_OPS)] {
        assert_eq!(
            list.lines()
                .filter(|line| {
                    let line = line.trim_start();
                    line.starts_with("\"aten.") || line.starts_with("\"avel.")
                })
                .count(),
            ops.len()
        );
        for op in ops {
            assert!(list.contains(&format!("\"{op}\"")), "{op}");
        }
    }
    // A preserved op reaches the compiler whole, so it must be supported.
    for op in PRESERVED_OPS {
        assert!(SUPPORTED_OPS.contains(op), "{op}");
    }
}

#[test]
fn rejects_overflowing_split_sizes() {
    assert!(invalid(shape_of(
        "aten.split_with_sizes.default",
        json!({"split_sizes": [i64::MAX, i64::MAX, 5], "dim": 0}),
        &[&[3]]
    )));
}

#[test]
fn rejects_comparisons_with_the_wrong_tensor_count() {
    for shapes in [vec![], vec![&[3][..], &[3][..], &[3][..]]] {
        assert!(invalid(shape_of("aten.eq.Tensor", json!({}), &shapes)));
    }
}

#[test]
fn checks_boolean_subtraction_and_alpha_types_like_pytorch() {
    for (op, attributes, dtypes) in [
        ("aten.sub.Tensor", json!({}), [Dtype::Bool, Dtype::I32]),
        (
            "aten.add.Tensor",
            json!({"alpha": true}),
            [Dtype::F32, Dtype::F32],
        ),
        (
            "aten.add.Tensor",
            json!({"alpha": 0.5}),
            [Dtype::I64, Dtype::I64],
        ),
    ] {
        assert!(resolved_with(op, attributes, &[&[3], &[3]], &dtypes, &HashMap::new()).is_err());
    }
}

#[test]
fn infers_addmm() {
    let attributes = json!({"beta": 1, "alpha": 1});
    let shape = shape_of("aten.addmm.default", attributes, &[&[8], &[3, 4], &[4, 8]]);
    assert_eq!(shape.unwrap(), [3, 8]);
}

#[test]
fn rejects_mismatched_matmul_and_bias() {
    let attributes = json!({"beta": 1, "alpha": 1});
    let wrong_inner = shape_of(
        "aten.addmm.default",
        attributes.clone(),
        &[&[8], &[3, 4], &[5, 8]],
    );
    let wrong_bias = shape_of("aten.addmm.default", attributes, &[&[7], &[3, 4], &[4, 8]]);
    assert!(invalid(wrong_inner));
    assert!(invalid(wrong_bias));
}

#[test]
fn infers_permute_with_negative_axes() {
    let shape = shape_of(
        "aten.permute.default",
        json!({"dims": [-1, 0, 1]}),
        &[&[2, 3, 4]],
    );
    assert_eq!(shape.unwrap(), [4, 2, 3]);
}

#[test]
fn rejects_repeated_permute_axes() {
    let shape = shape_of("aten.permute.default", json!({"dims": [0, 0]}), &[&[2, 3]]);
    assert!(invalid(shape));
}

#[test]
fn infers_reductions() {
    let keep = json!({"dim": [1], "keepdim": true});
    let drop = json!({"dim": [-1], "keepdim": false});
    let all = json!({"dim": [], "keepdim": false});
    assert_eq!(shape_of("aten.mean.dim", keep, &[&[2, 3]]).unwrap(), [2, 1]);
    assert_eq!(
        shape_of("aten.sum.dim_IntList", drop, &[&[2, 3]]).unwrap(),
        [2]
    );
    assert_eq!(
        shape_of("aten.sum.dim_IntList", all, &[&[2, 3]]).unwrap(),
        [0u64; 0]
    );
    assert_eq!(
        shape_of("aten.mean.default", json!({}), &[&[2, 3]]).unwrap(),
        [0u64; 0]
    );
}

#[test]
fn broadcasts_binary_operands_and_scalars() {
    let add = shape_of(
        "aten.add.Tensor",
        json!({"alpha": 2}),
        &[&[2, 1, 4], &[3, 1]],
    );
    assert_eq!(add.unwrap(), [2, 3, 4]);
    let (op, shape) = resolved("aten.div.Tensor", json!({"other": 6}), &[&[5]]).unwrap();
    assert_eq!(shape, [5]);
    assert_eq!(
        op,
        Op::Binary {
            kind: BinaryKind::Div,
            scalar: Some(Number::Int(6)),
            alpha: Number::Int(1)
        }
    );
    assert!(invalid(shape_of(
        "aten.mul.Tensor",
        json!({}),
        &[&[2, 3], &[4]]
    )));
    assert!(invalid(shape_of("aten.sub.Tensor", json!({}), &[&[2, 3]])));
}

#[test]
fn reads_clamp_bounds() {
    let (op, _) = resolved(
        "aten.clamp.default",
        json!({"min": 0, "max": null}),
        &[&[3]],
    )
    .unwrap();
    assert_eq!(
        op,
        Op::Unary(Unary::Clamp {
            min: Some(Number::Int(0)),
            max: None
        })
    );
    let (op, _) = resolved(
        "aten.hardtanh.default",
        json!({"min_val": 0, "max_val": 6}),
        &[&[3]],
    )
    .unwrap();
    assert_eq!(
        op,
        Op::Unary(Unary::Clamp {
            min: Some(Number::Int(0)),
            max: Some(Number::Int(6))
        })
    );
}

#[test]
fn infers_convolutions() {
    let conv = json!({
        "inputs": [true, true, false], "stride": [2, 2], "padding": [3, 3], "dilation": [1, 1],
        "transposed": false, "output_padding": [0, 0], "groups": 1
    });
    let shape = shape_of(
        "aten.convolution.default",
        conv.clone(),
        &[&[1, 3, 224, 224], &[64, 3, 7, 7]],
    );
    assert_eq!(shape.unwrap(), [1, 64, 112, 112]);

    let depthwise = json!({"stride": [1], "padding": [2], "dilation": [2], "groups": 8});
    let shape = shape_of(
        "aten.convolution.default",
        depthwise,
        &[&[2, 8, 10], &[16, 1, 3], &[16]],
    );
    assert_eq!(shape.unwrap(), [2, 16, 10]);

    let mut transposed = conv.clone();
    transposed["transposed"] = json!(true);
    let shapes: &[&[u64]] = &[&[1, 3, 8, 8], &[4, 3, 3, 3]];
    assert!(invalid(shape_of(
        "aten.convolution.default",
        transposed,
        shapes
    )));
    let mut grouped = conv;
    grouped["groups"] = json!(2);
    assert!(invalid(shape_of(
        "aten.convolution.default",
        grouped,
        shapes
    )));
}

#[test]
fn checks_batch_norm_statistics() {
    let attributes =
        json!({"inputs": [true, false, false, true, true], "momentum": 0.1, "eps": 1e-5});
    let (op, shape) = resolved(
        "aten._native_batch_norm_legit_no_training.default",
        attributes.clone(),
        &[&[2, 3, 4, 4], &[3], &[3]],
    )
    .unwrap();
    assert_eq!(shape, [2, 3, 4, 4]);
    assert_eq!(
        op,
        Op::BatchNorm {
            eps: 1e-5,
            weight: false,
            bias: false
        }
    );

    let wrong = shape_of(
        "aten._native_batch_norm_legit_no_training.default",
        attributes,
        &[&[2, 3, 4, 4], &[3], &[4]],
    );
    assert!(invalid(wrong));
}

#[test]
fn infers_pooling_windows() {
    let max = json!({"kernel_size": [3, 3], "stride": [2, 2], "padding": [1, 1]});
    let shape = shape_of(
        "aten.max_pool2d_with_indices.default",
        max,
        &[&[1, 64, 112, 112]],
    );
    assert_eq!(shape.unwrap(), [1, 64, 56, 56]);

    let ceil = json!({"kernel_size": [2, 2], "stride": [], "ceil_mode": true});
    let shape = shape_of("aten.max_pool2d_with_indices.default", ceil, &[&[3, 5, 7]]);
    assert_eq!(shape.unwrap(), [3, 3, 4]);

    let avg = json!({"kernel_size": [3], "stride": [2], "padding": [1], "ceil_mode": true});
    let shape = shape_of("aten.avg_pool2d.default", avg, &[&[1, 2, 6, 6]]);
    assert_eq!(shape.unwrap(), [1, 2, 4, 4]);

    // The last window would start in the right padding, so it is dropped.
    let avg = json!({"kernel_size": [2], "stride": [2], "padding": [1], "ceil_mode": true});
    let shape = shape_of("aten.avg_pool2d.default", avg, &[&[1, 1, 5, 5]]);
    assert_eq!(shape.unwrap(), [1, 1, 3, 3]);

    let padded = json!({"kernel_size": [2, 2], "padding": [2, 2]});
    assert!(invalid(shape_of(
        "aten.avg_pool2d.default",
        padded,
        &[&[1, 1, 4, 4]]
    )));
}

#[test]
fn infers_layer_norms() {
    let affine = json!({"normalized_shape": [64], "eps": 1e-5});
    let (op, shape) = resolved(
        "aten.native_layer_norm.default",
        affine.clone(),
        &[&[2, 17, 64], &[64], &[64]],
    )
    .unwrap();
    assert_eq!(shape, [2, 17, 64]);
    assert_eq!(
        op,
        Op::LayerNorm {
            eps: 1e-5,
            weight: true,
            bias: true,
            normalized: vec![64],
            rms: false,
        }
    );
    let bare = resolved(
        "aten.native_layer_norm.default",
        json!({"normalized_shape": [4, 8], "inputs": [true, false, false], "eps": 1e-5}),
        &[&[2, 3, 4, 8]],
    )
    .unwrap();
    assert_eq!(
        bare.0,
        Op::LayerNorm {
            eps: 1e-5,
            weight: false,
            bias: false,
            normalized: vec![4, 8],
            rms: false,
        }
    );
    assert!(invalid(shape_of(
        "aten.native_layer_norm.default",
        json!({"normalized_shape": [5], "eps": 1e-5}),
        &[&[2, 3, 5], &[5]],
    )));
    assert!(invalid(shape_of(
        "aten.native_layer_norm.default",
        json!({"normalized_shape": [4], "eps": 1e-5}),
        &[&[2, 3, 4], &[3]],
    )));
}

#[test]
fn infers_attention() {
    // Without a mask, the fourth tensor slot is empty.
    let (op, shape) = resolved(
        "aten.scaled_dot_product_attention.default",
        json!({"inputs": [true, true, true, false], "dropout_p": 0, "is_causal": false, "scale": null}),
        &[&[2, 4, 17, 16], &[2, 4, 17, 16], &[2, 4, 17, 32]],
    )
    .unwrap();
    assert_eq!(shape, [2, 4, 17, 32]);
    assert_eq!(
        op,
        Op::Attention {
            scale: 0.25,
            causal: false,
            mask: None,
        }
    );

    let causal = resolved(
        "aten.scaled_dot_product_attention.default",
        json!({"inputs": [true, true, true, false], "dropout_p": 0, "is_causal": true, "scale": 0.5}),
        &[&[1, 2, 9, 8], &[1, 2, 9, 8], &[1, 2, 9, 8]],
    )
    .unwrap();
    assert_eq!(
        causal.0,
        Op::Attention {
            scale: 0.5,
            causal: true,
            mask: None,
        }
    );

    // An additive f32 mask and a bool mask, broadcast over heads and queries.
    for (mask, kind, mask_dtype) in [
        (&[2, 1, 1, 17][..], MaskKind::Additive, Dtype::F32),
        (&[1, 1, 17, 17][..], MaskKind::Select, Dtype::Bool),
    ] {
        let dtypes = [Dtype::F32, Dtype::F32, Dtype::F32, mask_dtype];
        let (op, shape) = resolved_with(
            "aten.scaled_dot_product_attention.default",
            json!({"dropout_p": 0, "is_causal": false, "scale": null}),
            &[&[2, 4, 17, 16], &[2, 4, 17, 16], &[2, 4, 17, 16], mask],
            &dtypes,
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(shape, [2, 4, 17, 16]);
        assert_eq!(
            op,
            Op::Attention {
                scale: 0.25,
                causal: false,
                mask: Some(kind),
            }
        );
    }

    assert!(invalid(shape_of(
        "aten.scaled_dot_product_attention.default",
        json!({"inputs": [true, true, true, false], "dropout_p": 0, "is_causal": false, "enable_gqa": true}),
        &[&[1, 4, 9, 8], &[1, 2, 9, 8], &[1, 2, 9, 8]],
    )));
    assert!(invalid(shape_of(
        "aten.scaled_dot_product_attention.default",
        json!({"inputs": [true, true, true, false], "dropout_p": 0, "is_causal": false}),
        &[&[1, 4, 9, 8], &[1, 4, 9, 16], &[1, 4, 9, 8]],
    )));
}

#[test]
fn infers_embeddings() {
    let (op, shape) = resolved_with(
        "aten.embedding.default",
        json!({"padding_idx": -1, "scale_grad_by_freq": false, "sparse": false}),
        &[&[128, 64], &[2, 17]],
        &[Dtype::F32, Dtype::I32],
        &HashMap::new(),
    )
    .unwrap();
    assert_eq!(shape, [2, 17, 64]);
    assert_eq!(op, Op::Embedding { padding: None });

    let padded = resolved_with(
        "aten.embedding.default",
        json!({"padding_idx": 3, "scale_grad_by_freq": false, "sparse": false}),
        &[&[128, 64], &[17]],
        &[Dtype::F32, Dtype::I32],
        &HashMap::new(),
    )
    .unwrap();
    assert_eq!(padded, (Op::Embedding { padding: Some(3) }, vec![17, 64]));

    assert!(invalid(shape_of(
        "aten.embedding.default",
        json!({"padding_idx": -1}),
        &[&[128, 64], &[2, 17]],
    )));
}

#[test]
fn infers_concatenations() {
    let (op, shape) = resolved(
        "aten.cat.default",
        json!({"dim": 1}),
        &[&[2, 1, 8], &[2, 4, 8], &[2, 3, 8]],
    )
    .unwrap();
    assert_eq!(shape, [2, 8, 8]);
    assert_eq!(
        op,
        Op::Cat {
            axis: 1,
            sizes: vec![1, 4, 3]
        }
    );

    let negative = resolved("aten.cat.default", json!({"dim": -1}), &[&[2, 3], &[2, 5]]).unwrap();
    assert_eq!(negative.1, [2, 8]);
    assert!(invalid(shape_of(
        "aten.cat.default",
        json!({"dim": 0}),
        &[&[2, 3], &[4, 5]]
    )));
    assert!(invalid(shape_of(
        "aten.cat.default",
        json!({}),
        &[&[2, 3], &[2, 4, 5]]
    )));
}

#[test]
fn resolves_symbolic_attributes_against_input_dims() {
    let symbols: HashMap<String, u64> = [("s27".to_owned(), 17)].into_iter().collect();
    // `view` sizes and a slice bound written as expressions over `s27`.
    let shape = shape_of_with(
        "aten.view.default",
        json!({"size": [{"$sym": "2*s27"}, 64]}),
        &[&[2, 17, 64]],
        &symbols,
    )
    .unwrap();
    assert_eq!(shape, [34, 64]);
    let shape = shape_of_with(
        "aten.slice.Tensor",
        json!({"dim": 1, "start": 0, "end": {"$sym": "s27"}, "step": 1}),
        &[&[1, 64, 64]],
        &symbols,
    )
    .unwrap();
    assert_eq!(shape, [1, 17, 64]);
    assert!(
        shape_of_with(
            "aten.slice.Tensor",
            json!({"dim": 1, "start": 0, "end": {"$sym": "s28 + 1"}}),
            &[&[1, 64, 64]],
            &symbols,
        )
        .is_err()
    );
}

#[test]
fn evaluates_symbol_expressions() {
    let symbols: HashMap<String, u64> = [("s27".to_owned(), 17), ("b".to_owned(), 2)]
        .into_iter()
        .collect();
    assert_eq!(eval_symbol("s27", &symbols).unwrap(), 17);
    assert_eq!(eval_symbol("2*s27", &symbols).unwrap(), 34);
    assert_eq!(eval_symbol("b*s27 + 3", &symbols).unwrap(), 37);
    assert_eq!(eval_symbol("64", &symbols).unwrap(), 64);
    assert!(eval_symbol("s28", &symbols).is_err());
    assert!(eval_symbol("s27*", &symbols).is_err());
}

#[test]
fn infers_view_shapes() {
    let shape = shape_of("aten.view.default", json!({"size": [2, -1]}), &[&[2, 3, 4]]);
    assert_eq!(shape.unwrap(), [2, 12]);
    assert!(invalid(shape_of(
        "aten.view.default",
        json!({"size": [5, -1]}),
        &[&[2, 3]]
    )));

    let shape = shape_of(
        "aten.expand.default",
        json!({"size": [2, -1, 4]}),
        &[&[3, 1]],
    );
    assert_eq!(shape.unwrap(), [2, 3, 4]);
    assert!(invalid(shape_of(
        "aten.expand.default",
        json!({"size": [4, 4]}),
        &[&[3, 1]]
    )));

    let slice = json!({"dim": 1, "start": 1, "end": 9007199254740991_i64, "step": 2});
    let (op, shape) = resolved("aten.slice.Tensor", slice, &[&[2, 6]]).unwrap();
    assert_eq!(shape, [2, 3]);
    assert_eq!(
        op,
        Op::View(ViewOp::Slice {
            axis: 1,
            start: 1,
            step: 2
        })
    );
    let empty = json!({"dim": 0, "start": -1, "end": 0});
    assert!(invalid(shape_of("aten.slice.Tensor", empty, &[&[4]])));

    let (op, shape) = resolved(
        "aten.select.int",
        json!({"dim": 0, "index": -1}),
        &[&[3, 2]],
    )
    .unwrap();
    assert_eq!(
        (op, shape),
        (Op::View(ViewOp::Select { axis: 0, index: 2 }), vec![2])
    );

    let shape = shape_of("aten.unsqueeze.default", json!({"dim": -1}), &[&[2, 3]]);
    assert_eq!(shape.unwrap(), [2, 3, 1]);
    let shape = shape_of("aten.squeeze.dims", json!({"dim": [0, 1]}), &[&[1, 3, 1]]);
    assert_eq!(shape.unwrap(), [3, 1]);
}

#[test]
fn rejects_overflowing_shape_attributes_without_panicking() {
    for (op, attrs, shapes) in [
        (
            "aten.view.default",
            json!({"size": [i64::MAX, i64::MAX, 4]}),
            vec![&[4][..]],
        ),
        (
            "aten.convolution.default",
            json!({"inputs": [true, true, false], "stride": [1], "padding": [i64::MAX], "dilation": [1], "groups": 1}),
            vec![&[1, 1, 4][..], &[1, 1, 1][..]],
        ),
        (
            "aten.avg_pool2d.default",
            json!({"kernel_size": [i64::MAX, i64::MAX], "padding": [i64::MAX, i64::MAX]}),
            vec![&[1, 1, 4, 4][..]],
        ),
    ] {
        assert!(invalid(shape_of(op, attrs, &shapes)));
    }
}

#[test]
fn checks_tensor_slots() {
    let conv = |inputs: Value, shapes: &[&[u64]]| {
        shape_of(
            "aten.convolution.default",
            json!({"inputs": inputs, "stride": [1], "padding": [0], "dilation": [1], "groups": 1}),
            shapes,
        )
    };
    assert_eq!(
        conv(json!([true, true, false]), &[&[1, 2, 4], &[3, 2, 1]]).unwrap(),
        [1, 3, 4]
    );
    assert_eq!(
        conv(json!([true, true, true]), &[&[1, 2, 4], &[3, 2, 1], &[3]]).unwrap(),
        [1, 3, 4]
    );
    // Every slot is listed, even an empty trailing one.
    assert!(invalid(conv(
        json!([true, true]),
        &[&[1, 2, 4], &[3, 2, 1]]
    )));
    // Required tensors cannot be left out.
    assert!(invalid(conv(
        json!([true, false, true]),
        &[&[1, 2, 4], &[3]]
    )));
    assert!(invalid(shape_of(
        "aten.relu.default",
        json!({"inputs": [false]}),
        &[],
    )));
}

#[test]
fn rejects_attributes_of_the_wrong_kind() {
    assert!(invalid(shape_of(
        "aten.permute.default",
        json!({"dims": [1.5, 0]}),
        &[&[2, 3]],
    )));
    assert!(invalid(shape_of(
        "aten.mean.dim",
        json!({"dim": [0], "keepdim": 1}),
        &[&[2, 3]],
    )));
    assert!(invalid(shape_of(
        "aten.view.default",
        json!({"size": {"a": 1}}),
        &[&[2, 3]],
    )));
}

#[test]
fn every_supported_op_is_unique_and_dispatched() {
    let mut seen = std::collections::HashSet::new();
    for op in SUPPORTED_OPS {
        assert!(seen.insert(op), "duplicate op {op}");
        let node = Node {
            op: op.to_string(),
            inputs: vec![Some("t0".to_owned()), Some("t1".to_owned())],
            outputs: Vec::new(),
            attributes: BTreeMap::new(),
        };
        // Every listed op must reach its resolver arm: a missing one panics
        // (`unreachable!`). Placeholder inputs may fail shape checks, which
        // still proves the op was dispatched rather than rejected as unknown.
        if let Err(CompileError::UnsupportedOp(name)) = resolve(
            &node,
            &[&[2, 3], &[2, 3]],
            &[Dtype::F32, Dtype::F32],
            &HashMap::new(),
        ) {
            panic!("{name} is listed but not resolved");
        }
    }
}

#[test]
fn resolves_unary_math_ops_elementwise() {
    for op in [
        "aten.abs.default",
        "aten.neg.default",
        "aten.sign.default",
        "aten.tanh.default",
        "aten.exp.default",
        "aten.log.default",
        "aten.sqrt.default",
        "aten.rsqrt.default",
        "aten.reciprocal.default",
        "aten.erf.default",
        "aten.expm1.default",
        "aten.log1p.default",
        "aten.sin.default",
        "aten.cos.default",
        "aten.floor.default",
        "aten.ceil.default",
        "aten.round.default",
        "aten.trunc.default",
    ] {
        assert_eq!(
            shape_of(op, json!({}), &[&[2, 3]]).unwrap(),
            vec![2, 3],
            "{op}"
        );
    }
}

#[test]
fn resolves_scalar_arithmetic_with_its_scalar_operand() {
    let (op, shape) = resolved("aten.add.Scalar", json!({"other": 2}), &[&[2, 3]]).unwrap();
    assert_eq!(shape, [2, 3]);
    assert_eq!(
        op,
        Op::Binary {
            kind: BinaryKind::Add,
            scalar: Some(Number::Int(2)),
            alpha: Number::Int(1)
        }
    );
    assert!(invalid(shape_of("aten.mul.Scalar", json!({}), &[&[2, 3]])));
}

#[test]
fn comparisons_infer_a_bool_output() {
    let node = node("aten.lt.Scalar", json!({"other": 2}), 1);
    let (op, outputs) = resolve(&node, &[&[2, 3]], &[Dtype::F32], &HashMap::new()).unwrap();
    assert_eq!(
        op,
        Op::Compare {
            kind: CompareKind::Lt,
            scalar: Some(Number::Int(2))
        }
    );
    assert_eq!(
        outputs,
        vec![Output {
            shape: vec![2, 3],
            dtype: Dtype::Bool
        }]
    );
}

#[test]
fn infers_gather_and_order_ops() {
    // gather output takes the shape of its integer index tensor.
    let shape = resolved_with(
        "aten.gather.default",
        json!({"dim": 1}),
        &[&[4, 8], &[3, 5]],
        &[Dtype::F32, Dtype::I64],
        &HashMap::new(),
    )
    .unwrap()
    .1;
    assert_eq!(shape, [3, 5]);
    assert!(
        resolved_with(
            "aten.gather.default",
            json!({"dim": 1}),
            &[&[2, 8], &[3, 5]],
            &[Dtype::F32, Dtype::I64],
            &HashMap::new(),
        )
        .is_err()
    );

    let node = node(
        "aten.topk.default",
        json!({"k": 2, "dim": -1, "largest": true, "sorted": true}),
        1,
    );
    let (op, outputs) = resolve(&node, &[&[2, 5]], &[Dtype::F32], &HashMap::new()).unwrap();
    assert_eq!(
        op,
        Op::Order {
            axis: 1,
            largest: true
        }
    );
    assert_eq!(
        outputs,
        vec![
            Output {
                shape: vec![2, 2],
                dtype: Dtype::F32
            },
            Output {
                shape: vec![2, 2],
                dtype: Dtype::I64
            },
        ]
    );
    assert!(invalid(shape_of(
        "aten.topk.default",
        json!({"k": 6, "dim": -1}),
        &[&[2, 5]],
    )));
}
