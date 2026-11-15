use super::*;
use crate::graph::Attr;
use crate::ops::Movement;
use crate::plan::RequestedShapes;
use std::borrow::Cow;

/// Maps each symbolic input dimension name to its concrete size, so symbolic
/// integer attributes (`{"$sym": "2*s27"}`) resolve against the shapes the
/// plan is compiled for.
pub(super) fn symbols(
    graph: &Graph,
    input_shapes: &BTreeMap<String, Vec<u64>>,
) -> Result<HashMap<String, u64>, CompileError> {
    let mut symbols = HashMap::new();
    for input in &graph.inputs {
        let shape = input_shapes
            .get(&input.name)
            .ok_or_else(|| CompileError::MissingInputShape(input.name.clone()))?;
        for (dim, &size) in input.shape.iter().zip(shape) {
            if let Dim::Symbolic { name, .. } = dim {
                if let Some(&known) = symbols.get(name) {
                    if known != size {
                        return Err(CompileError::InconsistentSymbol(name.clone()));
                    }
                } else {
                    symbols.insert(name.clone(), size);
                }
            }
        }
    }
    Ok(symbols)
}

/// The shapes to compile for: the requested ones, with every symbolic input
/// dimension snapped up to a multiple of the bucket, capped by its declared
/// maximum.
pub(super) fn effective_shapes<'a>(
    graph: &Graph,
    options: &'a CompileOptions,
) -> Cow<'a, BTreeMap<String, Vec<u64>>> {
    let Some(bucket) = options.bucket else {
        return Cow::Borrowed(&options.input_shapes);
    };
    Cow::Owned(
        graph
            .inputs
            .iter()
            .map(|input| {
                let requested = &options.input_shapes[&input.name];
                let shape = input
                    .shape
                    .iter()
                    .zip(requested)
                    .map(|(dim, &size)| match dim {
                        Dim::Symbolic { max, .. } => {
                            let snapped = size.div_ceil(bucket).saturating_mul(bucket);
                            match max {
                                Some(max) if snapped > *max => *max,
                                _ => snapped,
                            }
                        }
                        Dim::Fixed(_) => size,
                    })
                    .collect();
                (input.name.clone(), shape)
            })
            .collect(),
    )
}

/// With a bucket, resolves the graph once at the requested shapes, to record
/// them in the plan: the runtime pads inputs up to the bucketed shapes and
/// trims outputs back down to these.
pub(super) fn requested_shapes(
    graph: &Graph,
    options: &CompileOptions,
    symbols: &HashMap<String, u64>,
) -> Result<Option<RequestedShapes>, CompileError> {
    let Some(bucket) = options.bucket else {
        return Ok(None);
    };
    if bucket == 0 {
        return Err(CompileError::InvalidOp {
            op: "compile options".to_owned(),
            message: "the bucket must be at least 1".to_owned(),
        });
    }
    let mut builder = Builder::new(&options.limits, graph);
    add_inputs(graph, &options.input_shapes, &mut builder)?;
    builder.add_parameters(graph)?;
    let mut custom = options.custom_kernels.clone();
    for kernel in custom.values_mut() {
        if let Some(shape) = kernel.requested_shape.take() {
            kernel.shape = shape;
        }
    }
    let resolved = resolve_nodes(graph, &builder, symbols, &custom)?;
    let mut shapes: HashMap<&str, Vec<u64>> = builder
        .values
        .iter()
        .map(|(&name, value)| (name, value.view.shape.clone()))
        .collect();
    for node in &resolved {
        for (name, output) in &node.outputs {
            if let Some(name) = name {
                shapes.insert(name, output.shape.clone());
            }
        }
    }
    let inputs = graph
        .inputs
        .iter()
        .map(|input| options.input_shapes[&input.name].clone())
        .collect();
    let outputs: Vec<Vec<u64>> = graph
        .outputs
        .iter()
        .map(|output| {
            shapes
                .get(output.name.as_str())
                .cloned()
                .ok_or_else(|| CompileError::UnknownOutput(output.name.clone()))
        })
        .collect::<Result<_, _>>()?;
    // Readback trimming evaluates this same bounded expression language in JS.
    // Reject expressions it cannot resolve instead of returning padded outputs.
    for (info, shape) in graph.outputs.iter().zip(&outputs) {
        check_shape(info, shape)?;
        for (dim, &size) in info.shape.iter().zip(shape) {
            if let Dim::Symbolic { name, .. } = dim {
                let expected =
                    ops::eval_symbol(name, symbols).map_err(|message| CompileError::InvalidOp {
                        op: "bucket output".into(),
                        message,
                    })?;
                if expected != size {
                    return Err(CompileError::InvalidOp {
                        op: "bucket output".into(),
                        message: format!(
                            "symbolic output `{name}` does not match inferred size {size}"
                        ),
                    });
                }
            }
        }
    }
    Ok(Some(RequestedShapes { inputs, outputs }))
}

/// Padding is only valid when real coordinates cannot read padded positions.
/// Keep unsupported cases explicit instead of silently changing model results.
pub(super) fn validate_bucketing(
    graph: &Graph,
    nodes: &[Resolved<'_>],
) -> Result<(), CompileError> {
    let info: HashMap<&str, &ValueInfo> = graph
        .inputs
        .iter()
        .chain(graph.nodes.iter().flat_map(|n| n.outputs.iter().flatten()))
        .map(|v| (v.name.as_str(), v))
        .collect();
    let symbolic = |name: &str, axis: usize| {
        info.get(name)
            .and_then(|v| v.shape.get(axis))
            .is_some_and(|d| matches!(d, Dim::Symbolic { .. }))
    };
    let symbol = |name: &str, axis: usize| info.get(name).and_then(|v| v.shape.get(axis));
    // Reshapes may split/merge fixed axes while preserving each symbolic
    // axis and its contiguous block. Moving a variable row boundary is unsafe.
    let pattern = |value: &ValueInfo| {
        let mut fixed = 1u64;
        let mut axes = Vec::new();
        for dim in value.shape.iter().rev() {
            match dim {
                Dim::Fixed(n) => fixed = fixed.saturating_mul(*n),
                Dim::Symbolic { name, .. } => {
                    axes.push((name.clone(), fixed));
                    fixed = 1;
                }
            }
        }
        (axes, fixed)
    };
    for node in nodes {
        let input = |i: usize| node.inputs[i];
        let variable = |i: usize, axis: usize| symbolic(input(i), axis);
        let rank = |i: usize| info.get(input(i)).map_or(0, |v| v.shape.len());
        let unsafe_attributes = node.node.attributes.iter().any(|(key, value)| {
            // These attributes only determine the extent of a view. The
            // shape/axis checks below still apply to the resulting view.
            let extent = matches!(
                (&node.op, key.as_str()),
                (Op::View(ViewOp::Reshape | ViewOp::Expand), "size")
                    | (Op::View(ViewOp::Slice { .. }), "end")
            );
            !extent && !matches!(node.op, Op::Custom(_)) && contains_symbol(value)
        });
        let unsafe_operation = match &node.op {
            Op::Reduce { axes, .. } => axes.iter().any(|&axis| variable(0, axis)),
            Op::Row { axes, kind } if !matches!(kind, ops::RowKind::Scan { .. }) => {
                axes.iter().any(|&axis| variable(0, axis))
            }
            Op::Rotary => variable(0, node.shape().len() - 1),
            Op::Matmul { bias, .. } => {
                // The reduced axis: the last of the left operand, the one
                // before the last of the right one.
                let left = usize::from(*bias);
                variable(left, rank(left).saturating_sub(1))
                    || variable(left + 1, rank(left + 1).saturating_sub(2))
            }
            Op::Convolution(_) => (1..rank(0)).any(|axis| variable(0, axis)),
            Op::Pool(pool) => {
                let spatial = pool.kernel.len().max(2);
                (rank(0).saturating_sub(spatial)..rank(0)).any(|axis| variable(0, axis))
            }
            Op::Order { axis, .. } => variable(0, *axis),
            Op::CacheWrite | Op::CachedAttention { .. } | Op::SampleCandidates | Op::Scatter(_) => {
                (0..node.inputs.len()).any(|i| (0..rank(i)).any(|axis| variable(i, axis)))
            }
            Op::BatchNorm { .. } => variable(0, 1),
            // Batch statistics span every axis but channels; group statistics
            // span channels and spatial axes.
            Op::GroupNorm { across_batch, .. } => (0..rank(0))
                .any(|axis| variable(0, axis) && if *across_batch { axis != 1 } else { axis != 0 }),
            Op::Movement(movement) => match movement {
                // Padding is appended: a later copy, a reversal, a wrap or an
                // indexed read (negative indices wrap) would expose it.
                Movement::Repeat => {
                    let repeats = match node.node.attributes.get("repeats") {
                        Some(Attr::List(items)) => items.iter().map(Attr::as_int).collect(),
                        _ => Vec::new(),
                    };
                    let lead = repeats.len().saturating_sub(rank(0));
                    (0..rank(0))
                        .any(|axis| variable(0, axis) && repeats.get(lead + axis) != Some(&Some(1)))
                }
                Movement::Flip(axes) => axes.iter().any(|&axis| variable(0, axis)),
                Movement::Roll { shifts, flattened } => {
                    (0..rank(0)).any(|axis| variable(0, axis))
                        && (*flattened || shifts.iter().any(|&(axis, _)| variable(0, axis)))
                }
                // Only right padding lands after the bucket padding; circular
                // left padding reads the end of the axis.
                Movement::Pad(pad) => {
                    pad.widths.iter().enumerate().any(|(axis, &(left, right))| {
                        variable(0, axis)
                            && (right != 0 || (left != 0 && pad.mode == ops::PadMode::Circular))
                    })
                }
                Movement::IndexSelect(axis) | Movement::Gather(axis) => variable(0, *axis),
                Movement::Index { axes, .. } => axes.iter().any(|&axis| variable(0, axis)),
                // Interpolation weights and adaptive bins depend on the size.
                Movement::Resize(_) | Movement::AdaptiveAverage => {
                    (rank(0).saturating_sub(2)..rank(0)).any(|axis| variable(0, axis))
                }
            },
            Op::LayerNorm { normalized, .. } => {
                (rank(0).saturating_sub(normalized.len())..rank(0)).any(|axis| variable(0, axis))
            }
            Op::Cat { axis, .. } => {
                node.inputs.len() > 1 && (0..node.inputs.len()).any(|i| variable(i, *axis))
            }
            Op::Attention { causal, mask, .. } => {
                let key_axis = rank(1).saturating_sub(2);
                let query_axis = rank(0).saturating_sub(2);
                let same_symbol = symbol(input(0), query_axis) == symbol(input(1), key_axis);
                let bool_mask = *mask == Some(ops::MaskKind::Select)
                    && symbol(input(3), rank(3).saturating_sub(1)) == symbol(input(1), key_axis);
                variable(0, rank(0).saturating_sub(1))
                    || variable(1, rank(1).saturating_sub(1))
                    || (variable(1, key_axis) && !(*causal && same_symbol || bool_mask))
            }
            // Intermediate flatten/unflatten pairs can restore row boundaries
            // (e.g. transformer projections); a returned reshape cannot.
            Op::View(ViewOp::Reshape) => {
                graph.outputs.iter().any(|v| v.name == node.name())
                    && info.get(input(0)).is_some_and(|source| {
                        source
                            .shape
                            .iter()
                            .any(|d| matches!(d, Dim::Symbolic { .. }))
                            && pattern(source) != pattern(info[node.name()])
                    })
            }
            Op::View(ViewOp::Select { axis, .. }) => {
                variable(0, *axis)
                    && node
                        .node
                        .attributes
                        .get("index")
                        .and_then(Attr::as_int)
                        .is_some_and(|n| n < 0)
            }
            Op::View(ViewOp::Slice { axis, .. }) => {
                variable(0, *axis)
                    && ["start", "end"].iter().any(|key| {
                        node.node
                            .attributes
                            .get(*key)
                            .and_then(Attr::as_int)
                            .is_some_and(|n| n < 0)
                    })
            }
            // Elementwise and per-element lookups keep each real element's
            // inputs real. List every op so new ones must decide.
            Op::Custom(_)
            | Op::Unary(_)
            | Op::Binary { .. }
            | Op::Power { .. }
            | Op::Row { .. }
            | Op::View(_)
            | Op::Embedding { .. }
            | Op::Compare { .. }
            | Op::Where
            | Op::Bitwise(_)
            | Op::Cast(_)
            | Op::Dequantize(_)
            | Op::Arange { .. }
            | Op::Full { .. } => false,
        };
        if unsafe_attributes || unsafe_operation {
            return Err(CompileError::InvalidOp { op: node.node.op.clone(), message: "bucket padding can change this operation's result; compile exact shapes instead".into() });
        }
    }
    Ok(())
}

fn contains_symbol(attr: &Attr) -> bool {
    match attr {
        Attr::Sym(_) => true,
        Attr::List(items) => items.iter().any(contains_symbol),
        Attr::Map(items) => items.values().any(contains_symbol),
        _ => false,
    }
}
