use super::*;
use crate::kernel::reduce_strategy;
use crate::ops::{Movement, PadMode};

/// A typed constant for a scalar argument.
fn constant(value: f64, ty: ScalarType) -> Result<Expr, String> {
    if ty == ScalarType::I32 && !(i32::MIN as f64..=i32::MAX as f64).contains(&value.trunc()) {
        return Err(format!("scalar {value} does not fit in 32 bits"));
    }
    if ty == ScalarType::F32 && !(value as f32).is_finite() {
        return Err(format!("scalar {value} is not finite in f32"));
    }
    Ok(Expr::Const(match ty {
        ScalarType::F32 => Scalar::F32(value as f32),
        ScalarType::I32 => Scalar::I32(value as i32),
        ScalarType::Bool => Scalar::Bool(value != 0.0),
    }))
}

/// `expr`, of type `from`, converted to `to`.
fn convert(expr: Expr, from: ScalarType, to: ScalarType) -> Expr {
    match from == to {
        true => expr,
        false => Expr::cast(to, expr),
    }
}

/// Lowers one node to views and kernels, in `layout` for computed tensors.
pub(super) fn lower<'a>(
    node: &Resolved<'a>,
    layout: Layout,
    builder: &mut Builder<'a>,
) -> Result<(), CompileError> {
    let (inputs, dtypes): (Vec<View>, Vec<Dtype>) = node
        .inputs
        .iter()
        .map(|name| {
            builder
                .lookup(name)
                .map(|value| (value.view.clone(), value.dtype))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .unzip();
    let name = node.name();
    let shape = node.shape().to_vec();
    let dtype = node.dtype();

    if let Op::View(op) = &node.op {
        let source = &inputs[0];
        let view = match op {
            ViewOp::Split { axis, sizes } => {
                let mut start = 0;
                for ((output, _), &size) in node.outputs.iter().zip(sizes) {
                    if let Some(output) = output {
                        builder.define(output, dtype, source.slice(*axis, start, size, 1));
                    }
                    start += size;
                }
                return Ok(());
            }
            ViewOp::Alias => source.clone(),
            ViewOp::Permute(dims) => source.permute(dims),
            ViewOp::Reshape => match source.reshape(&shape) {
                Some(view) => view,
                // The strides cannot express it: transpose into a row-major copy.
                None => builder
                    .materialize(name, source.clone())?
                    .reshape(&shape)
                    .expect("row-major tensors reshape freely"),
            },
            ViewOp::Expand => source.broadcast_to(&shape),
            &ViewOp::Slice { axis, start, step } => source.slice(axis, start, shape[axis], step),
            &ViewOp::Select { axis, index } => source.select(axis, index),
            &ViewOp::Unsqueeze(axis) => source.unsqueeze(axis),
            ViewOp::Squeeze(axes) => source.squeeze(axes),
        };
        builder.define(name, dtype, view);
        return Ok(());
    }

    if let Op::Custom(kernel) = &node.op {
        // Each declared binding gets a distinct dense f32 buffer, including
        // repeated inputs and strided/media views. No custom ABI knows layouts.
        let inputs = inputs
            .into_iter()
            .enumerate()
            .map(|(i, view)| builder.materialize(&format!("{name}.input{i}"), view))
            .collect::<Result<Vec<_>, _>>()?;
        let output =
            builder.allocate(name, Dtype::F32, shape, Layout::RowMajor, BufferRole::Arena)?;
        builder.emit(Kernel::Custom(kernel.clone()), inputs, output.clone(), name);
        builder.define(name, Dtype::F32, output);
        return Ok(());
    }

    // `i32` and `i64` values share a storage and a compute type: converting
    // between them is a view.
    if let &Op::Cast(target) = &node.op
        && ScalarType::of(target) == ScalarType::of(dtypes[0])
    {
        builder.define(name, target, inputs[0].clone());
        return Ok(());
    }

    if let Op::Movement(kind) = &node.op
        && let Some(view) = movement_view(kind, &inputs[0], &shape)
    {
        builder.define(name, dtype, view);
        return Ok(());
    }

    let output = builder.allocate(name, dtype, shape.clone(), layout, BufferRole::Arena)?;
    let x = Expr::Input(0);
    let ty = ScalarType::of(dtype);
    let constant = |value, ty| {
        constant(value, ty).map_err(|message| CompileError::InvalidOp {
            op: node.node.op.clone(),
            message,
        })
    };
    match &node.op {
        Op::View(_) | Op::Custom(_) => unreachable!("handled above"),
        Op::Unary(unary) => {
            let expr = match *unary {
                Unary::Relu => Expr::binary(BinaryFn::Max, x, Expr::f32(0.0)),
                Unary::Gelu { approximate } => {
                    let function = if approximate {
                        UnaryFn::GeluTanh
                    } else {
                        UnaryFn::Gelu
                    };
                    Expr::unary(function, x)
                }
                Unary::Sigmoid => Expr::unary(UnaryFn::Sigmoid, x),
                Unary::Math(UnaryFn::Sign) if ty == ScalarType::Bool => x,
                Unary::Math(UnaryFn::Floor | UnaryFn::Ceil | UnaryFn::Round | UnaryFn::Trunc)
                    if ty == ScalarType::I32 =>
                {
                    x
                }
                Unary::Math(function) => {
                    Expr::unary(function, convert(x, ScalarType::of(dtypes[0]), ty))
                }
                Unary::Elu {
                    alpha,
                    scale,
                    input_scale,
                } => {
                    let negative = Expr::binary(
                        BinaryFn::Mul,
                        Expr::f32(alpha),
                        Expr::unary(
                            UnaryFn::Expm1,
                            Expr::binary(BinaryFn::Mul, x.clone(), Expr::f32(input_scale)),
                        ),
                    );
                    let piecewise = Expr::select(
                        Expr::binary(BinaryFn::Gt, x.clone(), Expr::f32(0.0)),
                        x,
                        negative,
                    );
                    if scale == 1.0 {
                        piecewise
                    } else {
                        Expr::binary(BinaryFn::Mul, piecewise, Expr::f32(scale))
                    }
                }
                Unary::LeakyRelu { slope } => Expr::select(
                    Expr::binary(BinaryFn::Gt, x.clone(), Expr::f32(0.0)),
                    x.clone(),
                    Expr::binary(BinaryFn::Mul, x, Expr::f32(slope)),
                ),
                Unary::Clamp { min, max } => {
                    let x = convert(x, ScalarType::of(dtypes[0]), ty);
                    let lower = match min {
                        Some(min) => Expr::binary(BinaryFn::Max, x, constant(min.as_f64(), ty)?),
                        None => x,
                    };
                    match max {
                        Some(max) => {
                            Expr::binary(BinaryFn::Min, lower, constant(max.as_f64(), ty)?)
                        }
                        None => lower,
                    }
                }
            };
            let input = inputs[0].broadcast_to(&shape);
            builder.emit(Kernel::ElementWise(expr), vec![input], output.clone(), name);
        }
        &Op::Binary {
            kind,
            scalar,
            alpha,
        } => {
            if ty == ScalarType::F32
                && alpha.as_f64() == 1.0
                && matches!(kind, BinaryKind::Add | BinaryKind::Mul)
                && let Some(folded) = fold_binary_affine(builder, node, &inputs, &shape, kind)?
            {
                builder.define(name, dtype, folded);
                return Ok(());
            }
            // Operands are converted to the result type: bools add as a
            // logical or and multiply as a logical and, as in PyTorch.
            let function = match (kind, ty) {
                (BinaryKind::Add, ScalarType::Bool) => BinaryFn::Or,
                (BinaryKind::Mul, ScalarType::Bool) => BinaryFn::And,
                (BinaryKind::Add, _) => BinaryFn::Add,
                (BinaryKind::Sub, _) => BinaryFn::Sub,
                (BinaryKind::Mul, _) => BinaryFn::Mul,
                (BinaryKind::Div, _) => BinaryFn::Div,
                (BinaryKind::Power, ScalarType::F32) => BinaryFn::Pow,
                (BinaryKind::Power, _) => BinaryFn::PowI32,
                (BinaryKind::Remainder, ScalarType::F32) => BinaryFn::RemainderF32,
                (BinaryKind::Remainder, _) => BinaryFn::RemainderI32,
                (BinaryKind::Fmod, _) => BinaryFn::Fmod,
                (BinaryKind::Maximum, ScalarType::F32) => BinaryFn::MaximumF32,
                (BinaryKind::Minimum, ScalarType::F32) => BinaryFn::MinimumF32,
                (BinaryKind::Maximum, ScalarType::Bool) => BinaryFn::Or,
                (BinaryKind::Minimum, ScalarType::Bool) => BinaryFn::And,
                (BinaryKind::Maximum, _) => BinaryFn::Max,
                (BinaryKind::Minimum, _) => BinaryFn::Min,
            };
            let operand =
                |index: usize| convert(Expr::Input(index), ScalarType::of(dtypes[index]), ty);
            let views: Vec<View> = inputs
                .iter()
                .map(|view| view.broadcast_to(&shape))
                .collect();
            let alpha = alpha.as_f64();
            let other = match scalar {
                Some(scalar) => constant(alpha * scalar.as_f64(), ty)?,
                None if ty == ScalarType::Bool => {
                    Expr::binary(BinaryFn::And, operand(1), constant(alpha, ty)?)
                }
                None if alpha == 1.0 => operand(1),
                None => Expr::binary(BinaryFn::Mul, operand(1), constant(alpha, ty)?),
            };
            let expr = Expr::binary(function, operand(0), other);
            builder.emit(Kernel::ElementWise(expr), views, output.clone(), name);
        }
        &Op::Power {
            scalar,
            scalar_base,
        } => {
            let x = convert(x, ScalarType::of(dtypes[0]), ty);
            let number = scalar.as_f64();
            let literal = |number| constant(number, ty);
            let expr = match (scalar_base, ty, number) {
                (false, _, 0.0) => literal(1.0)?,
                (false, _, 1.0) => x,
                (false, _, 2.0) => Expr::binary(BinaryFn::Mul, x.clone(), x),
                (false, ScalarType::F32, 0.5) => Expr::unary(UnaryFn::Sqrt, x),
                (false, ScalarType::F32, -0.5) => Expr::unary(UnaryFn::InverseSqrt, x),
                (false, ScalarType::F32, -1.0) => Expr::unary(UnaryFn::Reciprocal, x),
                (true, ScalarType::F32, 1.0) => literal(1.0)?,
                (true, ScalarType::F32, n) if (n as f32).is_finite() && (n as f32) > 0.0 => {
                    // ln(base) is known at compile time. Positive scalar
                    // bases (notably RoPE) only need one exponential per element.
                    let logarithm = (n as f32).ln();
                    Expr::unary(
                        UnaryFn::Exp,
                        Expr::binary(BinaryFn::Mul, x, Expr::f32(logarithm)),
                    )
                }
                (true, ScalarType::F32, _) => Expr::binary(BinaryFn::Pow, literal(number)?, x),
                (true, _, _) => Expr::binary(BinaryFn::PowI32, literal(number)?, x),
                (false, ScalarType::F32, n) if n.fract() == 0.0 && n % 2.0 != 0.0 => {
                    Expr::binary(BinaryFn::PowOdd, x, literal(n)?)
                }
                (false, ScalarType::F32, n) if n.fract() == 0.0 => {
                    Expr::binary(BinaryFn::Pow, Expr::unary(UnaryFn::Abs, x), literal(n)?)
                }
                (false, ScalarType::F32, _) => Expr::binary(BinaryFn::Pow, x, literal(number)?),
                (false, _, _) => Expr::binary(BinaryFn::PowI32, x, literal(number)?),
            };
            builder.emit(Kernel::ElementWise(expr), inputs, output.clone(), name);
        }
        &Op::Compare { kind, scalar } => {
            // Both operands are converted to their promoted dtype.
            let compute = ScalarType::of(
                ops::infer_dtypes(
                    &Op::Binary {
                        kind: BinaryKind::Add,
                        scalar,
                        alpha: Number::Int(1),
                    },
                    &dtypes,
                    &inputs
                        .iter()
                        .map(|view| view.shape.as_slice())
                        .collect::<Vec<_>>(),
                )
                .map_err(|message| CompileError::InvalidOp {
                    op: node.node.op.clone(),
                    message,
                })?[0],
            );
            // WGSL only orders numbers: bools order as integers.
            let ordered = !matches!(kind, CompareKind::Eq | CompareKind::Ne);
            let compute = match compute {
                ScalarType::Bool if ordered => ScalarType::I32,
                compute => compute,
            };
            let operand =
                |index: usize| convert(Expr::Input(index), ScalarType::of(dtypes[index]), compute);
            let other = match scalar {
                Some(scalar) => constant(scalar.as_f64(), compute)?,
                None => operand(1),
            };
            let function = match kind {
                CompareKind::Eq => BinaryFn::Eq,
                CompareKind::Ne => BinaryFn::Ne,
                CompareKind::Lt => BinaryFn::Lt,
                CompareKind::Le => BinaryFn::Le,
                CompareKind::Gt => BinaryFn::Gt,
                CompareKind::Ge => BinaryFn::Ge,
            };
            let expr = Expr::binary(function, operand(0), other);
            let views = inputs
                .iter()
                .map(|view| view.broadcast_to(&shape))
                .collect();
            builder.emit(Kernel::ElementWise(expr), views, output.clone(), name);
        }
        Op::Where => {
            let branch =
                |index: usize| convert(Expr::Input(index), ScalarType::of(dtypes[index]), ty);
            let expr = Expr::select(Expr::Input(0), branch(1), branch(2));
            let views = inputs
                .iter()
                .map(|view| view.broadcast_to(&shape))
                .collect();
            builder.emit(Kernel::ElementWise(expr), views, output.clone(), name);
        }
        &Op::Bitwise(kind) => {
            let operand =
                |index: usize| convert(Expr::Input(index), ScalarType::of(dtypes[index]), ty);
            let expr = match (kind, ty) {
                (BitwiseKind::Not, _) => Expr::unary(UnaryFn::Not, operand(0)),
                (BitwiseKind::And, _) => Expr::binary(BinaryFn::And, operand(0), operand(1)),
                (BitwiseKind::Or, _) => Expr::binary(BinaryFn::Or, operand(0), operand(1)),
                (BitwiseKind::Xor, ScalarType::Bool) => {
                    Expr::binary(BinaryFn::Ne, operand(0), operand(1))
                }
                (BitwiseKind::Xor, _) => Expr::binary(BinaryFn::Xor, operand(0), operand(1)),
            };
            let views = inputs
                .iter()
                .map(|view| view.broadcast_to(&shape))
                .collect();
            builder.emit(Kernel::ElementWise(expr), views, output.clone(), name);
        }
        &Op::Cast(target) => {
            let expr = Expr::cast(ScalarType::of(target), x);
            let input = inputs[0].clone();
            builder.emit(Kernel::ElementWise(expr), vec![input], output.clone(), name);
        }
        &Op::Arange { start, step, .. } => {
            // Read nothing: the kernel folds into the plan's preparation.
            let index = convert(Expr::Index(0), ScalarType::I32, ty);
            let expr = Expr::binary(
                BinaryFn::Add,
                constant(start.as_f64(), ty)?,
                Expr::binary(BinaryFn::Mul, constant(step.as_f64(), ty)?, index),
            );
            builder.emit(Kernel::ElementWise(expr), Vec::new(), output.clone(), name);
        }
        &Op::Full { value, .. } => {
            let expr = if ty == ScalarType::F32 && !value.as_f64().is_finite() {
                Expr::f32(value.as_f64() as f32)
            } else {
                constant(value.as_f64(), ty)?
            };
            builder.emit(Kernel::ElementWise(expr), Vec::new(), output.clone(), name);
        }
        &Op::Matmul { beta, alpha, bias } => {
            let operands = usize::from(bias);
            let mut views = vec![inputs[operands].clone(), inputs[operands + 1].clone()];
            if bias {
                views.push(inputs[0].broadcast_to(&shape));
            }
            let kernel = Kernel::Matmul { beta, alpha, bias };
            builder.emit(kernel, views, output.clone(), name);
        }
        Op::Reduce { kind, axes } => {
            let kernel = Kernel::Reduction {
                kind: *kind,
                axes: axes.clone(),
                strategy: Strategy::Serial,
                split: 1,
            };
            builder.emit(kernel, inputs, output.clone(), name);
        }
        Op::Convolution(convolution) => {
            let path = conv_path(convolution, &inputs[0].shape, &inputs[1].shape, &shape);
            let kernel = Kernel::Convolution {
                convolution: convolution.clone(),
                path,
                tiling: Tiling::default(),
            };
            builder.emit(kernel, inputs, output.clone(), name);
        }
        &Op::BatchNorm { eps, weight, bias } => {
            let normalized =
                lower_batch_norm(builder, node, &inputs, eps, weight, bias, output.clone())?;
            builder.define(name, Dtype::F32, normalized);
            return Ok(());
        }
        Op::Movement(kind) => {
            let kernel = if let Movement::Pad(pad) = kind
                && let PadMode::Constant(value) = pad.mode
            {
                let Expr::Const(fill) = constant(value.as_f64(), ty)? else {
                    unreachable!()
                };
                Kernel::Movement {
                    kind: kind.clone(),
                    fill: Some(fill),
                }
            } else {
                Kernel::Movement {
                    kind: kind.clone(),
                    fill: None,
                }
            };
            builder.emit(kernel, inputs, output.clone(), name);
        }
        &Op::Dequantize(axis) => {
            let mut scale = inputs[1].clone();
            let stride = scale.strides[0];
            scale.shape.clone_from(&shape);
            scale.strides = vec![0; shape.len()];
            scale.strides[axis] = stride;
            let expr = Expr::binary(
                BinaryFn::Mul,
                Expr::cast(ScalarType::F32, Expr::Input(0)),
                Expr::Input(1),
            );
            builder.emit(
                Kernel::ElementWise(expr),
                vec![inputs[0].clone(), scale],
                output.clone(),
                name,
            );
        }
        Op::Scatter(scatter) => {
            let mut inputs = inputs;
            if let crate::ops::Scatter::Advanced {
                start,
                rank,
                selection,
                ..
            } = scatter
            {
                let values = inputs.len() - 1;
                for index in &mut inputs[1..values] {
                    *index = index.broadcast_to(&selection[*start..*start + *rank]);
                }
                inputs[values] = inputs[values].broadcast_to(selection);
            }
            builder.emit(
                Kernel::Scatter(scatter.clone()),
                inputs,
                output.clone(),
                name,
            );
        }
        Op::Rotary => {
            let views = inputs
                .iter()
                .map(|view| view.broadcast_to(&shape))
                .collect();
            builder.emit(Kernel::Rotary, views, output.clone(), name);
        }
        Op::CacheWrite => {
            builder.emit(
                Kernel::CacheWrite { in_place: false },
                inputs,
                output.clone(),
                name,
            );
        }
        &Op::CachedAttention { scale } => {
            builder.emit(
                Kernel::CachedAttention { scale },
                inputs,
                output.clone(),
                name,
            );
        }
        Op::SampleCandidates => {
            builder.emit(Kernel::SampleCandidates, inputs, output.clone(), name);
        }
        &Op::Order { axis, largest } => {
            lower_order(builder, node, &output, inputs[0].clone(), axis, largest)?;
        }
        Op::Pool(pool) => {
            emit_pair(
                builder,
                node,
                Kernel::Pool(pool.clone()),
                inputs,
                &output,
                layout,
            )?;
        }
        Op::Row { kind, axes } => {
            let kernel = Kernel::Row {
                kind: match kind {
                    // Preserve the scan's accumulator type even when an
                    // epilogue changes the storage type of its output.
                    crate::ops::RowKind::Scan { .. } => {
                        crate::ops::RowKind::Scan { dtype: Some(dtype) }
                    }
                    crate::ops::RowKind::Product { .. } => {
                        crate::ops::RowKind::Product { dtype: Some(dtype) }
                    }
                    _ => *kind,
                },
                axes: axes.clone(),
                strategy: Strategy::Serial,
            };
            emit_pair(builder, node, kernel, inputs, &output, layout)?;
        }
        &Op::GroupNorm {
            eps,
            groups,
            weight,
            bias,
            across_batch,
        } => {
            let group = |view: &View| {
                let mut view = view.clone();
                if across_batch {
                    view.shape.swap(0, 1);
                    view.strides.swap(0, 1);
                    return view;
                }
                let channels = view.shape[1] / groups;
                view.shape[1] = channels;
                view.shape.insert(1, groups);
                view.strides.insert(1, channels * view.strides[1]);
                view
            };
            let mut views = vec![group(&inputs[0])];
            if weight {
                views.push(group(&per_channel(&inputs[1], &shape)));
            }
            if bias {
                views.push(group(&per_channel(
                    &inputs[1 + usize::from(weight)],
                    &shape,
                )));
            }
            let grouped = group(&output);
            builder.emit(
                Kernel::LayerNorm {
                    eps,
                    normalized: grouped.shape[if across_batch { 1 } else { 2 }..].to_vec(),
                    rms: false,
                    weight,
                    strategy: Strategy::Serial,
                },
                views,
                grouped,
                name,
            );
        }
        &Op::LayerNorm {
            eps,
            weight,
            bias,
            ref normalized,
            rms,
        } => {
            let mut views = vec![inputs[0].clone()];
            if weight {
                views.push(inputs[1].broadcast_to(&shape));
            }
            if bias {
                views.push(inputs[1 + usize::from(weight)].broadcast_to(&shape));
            }
            builder.emit(
                Kernel::LayerNorm {
                    eps,
                    normalized: normalized.clone(),
                    rms,
                    weight,
                    strategy: Strategy::Serial,
                },
                views,
                output.clone(),
                name,
            );
        }
        &Op::Attention {
            scale,
            causal,
            mask,
        } => {
            let rank = shape.len();
            let outer = &shape[..rank - 3];
            let broadcast = |view: &View, tail: &[u64]| {
                let mut target = outer.to_vec();
                target.extend_from_slice(tail);
                view.broadcast_to(&target)
            };
            let views = vec![
                broadcast(&inputs[0], &inputs[0].shape[rank - 3..]),
                broadcast(&inputs[1], &inputs[1].shape[rank - 3..]),
                broadcast(&inputs[2], &inputs[2].shape[rank - 3..]),
            ];
            let mut views = views;
            if mask.is_some() {
                let tail = [shape[rank - 3], shape[rank - 2], inputs[1].shape[rank - 2]];
                views.push(broadcast(&inputs[3], &tail));
            }
            builder.emit(
                Kernel::Attention {
                    scale,
                    causal,
                    mask: mask.is_some(),
                },
                views,
                output.clone(),
                name,
            );
        }
        &Op::Embedding { padding } => {
            builder.emit(Kernel::Embedding { padding }, inputs, output.clone(), name);
        }
        &Op::Cat { axis, ref sizes } => {
            // One binding per input plus the output: wider concatenations go
            // through intermediate stages that each fit the device limit.
            let fan_in = (builder.limits.max_storage_buffers_per_shader_stage as usize)
                .saturating_sub(1)
                .max(2);
            let mut views = inputs;
            let mut sizes = sizes.clone();
            let mut stage = 0;
            while views.len() > fan_in {
                let mut next_views = Vec::new();
                let mut next_sizes = Vec::new();
                for (part, (chunk, chunk_sizes)) in
                    views.chunks(fan_in).zip(sizes.chunks(fan_in)).enumerate()
                {
                    if let [view] = chunk {
                        next_views.push(view.clone());
                        next_sizes.push(chunk_sizes[0]);
                        continue;
                    }
                    let total: u64 = chunk_sizes.iter().sum();
                    let mut part_shape = shape.clone();
                    part_shape[axis] = total;
                    let view = builder.allocate(
                        &format!("{name}.cat{stage}.{part}"),
                        dtype,
                        part_shape,
                        layout,
                        BufferRole::Arena,
                    )?;
                    builder.emit(
                        Kernel::Cat {
                            axis,
                            sizes: chunk_sizes.to_vec(),
                        },
                        chunk.to_vec(),
                        view.clone(),
                        name,
                    );
                    next_views.push(view);
                    next_sizes.push(total);
                }
                views = next_views;
                sizes = next_sizes;
                stage += 1;
            }
            builder.emit(Kernel::Cat { axis, sizes }, views, output.clone(), name);
        }
    }
    if let Some((Some(name), _)) = node.outputs.first() {
        builder.define(name, dtype, output);
    }
    Ok(())
}

/// Non-repeating axes, singleton broadcasts and pure crops need no dispatch.
fn movement_view(kind: &Movement, input: &View, shape: &[u64]) -> Option<View> {
    match kind {
        Movement::Repeat
            if crate::ops::broadcast(&input.shape, shape).as_deref() == Some(shape) =>
        {
            Some(input.broadcast_to(shape))
        }
        Movement::Flip(axes) if axes.iter().all(|&axis| input.shape[axis] == 1) => {
            Some(input.clone())
        }
        Movement::Roll { shifts, .. } if shifts.iter().all(|&(_, shift)| shift == 0) => {
            Some(input.clone())
        }
        Movement::Resize(_) | Movement::AdaptiveAverage if input.shape == shape => {
            Some(input.clone())
        }
        Movement::Pad(pad)
            if pad
                .widths
                .iter()
                .all(|&(left, right)| left <= 0 && right <= 0) =>
        {
            let mut view = input.clone();
            for (axis, &(left, _)) in pad.widths.iter().enumerate() {
                view = view.slice(axis, (-left) as u64, shape[axis], 1);
            }
            Some(view)
        }
        _ => None,
    }
}

/// Sort tiles locally, then merge pairs of runs in parallel. For topk every
/// intermediate run retains only the prefix that can reach the final result.
fn lower_order<'a>(
    builder: &mut Builder<'a>,
    node: &Resolved<'a>,
    output: &View,
    input: View,
    axis: usize,
    largest: bool,
) -> Result<(), CompileError> {
    let name = node.name();
    if input.shape.is_empty() {
        if let Some((Some(indices), _)) = node.outputs.get(1) {
            let index = builder.allocate(
                indices,
                Dtype::I64,
                vec![],
                Layout::RowMajor,
                BufferRole::Arena,
            )?;
            builder.emit(
                Kernel::ElementWise(Expr::Const(Scalar::I32(0))),
                vec![],
                index.clone(),
                indices,
            );
            builder.define(indices, Dtype::I64, index);
        }
        if node.outputs[0].0.is_some() {
            builder.emit(
                Kernel::ElementWise(Expr::Input(0)),
                vec![input],
                output.clone(),
                name,
            );
        }
        return Ok(());
    }
    let width = input.shape[axis];
    let rows = input.elements() / width;
    let k = output.shape[axis];
    let tile = width.min(1024).next_power_of_two();
    let allocate = |builder: &mut Builder, span: u64| {
        builder.allocate(
            name,
            Dtype::I32,
            vec![rows, width.div_ceil(span), k.min(span)],
            Layout::RowMajor,
            BufferRole::Arena,
        )
    };
    let mut indices = allocate(builder, tile)?;
    builder.emit(
        Kernel::OrderBlocks {
            axis,
            largest,
            tile,
        },
        vec![input.clone()],
        indices.clone(),
        name,
    );
    let mut span = tile;
    while span < width {
        let merged = allocate(builder, span * 2)?;
        builder.emit(
            Kernel::OrderMerge {
                axis,
                largest,
                span,
            },
            vec![input.clone(), indices],
            merged.clone(),
            name,
        );
        indices = merged;
        span *= 2;
    }
    emit_pair(
        builder,
        node,
        Kernel::OrderGather { axis },
        vec![input, indices],
        output,
        Layout::RowMajor,
    )
}

fn emit_pair<'a>(
    builder: &mut Builder<'a>,
    node: &Resolved<'a>,
    kernel: Kernel,
    inputs: Vec<View>,
    output: &View,
    layout: Layout,
) -> Result<(), CompileError> {
    let mut kernel = KernelNode::new(kernel, inputs, output.clone(), node.name());
    if let Some((Some(name), info)) = node.outputs.get(1) {
        let view = builder.allocate(
            name,
            info.dtype,
            info.shape.clone(),
            layout,
            BufferRole::Arena,
        )?;
        builder.define(name, info.dtype, view.clone());
        kernel.second = Some(view);
    }
    builder.emit_node(kernel);
    Ok(())
}

/// `x * scale + shift` per channel, where `scale = weight / sqrt(var + eps)`
/// and `shift = bias - mean * scale` only read parameters and fold away.
/// Returns the normalized tensor: `output`, or the output of the convolution
/// producing `x` when the normalization folds into its weights and bias.
fn lower_batch_norm(
    builder: &mut Builder,
    node: &Resolved,
    inputs: &[View],
    eps: f32,
    weight: bool,
    bias: bool,
    output: View,
) -> Result<View, CompileError> {
    let name = node.name();
    let mut statistics = inputs[1..].iter().cloned();
    let weight = weight.then(|| statistics.next()).flatten();
    let bias = bias.then(|| statistics.next()).flatten();
    let (Some(mean), Some(var)) = (statistics.next(), statistics.next()) else {
        unreachable!("the op-level pass checked the inputs");
    };
    let channels = mean.shape.clone();

    let inverse_deviation = Expr::unary(
        UnaryFn::InverseSqrt,
        Expr::binary(BinaryFn::Add, Expr::Input(0), Expr::f32(eps)),
    );
    let (scale_expr, scale_inputs) = match weight {
        Some(weight) => (
            Expr::binary(BinaryFn::Mul, Expr::Input(1), inverse_deviation),
            vec![var, weight],
        ),
        None => (inverse_deviation, vec![var]),
    };
    let scale_name = format!("{name}.scale");
    let scale = builder.allocate(
        &scale_name,
        Dtype::F32,
        channels.clone(),
        Layout::RowMajor,
        BufferRole::Arena,
    )?;
    builder.emit(
        Kernel::ElementWise(scale_expr),
        scale_inputs,
        scale.clone(),
        &scale_name,
    );

    let centered = Expr::binary(BinaryFn::Mul, Expr::Input(0), Expr::Input(1));
    let (shift_expr, shift_inputs) = match bias {
        Some(bias) => (
            Expr::binary(BinaryFn::Sub, Expr::Input(2), centered),
            vec![mean, scale.clone(), bias],
        ),
        None => (
            Expr::binary(BinaryFn::Sub, Expr::f32(0.0), centered),
            vec![mean, scale.clone()],
        ),
    };
    let shift_name = format!("{name}.shift");
    let shift = builder.allocate(
        &shift_name,
        Dtype::F32,
        channels,
        Layout::RowMajor,
        BufferRole::Arena,
    )?;
    builder.emit(
        Kernel::ElementWise(shift_expr),
        shift_inputs,
        shift.clone(),
        &shift_name,
    );

    if builder
        .producers
        .get(&inputs[0].buffer)
        .is_some_and(|&index| matches!(builder.kernels[index].kernel, Kernel::Convolution { .. }))
        && let Some(folded) = fold_output_affine(
            builder,
            name,
            node.inputs[0],
            &inputs[0],
            Some(&scale),
            Some(&shift),
        )?
    {
        return Ok(folded);
    }

    let per_channel = |view: &View| per_channel(view, &output.shape);
    let expr = Expr::binary(
        BinaryFn::Add,
        Expr::binary(BinaryFn::Mul, Expr::Input(0), Expr::Input(1)),
        Expr::Input(2),
    );
    let views = vec![inputs[0].clone(), per_channel(&scale), per_channel(&shift)];
    builder.emit(Kernel::ElementWise(expr), views, output.clone(), name);
    Ok(output)
}

/// Reads the per-channel vector `view` at every element of `shape`, along axis 1.
fn per_channel(view: &View, shape: &[u64]) -> View {
    View {
        buffer: view.buffer,
        shape: shape.to_vec(),
        strides: (0..shape.len())
            .map(|axis| if axis == 1 { view.strides[0] } else { 0 })
            .collect(),
        offset: view.offset,
    }
}

/// A broadcast vector can be folded only when it varies along the output
/// channel axis and is constant over every batch and spatial dimension.
fn channel_vector(view: &View, shape: &[u64], axis: usize) -> Option<View> {
    let view = view.broadcast_to(shape);
    if view
        .strides
        .iter()
        .enumerate()
        .any(|(i, &stride)| i != axis && shape[i] != 1 && stride != 0)
    {
        return None;
    }
    Some(View {
        buffer: view.buffer,
        shape: vec![shape[axis]],
        strides: vec![view.strides[axis]],
        offset: view.offset,
    })
}

fn fold_binary_affine(
    builder: &mut Builder,
    node: &Resolved,
    inputs: &[View],
    shape: &[u64],
    kind: BinaryKind,
) -> Result<Option<View>, CompileError> {
    let [a, b] = inputs else { return Ok(None) };
    for (data_name, data, constant) in [(node.inputs[0], a, b), (node.inputs[1], b, a)] {
        if data.shape != shape
            || builder.dtype(constant) != Dtype::F32
            || !builder.constant(constant)
        {
            continue;
        }
        let Some(&producer) = builder.producers.get(&data.buffer) else {
            continue;
        };
        let axis = match builder.kernels[producer].kernel {
            Kernel::Convolution { .. } => 1,
            Kernel::Matmul { .. } => shape.len() - 1,
            _ => continue,
        };
        let Some(vector) = channel_vector(constant, shape, axis) else {
            continue;
        };
        let (scale, shift) = if kind == BinaryKind::Mul {
            (Some(&vector), None)
        } else {
            (None, Some(&vector))
        };
        if let Some(output) =
            fold_output_affine(builder, node.name(), data_name, data, scale, shift)?
        {
            return Ok(Some(output));
        }
    }
    Ok(None)
}

/// Folds scale and shift into constant weights and bias of the sole
/// producer. The adjusted parameters are computed once during preparation.
fn fold_output_affine(
    builder: &mut Builder,
    name: &str,
    source: &str,
    x: &View,
    scale: Option<&View>,
    shift: Option<&View>,
) -> Result<Option<View>, CompileError> {
    let Some(&index) = builder.producers.get(&x.buffer) else {
        return Ok(None);
    };
    let node = &builder.kernels[index];
    let (channel, weight_axis, matmul) = match node.kernel {
        Kernel::Convolution {
            ref convolution, ..
        } if !convolution.transposed => (1, 0, false),
        Kernel::Matmul {
            alpha: 1.0,
            beta: 1.0,
            ..
        } => (x.shape.len() - 1, node.inputs[1].shape.len() - 1, true),
        _ => return Ok(None),
    };
    let single_use =
        node.ops.last().is_some_and(|op| op == source) && builder.uses.get(source) == Some(&1);
    if node.output != *x
        || !single_use
        || !node.inputs[1..].iter().all(|view| builder.constant(view))
    {
        return Ok(None);
    }
    let weight = node.inputs[1].clone();
    if scale.is_some() && builder.dequantized(&weight) {
        return Ok(None);
    }
    let old_bias = match node.inputs.get(2) {
        Some(bias) if matmul => {
            let Some(bias) = channel_vector(bias, &x.shape, channel) else {
                return Ok(None);
            };
            Some(bias)
        }
        bias => bias.cloned(),
    };

    let fold = |builder: &mut Builder, label, shape: Vec<u64>, expr, inputs| {
        let label = format!("{name}.{label}");
        let out = builder.allocate(
            &label,
            Dtype::F32,
            shape,
            Layout::RowMajor,
            BufferRole::Arena,
        )?;
        builder.emit(Kernel::ElementWise(expr), inputs, out.clone(), &label);
        Ok::<_, CompileError>(out)
    };
    let product = || Expr::binary(BinaryFn::Mul, Expr::Input(0), Expr::Input(1));
    let folded_weight = if let Some(scale) = scale {
        let mut channel_scale = scale.clone();
        channel_scale.shape.clone_from(&weight.shape);
        channel_scale.strides = vec![0; weight.shape.len()];
        channel_scale.strides[weight_axis] = scale.strides[0];
        fold(
            builder,
            "weight",
            weight.shape.clone(),
            product(),
            vec![weight, channel_scale],
        )?
    } else {
        weight
    };
    let bias = match (old_bias, scale, shift) {
        (Some(bias), Some(scale), Some(shift)) => Some(fold(
            builder,
            "bias",
            bias.shape.clone(),
            Expr::binary(BinaryFn::Add, product(), Expr::Input(2)),
            vec![bias, scale.clone(), shift.clone()],
        )?),
        (Some(bias), Some(scale), None) => Some(fold(
            builder,
            "bias",
            bias.shape.clone(),
            product(),
            vec![bias, scale.clone()],
        )?),
        (Some(bias), None, Some(shift)) => Some(fold(
            builder,
            "bias",
            bias.shape.clone(),
            Expr::binary(BinaryFn::Add, Expr::Input(0), Expr::Input(1)),
            vec![bias, shift.clone()],
        )?),
        (None, _, shift) => shift.cloned(),
        (bias, None, None) => bias,
    };
    let node = &mut builder.kernels[index];
    match &mut node.kernel {
        Kernel::Convolution { convolution, .. } => convolution.bias = bias.is_some(),
        Kernel::Matmul { bias: enabled, .. } => *enabled = bias.is_some(),
        _ => unreachable!("checked above"),
    }
    node.inputs.truncate(1);
    node.inputs.push(folded_weight);
    if let Some(bias) = bias {
        node.inputs.push(if matmul {
            bias.broadcast_to(&x.shape)
        } else {
            bias
        });
    }
    node.ops.push(name.to_owned());
    Ok(Some(node.output.clone()))
}

/// Turn a spatial pad read only by a convolution into an address calculation
/// in its GEMM load. A symmetric zero pad can use the convolution's own guard.
pub(super) fn fold_padding(graph: &Graph, builder: &mut Builder) -> Result<(), CompileError> {
    let mut protected = HashSet::new();
    for name in graph
        .outputs
        .iter()
        .map(|output| output.name.as_str())
        .chain(
            graph
                .parameters
                .iter()
                .filter_map(|parameter| parameter.update.as_deref()),
        )
    {
        protected.insert(builder.lookup(name)?.view.buffer);
    }
    let mut readers = HashMap::<usize, usize>::new();
    let mut pads = HashMap::new();
    for node in builder.prepare.iter().chain(&builder.kernels) {
        for read in node.reads() {
            *readers.entry(read.buffer).or_default() += 1;
        }
        if let Kernel::Movement {
            kind: Movement::Pad(pad),
            ..
        } = &node.kernel
            && !protected.contains(&node.output.buffer)
            && !matches!(pad.mode, PadMode::Constant(value) if value.as_f64() != 0.0)
            && pad.widths.len() >= 3
            && pad.widths[..2] == [(0, 0); 2]
            && pad
                .widths
                .iter()
                .all(|&(left, right)| left >= 0 && right >= 0)
        {
            pads.insert(
                node.output.buffer,
                (node.output.clone(), node.inputs[0].clone(), pad.clone()),
            );
        }
    }
    let mut removed = HashSet::new();
    for node in builder.prepare.iter_mut().chain(&mut builder.kernels) {
        let Kernel::Convolution {
            convolution, path, ..
        } = &mut node.kernel
        else {
            continue;
        };
        let buffer = node.inputs[0].buffer;
        if convolution.transposed || readers[&buffer] != 1 {
            continue;
        }
        let Some((padded, source, pad)) = pads.remove(&buffer) else {
            continue;
        };
        if padded != node.inputs[0] {
            continue;
        }
        let symmetric_zero = matches!(pad.mode, PadMode::Constant(_))
            && pad.widths[2..]
                .iter()
                .zip(&convolution.padding)
                .enumerate()
                .all(|(axis, (&(left, right), &existing))| {
                    let pad = existing + left as u64;
                    let dimension = source.shape[axis + 2] + 2 * pad;
                    left == right && dimension <= i32::MAX as u64
                });
        if symmetric_zero {
            for (pad, &(left, _)) in convolution.padding.iter_mut().zip(&pad.widths[2..]) {
                *pad += left as u64;
            }
        } else {
            convolution.input_pad = Some(pad);
        }
        node.inputs[0] = source;
        *path = conv_path(
            convolution,
            &node.inputs[0].shape,
            &node.inputs[1].shape,
            &node.output.shape,
        );
        removed.insert(buffer);
    }
    builder
        .prepare
        .retain(|node| !removed.contains(&node.output.buffer));
    builder
        .kernels
        .retain(|node| !removed.contains(&node.output.buffer));
    Ok(())
}

/// Stores the constant weights of GEMM convolutions with output channels
/// innermost: neighbouring invocations compute neighbouring output channels,
/// so they read neighbouring weights. A weight computed when the plan is
/// created, such as one folded from a batch norm, is written packed directly;
/// others are copied once.
pub(super) fn pack_weights(builder: &mut Builder) -> Result<(), CompileError> {
    for index in 0..builder.kernels.len() {
        let node = &builder.kernels[index];
        let Kernel::Convolution {
            convolution, path, ..
        } = &node.kernel
        else {
            continue;
        };
        let (input, weight) = (&node.inputs[0], node.inputs[1].clone());
        if matches!(path, ConvPath::Depthwise | ConvPath::TransposeDisjoint)
            || (!convolution.transposed && convolution.groups != 1)
            || !builder.constant(&weight)
            || builder.dequantized(&weight)
        {
            continue;
        }
        // The order the GEMM template walks the reduction in, then output channels.
        let rank = weight.shape.len();
        let (out, channel) = if convolution.transposed {
            (1, 0)
        } else {
            (0, 1)
        };
        let order: Vec<usize> = match input.strides[1] == 1 && rank > 2 {
            true => (2..rank).chain([channel, out]).collect(),
            false => [channel].into_iter().chain(2..rank).chain([out]).collect(),
        };
        let packed_strides =
            View::dense_in_order(weight.buffer, weight.shape.clone(), &order).strides;
        if weight.strides == packed_strides {
            continue;
        }

        let readers = builder
            .prepare
            .iter()
            .chain(&builder.kernels)
            .flat_map(KernelNode::reads)
            .filter(|view| view.buffer == weight.buffer)
            .count();
        let producer = builder.prepare.iter().position(|node| {
            node.output == weight && matches!(node.kernel, Kernel::ElementWise(_))
        });
        let packed = match producer {
            Some(producer) if readers == 1 => {
                builder.prepare[producer].output.strides = packed_strides.clone();
                View {
                    strides: packed_strides,
                    ..weight
                }
            }
            _ => {
                let name = format!("{}.packed", node.ops[0]);
                let buffer = builder
                    .allocate(
                        &name,
                        Dtype::F32,
                        weight.shape.clone(),
                        Layout::RowMajor,
                        BufferRole::Arena,
                    )?
                    .buffer;
                let packed = View::dense_in_order(buffer, weight.shape.clone(), &order);
                builder.emit(
                    Kernel::ElementWise(Expr::Input(0)),
                    vec![weight],
                    packed.clone(),
                    &name,
                );
                packed
            }
        };
        builder.kernels[index].inputs[1] = packed;
    }
    Ok(())
}

/// Picks the tiling of each GEMM convolution. One whose reduction is split
/// stores partial sums in a buffer of its own; a sum kernel then reduces them
/// into the convolution output and adds the bias, and fusion makes the rest of
/// the epilogue part of that sum.
pub(super) fn tile_convolutions(
    builder: &mut Builder,
    options: &CompileOptions,
) -> Result<(), CompileError> {
    for mut node in std::mem::take(&mut builder.kernels) {
        let Kernel::Convolution {
            convolution,
            path,
            tiling,
        } = &mut node.kernel
        else {
            builder.kernels.push(node);
            continue;
        };
        if *path == ConvPath::Depthwise {
            builder.kernels.push(node);
            continue;
        }
        let (input, weight, output) = (&node.inputs[0], &node.inputs[1], &node.output);
        let groups = convolution.groups;
        let disjoint = *path == ConvPath::TransposeDisjoint;
        let pixel_shape = if disjoint {
            &input.shape
        } else {
            &output.shape
        };
        let rows = pixel_shape[0] * pixel_shape[2..].iter().product::<u64>();
        let window = weight.shape[2..].iter().product::<u64>();
        let reduction = input.shape[1] / groups * if disjoint { 1 } else { window };
        let columns = output.shape[1] / groups * if disjoint { window } else { 1 };
        *tiling = gemm_tiling(rows, groups, columns, reduction, options.gpu_class);
        if let Some(config) = options.tiles.get(&node.ops[0]) {
            config.validate()?;
            *tiling = Tiling {
                rows: config.rows,
                columns: config.columns,
                split: config.split,
            };
        }
        let split = tiling.split;
        if split == 1 {
            builder.kernels.push(node);
            continue;
        }

        convolution.bias = false;
        let bias = node.inputs.get(2).cloned();
        node.inputs.truncate(2);
        let name = node.ops[0].clone();
        let output = node.output.clone();
        // Partial sums share the output layout, behind the slice axis.
        let mut order: Vec<usize> = (0..output.shape.len()).collect();
        order.sort_by_key(|&axis| std::cmp::Reverse(output.strides[axis]));
        let order: Vec<usize> = [0]
            .into_iter()
            .chain(order.iter().map(|axis| axis + 1))
            .collect();
        let shape: Vec<u64> = [split]
            .into_iter()
            .chain(output.shape.iter().copied())
            .collect();
        let partial_name = format!("{name}.partial");
        let buffer = builder
            .allocate(
                &partial_name,
                Dtype::F32,
                shape.clone(),
                Layout::RowMajor,
                BufferRole::Arena,
            )?
            .buffer;
        let partial = View::dense_in_order(buffer, shape, &order);
        node.output = partial.clone();
        builder.kernels.push(node);

        let sum_name = format!("{name}.sum");
        let reduce = Kernel::Reduction {
            kind: ReduceKind::Sum,
            axes: vec![0],
            strategy: Strategy::Serial,
            split: 1,
        };
        let Some(bias) = bias else {
            builder.emit(reduce, vec![partial], output, &sum_name);
            continue;
        };
        let buffer = builder
            .allocate(
                &sum_name,
                Dtype::F32,
                output.shape.clone(),
                Layout::RowMajor,
                BufferRole::Arena,
            )?
            .buffer;
        let sum = View::dense_in_order(
            buffer,
            output.shape.clone(),
            &order[1..].iter().map(|axis| axis - 1).collect::<Vec<_>>(),
        );
        builder.emit(reduce, vec![partial], sum.clone(), &sum_name);
        let add = Expr::binary(BinaryFn::Add, Expr::Input(0), Expr::Input(1));
        let channel_bias = per_channel(&bias, &output.shape);
        builder.emit(
            Kernel::ElementWise(add),
            vec![sum, channel_bias],
            output,
            &format!("{name}.bias"),
        );
    }
    Ok(())
}

/// Picks how each reduction and normalization covers its rows: serially,
/// one workgroup per row, or, for a few very long rows, in two passes. A
/// split reduction stores partial results in a buffer of its own, and a
/// serial sum over the slices then writes the reduction output.
pub(super) fn plan_reductions(
    builder: &mut Builder,
    options: &CompileOptions,
) -> Result<(), CompileError> {
    let subgroups = options.features.subgroup_lanes();
    for node in builder.prepare.iter_mut().chain(&mut builder.kernels) {
        if let Kernel::Row {
            kind,
            axes,
            strategy,
        } = &mut node.kernel
        {
            let extent: u64 = axes
                .iter()
                .map(|&axis| node.inputs[0].shape[axis])
                .product();
            let rows = node.inputs[0].elements() / extent;
            *strategy = if matches!(kind, crate::ops::RowKind::Scan { .. }) && extent >= 128 {
                Strategy::Workgroup {
                    lanes: crate::kernel::REDUCTION_LANES,
                    subgroup: false,
                }
            } else {
                reduce_strategy(rows, extent, None).0
            };
        }
    }
    for node in &mut builder.prepare {
        match &mut node.kernel {
            Kernel::Reduction { axes, strategy, .. } => {
                let extent = axes
                    .iter()
                    .map(|&axis| node.inputs[0].shape[axis])
                    .product();
                *strategy = reduce_strategy(node.output.elements(), extent, subgroups).0;
            }
            Kernel::LayerNorm {
                normalized,
                strategy,
                ..
            } => {
                let extent: u64 = normalized.iter().product();
                *strategy = reduce_strategy(node.output.elements() / extent, extent, subgroups).0;
            }
            _ => {}
        }
    }
    for mut node in std::mem::take(&mut builder.kernels) {
        let split = match &mut node.kernel {
            Kernel::Reduction {
                axes,
                strategy,
                split,
                ..
            } => {
                let extent = axes
                    .iter()
                    .map(|&axis| node.inputs[0].shape[axis])
                    .product();
                (*strategy, *split) = reduce_strategy(node.output.elements(), extent, subgroups);
                *split
            }
            Kernel::LayerNorm {
                normalized,
                strategy,
                ..
            } => {
                let extent: u64 = normalized.iter().product();
                *strategy = reduce_strategy(node.output.elements() / extent, extent, subgroups).0;
                1
            }
            _ => 1,
        };
        if split == 1 {
            builder.kernels.push(node);
            continue;
        }
        // Partial results share the output layout, behind the slice axis.
        let output = node.output.clone();
        let mut order: Vec<usize> = (0..output.shape.len()).collect();
        order.sort_by_key(|&axis| std::cmp::Reverse(output.strides[axis]));
        let order: Vec<usize> = [0]
            .into_iter()
            .chain(order.iter().map(|axis| axis + 1))
            .collect();
        let shape: Vec<u64> = [split]
            .into_iter()
            .chain(output.shape.iter().copied())
            .collect();
        let name = node.ops[0].clone();
        let partial_name = format!("{name}.partial");
        let buffer = builder
            .allocate(
                &partial_name,
                Dtype::F32,
                shape.clone(),
                Layout::RowMajor,
                BufferRole::Arena,
            )?
            .buffer;
        let partial = View::dense_in_order(buffer, shape, &order);
        node.output = partial.clone();
        builder.kernels.push(node);
        let sum = Kernel::Reduction {
            kind: ReduceKind::Sum,
            axes: vec![0],
            strategy: Strategy::Serial,
            split: 1,
        };
        builder.emit(sum, vec![partial], output, &format!("{name}.sum"));
    }
    Ok(())
}

/// Appends the copies that write each state update back into its state
/// buffer. They run last, after every kernel reading the previous state.
pub(super) fn write_state(graph: &Graph, builder: &mut Builder) -> Result<(), CompileError> {
    let mut updates = Vec::new();
    for parameter in &graph.parameters {
        let Some(update) = &parameter.update else {
            continue;
        };
        let state = builder.lookup(&parameter.name)?.view.clone();
        let value = builder.lookup(update)?.clone();
        let invalid = |message: String| CompileError::InvalidOp {
            op: format!("state `{}`", parameter.name),
            message,
        };
        if value.dtype != parameter.dtype || value.view.shape != parameter.shape {
            return Err(invalid(format!(
                "the update `{update}` is {:?} {:?}, the state {:?} {:?}",
                value.dtype, value.view.shape, parameter.dtype, parameter.shape
            )));
        }
        if value.view == state {
            continue;
        }
        // Snapshot any state source before writing any destination: updates
        // may swap buffers or read a view of a different state tensor.
        let source = match matches!(
            builder.buffers[value.view.buffer].role,
            BufferRole::State { .. }
        ) {
            true => builder.materialize(&format!("{update}.copy"), value.view)?,
            false => value.view,
        };
        // Never folded into the preparation, even when the update is
        // constant: the first run must still see the initial state.
        let node = KernelNode::new(
            Kernel::ElementWise(Expr::Input(0)),
            vec![source],
            state,
            &format!("{}.update", parameter.name),
        );
        updates.push(node);
    }
    builder.kernels.extend(updates);
    Ok(())
}

/// Reuse state storage when the patch is the final value and all readers of
/// the old state precede it. Readers of the patched value can then use state
/// directly, and only the incoming rows are written.
pub(super) fn patch_state_in_place(builder: &mut Builder) {
    let kernels = &mut builder.kernels;
    let mut remove = HashSet::new();
    for writer in 0..kernels.len() {
        let node = &kernels[writer];
        if !matches!(node.kernel, Kernel::CacheWrite { in_place: false }) {
            continue;
        }
        let state = &node.inputs[0];
        let temporary = node.output.buffer;
        if !matches!(builder.buffers[state.buffer].role, BufferRole::State { .. })
            || builder.buffers[temporary].role != BufferRole::Arena
            || state.shape != node.output.shape
            || state.strides != node.output.strides
            || state.offset != 0
            || node.inputs[1..]
                .iter()
                .any(|view| view.buffer == state.buffer)
        {
            continue;
        }
        let Some(writeback) = (writer + 1..kernels.len()).find(|&step| {
            let later = &kernels[step];
            later.output == *state
                && later.inputs == [node.output.clone()]
                && matches!(later.kernel, Kernel::ElementWise(Expr::Input(0)))
        }) else {
            continue;
        };
        let state = state.buffer;
        if kernels[writer + 1..]
            .iter()
            .enumerate()
            .any(|(index, later)| {
                writer + index + 1 != writeback
                    && later
                        .reads()
                        .chain(later.writes())
                        .any(|view| view.buffer == state)
            })
        {
            continue;
        }
        kernels[writer].output.buffer = state;
        kernels[writer].kernel = Kernel::CacheWrite { in_place: true };
        for later in &mut kernels[writer + 1..] {
            for view in &mut later.inputs {
                if view.buffer == temporary {
                    view.buffer = state;
                }
            }
        }
        remove.insert(writeback);
    }
    let mut step = 0;
    kernels.retain(|_| {
        let keep = !remove.contains(&step);
        step += 1;
        keep
    });
}
