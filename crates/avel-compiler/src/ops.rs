//! The op-level IR: parses graph nodes and infers their output shapes and dtypes.

use std::collections::HashMap;

mod shape;
use shape::{
    resolve_expand, resolve_permute, resolve_reduce, resolve_select, resolve_slice, resolve_split,
    resolve_squeeze, resolve_view,
};

mod normalization;
use normalization::{resolve_attention, resolve_group_norm, resolve_layer_norm, resolve_rms_norm};

mod spatial;
use spatial::{resolve_convolution, resolve_pool};

mod movement;
use movement::resolve_movement;
pub use movement::{Movement, PadMode, Padding, Resize, ResizeMode};
mod scatter;
pub use scatter::Scatter;

mod rows;
use rows::resolve_row;
pub use rows::{Extremum, RowKind};

mod creation;
use creation::{resolve_arange, resolve_cast, resolve_full};

use crate::error::CompileError;
use crate::graph::{Attr, Dtype, Node};

/// Ops the compiler accepts. Kept in sync with `avel_export/ops.py` by a test.
pub const SUPPORTED_OPS: &[&str] = &[
    "aten._adaptive_avg_pool2d.default",
    "aten._log_softmax.default",
    "aten._native_batch_norm_legit.no_stats",
    "aten._native_batch_norm_legit_no_training.default",
    "aten._softmax.default",
    "aten._to_copy.default",
    "aten.abs.default",
    "aten.adaptive_max_pool2d.default",
    "aten.add.Scalar",
    "aten.add.Tensor",
    "aten.addmm.default",
    "aten.amax.default",
    "aten.amin.default",
    "aten.any.dim",
    "aten.arange.start_step",
    "aten.argmax.default",
    "aten.argmin.default",
    "aten.avg_pool2d.default",
    "aten.avg_pool3d.default",
    "aten.bitwise_and.Tensor",
    "aten.bitwise_not.default",
    "aten.bitwise_or.Tensor",
    "aten.bitwise_xor.Tensor",
    "aten.bmm.default",
    "aten.cat.default",
    "aten.ceil.default",
    "aten.clamp.default",
    "aten.clone.default",
    "aten.constant_pad_nd.default",
    "aten.convolution.default",
    "aten.cos.default",
    "aten.cumsum.default",
    "aten.div.Scalar",
    "aten.div.Tensor",
    "aten.elu.default",
    "aten.embedding.default",
    "aten.eq.Scalar",
    "aten.eq.Tensor",
    "aten.erf.default",
    "aten.exp.default",
    "aten.expand.default",
    "aten.expm1.default",
    "aten.flip.default",
    "aten.floor.default",
    "aten.fmod.Scalar",
    "aten.full.default",
    "aten.full_like.default",
    "aten.gather.default",
    "aten.ge.Scalar",
    "aten.ge.Tensor",
    "aten.gelu.default",
    "aten.gt.Scalar",
    "aten.gt.Tensor",
    "aten.hardtanh.default",
    "aten.index.Tensor",
    "aten.index_put.default",
    "aten.index_select.default",
    "aten.le.Scalar",
    "aten.le.Tensor",
    "aten.leaky_relu.default",
    "aten.log.default",
    "aten.log1p.default",
    "aten.lt.Scalar",
    "aten.lt.Tensor",
    "aten.max.dim",
    "aten.max_pool2d_with_indices.default",
    "aten.max_pool3d_with_indices.default",
    "aten.maximum.default",
    "aten.mean.default",
    "aten.mean.dim",
    "aten.min.dim",
    "aten.minimum.default",
    "aten.mm.default",
    "aten.mul.Scalar",
    "aten.mul.Tensor",
    "aten.native_group_norm.default",
    "aten.native_layer_norm.default",
    "aten.ne.Scalar",
    "aten.ne.Tensor",
    "aten.neg.default",
    "aten.pad.default",
    "aten.permute.default",
    "aten.pow.Scalar",
    "aten.pow.Tensor_Scalar",
    "aten.pow.Tensor_Tensor",
    "aten.prod.dim_int",
    "aten.reciprocal.default",
    "aten.relu.default",
    "aten.remainder.Scalar",
    "aten.repeat.default",
    "aten.rms_norm.default",
    "aten.roll.default",
    "aten.round.default",
    "aten.rsqrt.default",
    "aten.scalar_tensor.default",
    "aten.scaled_dot_product_attention.default",
    "aten.scatter.src",
    "aten.scatter_add.default",
    "aten.select.int",
    "aten.sigmoid.default",
    "aten.sign.default",
    "aten.sin.default",
    "aten.slice.Tensor",
    "aten.sort.default",
    "aten.sort.stable",
    "aten.split_with_sizes.default",
    "aten.sqrt.default",
    "aten.squeeze.dims",
    "aten.sub.Scalar",
    "aten.sub.Tensor",
    "aten.sum.dim_IntList",
    "aten.tanh.default",
    "aten.topk.default",
    "aten.trunc.default",
    "aten.unsqueeze.default",
    "aten.upsample_bicubic2d.vec",
    "aten.upsample_bilinear2d.vec",
    "aten.upsample_nearest2d.vec",
    "aten.var.correction",
    "aten.view.default",
    "aten.where.self",
    "avel.cache_write.default",
    "avel.cached_attention.default",
    "avel.dequantize_per_channel.default",
    "avel.rotary.default",
    "avel.sample_candidates.default",
];

/// Composite ops the exporter keeps whole instead of decomposing them, as
/// their decomposition loses what the compiler needs. Kept in sync with
/// `PRESERVED_OPS` in `avel_export/ops.py` by a test; each is also supported.
pub const PRESERVED_OPS: &[&str] = &[
    "aten.adaptive_max_pool2d.default",
    "aten.avg_pool3d.default",
    "aten.index_put.default",
    "aten.max_pool3d_with_indices.default",
    "aten.pad.default",
    "aten.rms_norm.default",
    "aten.roll.default",
    "aten.scaled_dot_product_attention.default",
    "aten.scatter.src",
    "aten.scatter_add.default",
    "aten.sort.default",
    "aten.sort.stable",
    "aten.topk.default",
    "aten.upsample_bicubic2d.vec",
    "avel.cache_write.default",
    "avel.cached_attention.default",
    "avel.rotary.default",
    "avel.sample_candidates.default",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceKind {
    Sum,
    Mean,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Unary {
    Math(crate::kernel::UnaryFn),
    Elu {
        alpha: f32,
        scale: f32,
        input_scale: f32,
    },
    LeakyRelu {
        slope: f32,
    },
    Relu,
    Gelu {
        approximate: bool,
    },
    Sigmoid,
    Clamp {
        min: Option<Number>,
        max: Option<Number>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryKind {
    Power,
    Remainder,
    Fmod,
    Maximum,
    Minimum,
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareKind {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Bitwise ops, logical on bools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitwiseKind {
    And,
    Or,
    Xor,
    Not,
}

/// A scalar argument, keeping the kind PyTorch type promotion looks at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Number {
    Bool(bool),
    Int(i64),
    Float(f64),
}

impl Number {
    pub fn as_f64(self) -> f64 {
        match self {
            Number::Bool(value) => f64::from(u8::from(value)),
            Number::Int(value) => value as f64,
            Number::Float(value) => value,
        }
    }

    /// The dtype PyTorch gives a tensor created from this scalar alone.
    pub fn default_dtype(self) -> Dtype {
        match self {
            Number::Bool(_) => Dtype::Bool,
            Number::Int(_) => Dtype::I64,
            Number::Float(_) => Dtype::F32,
        }
    }
}

/// A convolution over the trailing spatial axes of `[batch, channels, ...]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Convolution {
    pub stride: Vec<u64>,
    pub padding: Vec<u64>,
    pub dilation: Vec<u64>,
    pub groups: u64,
    pub bias: bool,
    pub transposed: bool,
    pub input_pad: Option<Padding>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolKind {
    Max,
    AdaptiveMax,
    Avg {
        count_include_pad: bool,
        divisor: Option<u64>,
    },
}

/// A pooling window over the trailing spatial axes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pool {
    pub kind: PoolKind,
    pub kernel: Vec<u64>,
    pub stride: Vec<u64>,
    pub padding: Vec<u64>,
    pub dilation: Vec<u64>,
}

/// Ops that only change how a tensor is read, never its elements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViewOp {
    Alias,
    Permute(Vec<usize>),
    Reshape,
    Expand,
    Slice {
        axis: usize,
        start: u64,
        step: u64,
    },
    Select {
        axis: usize,
        index: u64,
    },
    Unsqueeze(usize),
    Squeeze(Vec<usize>),
    /// Consecutive slices along `axis`, one output each.
    Split {
        axis: usize,
        sizes: Vec<u64>,
    },
}

/// How an attention mask is applied to the scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskKind {
    /// An f32 mask added to the scores.
    Additive,
    /// A bool mask: false positions are masked out.
    Select,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Custom(crate::CustomKernel),
    /// `beta * bias + alpha * lhs @ rhs` over `[..., M, K]` and `[..., K, N]`
    /// operands with the same batch axes: `addmm` (with a bias), `mm` and
    /// `bmm`. Graph inputs are the bias, if any, then the operands.
    Matmul {
        beta: f32,
        alpha: f32,
        bias: bool,
    },
    Unary(Unary),
    /// `self op alpha * other`, where `other` is a tensor input or a scalar.
    Binary {
        kind: BinaryKind,
        scalar: Option<Number>,
        alpha: Number,
    },
    Power {
        scalar: Number,
        scalar_base: bool,
    },
    Reduce {
        kind: ReduceKind,
        axes: Vec<usize>,
    },
    Row {
        kind: RowKind,
        axes: Vec<usize>,
    },
    Convolution(Convolution),
    /// Inference batch norm over axis 1, with optional affine weight and bias.
    BatchNorm {
        eps: f32,
        weight: bool,
        bias: bool,
    },
    /// Pooling; a max pooling also returns the index of each maximum in its
    /// input plane.
    Pool(Pool),
    Order {
        axis: usize,
        largest: bool,
    },
    Movement(Movement),
    Scatter(Scatter),
    View(ViewOp),
    /// Layer norm over the trailing `normalized` axes, with optional affine
    /// weight and bias. RMS norm divides by the root mean square, without
    /// centering, and has no bias.
    LayerNorm {
        eps: f32,
        weight: bool,
        bias: bool,
        normalized: Vec<u64>,
        rms: bool,
    },
    GroupNorm {
        eps: f32,
        groups: u64,
        weight: bool,
        bias: bool,
        across_batch: bool,
    },
    /// Scaled dot-product attention over `[..., heads, seq, depth]` inputs.
    Attention {
        scale: f32,
        causal: bool,
        mask: Option<MaskKind>,
    },
    /// `weight[indices]`, gathering rows of an `[vocab, dim]` weight.
    Embedding {
        padding: Option<u64>,
    },
    Cat {
        axis: usize,
        /// Size of each input along `axis`.
        sizes: Vec<u64>,
    },
    /// `self op other` as a bool, comparing in the promoted dtype of both;
    /// `other` is a tensor input or a scalar.
    Compare {
        kind: CompareKind,
        scalar: Option<Number>,
    },
    /// `condition ? self : other`, element-wise with broadcasting.
    Where,
    Bitwise(BitwiseKind),
    /// A dtype conversion.
    Cast(Dtype),
    Dequantize(usize),
    CacheWrite,
    Rotary,
    CachedAttention {
        scale: f32,
    },
    SampleCandidates,
    /// `start + step * i` along the only axis. Reads no tensor.
    Arange {
        start: Number,
        step: Number,
        dtype: Dtype,
    },
    /// A tensor filled with `value`. Reads no tensor, or only the shape of
    /// its input, for `full_like`.
    Full {
        value: Number,
        dtype: Dtype,
    },
}

/// One output of a node: its shape and dtype.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Output {
    pub shape: Vec<u64>,
    pub dtype: Dtype,
}

/// The dtypes values may have in a plan: `f16` and `i8` only exist as storage.
const VALUE_DTYPES: &[Dtype] = &[Dtype::F32, Dtype::I32, Dtype::I64, Dtype::Bool];

/// Rank of a dtype in PyTorch type promotion: bool, then integers, then floats.
fn category(dtype: Dtype) -> u8 {
    match dtype {
        Dtype::Bool => 0,
        Dtype::I8 | Dtype::I32 | Dtype::I64 => 1,
        Dtype::F16 | Dtype::F32 => 2,
    }
}

/// Promotes two dtypes of the same priority, as `torch.promote_types` does.
fn promote(a: Dtype, b: Dtype) -> Dtype {
    match (category(a), category(b)) {
        (x, y) if x != y => {
            if x > y {
                a
            } else {
                b
            }
        }
        _ if a == Dtype::I64 || b == Dtype::I64 => Dtype::I64,
        _ => a,
    }
}

/// The result dtype of an arithmetic op, as `torch.result_type`: tensors with
/// dimensions decide, unless a zero-dimensional tensor, then a scalar, belongs
/// to a higher category.
fn result_type(dtypes: &[Dtype], shapes: &[&[u64]], scalar: Option<Number>) -> Dtype {
    let tier = |zero_dim: bool| {
        dtypes
            .iter()
            .zip(shapes)
            .filter(|(_, shape)| shape.is_empty() == zero_dim)
            .map(|(&dtype, _)| dtype)
            .reduce(promote)
    };
    let tiers = [tier(false), tier(true), scalar.map(Number::default_dtype)];
    let mut result: Option<Dtype> = None;
    for dtype in tiers.into_iter().flatten() {
        result = match result {
            Some(current) if category(dtype) <= category(current) => Some(current),
            _ => Some(dtype),
        };
    }
    result.expect("an arithmetic op has a tensor operand")
}

/// The dtypes of the outputs of `op`, given those of its inputs, or why it
/// rejects them. This is the dtype half of the op-level pass: every op lists
/// the input dtypes it accepts and follows PyTorch for its outputs.
pub fn infer_dtypes(op: &Op, inputs: &[Dtype], shapes: &[&[u64]]) -> Result<Vec<Dtype>, String> {
    let reject = |index: usize| {
        format!(
            "input {index} is {:?}, which this op does not accept",
            inputs[index]
        )
    };
    let only = |accepted: &[Dtype]| match inputs.iter().position(|d| !accepted.contains(d)) {
        Some(index) => Err(reject(index)),
        None => Ok(()),
    };
    let f32 = || only(&[Dtype::F32]).map(|()| vec![Dtype::F32]);
    match op {
        Op::CacheWrite | Op::CachedAttention { .. } | Op::SampleCandidates => {
            let i32_scalar = matches!(inputs.last(), Some(Dtype::I32 | Dtype::I64));
            let valid = i32_scalar
                && match op {
                    Op::CacheWrite => inputs[..2] == [Dtype::F32; 2],
                    Op::CachedAttention { .. } => inputs[..3] == [Dtype::F32; 3],
                    Op::SampleCandidates => {
                        inputs[0] == Dtype::F32 && matches!(inputs[1], Dtype::I32 | Dtype::I64)
                    }
                    _ => unreachable!(),
                };
            if !valid {
                return Err("expected f32 data and an integer position or seed".to_owned());
            }
            Ok(vec![if matches!(op, Op::SampleCandidates) {
                Dtype::I64
            } else {
                Dtype::F32
            }])
        }
        Op::Dequantize(_) => {
            if inputs != [Dtype::I8, Dtype::F32] {
                return Err("dequantization needs packed i8 weights and f32 scales".to_owned());
            }
            Ok(vec![Dtype::F32])
        }
        Op::Custom(_)
        | Op::Rotary
        | Op::Matmul { .. }
        | Op::Reduce { .. }
        | Op::Convolution(_)
        | Op::BatchNorm { .. }
        | Op::LayerNorm { .. }
        | Op::GroupNorm { .. } => f32(),
        Op::Scatter(_) => {
            only(VALUE_DTYPES)?;
            let values = inputs.len() - 1;
            if inputs[0] != inputs[values]
                || inputs[1..values]
                    .iter()
                    .any(|d| !matches!(d, Dtype::I32 | Dtype::I64))
            {
                return Err(
                    "scatter needs integer indices and values matching the destination dtype"
                        .to_owned(),
                );
            }
            Ok(vec![inputs[0]])
        }
        Op::Movement(kind) => {
            only(VALUE_DTYPES)?;
            if matches!(
                kind,
                Movement::AdaptiveAverage
                    | Movement::Resize(Resize {
                        mode: ResizeMode::Bilinear { .. } | ResizeMode::Bicubic { .. },
                        ..
                    })
            ) {
                return f32();
            }
            if matches!(
                kind,
                Movement::Gather(_) | Movement::IndexSelect(_) | Movement::Index { .. }
            ) {
                for (index, &dtype) in inputs.iter().enumerate().skip(1) {
                    if dtype != Dtype::I64
                        && !(dtype == Dtype::I32 && !matches!(kind, Movement::Gather(_)))
                    {
                        return Err(format!(
                            "index tensor {index} must be an integer index (bool masks have dynamic output shapes)"
                        ));
                    }
                }
            }
            Ok(vec![inputs[0]])
        }
        Op::Row { kind, .. } => match kind {
            RowKind::Softmax { .. } | RowKind::Variance { .. } => f32(),
            RowKind::Extrema { output, .. } => {
                only(VALUE_DTYPES)?;
                match output {
                    Extremum::Both => Ok(vec![inputs[0], Dtype::I64]),
                    Extremum::Value => Ok(vec![inputs[0]]),
                    Extremum::Index if inputs[0] == Dtype::Bool => {
                        Err("argmax/argmin do not accept bool inputs".to_owned())
                    }
                    Extremum::Index => Ok(vec![Dtype::I64]),
                }
            }
            RowKind::Any => {
                only(VALUE_DTYPES)?;
                Ok(vec![Dtype::Bool])
            }
            RowKind::Scan { dtype } | RowKind::Product { dtype } => {
                only(VALUE_DTYPES)?;
                let dtype = dtype.unwrap_or(match inputs[0] {
                    Dtype::F32 => Dtype::F32,
                    _ => Dtype::I64,
                });
                if dtype == Dtype::Bool && matches!(kind, RowKind::Scan { .. }) {
                    return Err("cumsum does not support bool accumulation".to_owned());
                }
                Ok(vec![dtype])
            }
        },
        Op::Unary(unary) => {
            use crate::kernel::UnaryFn as F;
            match unary {
                Unary::Clamp { min, max } => {
                    only(VALUE_DTYPES)?;
                    let dtype =
                        [*min, *max]
                            .into_iter()
                            .flatten()
                            .fold(inputs[0], |dtype, bound| {
                                let bound = bound.default_dtype();
                                if category(bound) > category(dtype) {
                                    bound
                                } else {
                                    dtype
                                }
                            });
                    if dtype == Dtype::Bool {
                        return Err("clamp requires a numeric result".to_owned());
                    }
                    Ok(vec![dtype])
                }
                Unary::Math(F::Sign) => {
                    only(VALUE_DTYPES)?;
                    Ok(vec![inputs[0]])
                }
                Unary::Math(F::Abs | F::Neg | F::Floor | F::Ceil | F::Round | F::Trunc) => {
                    only(&[Dtype::F32, Dtype::I32, Dtype::I64])?;
                    Ok(vec![inputs[0]])
                }
                Unary::Math(_) => {
                    only(VALUE_DTYPES)?;
                    Ok(vec![Dtype::F32])
                }
                _ => f32(),
            }
        }
        Op::Power {
            scalar,
            scalar_base,
        } => {
            only(VALUE_DTYPES)?;
            let dtype = result_type(inputs, shapes, Some(*scalar));
            if dtype == Dtype::Bool {
                return Err("bool power is not supported".to_owned());
            }
            if !*scalar_base && !dtype.is_float() && scalar.as_f64() < 0.0 {
                return Err("integer power cannot use a negative scalar exponent".to_owned());
            }
            Ok(vec![dtype])
        }
        Op::Order { .. } => {
            only(VALUE_DTYPES)?;
            Ok(vec![inputs[0], Dtype::I64])
        }
        Op::Pool(pool) => {
            only(&[Dtype::F32])?;
            Ok(match pool.kind {
                PoolKind::Max | PoolKind::AdaptiveMax => vec![Dtype::F32, Dtype::I64],
                PoolKind::Avg { .. } => vec![Dtype::F32],
            })
        }
        Op::Binary {
            kind,
            scalar,
            alpha,
        } => {
            only(VALUE_DTYPES)?;
            let dtype = result_type(inputs, shapes, *scalar);
            if dtype == Dtype::Bool
                && matches!(
                    kind,
                    BinaryKind::Power | BinaryKind::Remainder | BinaryKind::Fmod
                )
            {
                return Err("power and remainder require a numeric result".to_owned());
            }
            if !dtype.is_float()
                && matches!(kind, BinaryKind::Remainder | BinaryKind::Fmod)
                && scalar.is_some_and(|number| number.as_f64() == 0.0)
            {
                return Err("integer remainder divisor is zero".to_owned());
            }
            if *kind == BinaryKind::Sub
                && (inputs.contains(&Dtype::Bool) || matches!(scalar, Some(Number::Bool(_))))
            {
                return Err("subtracting bools is not supported, as in PyTorch".to_owned());
            }
            if matches!(kind, BinaryKind::Add | BinaryKind::Sub) {
                if matches!(alpha, Number::Bool(_)) && dtype != Dtype::Bool {
                    return Err("a bool `alpha` requires a bool result".to_owned());
                }
                if matches!(alpha, Number::Float(_)) && !dtype.is_float() {
                    return Err("a float `alpha` requires a floating-point result".to_owned());
                }
            }
            Ok(vec![match kind {
                BinaryKind::Div => Dtype::F32,
                _ => dtype,
            }])
        }
        Op::Compare { scalar, .. } => {
            only(VALUE_DTYPES)?;
            result_type(inputs, shapes, *scalar);
            Ok(vec![Dtype::Bool])
        }
        Op::Where => {
            only(VALUE_DTYPES)?;
            if inputs[0] != Dtype::Bool {
                return Err(reject(0));
            }
            Ok(vec![result_type(&inputs[1..], &shapes[1..], None)])
        }
        Op::Bitwise(kind) => {
            only(&[Dtype::Bool, Dtype::I32, Dtype::I64])?;
            if *kind == BitwiseKind::Not && inputs[0] != Dtype::Bool {
                return Err("`bitwise_not` is only supported on bools".to_owned());
            }
            Ok(vec![result_type(inputs, shapes, None)])
        }
        Op::View(ViewOp::Split { sizes, .. }) => {
            only(VALUE_DTYPES)?;
            Ok(vec![inputs[0]; sizes.len()])
        }
        Op::View(_) => {
            only(VALUE_DTYPES)?;
            Ok(vec![inputs[0]])
        }
        Op::Cat { .. } => {
            only(VALUE_DTYPES)?;
            match inputs.iter().position(|&dtype| dtype != inputs[0]) {
                Some(index) => Err(format!(
                    "input {index} is {:?}, the first one {:?}: cast them to one dtype",
                    inputs[index], inputs[0]
                )),
                None => Ok(vec![inputs[0]]),
            }
        }
        Op::Attention { mask, .. } => {
            if let Some(index) = (0..3).find(|&index| inputs[index] != Dtype::F32) {
                return Err(reject(index));
            }
            if mask.is_some() && !matches!(inputs[3], Dtype::F32 | Dtype::Bool) {
                return Err(reject(3));
            }
            Ok(vec![Dtype::F32])
        }
        Op::Embedding { .. } => {
            if inputs[0] != Dtype::F32 {
                return Err(reject(0));
            }
            if !matches!(inputs[1], Dtype::I32 | Dtype::I64) {
                return Err(reject(1));
            }
            Ok(vec![Dtype::F32])
        }
        &Op::Cast(dtype) => {
            only(VALUE_DTYPES)?;
            Ok(vec![dtype])
        }
        &Op::Arange { dtype, .. } | &Op::Full { dtype, .. } => {
            only(VALUE_DTYPES)?;
            Ok(vec![dtype])
        }
    }
}

/// Parses `node` for the given input shapes, dtypes and symbol sizes,
/// returning the op and every output it computes, in schema order. An op may
/// compute fewer outputs than its schema lists, such as the statistics of a
/// layer norm: the graph must not read the others.
pub fn resolve(
    node: &Node,
    shapes: &[&[u64]],
    dtypes: &[Dtype],
    symbols: &HashMap<String, u64>,
) -> Result<(Op, Vec<Output>), CompileError> {
    if !SUPPORTED_OPS.contains(&node.op.as_str()) {
        return Err(CompileError::UnsupportedOp(node.op.clone()));
    }
    let resolve_checked = || {
        if shapes.len() != dtypes.len() {
            return Err("each input needs a dtype".to_owned());
        }
        for shape in shapes {
            checked_elements(shape)?;
        }
        let (op, output_shapes) = resolve_supported(node, shapes, dtypes, symbols)?;
        let output_dtypes = infer_dtypes(&op, dtypes, shapes)?;
        let mut outputs = Vec::with_capacity(output_shapes.len());
        for (shape, dtype) in output_shapes.into_iter().zip(output_dtypes) {
            checked_elements(&shape)?;
            outputs.push(Output { shape, dtype });
        }
        Ok((op, outputs))
    };
    resolve_checked().map_err(|message| CompileError::InvalidOp {
        op: node.op.clone(),
        message,
    })
}

fn checked_elements(shape: &[u64]) -> Result<u64, String> {
    shape.iter().try_fold(1u64, |count, &dim| {
        count
            .checked_mul(dim)
            .filter(|&n| dim > 0 && n <= u64::from(u32::MAX))
            .ok_or_else(|| format!("empty or oversized shape {shape:?}"))
    })
}

/// Evaluates a symbolic size expression such as `s27` or `2*s27` against the
/// concrete sizes of the input dimensions, named as in the graph.
pub fn eval_symbol(expr: &str, symbols: &HashMap<String, u64>) -> Result<u64, String> {
    let mut total: u64 = 0;
    for sum_term in expr.split('+') {
        let mut product: u64 = 1;
        for factor in sum_term.split('*') {
            let factor = factor.trim();
            if factor.is_empty() {
                return Err(format!("invalid expression `{expr}`"));
            }
            let value = match factor.parse::<u64>() {
                Ok(literal) => literal,
                Err(_) => *symbols
                    .get(factor)
                    .ok_or_else(|| format!("unknown symbol `{factor}` in `{expr}`"))?,
            };
            product = product
                .checked_mul(value)
                .ok_or_else(|| format!("`{expr}` overflows"))?;
        }
        total = total
            .checked_add(product)
            .ok_or_else(|| format!("`{expr}` overflows"))?;
    }
    Ok(total)
}

/// Evaluates a symbolic integer attribute.
pub fn eval_int(expr: &str, symbols: &HashMap<String, u64>) -> Result<i64, String> {
    let value = eval_symbol(expr, symbols)?;
    i64::try_from(value).map_err(|_| format!("`{expr}` is too large: {value}"))
}

/// One output of the given shape.
fn single((op, shape): (Op, Vec<u64>)) -> (Op, Vec<Vec<u64>>) {
    (op, vec![shape])
}

fn resolve_supported(
    node: &Node,
    shapes: &[&[u64]],
    dtypes: &[Dtype],
    symbols: &HashMap<String, u64>,
) -> Result<(Op, Vec<Vec<u64>>), String> {
    if shapes.len() != node.tensors().count() {
        return Err("each tensor argument needs a shape".to_owned());
    }
    // A node lists one slot per tensor argument. `schema` checks their count,
    // and that every slot but the `optional` ones holds a tensor.
    let schema = |count: usize, optional: &[usize]| {
        if node.inputs.len() != count {
            return Err(format!(
                "expected {count} tensor arguments, got {}",
                node.inputs.len()
            ));
        }
        match (0..count).find(|i| node.inputs[*i].is_none() && !optional.contains(i)) {
            Some(index) => Err(format!("tensor argument {index} is required")),
            None => Ok(()),
        }
    };
    let arity = |count: usize| schema(count, &[]);
    let present = |index: usize| node.inputs[index].is_some();
    let attributes = Attributes { node, symbols };

    match node.op.as_str() {
        "aten._native_batch_norm_legit_no_training.default" => {
            // input, weight?, bias?, running_mean, running_var
            schema(5, &[1, 2])?;
            let (weight, bias) = (present(1), present(2));
            resolve_batch_norm(&attributes, shapes, weight, bias).map(single)
        }
        "avel.rotary.default" => {
            arity(3)?;
            let x = shapes[0];
            if x.last().is_none_or(|depth| !depth.is_multiple_of(2))
                || !broadcasts_to(shapes[1], x)
                || !broadcasts_to(shapes[2], x)
            {
                return Err(
                    "rotary needs even depth and cosine/sine tables broadcastable to the input"
                        .to_owned(),
                );
            }
            Ok(single((Op::Rotary, x.to_vec())))
        }
        "avel.cache_write.default" => {
            arity(3)?;
            let (cache, patch) = (shapes[0], shapes[1]);
            let Some((axis, _)) = cache.len().checked_sub(2).zip(patch.len().checked_sub(2)) else {
                return Err("cache writes need [..., sequence, depth] tensors".to_owned());
            };
            if cache.len() != patch.len()
                || cache
                    .iter()
                    .zip(patch)
                    .enumerate()
                    .any(|(i, (&a, &b))| if i == axis { a < b } else { a != b })
                || cache[axis] > i32::MAX as u64
                || !matches!(shapes[2], [] | [1])
            {
                return Err("cache writes need a matching patch and a scalar position".to_owned());
            }
            Ok(single((Op::CacheWrite, cache.to_vec())))
        }
        "avel.cached_attention.default" => {
            arity(4)?;
            let (query, key, value) = (shapes[0], shapes[1], shapes[2]);
            if query.len() < 3 || query.len() != key.len() || query.len() != value.len() {
                return Err(
                    "cached attention needs matching [..., heads, sequence, depth] tensors"
                        .to_owned(),
                );
            }
            let axis = query.len() - 2;
            if query[..axis] != key[..axis]
                || key[..axis] != value[..axis]
                || query[axis + 1] != key[axis + 1]
                || query[axis] > key[axis]
                || key[axis] != value[axis]
                || key[axis] > i32::MAX as u64
                || !matches!(shapes[3], [] | [1])
            {
                return Err("invalid cached attention shape or position".to_owned());
            }
            let scale = attributes
                .scalar("scale")?
                .unwrap_or_else(|| 1.0 / (query[axis + 1] as f32).sqrt());
            let mut shape = query.to_vec();
            shape[axis + 1] = value[axis + 1];
            Ok(single((Op::CachedAttention { scale }, shape)))
        }
        "avel.sample_candidates.default" => {
            arity(3)?;
            let (Some((_, rows)), true) = (shapes[0].split_last(), shapes[0] == shapes[1]) else {
                return Err("sampling needs matching non-scalar logits and indices".to_owned());
            };
            if !matches!(shapes[2], [] | [1]) {
                return Err("sampling needs a scalar seed".to_owned());
            }
            Ok(single((Op::SampleCandidates, rows.to_vec())))
        }
        "avel.dequantize_per_channel.default" => {
            arity(2)?;
            let axis = resolve_axis(attributes.required_int("axis")?, shapes[0].len())?;
            if shapes[1] != [shapes[0][axis]] {
                return Err("dequantization needs one scale per channel".to_owned());
            }
            Ok(single((Op::Dequantize(axis), shapes[0].to_vec())))
        }
        "aten._to_copy.default" => {
            arity(1)?;
            resolve_cast(&attributes, shapes[0], dtypes[0]).map(single)
        }
        "aten.add.Tensor"
        | "aten.sub.Tensor"
        | "aten.mul.Tensor"
        | "aten.div.Tensor"
        | "aten.add.Scalar"
        | "aten.sub.Scalar"
        | "aten.mul.Scalar"
        | "aten.div.Scalar"
        | "aten.maximum.default"
        | "aten.minimum.default"
        | "aten.pow.Tensor_Tensor"
        | "aten.remainder.Scalar"
        | "aten.fmod.Scalar" => {
            let kind = match node.op.split('.').nth(1).unwrap() {
                "add" => BinaryKind::Add,
                "sub" => BinaryKind::Sub,
                "mul" => BinaryKind::Mul,
                "div" => BinaryKind::Div,
                "maximum" => BinaryKind::Maximum,
                "minimum" => BinaryKind::Minimum,
                "pow" => BinaryKind::Power,
                "remainder" => BinaryKind::Remainder,
                "fmod" => BinaryKind::Fmod,
                _ => unreachable!(),
            };
            let count = if node.op.ends_with(".Scalar") {
                1
            } else if matches!(
                kind,
                BinaryKind::Maximum | BinaryKind::Minimum | BinaryKind::Power
            ) {
                2
            } else {
                node.inputs.len()
            };
            arity(count)?;
            resolve_binary(&attributes, kind, shapes).map(single)
        }
        "aten.pow.Tensor_Scalar" | "aten.pow.Scalar" => {
            arity(1)?;
            let scalar_base = node.op == "aten.pow.Scalar";
            let scalar =
                attributes.required_number(if scalar_base { "self" } else { "exponent" })?;
            Ok(single((
                Op::Power {
                    scalar,
                    scalar_base,
                },
                shapes[0].to_vec(),
            )))
        }
        "aten.elu.default" | "aten.leaky_relu.default" => {
            arity(1)?;
            let unary = if node.op == "aten.elu.default" {
                Unary::Elu {
                    alpha: attributes.scalar("alpha")?.unwrap_or(1.0),
                    scale: attributes.scalar("scale")?.unwrap_or(1.0),
                    input_scale: attributes.scalar("input_scale")?.unwrap_or(1.0),
                }
            } else {
                Unary::LeakyRelu {
                    slope: attributes.scalar("negative_slope")?.unwrap_or(0.01),
                }
            };
            Ok(single((Op::Unary(unary), shapes[0].to_vec())))
        }
        "aten.tanh.default"
        | "aten.exp.default"
        | "aten.log.default"
        | "aten.sqrt.default"
        | "aten.rsqrt.default"
        | "aten.reciprocal.default"
        | "aten.abs.default"
        | "aten.neg.default"
        | "aten.sign.default"
        | "aten.erf.default"
        | "aten.expm1.default"
        | "aten.log1p.default"
        | "aten.sin.default"
        | "aten.cos.default"
        | "aten.floor.default"
        | "aten.ceil.default"
        | "aten.round.default"
        | "aten.trunc.default" => {
            use crate::kernel::UnaryFn as F;
            arity(1)?;
            let function = match node.op.as_str() {
                "aten.tanh.default" => F::Tanh,
                "aten.exp.default" => F::Exp,
                "aten.log.default" => F::Log,
                "aten.sqrt.default" => F::Sqrt,
                "aten.rsqrt.default" => F::InverseSqrt,
                "aten.reciprocal.default" => F::Reciprocal,
                "aten.abs.default" => F::Abs,
                "aten.neg.default" => F::Neg,
                "aten.sign.default" => F::Sign,
                "aten.erf.default" => F::Erf,
                "aten.expm1.default" => F::Expm1,
                "aten.log1p.default" => F::Log1p,
                "aten.sin.default" => F::Sin,
                "aten.cos.default" => F::Cos,
                "aten.floor.default" => F::Floor,
                "aten.ceil.default" => F::Ceil,
                "aten.round.default" => F::Round,
                "aten.trunc.default" => F::Trunc,
                _ => unreachable!(),
            };
            Ok(single((
                Op::Unary(Unary::Math(function)),
                shapes[0].to_vec(),
            )))
        }
        "aten.eq.Scalar" | "aten.ne.Scalar" | "aten.lt.Scalar" | "aten.le.Scalar"
        | "aten.gt.Scalar" | "aten.ge.Scalar" | "aten.eq.Tensor" | "aten.ne.Tensor"
        | "aten.lt.Tensor" | "aten.le.Tensor" | "aten.gt.Tensor" | "aten.ge.Tensor" => {
            let kind = match &node.op[5..7] {
                "eq" => CompareKind::Eq,
                "ne" => CompareKind::Ne,
                "lt" => CompareKind::Lt,
                "le" => CompareKind::Le,
                "gt" => CompareKind::Gt,
                _ => CompareKind::Ge,
            };
            arity(node.inputs.len())?;
            let scalar = match shapes.len() {
                1 => Some(attributes.required_number("other")?),
                2 => None,
                _ => {
                    return Err(
                        "a comparison needs one tensor and a scalar, or two tensors".to_owned()
                    );
                }
            };
            let shape = broadcast_all(shapes)?;
            Ok(single((Op::Compare { kind, scalar }, shape)))
        }
        "aten.where.self" => {
            arity(3)?;
            Ok(single((Op::Where, broadcast_all(shapes)?)))
        }
        "aten.bitwise_and.Tensor" | "aten.bitwise_or.Tensor" | "aten.bitwise_xor.Tensor" => {
            arity(2)?;
            let kind = match node.op.as_str() {
                "aten.bitwise_and.Tensor" => BitwiseKind::And,
                "aten.bitwise_or.Tensor" => BitwiseKind::Or,
                _ => BitwiseKind::Xor,
            };
            Ok(single((Op::Bitwise(kind), broadcast_all(shapes)?)))
        }
        "aten.bitwise_not.default" => {
            arity(1)?;
            Ok(single((Op::Bitwise(BitwiseKind::Not), shapes[0].to_vec())))
        }
        "aten.addmm.default" => {
            arity(3)?;
            resolve_matmul(&attributes, Some(shapes[0]), shapes[1], shapes[2], false).map(single)
        }
        "aten.arange.start_step" => {
            arity(0)?;
            resolve_arange(&attributes).map(single)
        }
        "aten.avg_pool2d.default" | "aten.avg_pool3d.default" => {
            arity(1)?;
            let rank = if node.op == "aten.avg_pool3d.default" {
                3
            } else {
                2
            };
            resolve_pool(&attributes, shapes[0], rank, false).map(single)
        }
        "aten.bmm.default" => {
            arity(2)?;
            resolve_matmul(&attributes, None, shapes[0], shapes[1], true).map(single)
        }
        "aten.cat.default" => {
            arity(node.inputs.len())?;
            resolve_cat(&attributes, shapes).map(single)
        }
        "aten.clamp.default" | "aten.hardtanh.default" => {
            arity(1)?;
            let (min, max) = match node.op.as_str() {
                "aten.clamp.default" => (attributes.number("min")?, attributes.number("max")?),
                _ => (
                    Some(attributes.required_number("min_val")?),
                    Some(attributes.required_number("max_val")?),
                ),
            };
            if min.is_none() && max.is_none() {
                return Err("clamp needs at least one bound".to_owned());
            }
            Ok(single((
                Op::Unary(Unary::Clamp { min, max }),
                shapes[0].to_vec(),
            )))
        }
        "aten.clone.default" => {
            arity(1)?;
            Ok(single((Op::View(ViewOp::Alias), shapes[0].to_vec())))
        }
        "aten.convolution.default" => {
            // input, weight, bias?
            schema(3, &[2])?;
            let bias = present(2);
            resolve_convolution(&attributes, shapes, bias).map(single)
        }
        "aten.embedding.default" => {
            arity(2)?;
            resolve_embedding(&attributes, shapes).map(single)
        }
        "aten.expand.default" => {
            arity(1)?;
            resolve_expand(&attributes, shapes[0]).map(single)
        }
        "aten.full.default" => {
            arity(0)?;
            resolve_full(&attributes, None).map(single)
        }
        "aten.full_like.default" => {
            arity(1)?;
            resolve_full(&attributes, Some((shapes[0], dtypes[0]))).map(single)
        }
        "aten.gelu.default" => {
            arity(1)?;
            let approximate = match attributes.get("approximate") {
                Some(Attr::Str(mode)) if mode == "none" => false,
                Some(Attr::Str(mode)) if mode == "tanh" => true,
                other => return Err(format!("invalid `approximate`: {other:?}")),
            };
            Ok(single((
                Op::Unary(Unary::Gelu { approximate }),
                shapes[0].to_vec(),
            )))
        }
        "aten.repeat.default"
        | "aten.flip.default"
        | "aten.roll.default"
        | "aten.pad.default"
        | "aten.constant_pad_nd.default"
        | "aten.upsample_nearest2d.vec"
        | "aten.upsample_bilinear2d.vec"
        | "aten.upsample_bicubic2d.vec"
        | "aten._adaptive_avg_pool2d.default" => {
            arity(1)?;
            resolve_movement(&attributes, shapes).map(single)
        }
        "aten.index_select.default" | "aten.gather.default" => {
            arity(2)?;
            resolve_movement(&attributes, shapes).map(single)
        }
        "aten.scatter.src" | "aten.scatter_add.default" => {
            arity(3)?;
            scatter::resolve_scatter(&attributes, shapes).map(single)
        }
        "aten.index_put.default" => {
            let count = node.inputs.len();
            if count < 3 {
                return Err("index_put needs a destination, indices and values".to_owned());
            }
            schema(count, &(1..count - 1).collect::<Vec<_>>())?;
            scatter::resolve_index_put(&attributes, shapes).map(single)
        }
        "aten.index.Tensor" => {
            schema(
                node.inputs.len().max(1),
                &(1..node.inputs.len()).collect::<Vec<_>>(),
            )?;
            resolve_movement(&attributes, shapes).map(single)
        }
        "aten._softmax.default"
        | "aten._log_softmax.default"
        | "aten.amax.default"
        | "aten.amin.default"
        | "aten.max.dim"
        | "aten.min.dim"
        | "aten.argmax.default"
        | "aten.argmin.default"
        | "aten.var.correction"
        | "aten.cumsum.default"
        | "aten.prod.dim_int"
        | "aten.any.dim" => {
            arity(1)?;
            resolve_row(&attributes, shapes[0])
        }
        "aten.topk.default" | "aten.sort.default" | "aten.sort.stable" => {
            arity(1)?;
            let topk = node.op == "aten.topk.default";
            let mut shape = shapes[0].to_vec();
            let axis = resolve_axis(attributes.int("dim")?.unwrap_or(-1), shape.len().max(1))?;
            let largest = if topk {
                let k = attributes.required_int("k")?;
                if !(1..=shape.get(axis).copied().unwrap_or(1) as i64).contains(&k) {
                    return Err("topk requires 0 < k <= dimension size".to_owned());
                }
                if !shape.is_empty() {
                    shape[axis] = k as u64;
                }
                attributes.bool("largest", true)?
            } else {
                attributes.bool("descending", false)?
            };
            Ok((Op::Order { axis, largest }, vec![shape.clone(), shape]))
        }
        "aten.adaptive_max_pool2d.default" => {
            arity(1)?;
            let (op, shape) = spatial::resolve_adaptive_max_pool(&attributes, shapes[0])?;
            Ok((op, vec![shape.clone(), shape]))
        }
        "aten.max_pool2d_with_indices.default" | "aten.max_pool3d_with_indices.default" => {
            arity(1)?;
            // The values, then the indices of the maxima.
            let rank = if node.op == "aten.max_pool3d_with_indices.default" {
                3
            } else {
                2
            };
            let (op, shape) = resolve_pool(&attributes, shapes[0], rank, true)?;
            Ok((op, vec![shape.clone(), shape]))
        }
        "aten.mean.default" | "aten.mean.dim" => {
            arity(1)?;
            resolve_reduce(&attributes, ReduceKind::Mean, shapes[0]).map(single)
        }
        "aten.mm.default" => {
            arity(2)?;
            resolve_matmul(&attributes, None, shapes[0], shapes[1], false).map(single)
        }
        "aten.native_group_norm.default" | "aten._native_batch_norm_legit.no_stats" => {
            schema(3, &[1, 2])?;
            resolve_group_norm(&attributes, shapes, present(1), present(2)).map(single)
        }
        "aten.native_layer_norm.default" => {
            // input, weight?, bias?
            schema(3, &[1, 2])?;
            let (weight, bias) = (present(1), present(2));
            resolve_layer_norm(&attributes, shapes, weight, bias).map(single)
        }
        "aten.permute.default" => {
            arity(1)?;
            resolve_permute(&attributes, shapes[0]).map(single)
        }
        "aten.relu.default" => {
            arity(1)?;
            Ok(single((Op::Unary(Unary::Relu), shapes[0].to_vec())))
        }
        "aten.rms_norm.default" => {
            // input, weight?
            schema(2, &[1])?;
            resolve_rms_norm(&attributes, shapes, present(1)).map(single)
        }
        "aten.scalar_tensor.default" => {
            arity(0)?;
            resolve_full(&attributes, None).map(single)
        }
        "aten.scaled_dot_product_attention.default" => {
            // query, key, value, attn_mask?
            schema(4, &[3])?;
            let mask = present(3);
            resolve_attention(&attributes, shapes, dtypes, mask).map(single)
        }
        "aten.select.int" => {
            arity(1)?;
            resolve_select(&attributes, shapes[0]).map(single)
        }
        "aten.sigmoid.default" => {
            arity(1)?;
            Ok(single((Op::Unary(Unary::Sigmoid), shapes[0].to_vec())))
        }
        "aten.slice.Tensor" => {
            arity(1)?;
            resolve_slice(&attributes, shapes[0]).map(single)
        }
        "aten.split_with_sizes.default" => {
            arity(1)?;
            resolve_split(&attributes, shapes[0])
        }
        "aten.squeeze.dims" => {
            arity(1)?;
            resolve_squeeze(&attributes, shapes[0]).map(single)
        }
        "aten.sum.dim_IntList" => {
            arity(1)?;
            resolve_reduce(&attributes, ReduceKind::Sum, shapes[0]).map(single)
        }
        "aten.unsqueeze.default" => {
            arity(1)?;
            let rank = shapes[0].len();
            let axis = resolve_axis(attributes.required_int("dim")?, rank + 1)?;
            let mut shape = shapes[0].to_vec();
            shape.insert(axis, 1);
            Ok(single((Op::View(ViewOp::Unsqueeze(axis)), shape)))
        }
        "aten.view.default" => {
            arity(1)?;
            resolve_view(&attributes, shapes[0]).map(single)
        }
        _ => unreachable!("`{}` is in SUPPORTED_OPS but not resolved", node.op),
    }
}

/// Typed access to the attributes of a node. Integer attributes may be
/// symbolic, resolved against `symbols`, as the exporter writes sizes that
/// depend on a dynamic input dimension.
struct Attributes<'a> {
    node: &'a Node,
    symbols: &'a HashMap<String, u64>,
}

impl Attributes<'_> {
    fn get(&self, name: &str) -> Option<&Attr> {
        self.node.attributes.get(name)
    }

    /// One integer attribute: a plain number, or a symbolic expression.
    fn int_value(&self, value: &Attr) -> Result<i64, String> {
        match value {
            Attr::Int(value) => Ok(*value),
            Attr::Sym(expr) => eval_int(expr, self.symbols),
            _ => Err(format!("invalid integer: {value:?}")),
        }
    }

    fn scalar(&self, name: &str) -> Result<Option<f32>, String> {
        let scalar = match self.get(name) {
            None | Some(Attr::Null) => return Ok(None),
            Some(Attr::Float(value)) => *value as f32,
            Some(Attr::Int(value)) => *value as f32,
            Some(Attr::Bool(value)) => f32::from(u8::from(*value)),
            Some(value) => return Err(format!("invalid `{name}`: {value:?}")),
        };
        if !scalar.is_finite() {
            return Err(format!("`{name}` is not finite: {scalar}"));
        }
        Ok(Some(scalar))
    }

    /// A scalar keeping whether it is a bool, an integer or a float.
    fn number(&self, name: &str) -> Result<Option<Number>, String> {
        let number = match self.get(name) {
            None | Some(Attr::Null) => return Ok(None),
            Some(Attr::Bool(value)) => Number::Bool(*value),
            Some(Attr::Int(value)) => Number::Int(*value),
            Some(Attr::Sym(_)) => Number::Int(self.required_int(name)?),
            Some(Attr::Float(value)) if value.is_finite() => Number::Float(*value),
            Some(value) => return Err(format!("invalid `{name}`: {value:?}")),
        };
        Ok(Some(number))
    }

    fn required_number(&self, name: &str) -> Result<Number, String> {
        self.number(name)?
            .ok_or_else(|| format!("missing `{name}`"))
    }

    /// A `torch.dtype` argument, exported as its name.
    fn dtype(&self, name: &str) -> Result<Option<Dtype>, String> {
        match self.get(name) {
            None | Some(Attr::Null) => Ok(None),
            Some(Attr::Str(dtype)) => match dtype.as_str() {
                "torch.float32" => Ok(Some(Dtype::F32)),
                "torch.int32" => Ok(Some(Dtype::I32)),
                "torch.int64" => Ok(Some(Dtype::I64)),
                "torch.bool" => Ok(Some(Dtype::Bool)),
                other => Err(format!("unsupported dtype `{other}`")),
            },
            Some(value) => Err(format!("invalid `{name}`: {value:?}")),
        }
    }

    fn required_scalar(&self, name: &str) -> Result<f32, String> {
        self.scalar(name)?
            .ok_or_else(|| format!("missing `{name}`"))
    }

    fn int(&self, name: &str) -> Result<Option<i64>, String> {
        match self.get(name) {
            None | Some(Attr::Null) => Ok(None),
            Some(value) => self.int_value(value).map(Some),
        }
    }

    fn required_int(&self, name: &str) -> Result<i64, String> {
        self.int(name)?.ok_or_else(|| format!("missing `{name}`"))
    }

    fn bool(&self, name: &str, default: bool) -> Result<bool, String> {
        match self.get(name) {
            None | Some(Attr::Null) => Ok(default),
            Some(Attr::Bool(value)) => Ok(*value),
            Some(value) => Err(format!("invalid `{name}`: {value:?}")),
        }
    }

    fn ints(&self, name: &str) -> Result<Option<Vec<i64>>, String> {
        let Some(value) = self.get(name).filter(|value| !matches!(value, Attr::Null)) else {
            return Ok(None);
        };
        let values = match value {
            Attr::Int(_) | Attr::Sym(_) => {
                return self.int_value(value).map(|single| Some(vec![single]));
            }
            Attr::List(values) => values,
            _ => return Err(format!("invalid `{name}`: {value:?}")),
        };
        values
            .iter()
            .map(|item| self.int_value(item))
            .collect::<Result<Vec<_>, _>>()
            .map(Some)
    }

    /// Reads a list of distinct axes, resolving negative ones against `rank`.
    fn axes(&self, name: &str, rank: usize) -> Result<Option<Vec<usize>>, String> {
        let Some(values) = self.ints(name)? else {
            return Ok(None);
        };
        let mut axes = Vec::with_capacity(values.len());
        for value in values {
            let axis = resolve_axis(value, rank)?;
            if axes.contains(&axis) {
                return Err(format!("axis {value} is repeated"));
            }
            axes.push(axis);
        }
        Ok(Some(axes))
    }

    /// Reads a per-spatial-axis list of `spatial` non-negative sizes. One
    /// value applies to every axis; `default` fills in a missing or empty list.
    fn spatial(&self, name: &str, spatial: usize, default: &[u64]) -> Result<Vec<u64>, String> {
        let values = match self.ints(name)? {
            Some(values) if !values.is_empty() => values,
            _ => return Ok(default.to_vec()),
        };
        let values = match values.as_slice() {
            &[single] => vec![single; spatial],
            _ => values,
        };
        if values.len() != spatial || values.iter().any(|&value| value < 0) {
            return Err(format!(
                "invalid `{name}` {values:?} for {spatial} spatial axes"
            ));
        }
        Ok(values.into_iter().map(|value| value as u64).collect())
    }
}

fn resolve_axis(axis: i64, rank: usize) -> Result<usize, String> {
    let resolved = if axis < 0 { axis + rank as i64 } else { axis };
    if resolved < 0 || resolved >= rank as i64 {
        return Err(format!("axis {axis} is out of range for rank {rank}"));
    }
    Ok(resolved as usize)
}

/// The shape both operands broadcast to, as in PyTorch.
pub fn broadcast(left: &[u64], right: &[u64]) -> Option<Vec<u64>> {
    let rank = left.len().max(right.len());
    let size = |shape: &[u64], axis: usize| {
        let lead = rank - shape.len();
        axis.checked_sub(lead).map_or(1, |own| shape[own])
    };
    (0..rank)
        .map(|axis| match (size(left, axis), size(right, axis)) {
            (a, b) if a == b || b == 1 => Some(a),
            (1, b) => Some(b),
            _ => None,
        })
        .collect()
}

/// The shape every input broadcasts to.
fn broadcast_all(shapes: &[&[u64]]) -> Result<Vec<u64>, String> {
    shapes.iter().try_fold(Vec::new(), |shape, input| {
        broadcast(&shape, input).ok_or_else(|| format!("cannot broadcast {shape:?} with {input:?}"))
    })
}

fn broadcasts_to(shape: &[u64], target: &[u64]) -> bool {
    broadcast(shape, target).is_some_and(|result| result == target)
}

/// `addmm` (with `bias`), `mm` and `bmm` (with `batched`).
fn resolve_matmul(
    attributes: &Attributes,
    bias: Option<&[u64]>,
    left: &[u64],
    right: &[u64],
    batched: bool,
) -> Result<(Op, Vec<u64>), String> {
    let rank = if batched { 3 } else { 2 };
    if left.len() != rank || right.len() != rank {
        return Err(format!(
            "expected rank {rank} operands, got {left:?} and {right:?}"
        ));
    }
    let (batch, rows, inner) = (&left[..rank - 2], left[rank - 2], left[rank - 1]);
    if right[..rank - 2] != *batch || right[rank - 2] != inner {
        return Err(format!("cannot multiply {left:?} by {right:?}"));
    }
    let mut shape = batch.to_vec();
    shape.extend([rows, right[rank - 1]]);
    let op = match bias {
        Some(bias) => {
            if !broadcasts_to(bias, &shape) {
                return Err(format!("cannot broadcast {bias:?} to {shape:?}"));
            }
            Op::Matmul {
                beta: attributes.required_scalar("beta")?,
                alpha: attributes.required_scalar("alpha")?,
                bias: true,
            }
        }
        None => Op::Matmul {
            beta: 0.0,
            alpha: 1.0,
            bias: false,
        },
    };
    Ok((op, shape))
}

fn resolve_binary(
    attributes: &Attributes,
    kind: BinaryKind,
    shapes: &[&[u64]],
) -> Result<(Op, Vec<u64>), String> {
    let alpha = attributes.number("alpha")?.unwrap_or(Number::Int(1));
    match *shapes {
        [left, right] => {
            let shape = broadcast(left, right)
                .ok_or_else(|| format!("cannot broadcast {left:?} with {right:?}"))?;
            let op = Op::Binary {
                kind,
                scalar: None,
                alpha,
            };
            Ok((op, shape))
        }
        [shape] => {
            let scalar = attributes.required_number("other")?;
            let op = Op::Binary {
                kind,
                scalar: Some(scalar),
                alpha,
            };
            Ok((op, shape.to_vec()))
        }
        _ => Err(format!("expected 1 or 2 inputs, got {}", shapes.len())),
    }
}

fn resolve_batch_norm(
    attributes: &Attributes,
    shapes: &[&[u64]],
    weight: bool,
    bias: bool,
) -> Result<(Op, Vec<u64>), String> {
    let input = shapes[0];
    let &[_, channels, ..] = input else {
        return Err(format!(
            "input has shape {input:?}, expected [batch, channels, ...]"
        ));
    };
    if let Some(shape) = shapes[1..].iter().find(|shape| **shape != [channels]) {
        return Err(format!(
            "expected per-channel [{channels}] statistics, got {shape:?}"
        ));
    }
    let op = Op::BatchNorm {
        eps: attributes.required_scalar("eps")?,
        weight,
        bias,
    };
    Ok((op, input.to_vec()))
}

fn resolve_cat(attributes: &Attributes, shapes: &[&[u64]]) -> Result<(Op, Vec<u64>), String> {
    if shapes.is_empty() {
        return Err("cat needs at least one tensor".to_owned());
    }
    let rank = shapes[0].len();
    let axis = resolve_axis(attributes.int("dim")?.unwrap_or(0), rank)?;
    let mut output = shapes[0].to_vec();
    output[axis] = 0;
    let mut sizes = Vec::with_capacity(shapes.len());
    for shape in shapes {
        if shape.len() != rank {
            return Err(format!(
                "cannot concatenate {shape:?} with rank {rank} tensors"
            ));
        }
        for (other, (&size, &kept)) in shape.iter().zip(&output).enumerate() {
            if other != axis && size != kept {
                return Err(format!(
                    "cannot concatenate {shape:?} along axis {axis} with {output:?}"
                ));
            }
        }
        output[axis] = output[axis]
            .checked_add(shape[axis])
            .ok_or("cat shape overflows")?;
        sizes.push(shape[axis]);
    }
    Ok((Op::Cat { axis, sizes }, output))
}

fn resolve_embedding(attributes: &Attributes, shapes: &[&[u64]]) -> Result<(Op, Vec<u64>), String> {
    let (weight, indices) = (shapes[0], shapes[1]);
    if weight.len() != 2 {
        return Err(format!(
            "weight has shape {weight:?}, expected [vocab, dim]"
        ));
    }
    if attributes.bool("scale_grad_by_freq", false)? || attributes.bool("sparse", false)? {
        return Err("`scale_grad_by_freq` and `sparse` are not supported".to_owned());
    }
    let padding = attributes.int("padding_idx")?.unwrap_or(-1);
    let padding = (padding >= 0).then_some(padding as u64);
    let mut output = indices.to_vec();
    output.push(weight[1]);
    Ok((Op::Embedding { padding }, output))
}

#[cfg(test)]
mod tests;
