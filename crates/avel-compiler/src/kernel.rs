//! The kernel-level IR: one node per dispatch.

use crate::graph::Dtype;
use crate::ops::{Convolution, Movement, Pool, ReduceKind, Resize, RowKind, Scatter};
use crate::view::View;

/// The type an expression computes in. Storage is separate: f32 values may be
/// stored in f16, and i32, i64 and bool values are stored in `u32` words.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    F32,
    I32,
    Bool,
}

impl ScalarType {
    /// The compute type of values stored as `dtype`.
    pub fn of(dtype: Dtype) -> Self {
        match dtype {
            Dtype::F32 | Dtype::F16 => ScalarType::F32,
            Dtype::I32 | Dtype::I64 | Dtype::I8 => ScalarType::I32,
            Dtype::Bool => ScalarType::Bool,
        }
    }

    pub fn wgsl(self) -> &'static str {
        match self {
            ScalarType::F32 => "f32",
            ScalarType::I32 => "i32",
            ScalarType::Bool => "bool",
        }
    }
}

/// A typed constant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    F32(f32),
    I32(i32),
    Bool(bool),
}

impl Scalar {
    pub fn ty(self) -> ScalarType {
        match self {
            Scalar::F32(_) => ScalarType::F32,
            Scalar::I32(_) => ScalarType::I32,
            Scalar::Bool(_) => ScalarType::Bool,
        }
    }
}

/// An element-wise expression over the inputs of a kernel, all read at the
/// coordinates of the output element. Expressions are typed: lowering inserts
/// every conversion as a `Cast`, so each operation sees operands of one type,
/// and the expression of a kernel computes the type of its output.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Input(usize),
    /// The value a kernel computed for the element, in an epilogue.
    Acc,
    Const(Scalar),
    /// The coordinate of the output element along an axis, as an `i32`. Only
    /// in kernels with one invocation per output element, such as generators.
    Index(usize),
    Cast(ScalarType, Box<Expr>),
    Unary(UnaryFn, Box<Expr>),
    Binary(BinaryFn, Box<Expr>, Box<Expr>),
    /// `if condition { then } else { otherwise }`; both branches are evaluated.
    Select(Box<Expr>, Box<Expr>, Box<Expr>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryFn {
    Gelu,
    GeluTanh,
    Sigmoid,
    Tanh,
    Exp,
    Log,
    Sqrt,
    InverseSqrt,
    Reciprocal,
    Abs,
    Neg,
    Sign,
    Erf,
    Expm1,
    Log1p,
    Sin,
    Cos,
    Floor,
    Ceil,
    Round,
    Trunc,

    /// Logical not, on bools.
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryFn {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    MaximumF32,
    MinimumF32,
    Pow,
    PowI32,
    RemainderF32,
    RemainderI32,
    Fmod,
    /// Power on abs(base), with the original sign for odd scalar exponents.
    PowOdd,
    Min,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    /// Logical on bools, bitwise on integers.
    And,
    Or,
    /// Bitwise on integers; bools compare with `Ne` instead.
    Xor,
}

impl BinaryFn {
    fn compares(self) -> bool {
        matches!(
            self,
            BinaryFn::Eq | BinaryFn::Ne | BinaryFn::Lt | BinaryFn::Le | BinaryFn::Gt | BinaryFn::Ge
        )
    }
}

impl Expr {
    pub fn f32(value: f32) -> Self {
        Self::Const(Scalar::F32(value))
    }

    pub fn unary(function: UnaryFn, operand: Expr) -> Self {
        Self::Unary(function, Box::new(operand))
    }

    pub fn binary(function: BinaryFn, left: Expr, right: Expr) -> Self {
        Self::Binary(function, Box::new(left), Box::new(right))
    }

    pub fn cast(ty: ScalarType, operand: Expr) -> Self {
        Self::Cast(ty, Box::new(operand))
    }

    pub fn select(condition: Expr, then: Expr, otherwise: Expr) -> Self {
        Self::Select(Box::new(condition), Box::new(then), Box::new(otherwise))
    }

    /// The type the expression computes, given those of its inputs and `Acc`.
    pub fn ty(&self, input: &impl Fn(usize) -> ScalarType, acc: ScalarType) -> ScalarType {
        match self {
            Expr::Input(index) => input(*index),
            Expr::Acc => acc,
            Expr::Const(value) => value.ty(),
            Expr::Index(_) => ScalarType::I32,
            Expr::Cast(ty, _) => *ty,
            Expr::Unary(UnaryFn::Not | UnaryFn::Abs | UnaryFn::Neg | UnaryFn::Sign, operand) => {
                operand.ty(input, acc)
            }
            Expr::Unary(..) => ScalarType::F32,
            Expr::Binary(function, _, _) if function.compares() => ScalarType::Bool,
            Expr::Binary(_, left, _) => left.ty(input, acc),
            Expr::Select(_, then, _) => then.ty(input, acc),
        }
    }

    /// Replaces every `Input(i)` with `input(i)` and `Acc` with `acc`.
    pub fn substitute(&self, input: &impl Fn(usize) -> Expr, acc: &Expr) -> Self {
        let recurse = |expr: &Expr| Box::new(expr.substitute(input, acc));
        match self {
            Expr::Input(index) => input(*index),
            Expr::Acc => acc.clone(),
            Expr::Const(_) | Expr::Index(_) => self.clone(),
            Expr::Cast(ty, operand) => Expr::Cast(*ty, recurse(operand)),
            Expr::Unary(function, operand) => Expr::Unary(*function, recurse(operand)),
            Expr::Binary(function, left, right) => {
                Expr::Binary(*function, recurse(left), recurse(right))
            }
            Expr::Select(condition, then, otherwise) => {
                Expr::Select(recurse(condition), recurse(then), recurse(otherwise))
            }
        }
    }

    fn children(&self) -> impl Iterator<Item = &Expr> {
        match self {
            Expr::Input(_) | Expr::Acc | Expr::Const(_) | Expr::Index(_) => [None; 3],
            Expr::Cast(_, operand) | Expr::Unary(_, operand) => {
                [Some(operand.as_ref()), None, None]
            }
            Expr::Binary(_, left, right) => [Some(left.as_ref()), Some(right.as_ref()), None],
            Expr::Select(condition, then, otherwise) => [
                Some(condition.as_ref()),
                Some(then.as_ref()),
                Some(otherwise.as_ref()),
            ],
        }
        .into_iter()
        .flatten()
    }

    /// Whether the expression reads output coordinates, which only kernels
    /// with one invocation per output element provide.
    pub fn uses_index(&self) -> bool {
        matches!(self, Expr::Index(_)) || self.children().any(Expr::uses_index)
    }

    /// Whether recomputing the expression on every read costs little: a few
    /// arithmetic operations, no transcendental function.
    pub fn is_cheap(&self) -> bool {
        fn walk(expr: &Expr, budget: &mut u32) -> bool {
            if *budget == 0 {
                return false;
            }
            *budget -= 1;
            match expr {
                Expr::Unary(UnaryFn::Not | UnaryFn::Abs | UnaryFn::Neg | UnaryFn::Sign, _) => {}
                Expr::Unary(..)
                | Expr::Binary(BinaryFn::Pow | BinaryFn::PowOdd | BinaryFn::PowI32, ..) => {
                    return false;
                }
                _ => {}
            }
            expr.children().all(|child| walk(child, budget))
        }
        walk(self, &mut 12)
    }
}

/// How a reduction over a row is computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// One invocation loops over the whole row: many rows, short ones.
    Serial,
    /// One workgroup of `lanes` invocations per row: strided loads, then a
    /// tree in workgroup memory. With `subgroup`, the workgroup is no larger
    /// than the smallest subgroup of the device, so it is a single subgroup
    /// and subgroup operations replace the tree, without assuming how
    /// invocations map to subgroups.
    Workgroup { lanes: u32, subgroup: bool },
}

/// Invocations of a cooperative reduction without subgroups.
pub const REDUCTION_LANES: u32 = 64;

/// How a convolution is computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvPath {
    /// Implicit GEMM: a tiled matmul whose left operand is gathered from the
    /// input windows on the fly, one GEMM per group.
    Gemm,
    /// A 1×1 convolution without stride or padding: a plain tiled GEMM.
    Pointwise,
    /// Non-overlapping transpose: GEMM rows are input pixels and columns
    /// are output channels times window positions.
    TransposeDisjoint,
    /// One input channel per group: a direct loop over the window.
    Depthwise,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Kernel {
    Custom(crate::CustomKernel),
    /// [`Kernel::Matmul`] staged through workgroup memory in square tiles.
    TiledMatmul {
        beta: f32,
        alpha: f32,
        bias: bool,
        tile: u32,
    },
    ElementWise(Expr),
    /// A sum or mean over `axes`. With a split, the output gains a leading
    /// axis of partial results, one per slice of the reduced elements, each
    /// already divided by the full count for a mean; a second reduction sums
    /// them.
    Reduction {
        kind: ReduceKind,
        axes: Vec<usize>,
        strategy: Strategy,
        split: u64,
    },
    Row {
        kind: RowKind,
        axes: Vec<usize>,
        strategy: Strategy,
    },
    /// `beta * bias + alpha * lhs @ rhs` over `[..., M, K]` and `[..., K, N]`
    /// operands, the leading batch axes broadcast to the output. Inputs are
    /// the operands, then the optional bias broadcast to the output.
    Matmul {
        beta: f32,
        alpha: f32,
        bias: bool,
    },
    /// Inputs are the input, the weight and an optional bias. With a split
    /// tiling, the output gains a leading axis of partial sums, one per slice
    /// of the reduction, and there is no bias.
    Convolution {
        convolution: Convolution,
        path: ConvPath,
        tiling: Tiling,
    },
    Pool(Pool),
    Scatter(Scatter),
    OrderBlocks {
        axis: usize,
        largest: bool,
        tile: u64,
    },
    OrderMerge {
        axis: usize,
        largest: bool,
        span: u64,
    },
    OrderGather {
        axis: usize,
    },
    Rotary,
    CacheWrite {
        in_place: bool,
    },
    CachedAttention {
        scale: f32,
    },
    SampleCandidates,
    Movement {
        kind: Movement,
        fill: Option<Scalar>,
    },
    /// Layer norm over the trailing `normalized` axes of the input, one row
    /// per outer index; inputs are the input, then the optional weight and
    /// bias broadcast to the output shape. RMS norm skips the mean.
    LayerNorm {
        eps: f32,
        normalized: Vec<u64>,
        rms: bool,
        weight: bool,
        strategy: Strategy,
    },
    /// Fused scaled dot-product attention over `[..., heads, seq, depth]`.
    /// One invocation per query row; inputs are the query, key, value
    /// broadcast to the output rank, then an optional mask.
    Attention {
        scale: f32,
        causal: bool,
        mask: bool,
    },
    /// `weight[indices]`: inputs are the `[vocab, dim]` weight and the integer
    /// indices. One invocation per output element.
    Embedding {
        padding: Option<u64>,
    },
    /// Concatenation along `axis`; one invocation per output element.
    Cat {
        axis: usize,
        sizes: Vec<u64>,
    },
}

/// Element-wise work fused after a kernel that is not element-wise: `expr`
/// maps the computed value (`Acc`) and `inputs`, read at the coordinates of
/// the output element, to the value stored.
#[derive(Debug, Clone, PartialEq)]
pub struct Epilogue {
    pub expr: Expr,
    pub inputs: Vec<View>,
}

/// Element-wise work fused before a kernel that is not element-wise: the
/// kernel reads `expr` over `inputs` wherever it would have read the element
/// of `source`, the dense output of the element-wise kernel fused away.
#[derive(Debug, Clone, PartialEq)]
pub struct Prologue {
    pub kind: PrologueKind,
    pub inputs: Vec<View>,
    pub source: View,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PrologueKind {
    Rotary,
    ElementWise(Expr),
    Nearest(Resize),
}

/// Inputs are read through views; outputs always fill their own buffers.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelNode {
    pub kernel: Kernel,
    pub inputs: Vec<View>,
    pub output: View,
    /// A second output, written at the same element as `output`, such as the
    /// indices of a max pooling. A kernel with one never becomes an epilogue.
    pub second: Option<View>,
    pub epilogue: Option<Epilogue>,
    /// One entry per input, or none at all: the prologue computing it.
    pub prologues: Vec<Option<Prologue>>,
    /// Graph values the kernel computes, in order; several once fused.
    pub ops: Vec<String>,
}

impl KernelNode {
    pub fn new(kernel: Kernel, inputs: Vec<View>, output: View, op: &str) -> Self {
        Self {
            kernel,
            inputs,
            output,
            second: None,
            epilogue: None,
            prologues: Vec::new(),
            ops: vec![op.to_owned()],
        }
    }

    /// The prologue computing input `index`, if any.
    pub fn prologue(&self, index: usize) -> Option<&Prologue> {
        self.prologues.get(index).and_then(Option::as_ref)
    }

    /// Every view the kernel reads, in binding order: its inputs but those a
    /// prologue computes, then those of its epilogue, then of its prologues.
    pub fn reads(&self) -> impl Iterator<Item = &View> {
        let inputs = (0..self.inputs.len())
            .filter(|&index| self.prologue(index).is_none())
            .map(|index| &self.inputs[index]);
        let epilogue = self.epilogue.iter().flat_map(|epilogue| &epilogue.inputs);
        let prologues = self.prologues.iter().flatten().flat_map(|p| &p.inputs);
        inputs.chain(epilogue).chain(prologues)
    }

    /// Every view the kernel writes.
    pub fn writes(&self) -> impl Iterator<Item = &View> {
        std::iter::once(&self.output).chain(self.second.as_ref())
    }
}

/// How a GEMM convolution divides its work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tiling {
    /// Output rows (pixels) per invocation.
    pub rows: u64,
    /// Output columns (channels) per invocation.
    pub columns: u64,
    /// Slices the reduction is split into, summed by a separate kernel.
    pub split: u64,
}

impl Default for Tiling {
    fn default() -> Self {
        Self {
            rows: 1,
            columns: 1,
            split: 1,
        }
    }
}

/// Side of the square block of invocations of a GEMM workgroup, and the depth
/// of the reduction slices it stages in workgroup memory.
pub const TILE: u64 = 16;

/// Workgroups a GEMM should dispatch to keep the GPU busy. A heuristic,
/// tuned on ResNet-18 and MobileNetV3 at 224×224.
const MIN_WORKGROUPS: u64 = 96;

/// The shortest reduction slice worth a split: 16 steps of `TILE`.
const MIN_SPLIT_REDUCTION: u64 = 16 * TILE;

/// Outputs per invocation, as rows × columns, from the most to the least.
const BLOCKS: [(u64, u64); 9] = [
    (4, 4),
    (2, 4),
    (4, 2),
    (2, 2),
    (1, 4),
    (4, 1),
    (1, 2),
    (2, 1),
    (1, 1),
];

/// Picks the largest block of outputs per invocation that, with a split of
/// the reduction if needed, still dispatches enough workgroups.
pub fn gemm_tiling(
    rows: u64,
    groups: u64,
    group_outputs: u64,
    reduction: u64,
    gpu: crate::GpuClass,
) -> Tiling {
    let target = match gpu {
        crate::GpuClass::Integrated => MIN_WORKGROUPS,
        crate::GpuClass::Discrete => 192,
        crate::GpuClass::Mobile => 32,
    };
    let max_split = (reduction / MIN_SPLIT_REDUCTION).max(1);
    for (block_rows, block_columns) in BLOCKS {
        let workgroups = rows.div_ceil(TILE * block_rows)
            * groups
            * group_outputs.div_ceil(TILE * block_columns);
        let mut split = 1;
        while workgroups * split < target && split * 2 <= max_split {
            split *= 2;
        }
        if workgroups * split >= target {
            return Tiling {
                rows: block_rows,
                columns: block_columns,
                split,
            };
        }
    }
    Tiling::default()
}

/// Picks the convolution path from the input and weight shapes.
pub fn conv_path(
    convolution: &Convolution,
    input: &[u64],
    weight: &[u64],
    output: &[u64],
) -> ConvPath {
    if convolution.transposed {
        return if convolution.stride == weight[2..]
            && convolution.padding.iter().all(|&v| v == 0)
            && convolution.dilation.iter().all(|&v| v == 1)
            && output[2..]
                .iter()
                .zip(input[2..].iter().zip(&weight[2..]))
                .all(|(&out, (&input, &kernel))| out == input * kernel)
        {
            ConvPath::TransposeDisjoint
        } else {
            ConvPath::Gemm
        };
    }
    let channels = input[1];
    if convolution.groups > 1 && convolution.groups == channels {
        ConvPath::Depthwise
    } else if convolution.groups == 1
        && convolution.input_pad.is_none()
        && weight[2..].iter().all(|&size| size == 1)
        && convolution.stride.iter().all(|&stride| stride == 1)
        && convolution.padding.iter().all(|&padding| padding == 0)
    {
        ConvPath::Pointwise
    } else {
        ConvPath::Gemm
    }
}

/// Rows shorter than this stay serial: most lanes of a workgroup would idle.
const MIN_COOPERATIVE_ROW: u64 = 256;

/// Rows at which a serial reduction already keeps the GPU busy.
const MAX_COOPERATIVE_ROWS: u64 = 4096;

/// Reduced elements per workgroup of a split reduction.
const SPLIT_SLICE: u64 = 4096;

/// The most slices a reduction is split into; the second pass loops over them.
const MAX_SPLIT: u64 = 64;

/// Picks how `rows` reductions of `extent` elements each are computed, and
/// in how many slices: one workgroup per row, unless rows are short or many
/// enough for one invocation each; a few very long rows are also split
/// across workgroups, and summed by a second pass. `subgroup` is the smallest
/// subgroup size of the device, when it has subgroups.
pub fn reduce_strategy(rows: u64, extent: u64, subgroup: Option<u32>) -> (Strategy, u64) {
    if extent < MIN_COOPERATIVE_ROW || rows >= MAX_COOPERATIVE_ROWS {
        return (Strategy::Serial, 1);
    }
    let mut split = 1;
    while rows * split * 2 <= MIN_WORKGROUPS
        && split * 2 <= MAX_SPLIT
        && extent / (split * 2) >= SPLIT_SLICE
    {
        split *= 2;
    }
    let strategy = match subgroup {
        Some(lanes) => Strategy::Workgroup {
            lanes,
            subgroup: true,
        },
        None => Strategy::Workgroup {
            lanes: REDUCTION_LANES,
            subgroup: false,
        },
    };
    (strategy, split)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiling(rows: u64, columns: u64, split: u64) -> Tiling {
        Tiling {
            rows,
            columns,
            split,
        }
    }

    #[test]
    fn tiles_gemms_by_parallelism() {
        // The ResNet-18 convolutions at 224×224: (rows, outputs, reduction).
        let layers = [
            ((12544, 64, 147), tiling(4, 4, 1)),
            ((3136, 64, 576), tiling(4, 4, 2)),
            ((784, 128, 1152), tiling(4, 4, 4)),
            ((196, 256, 2304), tiling(4, 4, 8)),
            ((49, 512, 4608), tiling(4, 4, 16)),
        ];
        for ((rows, outputs, reduction), expected) in layers {
            assert_eq!(
                gemm_tiling(rows, 1, outputs, reduction, crate::GpuClass::Integrated),
                expected
            );
        }
        // Too short to split: smaller blocks instead.
        assert_eq!(
            gemm_tiling(49, 1, 960, 160, crate::GpuClass::Integrated),
            tiling(1, 2, 1)
        );
        assert_eq!(
            gemm_tiling(1, 1, 1, 1, crate::GpuClass::Integrated),
            tiling(1, 1, 1)
        );
    }
    #[test]
    fn reduces_long_rows_cooperatively() {
        let workgroup = Strategy::Workgroup {
            lanes: REDUCTION_LANES,
            subgroup: false,
        };
        // Short rows, or enough of them, stay serial.
        assert_eq!(reduce_strategy(100, 64, None), (Strategy::Serial, 1));
        assert_eq!(reduce_strategy(65536, 1024, None), (Strategy::Serial, 1));
        // A layer norm over 768 channels for 197 tokens: one workgroup each.
        assert_eq!(reduce_strategy(197, 768, None), (workgroup, 1));
        // An LM head over a 32k vocabulary: one row, split in slices.
        assert_eq!(reduce_strategy(1, 32000, None), (workgroup, 4));
        // Global average pooling of a large map: few rows, long ones, on a
        // device whose subgroups have at least 32 invocations.
        let (strategy, split) = reduce_strategy(3, 1 << 20, Some(32));
        assert_eq!(split, 32);
        assert_eq!(
            strategy,
            Strategy::Workgroup {
                lanes: 32,
                subgroup: true
            }
        );
    }

    #[test]
    fn gpu_classes_choose_different_parallelism() {
        let integrated = gemm_tiling(784, 1, 128, 1152, crate::GpuClass::Integrated);
        let discrete = gemm_tiling(784, 1, 128, 1152, crate::GpuClass::Discrete);
        let mobile = gemm_tiling(784, 1, 128, 1152, crate::GpuClass::Mobile);
        assert!(discrete.split > integrated.split || discrete.rows < integrated.rows);
        assert!(mobile.split < integrated.split);
    }
}
