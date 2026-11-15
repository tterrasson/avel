use super::*;
use crate::kernel::{BinaryFn, Expr, Scalar, ScalarType, UnaryFn};

pub(super) enum Expression {
    Copy,
    Fill(u32),
    Evaluate(Eval),
}

pub(super) enum Eval {
    Input(usize),
    Constant(u32),
    Index { stride: usize, size: usize },
    Cast(ScalarType, ScalarType, Box<Eval>),
    Unary(UnaryFn, ScalarType, Box<Eval>),
    Binary(BinaryFn, ScalarType, Box<Eval>, Box<Eval>),
    Select(Box<Eval>, Box<Eval>, Box<Eval>),
}

impl Expression {
    pub(super) fn new(expr: Expr, inputs: &[Tensor], shape: &[u64]) -> Self {
        match expr {
            Expr::Input(0) if !matches!(inputs[0].dtype, Dtype::Bool | Dtype::I8) => Self::Copy,
            Expr::Const(value) => Self::Fill(word(value)),
            expr => Self::Evaluate(Eval::new(expr, &|i| ScalarType::of(inputs[i].dtype), shape)),
        }
    }

    pub(super) fn run(&self, memory: &mut Memory, inputs: &[Tensor], output: &Tensor) {
        let count = output.index.count;
        match self {
            Self::Copy if inputs[0].index.contiguous && output.index.contiguous => {
                let source = &inputs[0];
                let Buffer::Arena(target) = &memory.buffers[output.buffer] else {
                    unreachable!()
                };
                let start = target.start + output.offset;
                match &memory.buffers[source.buffer] {
                    Buffer::Arena(range) => {
                        let from = range.start + source.offset;
                        memory.words.copy_within(from..from + count, start);
                    }
                    Buffer::Parameter(words) => memory.words[start..start + count]
                        .copy_from_slice(&words[source.offset..source.offset + count]),
                }
            }
            Self::Copy => {
                for index in 0..count {
                    output.put(memory, index, inputs[0].get(memory, index));
                }
            }
            Self::Fill(value) => {
                memory.write(output.buffer)[output.offset..output.offset + count].fill(*value);
            }
            Self::Evaluate(expr) => {
                for index in 0..count {
                    output.put(memory, index, expr.eval(memory, inputs, index));
                }
            }
        }
    }
}

impl Eval {
    fn new(expr: Expr, input: &impl Fn(usize) -> ScalarType, shape: &[u64]) -> Self {
        let build = |expr: Box<Expr>| Box::new(Self::new(*expr, input, shape));
        match expr {
            Expr::Input(index) => Self::Input(index),
            Expr::Acc => unreachable!("CPU element-wise fusion has no epilogues"),
            Expr::Const(value) => Self::Constant(word(value)),
            Expr::Index(axis) => Self::Index {
                stride: shape[axis + 1..].iter().product::<u64>() as usize,
                size: shape[axis] as usize,
            },
            Expr::Cast(to, operand) => {
                Self::Cast(operand.ty(input, ScalarType::F32), to, build(operand))
            }
            Expr::Unary(op, operand) => {
                Self::Unary(op, operand.ty(input, ScalarType::F32), build(operand))
            }
            Expr::Binary(op, lhs, rhs) => {
                Self::Binary(op, lhs.ty(input, ScalarType::F32), build(lhs), build(rhs))
            }
            Expr::Select(cond, yes, no) => Self::Select(build(cond), build(yes), build(no)),
        }
    }

    fn eval(&self, memory: &Memory, inputs: &[Tensor], index: usize) -> u32 {
        use ScalarType::{Bool, F32, I32};
        let eval = |expr: &Self| expr.eval(memory, inputs, index);
        match self {
            Self::Input(i) => {
                let input = &inputs[*i];
                let value = input.get(memory, index);
                if input.dtype == Dtype::Bool {
                    u32::from(value != 0)
                } else {
                    value
                }
            }
            Self::Constant(value) => *value,
            Self::Index { stride, size } => (index / stride % size) as u32,
            Self::Cast(from, to, expr) => {
                let value = eval(expr);
                match (from, to) {
                    (F32, I32) => f32::from_bits(value) as i32 as u32,
                    (F32, Bool) => u32::from(f32::from_bits(value) != 0.0),
                    (I32 | Bool, F32) => (value as i32 as f32).to_bits(),
                    (I32, Bool) => u32::from(value != 0),
                    _ => value,
                }
            }
            Self::Unary(op, F32, expr) => unary(*op, f32::from_bits(eval(expr))).to_bits(),
            Self::Unary(op, _, expr) => {
                let x = eval(expr) as i32;
                match op {
                    UnaryFn::Abs => x.wrapping_abs() as u32,
                    UnaryFn::Neg => x.wrapping_neg() as u32,
                    UnaryFn::Sign => x.signum() as u32,
                    UnaryFn::Not => u32::from(x == 0),
                    _ => unreachable!("lowering types float functions"),
                }
            }
            Self::Binary(op, F32, lhs, rhs) => {
                float_binary(*op, f32::from_bits(eval(lhs)), f32::from_bits(eval(rhs)))
            }
            Self::Binary(op, _, lhs, rhs) => int_binary(*op, eval(lhs) as i32, eval(rhs) as i32),
            Self::Select(cond, yes, no) => eval(if eval(cond) != 0 { yes } else { no }),
        }
    }
}

pub(super) fn word(value: Scalar) -> u32 {
    match value {
        Scalar::F32(x) => x.to_bits(),
        Scalar::I32(x) => x as u32,
        Scalar::Bool(x) => u32::from(x),
    }
}

fn erf(x: f32) -> f32 {
    if x.is_nan() || x == 0.0 {
        return x;
    }
    // The same approximation as the WGSL kernel.
    let a = x.abs().min(6.0);
    let t = 1.0 / (1.0 + 0.327_591_1 * a);
    let p = t
        * (0.254_829_6
            + t * (-0.284_496_72 + t * (1.421_413_8 + t * (-1.453_152 + t * 1.061_405_4))));
    (1.0 - p * (-a * a).exp()).copysign(x)
}

fn unary(op: UnaryFn, x: f32) -> f32 {
    match op {
        UnaryFn::Gelu => 0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2)),
        UnaryFn::GeluTanh => 0.5 * x * (1.0 + (0.797_884_6 * x * (1.0 + 0.044_715 * x * x)).tanh()),
        UnaryFn::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        UnaryFn::Tanh => x.tanh(),
        UnaryFn::Exp => x.exp(),
        UnaryFn::Log => x.ln(),
        UnaryFn::Sqrt => x.sqrt(),
        UnaryFn::InverseSqrt => x.sqrt().recip(),
        UnaryFn::Reciprocal => x.recip(),
        UnaryFn::Abs => x.abs(),
        UnaryFn::Neg => -x,
        UnaryFn::Sign => f32::from(u8::from(x > 0.0)) - f32::from(u8::from(x < 0.0)),
        UnaryFn::Erf => erf(x),
        UnaryFn::Expm1 => x.exp_m1(),
        UnaryFn::Log1p => x.ln_1p(),
        UnaryFn::Sin => x.sin(),
        UnaryFn::Cos => x.cos(),
        UnaryFn::Floor => x.floor(),
        UnaryFn::Ceil => x.ceil(),
        UnaryFn::Round => x.round_ties_even(),
        UnaryFn::Trunc => x.trunc(),
        UnaryFn::Not => unreachable!("lowering types logical not"),
    }
}

fn float_binary(op: BinaryFn, x: f32, y: f32) -> u32 {
    let value = match op {
        BinaryFn::Add => x + y,
        BinaryFn::Sub => x - y,
        BinaryFn::Mul => x * y,
        BinaryFn::Div => x / y,
        BinaryFn::Max => x.max(y),
        BinaryFn::Min => x.min(y),
        BinaryFn::MaximumF32 => {
            if y > x || y.is_nan() {
                y
            } else {
                x
            }
        }
        BinaryFn::MinimumF32 => {
            if y < x || y.is_nan() {
                y
            } else {
                x
            }
        }
        BinaryFn::Pow => x.powf(y),
        BinaryFn::PowOdd => x.abs().powf(y).copysign(x),
        BinaryFn::Fmod => x % y,
        BinaryFn::RemainderF32 => {
            let r = x % y;
            if r != 0.0 && ((r < 0.0) != (y < 0.0)) {
                r + y
            } else {
                r
            }
        }
        BinaryFn::Eq => return u32::from(x == y),
        BinaryFn::Ne => return u32::from(x != y),
        BinaryFn::Lt => return u32::from(x < y),
        BinaryFn::Le => return u32::from(x <= y),
        BinaryFn::Gt => return u32::from(x > y),
        BinaryFn::Ge => return u32::from(x >= y),
        _ => unreachable!("lowering types integer and boolean functions"),
    };
    value.to_bits()
}

fn int_binary(op: BinaryFn, x: i32, y: i32) -> u32 {
    let value = match op {
        BinaryFn::Add => x.wrapping_add(y),
        BinaryFn::Sub => x.wrapping_sub(y),
        BinaryFn::Mul => x.wrapping_mul(y),
        BinaryFn::Div => x.checked_div(y).unwrap_or(x),
        BinaryFn::Max => x.max(y),
        BinaryFn::Min => x.min(y),
        BinaryFn::PowI32 if y >= 0 => x.wrapping_pow(y as u32),
        BinaryFn::PowI32 if x == 1 || x == -1 => x.wrapping_pow((y & 1) as u32),
        BinaryFn::PowI32 => 0,
        BinaryFn::Fmod => {
            if y == 0 {
                0
            } else {
                x.wrapping_rem(y)
            }
        }
        BinaryFn::RemainderI32 => {
            let r = if y == 0 { 0 } else { x.wrapping_rem(y) };
            if r != 0 && ((r < 0) != (y < 0)) {
                r.wrapping_add(y)
            } else {
                r
            }
        }
        BinaryFn::And => x & y,
        BinaryFn::Or => x | y,
        BinaryFn::Xor => x ^ y,
        BinaryFn::Eq => return u32::from(x == y),
        BinaryFn::Ne => return u32::from(x != y),
        BinaryFn::Lt => return u32::from(x < y),
        BinaryFn::Le => return u32::from(x <= y),
        BinaryFn::Gt => return u32::from(x > y),
        BinaryFn::Ge => return u32::from(x >= y),
        _ => unreachable!("lowering types float functions"),
    };
    value as u32
}
