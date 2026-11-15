//! Generates one WGSL compute shader per kernel, specialized for concrete
//! shapes, strides and offsets.
//!
//! Outputs are dense, so most shaders run one invocation per output element,
//! numbered `k` in memory order, and recover its coordinates from the output
//! strides. Convolutions that reduce to a GEMM use a tiled template instead,
//! and long reductions one workgroup per row. Every kernel loads through its
//! prologues and stores through its epilogue, if any. Values are typed: f32,
//! i32 or bool. Floats may be stored in f16, and integers and bools are
//! stored in `u32` words; loads and stores convert.
//! The templates in `codegen/wgsl` hold the fixed control flow. Rust supplies
//! bindings, specialized indices, unrolled statements and fused expressions.

use std::borrow::Cow;
use std::cell::RefCell;
use std::fmt::Write;

mod convolution;
use convolution::gemm_convolution;

mod normalization;
use normalization::{attention, layer_norm};

mod matmul;
use matmul::{matmul_body, tiled_matmul};

mod movement;
use movement::movement_body;

mod generation;
mod order;
mod rows;
mod scatter;
use rows::row_kernel;

mod reduction;
use reduction::reduction;

mod window;
use window::{depthwise_body, pool_body};

mod template;
pub(crate) use template::wgsl;

use crate::kernel::{
    BinaryFn, ConvPath, Expr, Kernel, KernelNode, PrologueKind, Scalar, ScalarType, Strategy, TILE,
    Tiling, UnaryFn,
};
use crate::ops::{Convolution, Pool, PoolKind, ReduceKind};
use crate::plan::Shader;
use crate::view::{View, row_major_strides};

const ENTRY_POINT: &str = "main";
/// A finite score low enough to mask a position out: `f32::MIN` does not
/// round-trip through its shortest decimal form, so spell a safe literal.
const LOWEST: f32 = -3.4e38;
const WORKGROUP_SIZE: u64 = 64;
/// The WebGPU minimum for `maxComputeWorkgroupsPerDimension`.
const MAX_WORKGROUPS: u64 = 65535;

/// Exact GELU needs `erf`, which WGSL lacks. Abramowitz & Stegun 7.1.26, with
/// the input bounded where `erf` has saturated to keep `a * a` finite.
const GELU: &str = include_str!("codegen/wgsl/gelu.wgsl");

/// `tanh` returns NaN on large inputs, so `x` is bounded where it has saturated.
const GELU_TANH: &str = include_str!("codegen/wgsl/gelu_tanh.wgsl");

const FLOAT_BITS: &str = include_str!("codegen/wgsl/float_bits.wgsl");
const MATH: &str = include_str!("codegen/wgsl/math.wgsl");
const TRIG: &str = include_str!("codegen/wgsl/trig.wgsl");
const POWER: &str = include_str!("codegen/wgsl/power.wgsl");
const EXTREMA: &str = include_str!("codegen/wgsl/extrema.wgsl");
const SIGMOID: &str = include_str!("codegen/wgsl/sigmoid.wgsl");

/// How a buffer is bound. Arena buffers are bound `read_write` even when only
/// read: within one compute pass, a storage buffer must not mix writable and
/// read-only usages.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Storage {
    /// Floats are stored as f16 rather than f32.
    pub f16: bool,
    /// Read-only i8 weights packed four to one storage word.
    pub packed_i8: bool,
    pub media: Option<crate::media::MediaRead>,
    /// The type of the values: i32 and bool values are stored as `u32`.
    pub ty: ScalarType,
    /// Bound `read_write`, for arena buffers a dispatch may also write.
    pub writable: bool,
}

impl Default for Storage {
    fn default() -> Self {
        Self {
            f16: false,
            packed_i8: false,
            media: None,
            ty: ScalarType::F32,
            writable: false,
        }
    }
}

impl Storage {
    fn element(&self) -> &'static str {
        match self.ty {
            ScalarType::I32 | ScalarType::Bool => "u32",
            ScalarType::F32 if self.f16 => "f16",
            ScalarType::F32 => "f32",
        }
    }

    /// Loads `name[index]` as a value of its type.
    fn load(&self, name: &str, index: &str) -> String {
        match self.ty {
            ScalarType::I32 if self.packed_i8 => format!(
                "extractBits(bitcast<i32>({name}[({index}) >> 2u]), (({index}) & 3u) << 3u, 8u)"
            ),
            ScalarType::I32 => format!("bitcast<i32>({name}[{index}])"),
            ScalarType::Bool => format!("({name}[{index}] != 0u)"),
            ScalarType::F32 if self.f16 => format!("f32({name}[{index}])"),
            ScalarType::F32 => format!("{name}[{index}]"),
        }
    }

    /// Converts `value`, of the storage's type, to its element type.
    fn encode(&self, value: &str) -> String {
        match self.ty {
            ScalarType::I32 => format!("bitcast<u32>({value})"),
            ScalarType::Bool => format!("select(0u, 1u, {value})"),
            ScalarType::F32 if self.f16 => format!("f16({value})"),
            ScalarType::F32 => value.to_owned(),
        }
    }
}

/// Where a kernel reads one of its views from.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Slot {
    /// A binding, by index among the distinct buffers read.
    Binding(usize),
    /// The output binding, for a buffer the kernel also writes.
    Output,
}

/// The bindings of a kernel: the buffers it reads (its inputs but those
/// prologues compute, the inputs of its epilogue, then of its prologues),
/// then `dst` and the optional `dst1`. Views of the same buffer share one
/// binding — WebGPU rejects aliased writable storage bindings — their offsets
/// staying in the read expressions.
struct Io<'a> {
    node: &'a KernelNode,
    /// One name per distinct buffer read.
    names: Vec<String>,
    storage: Vec<Storage>,
    /// Slot of each read position: inputs, epilogue inputs, then prologue
    /// inputs. `None` for inputs a prologue computes.
    slots: Vec<Option<Slot>>,
    output: Storage,
    second: Option<Storage>,
    /// Enables `subgroups` in the module.
    subgroups: bool,
    f16_compute: bool,
    /// Helper functions the expressions need, emitted before the entry point.
    helpers: RefCell<String>,
}

impl<'a> Io<'a> {
    fn new(node: &'a KernelNode, names: &[&str], storage: &impl Fn(usize) -> Storage) -> Self {
        let epilogue = node.epilogue.iter().flat_map(|epilogue| &epilogue.inputs);
        let prologues = node.prologues.iter().flatten().flat_map(|p| &p.inputs);
        let inputs = node.inputs.iter().enumerate().map(|(index, view)| {
            (
                node.prologue(index).is_none().then_some(view),
                names[index].to_owned(),
            )
        });
        let reads = inputs
            .chain(
                epilogue
                    .enumerate()
                    .map(|(i, view)| (Some(view), format!("ep{i}"))),
            )
            .chain(
                prologues
                    .enumerate()
                    .map(|(i, view)| (Some(view), format!("pro{i}"))),
            );
        let mut slots = Vec::new();
        let mut buffers: Vec<usize> = Vec::new();
        let mut bound_names = Vec::new();
        for (view, name) in reads {
            let slot = view.map(|view| {
                if view.buffer == node.output.buffer {
                    return Slot::Output;
                }
                match buffers.iter().position(|&buffer| buffer == view.buffer) {
                    Some(slot) => Slot::Binding(slot),
                    None => {
                        buffers.push(view.buffer);
                        bound_names.push(name);
                        Slot::Binding(buffers.len() - 1)
                    }
                }
            });
            slots.push(slot);
        }
        Self {
            node,
            names: bound_names,
            storage: buffers.iter().map(|&buffer| storage(buffer)).collect(),
            slots,
            output: storage(node.output.buffer),
            second: node.second.as_ref().map(|view| storage(view.buffer)),
            subgroups: false,
            f16_compute: false,
            helpers: RefCell::new(String::new()),
        }
    }

    fn with_f16_compute(mut self, enabled: bool) -> Self {
        self.f16_compute = enabled;
        self
    }

    fn operand_type(&self) -> &'static str {
        if self.f16_compute { "f16" } else { "f32" }
    }

    fn operand(&self, value: &str) -> String {
        if self.f16_compute {
            format!("f16({value})")
        } else {
            value.to_owned()
        }
    }

    fn accumulate_product(&self, left: &str, right: &str) -> String {
        if self.f16_compute {
            format!("f32({left} * {right})")
        } else {
            format!("{left} * {right}")
        }
    }

    fn output(&self) -> &View {
        &self.node.output
    }

    fn helpers(&self) -> String {
        self.helpers.borrow().clone()
    }

    fn declarations(&self) -> String {
        let mut wgsl = String::new();
        let stored = self
            .storage
            .iter()
            .chain([&self.output])
            .chain(&self.second);
        if self.f16_compute || stored.clone().any(|storage| storage.f16) {
            wgsl.push_str("enable f16;\n");
        }
        if self.subgroups {
            wgsl.push_str("enable subgroups;\n");
        }
        wgsl.push('\n');
        for (binding, (name, storage)) in self.names.iter().zip(&self.storage).enumerate() {
            if let Some(media) = storage.media {
                wgsl.push_str(&media.declaration(name, binding));
                continue;
            }
            let access = if storage.writable {
                "read_write"
            } else {
                "read"
            };
            writeln!(
                wgsl,
                "@group(0) @binding({binding}) var<storage, {access}> {name}: array<{}>;",
                storage.element()
            )
            .unwrap();
        }
        let outputs = [("dst", Some(&self.output)), ("dst1", self.second.as_ref())];
        for (offset, (name, storage)) in outputs.into_iter().enumerate() {
            if let Some(storage) = storage {
                writeln!(
                    wgsl,
                    "@group(0) @binding({}) var<storage, read_write> {name}: array<{}>;",
                    self.names.len() + offset,
                    storage.element()
                )
                .unwrap();
            }
        }
        wgsl
    }

    /// The read position of the first input of prologue `input`.
    fn prologue_base(&self, input: usize) -> usize {
        let node = self.node;
        let epilogue = node
            .epilogue
            .as_ref()
            .map_or(0, |epilogue| epilogue.inputs.len());
        let before: usize = (0..input)
            .filter_map(|index| node.prologue(index))
            .map(|prologue| prologue.inputs.len())
            .sum();
        node.inputs.len() + epilogue + before
    }

    /// The type of the value read at `position`.
    fn ty(&self, position: usize) -> ScalarType {
        if let Some(prologue) = self
            .node
            .inputs
            .get(position)
            .and(self.node.prologue(position))
        {
            let base = self.prologue_base(position);
            return match &prologue.kind {
                PrologueKind::ElementWise(expr) => {
                    expr.ty(&|index| self.ty(base + index), ScalarType::F32)
                }
                PrologueKind::Nearest(_) => self.ty(base),
                PrologueKind::Rotary => ScalarType::F32,
            };
        }
        match self.slots[position].expect("read positions are bound") {
            Slot::Output => self.output.ty,
            Slot::Binding(slot) => self.storage[slot].ty,
        }
    }

    /// Loads element `index` of the view read at `position`, as a value of
    /// its type. An input a prologue computes is evaluated from the prologue
    /// inputs, at the coordinates `index` has in the fused-away tensor.
    fn read(&self, position: usize, index: &str) -> String {
        if let Some(prologue) = self
            .node
            .inputs
            .get(position)
            .and(self.node.prologue(position))
        {
            let base = self.prologue_base(position);
            let index = format!("({index})");
            if matches!(prologue.kind, PrologueKind::Rotary) {
                return generation::rotary(self, &prologue.source, &prologue.inputs, base, &index);
            }
            let PrologueKind::ElementWise(expr) = &prologue.kind else {
                let PrologueKind::Nearest(resize) = &prologue.kind else {
                    unreachable!()
                };
                let mapped =
                    movement::nearest_index(&index, &prologue.source, &prologue.inputs[0], resize);
                return self.read(base, &mapped);
            };
            return render(
                expr,
                &|input| {
                    let view = &prologue.inputs[input];
                    self.read(base + input, &gather(&index, &prologue.source, view))
                },
                "",
                &|_| unreachable!("prologues have no output coordinates"),
                self,
            );
        }
        match self.slots[position].expect("read positions are bound") {
            Slot::Output => self.output.load("dst", index),
            Slot::Binding(slot) => {
                let name = &self.names[slot];
                match self.storage[slot].media {
                    Some(_) => format!("load_{name}({index})"),
                    None => self.storage[slot].load(name, index),
                }
            }
        }
    }

    /// Stores `value` at `dst[index]` through the epilogue, which reads its
    /// inputs at `epilogue_index(view)`.
    fn store(&self, index: &str, value: &str, epilogue_index: &impl Fn(&View) -> String) -> String {
        let node = self.node;
        let value = match &node.epilogue {
            None => value.to_owned(),
            Some(epilogue) => {
                let first = node.inputs.len();
                render(
                    &epilogue.expr,
                    &|input| self.read(first + input, &epilogue_index(&epilogue.inputs[input])),
                    &format!("({value})"),
                    &|_| unreachable!("epilogues have no output coordinates"),
                    self,
                )
            }
        };
        format!("dst[{index}] = {};", self.output.encode(&value))
    }

    /// Stores `value` at `dst[k]`, for kernels with one invocation per output element.
    fn store_k(&self, value: &str) -> String {
        let output = &self.node.output;
        self.store("k", value, &|view| gather("k", output, view))
    }

    /// Stores `value` in the second output, at the element output element `k`
    /// is at.
    fn store_second_k(&self, value: &str) -> String {
        let (Some(view), Some(storage)) = (&self.node.second, &self.second) else {
            return String::new();
        };
        let index = gather("k", &self.node.output, view);
        format!("dst1[{index}] = {};", storage.encode(value))
    }

    /// Adds a helper function written for this shader.
    fn add_helper(&self, helper: &str) {
        if helper.is_empty() {
            return;
        }
        if helper != FLOAT_BITS && helper.contains("avel_bits(") {
            self.add_helper(FLOAT_BITS);
        }
        let mut own = self.helpers.borrow_mut();
        if !own.contains(helper) {
            own.push_str(helper);
        }
    }
}

#[cfg(test)]
pub fn generate(node: &KernelNode, storage: &impl Fn(usize) -> Storage) -> (Shader, [u32; 3]) {
    generate_with_compute(node, storage, false)
}

pub fn generate_with_compute(
    node: &KernelNode,
    storage: &impl Fn(usize) -> Storage,
    f16_compute: bool,
) -> (Shader, [u32; 3]) {
    if let Kernel::Custom(kernel) = &node.kernel {
        return (
            Shader {
                wgsl: kernel.wgsl.clone(),
                entry_point: "main".to_owned(),
            },
            kernel.workgroups,
        );
    }
    let (wgsl, workgroups) = match &node.kernel {
        Kernel::Custom(_) => unreachable!(),
        &Kernel::TiledMatmul {
            beta,
            alpha,
            bias,
            tile,
        } => {
            let names = &["lhs", "rhs", "bias"][..node.inputs.len()];
            tiled_matmul(
                Io::new(node, names, storage).with_f16_compute(f16_compute),
                beta,
                alpha,
                bias,
                tile,
            )
        }
        Kernel::ElementWise(expr) => {
            let names = input_names("in", node.inputs.len());
            let names: Vec<&str> = names.iter().map(String::as_str).collect();
            let io = Io::new(node, &names, storage);
            let output = &node.output;
            let value = render(
                expr,
                &|input| io.read(input, &gather("k", output, &node.inputs[input])),
                "",
                &|axis| {
                    let coordinate = coordinate("k", output.strides[axis], output.shape[axis]);
                    format!("i32({coordinate})")
                },
                &io,
            );
            let body = io.store_k(&value);
            per_element(io, &body)
        }
        &Kernel::Matmul { beta, alpha, bias } => {
            let names = &["lhs", "rhs", "bias"][..node.inputs.len()];
            let io = Io::new(node, names, storage).with_f16_compute(f16_compute);
            let body = matmul_body(&io, beta, alpha, bias);
            per_element(io, &body)
        }
        Kernel::Reduction {
            kind,
            axes,
            strategy,
            split,
        } => reduction(
            Io::new(node, &["src"], storage),
            axes,
            *kind,
            *strategy,
            *split,
        ),
        Kernel::Row {
            kind,
            axes,
            strategy,
        } => row_kernel(Io::new(node, &["src"], storage), axes, *kind, *strategy),
        Kernel::Movement { kind, fill } => {
            let names = input_names("src", node.inputs.len());
            let names: Vec<&str> = names.iter().map(String::as_str).collect();
            let io = Io::new(node, &names, storage);
            let body = movement_body(&io, kind, *fill);
            per_element(io, &body)
        }
        &Kernel::OrderBlocks {
            axis,
            largest,
            tile,
        } => order::blocks(Io::new(node, &["src"], storage), axis, largest, tile),
        &Kernel::OrderMerge {
            axis,
            largest,
            span,
        } => {
            let io = Io::new(node, &["src", "indices"], storage);
            let body = order::merge(&io, axis, largest, span);
            per_element(io, &body)
        }
        &Kernel::OrderGather { axis } => {
            let io = Io::new(node, &["src", "indices"], storage);
            let body = order::gather_values(&io, axis);
            per_element(io, &body)
        }
        Kernel::Pool(pool) => {
            let io = Io::new(node, &["src"], storage);
            let body = pool_body(&io, pool);
            per_element(io, &body)
        }
        Kernel::Convolution {
            convolution,
            path: ConvPath::Depthwise,
            ..
        } => {
            let names = &["src", "weight", "bias"][..node.inputs.len()];
            let io = Io::new(node, names, storage);
            let body = depthwise_body(&io, convolution);
            per_element(io, &body)
        }
        Kernel::Convolution {
            convolution,
            path,
            tiling,
        } => {
            let names = &["src", "weight", "bias"][..node.inputs.len()];
            gemm_convolution(
                Io::new(node, names, storage).with_f16_compute(f16_compute),
                convolution,
                *path,
                *tiling,
            )
        }
        Kernel::LayerNorm {
            eps,
            normalized,
            rms,
            weight,
            strategy,
        } => {
            let names: &[&str] = if *weight {
                &["src", "weight", "bias"]
            } else {
                &["src", "bias"]
            };
            let names = &names[..node.inputs.len()];
            layer_norm(
                Io::new(node, names, storage),
                *eps,
                normalized,
                *rms,
                *weight,
                *strategy,
            )
        }
        Kernel::Scatter(spec) => {
            let names = input_names("src", node.inputs.len());
            let names: Vec<_> = names.iter().map(String::as_str).collect();
            let io = Io::new(node, &names, storage);
            let (body, rows) = scatter::scatter(&io, spec);
            linear_dispatch(io, rows, &body)
        }
        Kernel::Rotary => {
            let io = Io::new(node, &["src", "cosine", "sine"], storage);
            let value = generation::rotary(&io, io.output(), &node.inputs, 0, "k");
            let body = io.store_k(&value);
            per_element(io, &body)
        }
        &Kernel::CacheWrite { in_place } => {
            let io = Io::new(node, &["cache", "incoming", "position"], storage);
            let (body, count) = generation::cache_write(&io, in_place);
            linear_dispatch(io, count, &body)
        }
        &Kernel::CachedAttention { scale } => {
            let io = Io::new(node, &["query", "key", "value", "position"], storage);
            attention(io, scale, true, false, true)
        }
        Kernel::SampleCandidates => {
            let io = Io::new(node, &["logits", "indices", "seed"], storage);
            let body = generation::sample(&io);
            per_element(io, &body)
        }
        Kernel::Attention {
            scale,
            causal,
            mask,
        } => {
            // Not `k`: the invocation index of the row shaders is `k`.
            let names = &["query", "key", "value", "mask"][..node.inputs.len()];
            attention(Io::new(node, names, storage), *scale, *causal, *mask, false)
        }
        Kernel::Embedding { padding } => {
            let io = Io::new(node, &["weight", "indices"], storage);
            let body = embedding_body(&io, *padding);
            per_element(io, &body)
        }
        Kernel::Cat { axis, sizes } => {
            let names = input_names("in", node.inputs.len());
            let names: Vec<&str> = names.iter().map(String::as_str).collect();
            let io = Io::new(node, &names, storage);
            let body = cat_body(&io, *axis, sizes);
            per_element(io, &body)
        }
    };
    let shader = Shader {
        wgsl,
        entry_point: ENTRY_POINT.to_owned(),
    };
    (shader, workgroups)
}

fn input_names(prefix: &str, count: usize) -> Vec<String> {
    (0..count).map(|index| format!("{prefix}{index}")).collect()
}

fn per_element(io: Io, body: &str) -> (String, [u32; 3]) {
    let elements = io.output().elements();
    linear_dispatch(io, elements, body)
}

/// A shader with one invocation per element or row of the output.
fn linear_dispatch(io: Io, invocations: u64, body: &str) -> (String, [u32; 3]) {
    let wgsl = wgsl!(
        "linear_dispatch.wgsl",
        declarations = io.declarations(),
        helpers = io.helpers(),
        workgroup_size = WORKGROUP_SIZE,
        entry_point = ENTRY_POINT,
        row_stride = WORKGROUP_SIZE * MAX_WORKGROUPS,
        invocations = invocations,
        body = body,
    );
    (wgsl, workgroups(invocations))
}

fn workgroups(elements: u64) -> [u32; 3] {
    let groups = elements.div_ceil(WORKGROUP_SIZE);
    [
        groups.min(MAX_WORKGROUPS) as u32,
        groups.div_ceil(MAX_WORKGROUPS) as u32,
        1,
    ]
}

fn float(value: f32) -> String {
    if !value.is_finite() {
        format!("avel_bits({:#010x}u)", value.to_bits())
    } else if value.is_sign_negative() {
        format!("({value:?})")
    } else {
        format!("{value:?}")
    }
}

fn int(value: i32) -> String {
    match value {
        // The literal 2147483648 does not fit in an i32.
        i32::MIN => "(-2147483647i - 1i)".to_owned(),
        _ if value < 0 => format!("({value}i)"),
        _ => format!("{value}i"),
    }
}

/// The coordinate along an axis of `size` elements and `stride`, from a dense index.
fn coordinate(index: &str, stride: u64, size: u64) -> String {
    match (size, stride) {
        (1, _) => "0u".to_owned(),
        (_, 1) => format!("{index} % {size}u"),
        _ => format!("({index} / {stride}u) % {size}u"),
    }
}

/// `offset + sum(term * stride)`, leaving out the terms that are always zero.
fn linear(offset: u64, terms: &[(String, u64)]) -> String {
    let mut sum = String::new();
    for (term, stride) in terms {
        if *stride == 0 || term == "0u" {
            continue;
        }
        if !sum.is_empty() {
            sum.push_str(" + ");
        }
        sum.push_str(term);
        if *stride != 1 {
            write!(sum, " * {stride}u").unwrap();
        }
    }
    if offset != 0 || sum.is_empty() {
        if !sum.is_empty() {
            sum.push_str(" + ");
        }
        write!(sum, "{offset}u").unwrap();
    }
    sum
}

/// Index of the element of `input` at the coordinates of output element `k`.
/// `input` has the shape of `output`, broadcast axes having stride 0.
fn gather(index: &str, output: &View, input: &View) -> String {
    let aligned = input.offset == 0
        && output
            .shape
            .iter()
            .zip(&output.strides)
            .zip(&input.strides)
            .all(|((&size, a), b)| size == 1 || a == b);
    if aligned {
        return index.to_owned();
    }
    // A row kernel can supply a compound address, not just an identifier.
    let index = if index
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_')
    {
        Cow::Borrowed(index)
    } else {
        Cow::Owned(format!("({index})"))
    };
    let terms: Vec<(String, u64)> = output
        .shape
        .iter()
        .zip(&output.strides)
        .zip(&input.strides)
        .map(|((&size, &stride), &target)| (coordinate(&index, stride, size), target))
        .collect();
    linear(input.offset, &terms)
}

/// Declares `{prefix}{axis}` for each coordinate of output element `k`.
fn output_coordinates(output: &View, prefix: &str) -> String {
    let mut lines = String::new();
    for (axis, (&size, &stride)) in output.shape.iter().zip(&output.strides).enumerate() {
        writeln!(
            lines,
            "let {prefix}{axis} = {};",
            coordinate("k", stride, size)
        )
        .unwrap();
    }
    lines
}

/// Renders `expr`, loading input `i` with `input(i)`, reading `Acc` as `acc`
/// and the output coordinate along `axis` as `index(axis)`.
fn render(
    expr: &Expr,
    input: &impl Fn(usize) -> String,
    acc: &str,
    index: &impl Fn(usize) -> String,
    io: &Io,
) -> String {
    let recurse = |expr: &Expr| render(expr, input, acc, index, io);
    match expr {
        Expr::Input(position) => input(*position),
        Expr::Acc => acc.to_owned(),
        Expr::Const(Scalar::F32(value)) => {
            if !value.is_finite() {
                io.add_helper(FLOAT_BITS);
            }
            float(*value)
        }
        Expr::Const(Scalar::I32(value)) => int(*value),
        Expr::Const(Scalar::Bool(value)) => value.to_string(),
        Expr::Index(axis) => index(*axis),
        Expr::Cast(ty, operand) => format!("{}({})", ty.wgsl(), recurse(operand)),
        Expr::Select(condition, then, otherwise) => {
            let (condition, then, otherwise) =
                (recurse(condition), recurse(then), recurse(otherwise));
            format!("select({otherwise}, {then}, {condition})")
        }
        Expr::Unary(function, operand) => {
            let operand = recurse(operand);
            let (helper, name) = match function {
                UnaryFn::Gelu => (GELU, "gelu"),
                UnaryFn::GeluTanh => (GELU_TANH, "gelu_tanh"),
                UnaryFn::Sigmoid => (SIGMOID, "sigmoid"),
                UnaryFn::InverseSqrt => ("", "inverseSqrt"),
                UnaryFn::Not => return format!("(!{operand})"),
                UnaryFn::Neg => return format!("(-({operand}))"),
                UnaryFn::Reciprocal => return format!("(1.0 / ({operand}))"),
                UnaryFn::Abs => ("", "abs"),
                UnaryFn::Sign => ("", "sign"),
                UnaryFn::Log => (MATH, "avel_log"),
                UnaryFn::Sqrt => ("", "sqrt"),
                UnaryFn::Sin => (TRIG, "avel_sin"),
                UnaryFn::Cos => (TRIG, "avel_cos"),
                UnaryFn::Floor => ("", "floor"),
                UnaryFn::Ceil => ("", "ceil"),
                UnaryFn::Round => (MATH, "avel_round_even"),
                UnaryFn::Trunc => ("", "trunc"),
                UnaryFn::Erf => (GELU, "erf"),
                UnaryFn::Exp => (MATH, "avel_exp"),
                UnaryFn::Tanh => (MATH, "avel_tanh"),
                UnaryFn::Expm1 => (MATH, "avel_expm1"),
                UnaryFn::Log1p => (MATH, "avel_log1p"),
            };
            io.add_helper(helper);
            format!("{name}({operand})")
        }
        Expr::Binary(function, left, right) => {
            let left = recurse(left);
            let right = recurse(right);
            let helper = match function {
                BinaryFn::MaximumF32 => Some((EXTREMA, "avel_max")),
                BinaryFn::MinimumF32 => Some((EXTREMA, "avel_min")),
                BinaryFn::Pow => Some((POWER, "avel_pow")),
                BinaryFn::PowOdd => Some((POWER, "avel_pow_odd")),
                BinaryFn::PowI32 => Some((POWER, "avel_pow_int")),
                BinaryFn::RemainderF32 => Some((MATH, "avel_remainder_f32")),
                BinaryFn::RemainderI32 => Some((MATH, "avel_remainder_i32")),
                _ => None,
            };
            if let Some((helper, name)) = helper {
                io.add_helper(helper);
                return format!("{name}({left}, {right})");
            }
            let operator = match function {
                BinaryFn::MaximumF32
                | BinaryFn::MinimumF32
                | BinaryFn::Pow
                | BinaryFn::PowOdd
                | BinaryFn::PowI32
                | BinaryFn::RemainderF32
                | BinaryFn::RemainderI32 => unreachable!(),
                BinaryFn::Max => return format!("max({left}, {right})"),
                BinaryFn::Min => return format!("min({left}, {right})"),
                BinaryFn::Add => "+",
                BinaryFn::Sub => "-",
                BinaryFn::Mul => "*",
                BinaryFn::Div => "/",
                BinaryFn::Fmod => "%",
                BinaryFn::Eq => "==",
                BinaryFn::Ne => "!=",
                BinaryFn::Lt => "<",
                BinaryFn::Le => "<=",
                BinaryFn::Gt => ">",
                BinaryFn::Ge => ">=",
                BinaryFn::And => "&",
                BinaryFn::Or => "|",
                BinaryFn::Xor => "^",
            };
            format!("({left} {operator} {right})")
        }
    }
}

/// `weight[indices]`: one invocation per output element `k`; the row is
/// `k / depth` and the column `k % depth`, and the index is read through the
/// indices view expanded with a trailing unit axis.
fn embedding_body(io: &Io, _padding: Option<u64>) -> String {
    let node = io.node;
    let (weight, indices) = (&node.inputs[0], &node.inputs[1]);
    let output = &node.output;
    let depth = weight.shape[1];
    let expanded = indices.unsqueeze(indices.shape.len());
    let index = format!("u32({})", io.read(1, &gather("k", output, &expanded)));
    let column = coordinate("k", output.strides[output.shape.len() - 1], depth);
    let terms = vec![
        (index.clone(), weight.strides[0]),
        (column, weight.strides[1]),
    ];
    // padding_idx affects gradients only; inference reads the stored row.
    let value = io.read(0, &linear(weight.offset, &terms));
    io.store_k(&value)
}

/// Concatenation along `axis`: one invocation per output element, which reads
/// the one input whose slice along `axis` contains its coordinate.
fn cat_body(io: &Io, axis: usize, sizes: &[u64]) -> String {
    let node = io.node;
    let output = &node.output;
    let mut select_input = String::new();
    let position = format!("o{axis}");
    let mut cumulative = 0u64;
    let last = sizes.len() - 1;
    for (index, (&size, input)) in sizes.iter().zip(&node.inputs).enumerate() {
        let local = match cumulative {
            0 => position.clone(),
            _ => format!("({position} - {cumulative}u)"),
        };
        let terms: Vec<(String, u64)> = (0..output.shape.len())
            .map(|other| {
                let coordinate = if other == axis {
                    local.clone()
                } else {
                    format!("o{other}")
                };
                (coordinate, input.strides[other])
            })
            .collect();
        let read = io.read(index, &linear(input.offset, &terms));
        if sizes.len() == 1 {
            writeln!(select_input, "value = {read};").unwrap();
        } else if index == last {
            writeln!(select_input, "else {{ value = {read}; }}").unwrap();
        } else {
            let keyword = if index == 0 { "if" } else { "else if" };
            writeln!(
                select_input,
                "{keyword} ({position} < {}u) {{ value = {read}; }}",
                cumulative + size
            )
            .unwrap();
        }
        cumulative += size;
    }
    wgsl!(
        "cat.wgsl",
        coordinates = output_coordinates(output, "o"),
        ty = io.ty(0).wgsl(),
        select_input = select_input,
        store = io.store_k("value"),
    )
}

#[cfg(test)]
pub(crate) mod tests;
