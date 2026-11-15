use std::collections::{BTreeMap, HashMap, HashSet};

use serde::{Deserialize, Serialize};
#[cfg(feature = "ts")]
use tsify::Tsify;

use crate::error::CompileError;
use crate::graph::{Dim, Dtype, FORMAT_VERSION, Graph, Node, ValueInfo};
use crate::kernel::{
    BinaryFn, ConvPath, Expr, Kernel, KernelNode, Scalar, ScalarType, Strategy, Tiling, UnaryFn,
    conv_path, gemm_tiling,
};
use crate::layout::{self, Producer};
use crate::ops::{
    self, BinaryKind, BitwiseKind, CompareKind, Number, Op, Output, ReduceKind, Unary, ViewOp,
};
use crate::plan::{BufferDesc, BufferRole, PLAN_VERSION, Plan, TensorDesc};
use crate::view::{Layout, View};

#[cfg(feature = "cpu")]
mod cpu;
mod lowering;
#[cfg(feature = "cpu")]
pub(crate) use cpu::compile_cpu;
use lowering::{
    fold_padding, lower, pack_weights, patch_state_in_place, plan_reductions, tile_convolutions,
    write_state,
};

mod options;
pub use options::{CompileOptions, Features, Limits};

/// A node after the op-level pass: parsed, shape- and dtype-checked.
struct Resolved<'a> {
    node: &'a Node,
    /// The tensor arguments present, in schema order.
    inputs: Vec<&'a str>,
    op: Op,
    /// Every output the op computes, with the graph name of those read.
    outputs: Vec<(Option<&'a str>, Output)>,
}

impl<'a> Resolved<'a> {
    /// The name of the first output the graph reads, which labels the node.
    fn name(&self) -> &'a str {
        self.outputs
            .iter()
            .find_map(|(name, _)| *name)
            .expect("the op-level pass checks a node has an output")
    }

    fn shape(&self) -> &[u64] {
        &self.outputs[0].1.shape
    }

    fn dtype(&self) -> Dtype {
        self.outputs[0].1.dtype
    }
}

#[derive(Debug, Clone)]
struct Value {
    dtype: Dtype,
    view: View,
}

/// Buffers, named values and the kernels lowered so far. Each computed tensor
/// gets a buffer of its own, with the `Arena` role, until memory planning
/// gathers them in the arena.
struct Builder<'a> {
    limits: &'a Limits,
    buffers: Vec<BufferDesc>,
    values: HashMap<&'a str, Value>,
    prepare: Vec<KernelNode>,
    kernels: Vec<KernelNode>,
    /// The kernel that writes each buffer, by index into `kernels`.
    producers: HashMap<usize, usize>,
    /// How many nodes and graph outputs read each value.
    uses: HashMap<&'a str, usize>,
}

impl<'a> Builder<'a> {
    fn new(limits: &'a Limits, graph: &'a Graph) -> Self {
        let mut uses = HashMap::new();
        let names = graph.nodes.iter().flat_map(Node::tensors);
        let updates = graph
            .parameters
            .iter()
            .filter_map(|parameter| parameter.update.as_deref());
        for name in names
            .chain(graph.outputs.iter().map(|output| output.name.as_str()))
            .chain(updates)
        {
            *uses.entry(name).or_insert(0) += 1;
        }
        Self {
            limits,
            buffers: Vec::new(),
            values: HashMap::new(),
            prepare: Vec::new(),
            kernels: Vec::new(),
            producers: HashMap::new(),
            uses,
        }
    }

    /// Creates a buffer holding a dense tensor; `name` only labels errors.
    fn allocate(
        &mut self,
        name: &str,
        dtype: Dtype,
        shape: Vec<u64>,
        layout: Layout,
        role: BufferRole,
    ) -> Result<View, CompileError> {
        let size = shape
            .iter()
            .fold(dtype.size(), |size, &dim| size.saturating_mul(dim));
        let size = if dtype == Dtype::I8 {
            size.saturating_add(3) & !3
        } else {
            size
        };
        if size == 0 {
            return Err(CompileError::EmptyTensor(name.to_owned()));
        }
        let limit = self
            .limits
            .max_storage_buffer_binding_size
            .min(self.limits.max_buffer_size)
            .min(u64::from(u32::MAX));
        if size > limit {
            return Err(CompileError::BufferTooLarge {
                name: name.to_owned(),
                size,
                limit,
            });
        }
        self.buffers.push(BufferDesc { size, role, dtype });
        Ok(View::dense(self.buffers.len() - 1, shape, layout))
    }

    fn define(&mut self, name: &'a str, dtype: Dtype, view: View) {
        self.values.insert(name, Value { dtype, view });
    }

    fn add_parameters(&mut self, graph: &'a Graph) -> Result<(), CompileError> {
        for parameter in &graph.parameters {
            if !matches!(
                parameter.dtype,
                Dtype::F32 | Dtype::I32 | Dtype::I64 | Dtype::Bool | Dtype::I8
            ) {
                return Err(CompileError::UnsupportedDtype {
                    name: parameter.name.clone(),
                    dtype: parameter.dtype,
                });
            }
            let tensor = parameter.tensor.clone();
            let role = match parameter.update {
                Some(_) => BufferRole::State { tensor },
                None => BufferRole::Parameter { tensor },
            };
            let view = self.allocate(
                &parameter.name,
                parameter.dtype,
                parameter.shape.clone(),
                Layout::RowMajor,
                role,
            )?;
            self.define(&parameter.name, parameter.dtype, view);
        }
        Ok(())
    }

    fn lookup(&self, name: &str) -> Result<&Value, CompileError> {
        self.values
            .get(name)
            .ok_or_else(|| CompileError::UnknownValue(name.to_owned()))
    }

    /// Whether `view` holds the same data on every run: a parameter, or a
    /// constant computed from parameters and generators alone.
    fn constant(&self, view: &View) -> bool {
        matches!(
            self.buffers[view.buffer].role,
            BufferRole::Parameter { .. } | BufferRole::Constant
        )
    }

    /// Whether `view` was dequantized from i8 by a prepare kernel.
    fn dequantized(&self, view: &View) -> bool {
        self.prepare.iter().any(|producer| {
            producer.output.buffer == view.buffer
                && producer
                    .inputs
                    .first()
                    .is_some_and(|input| self.dtype(input) == Dtype::I8)
        })
    }

    /// Adds a kernel computing the value `op`. One that only reads parameters
    /// and constants, or nothing at all as generators do, is folded: it runs
    /// once when the plan is created, and its output is a constant.
    fn emit(&mut self, kernel: Kernel, inputs: Vec<View>, output: View, op: &str) {
        self.emit_node(KernelNode::new(kernel, inputs, output, op));
    }

    fn emit_node(&mut self, node: KernelNode) {
        let constant = node.inputs.iter().all(|view| self.constant(view));
        if constant {
            for view in node.writes() {
                self.buffers[view.buffer].role = BufferRole::Constant;
            }
            self.prepare.push(node);
        } else {
            for view in node.writes() {
                self.producers.insert(view.buffer, self.kernels.len());
            }
            self.kernels.push(node);
        }
    }

    /// The dtype of the buffer behind `view`.
    fn dtype(&self, view: &View) -> Dtype {
        self.buffers[view.buffer].dtype
    }

    /// Copies `view` into a new row-major buffer.
    fn materialize(&mut self, name: &str, view: View) -> Result<View, CompileError> {
        let output = self.allocate(
            name,
            self.dtype(&view),
            view.shape.clone(),
            Layout::RowMajor,
            BufferRole::Arena,
        )?;
        self.emit(
            Kernel::ElementWise(Expr::Input(0)),
            vec![view],
            output.clone(),
            name,
        );
        Ok(output)
    }
}

mod schedule;
mod shapes;
use shapes::{effective_shapes, requested_shapes, symbols};

/// Compiles `graph` for the concrete shapes in `options`.
///
/// Passes, in order: shape and dtype inference, layout choice, lowering to
/// kernels (views for layout ops, folding of weight-only and generator
/// kernels and of batch norms into convolutions), packing of convolution
/// weights, convolution tiling, reduction strategies, state write-back,
/// fusion (epilogues, then prologues), storage types, memory planning, then
/// WGSL generation.
pub fn compile(graph: &Graph, options: &CompileOptions) -> Result<Plan, CompileError> {
    validate(graph)?;
    if (options.f16 || options.f16_compute) && !options.features.shader_f16 {
        return Err(CompileError::F16Unavailable(
            "the device lacks `shader-f16`",
        ));
    }
    if (options.f16 || options.f16_compute) && !graph.f16 {
        return Err(CompileError::F16Unavailable(
            "the exporter did not validate the model in f16",
        ));
    }

    let GraphSetup {
        mut builder,
        inputs,
        symbols,
        requested,
    } = setup(graph, options)?;

    let mut media = graph.media.clone();
    media.extend(options.media.clone());
    let mut media_reads = HashMap::new();
    for (name, config) in &media {
        let input = inputs
            .iter()
            .find(|input| &input.name == name)
            .ok_or_else(|| CompileError::UnknownInput(name.clone()))?;
        media_reads.insert(input.buffer as usize, config.resolve(input)?);
    }
    let resolved = resolve_nodes(graph, &builder, &symbols, &options.custom_kernels)?;
    if options.bucket.is_some() {
        shapes::validate_bucketing(graph, &resolved)?;
    }
    let producers: Vec<Producer> = resolved
        .iter()
        .map(|node| Producer {
            op: &node.op,
            inputs: &node.inputs,
            output: node.name(),
            rank: node.shape().len(),
        })
        .collect();
    let ranks: HashMap<&str, usize> = builder
        .values
        .iter()
        .map(|(&name, value)| (name, value.view.shape.len()))
        .chain(resolved.iter().flat_map(|node| {
            node.outputs
                .iter()
                .filter_map(|(name, output)| Some(((*name)?, output.shape.len())))
        }))
        .collect();
    let layouts = layout::choose(&producers, |name| ranks[name]);

    for (node, &layout) in resolved.iter().zip(&layouts) {
        lower(node, layout, &mut builder)?;
    }
    for config in options.tiles.values() {
        config.validate()?;
    }
    for node in builder.kernels.iter_mut().chain(&mut builder.prepare) {
        if let Kernel::Matmul { beta, alpha, bias } = node.kernel {
            let tile = options.tiles.get(&node.ops[0]).map(|c| c.matmul).unwrap_or(
                if options.gpu_class == crate::GpuClass::Mobile {
                    8
                } else {
                    16
                },
            );
            if tile != 0 {
                node.kernel = Kernel::TiledMatmul {
                    beta,
                    alpha,
                    bias,
                    tile,
                };
            }
        }
    }
    fold_padding(graph, &mut builder)?;
    pack_weights(&mut builder)?;
    tile_convolutions(&mut builder, options)?;
    plan_reductions(&mut builder, options)?;
    // Even an identity/view output must execute its texture prologue.
    for output in &graph.outputs {
        let value = builder.lookup(&output.name)?.clone();
        if media_reads.contains_key(&value.view.buffer) {
            let view = builder.materialize(&output.name, value.view)?;
            builder.define(&output.name, value.dtype, view);
        }
    }
    let mut outputs = resolve_outputs(graph, &mut builder)?;
    // Last: every kernel reading the previous state runs before it is replaced.
    write_state(graph, &mut builder)?;
    patch_state_in_place(&mut builder);

    let mut inputs = inputs;
    let schedule = schedule::schedule(
        builder,
        options.f16,
        options.f16_compute,
        &media_reads,
        &mut inputs,
        &mut outputs,
    )?;
    Ok(Plan {
        version: PLAN_VERSION,
        media,
        inputs,
        outputs,
        shaders: schedule.shaders,
        buffers: schedule.buffers,
        prepare: schedule.prepare,
        dispatches: schedule.dispatches,
        requested,
    })
}

struct GraphSetup<'a> {
    builder: Builder<'a>,
    inputs: Vec<TensorDesc>,
    symbols: HashMap<String, u64>,
    requested: Option<crate::RequestedShapes>,
}

fn setup<'a>(
    graph: &'a Graph,
    options: &'a CompileOptions,
) -> Result<GraphSetup<'a>, CompileError> {
    // Validate the requested shapes before rounding: a bucket must not hide
    // an invalid rank, fixed dimension, bound or shared symbolic size.
    for input in &graph.inputs {
        let shape = options
            .input_shapes
            .get(&input.name)
            .ok_or_else(|| CompileError::MissingInputShape(input.name.clone()))?;
        check_shape(input, shape)?;
    }
    let requested_symbols = symbols(graph, &options.input_shapes)?;
    let requested = requested_shapes(graph, options, &requested_symbols)?;
    let effective = effective_shapes(graph, options);
    let symbols = if options.bucket.is_some() {
        symbols(graph, &effective)?
    } else {
        requested_symbols
    };
    let mut builder = Builder::new(&options.limits, graph);
    let inputs = add_inputs(graph, &effective, &mut builder)?;
    builder.add_parameters(graph)?;
    Ok(GraphSetup {
        builder,
        inputs,
        symbols,
        requested,
    })
}

/// The checks that do not depend on the input shapes: the format version, the
/// opset, and unique value names. [`compile`] runs them too; the runtime runs
/// them once when it loads a graph, to reject it before downloading weights.
pub fn validate(graph: &Graph) -> Result<(), CompileError> {
    if graph.format_version != FORMAT_VERSION {
        return Err(CompileError::UnsupportedFormat {
            found: graph.format_version,
            expected: FORMAT_VERSION,
        });
    }
    if graph.outputs.is_empty() {
        return Err(CompileError::InvalidOp {
            op: "graph".into(),
            message: "a graph needs an output".into(),
        });
    }
    if graph.opset != "core-aten/1" {
        return Err(CompileError::InvalidOp {
            op: "graph".into(),
            message: format!("unsupported opset `{}`", graph.opset),
        });
    }
    // Mutable aliases would share one GPU buffer while the compiler sees
    // independent values. A state tensor has one graph definition.
    let mut tensors = HashMap::new();
    let packed: HashSet<_> = graph
        .parameters
        .iter()
        .filter(|parameter| parameter.dtype == Dtype::I8)
        .map(|parameter| parameter.name.as_str())
        .collect();
    if !packed.is_empty() {
        let mut dequantized = HashSet::new();
        for node in &graph.nodes {
            for (slot, input) in node.inputs.iter().enumerate() {
                if let Some(input) = input.as_deref().filter(|name| packed.contains(name)) {
                    if slot != 0 || node.op != "avel.dequantize_per_channel.default" {
                        return Err(CompileError::UnsupportedDtype {
                            name: input.to_owned(),
                            dtype: Dtype::I8,
                        });
                    }
                    dequantized.insert(input);
                }
            }
        }
        for parameter in &graph.parameters {
            if packed.contains(parameter.name.as_str())
                && (parameter.update.is_some()
                    || !dequantized.contains(parameter.name.as_str())
                    || graph
                        .outputs
                        .iter()
                        .any(|output| output.name == parameter.name))
            {
                return Err(CompileError::UnsupportedDtype {
                    name: parameter.name.clone(),
                    dtype: Dtype::I8,
                });
            }
        }
    }
    for parameter in &graph.parameters {
        if tensors
            .insert(&parameter.tensor, parameter.update.is_some())
            .is_some_and(|state| state || parameter.update.is_some())
        {
            return Err(CompileError::InvalidOp {
                op: "graph".to_owned(),
                message: format!("duplicate state tensor `{}`", parameter.tensor),
            });
        }
    }
    let mut names = HashSet::new();
    let definitions = graph
        .inputs
        .iter()
        .map(|v| &v.name)
        .chain(graph.parameters.iter().map(|p| &p.name))
        .chain(
            graph
                .nodes
                .iter()
                .flat_map(|n| n.outputs.iter().flatten().map(|v| &v.name)),
        );
    for name in definitions {
        if name.is_empty() || !names.insert(name) {
            return Err(CompileError::InvalidOp {
                op: "graph".into(),
                message: format!("duplicate or empty value name `{name}`"),
            });
        }
    }
    Ok(())
}

fn add_inputs<'a>(
    graph: &'a Graph,
    input_shapes: &BTreeMap<String, Vec<u64>>,
    builder: &mut Builder<'a>,
) -> Result<Vec<TensorDesc>, CompileError> {
    if let Some(name) = input_shapes
        .keys()
        .find(|name| !graph.inputs.iter().any(|input| input.name == **name))
    {
        return Err(CompileError::UnknownInput(name.clone()));
    }

    graph
        .inputs
        .iter()
        .map(|input| {
            let shape = input_shapes
                .get(&input.name)
                .ok_or_else(|| CompileError::MissingInputShape(input.name.clone()))?;
            check_shape(input, shape)?;
            let view = builder.allocate(
                &input.name,
                input.dtype,
                shape.clone(),
                Layout::RowMajor,
                BufferRole::Input,
            )?;
            // The op-level pass checks which dtypes each node reads.
            if !matches!(
                input.dtype,
                Dtype::F32 | Dtype::I32 | Dtype::I64 | Dtype::Bool
            ) {
                return Err(CompileError::UnsupportedDtype {
                    name: input.name.clone(),
                    dtype: input.dtype,
                });
            }
            let buffer = view.buffer as u32;
            builder.define(&input.name, input.dtype, view);
            Ok(TensorDesc {
                name: input.name.clone(),
                dtype: input.dtype,
                shape: shape.clone(),
                buffer,
            })
        })
        .collect()
}

/// The op-level pass: parses each node, infers the shape and dtype of its
/// outputs and checks them against the exported metadata.
fn resolve_nodes<'a>(
    graph: &'a Graph,
    builder: &Builder,
    symbols: &HashMap<String, u64>,
    custom: &BTreeMap<String, crate::CustomKernel>,
) -> Result<Vec<Resolved<'a>>, CompileError> {
    let mut known: HashMap<&str, (Dtype, Vec<u64>)> = builder
        .values
        .iter()
        .map(|(&name, value)| (name, (value.dtype, value.view.shape.clone())))
        .collect();
    let mut resolved = Vec::with_capacity(graph.nodes.len());
    for node in &graph.nodes {
        let inputs = node
            .tensors()
            .map(|name| {
                known
                    .get(name)
                    .map(|(dtype, shape)| (*dtype, shape.as_slice()))
                    .ok_or_else(|| CompileError::UnknownValue(name.to_owned()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let shapes: Vec<&[u64]> = inputs.iter().map(|&(_, shape)| shape).collect();
        let dtypes: Vec<Dtype> = inputs.iter().map(|&(dtype, _)| dtype).collect();
        let invalid = |message: String| CompileError::InvalidOp {
            op: node.op.clone(),
            message,
        };
        let (op, outputs) = if node.op.starts_with("aten.") || node.op.starts_with("avel.") {
            ops::resolve(node, &shapes, &dtypes, symbols)?
        } else {
            let name = node
                .outputs
                .first()
                .and_then(Option::as_ref)
                .ok_or_else(|| CompileError::UnsupportedOp(node.op.clone()))?;
            let kernel = custom
                .get(&name.name)
                .ok_or_else(|| CompileError::UnsupportedOp(node.op.clone()))?;
            if node.inputs.iter().any(Option::is_none) {
                return Err(invalid("custom ops take no optional tensors".to_owned()));
            }
            kernel.validate(node.inputs.len(), builder.limits)?;
            let op = Op::Custom(kernel.clone());
            let dtypes = ops::infer_dtypes(&op, &dtypes, &shapes).map_err(invalid)?;
            let output = Output {
                shape: kernel.shape.clone(),
                dtype: dtypes[0],
            };
            (op, vec![output])
        };

        if node.outputs.iter().all(Option::is_none) {
            return Err(invalid("the node has no output".to_owned()));
        }
        if let Some(index) =
            (outputs.len()..node.outputs.len()).find(|&i| node.outputs[i].is_some())
        {
            return Err(invalid(format!("output {index} is not supported")));
        }
        // Pools compute values alongside indices even when only indices are
        // read. Other kernels currently need their first output named.
        if node.outputs[0].is_none()
            && !matches!(
                op,
                Op::View(_)
                    | Op::Pool(_)
                    | Op::Order { .. }
                    | Op::Row {
                        kind: ops::RowKind::Extrema {
                            output: ops::Extremum::Both,
                            ..
                        },
                        ..
                    }
            )
        {
            return Err(invalid("output 0 must be read".to_owned()));
        }
        let mut named = Vec::with_capacity(outputs.len());
        for (info, output) in node.outputs.iter().map(Option::as_ref).zip(outputs) {
            if let Some(info) = info {
                check_shape(info, &output.shape)?;
                check_dtype(info, output.dtype)?;
                known.insert(&info.name, (output.dtype, output.shape.clone()));
            }
            named.push((info.map(|info| info.name.as_str()), output));
        }
        resolved.push(Resolved {
            node,
            inputs: node.tensors().collect(),
            op,
            outputs: named,
        });
    }
    Ok(resolved)
}

/// Resolves the graph outputs. Outputs are read back as whole buffers, so one
/// that is a strided view, or part of a larger buffer, is copied first.
fn resolve_outputs(graph: &Graph, builder: &mut Builder) -> Result<Vec<TensorDesc>, CompileError> {
    graph
        .outputs
        .iter()
        .map(|output| {
            let Value { dtype, view } = builder
                .values
                .get(output.name.as_str())
                .cloned()
                .ok_or_else(|| CompileError::UnknownOutput(output.name.clone()))?;
            if output.dtype != dtype {
                return Err(CompileError::OutputTypeMismatch {
                    name: output.name.clone(),
                    found: dtype,
                    expected: output.dtype,
                });
            }
            check_shape(output, &view.shape)?;

            // Outputs are read back after the run, once state is rewritten:
            // a state tensor returned as is is copied first.
            let whole = view.is_row_major()
                && view.elements() * dtype.size() == builder.buffers[view.buffer].size
                && !matches!(builder.buffers[view.buffer].role, BufferRole::State { .. });
            let view = match whole {
                true => view,
                false => builder.materialize(&output.name, view)?,
            };
            let role = &mut builder.buffers[view.buffer].role;
            if *role == BufferRole::Arena {
                *role = BufferRole::Output;
            }
            Ok(TensorDesc {
                name: output.name.clone(),
                dtype,
                shape: view.shape,
                buffer: view.buffer as u32,
            })
        })
        .collect()
}

fn check_dtype(value: &ValueInfo, dtype: Dtype) -> Result<(), CompileError> {
    if value.dtype != dtype {
        return Err(CompileError::DtypeMismatch {
            name: value.name.clone(),
            found: dtype,
            expected: value.dtype,
        });
    }
    Ok(())
}

fn check_shape(value: &ValueInfo, shape: &[u64]) -> Result<(), CompileError> {
    if shape.len() != value.shape.len() {
        return Err(CompileError::RankMismatch {
            name: value.name.clone(),
            found: shape.len(),
            expected: value.shape.len(),
        });
    }
    for (axis, (dim, &size)) in value.shape.iter().zip(shape).enumerate() {
        if !dim.accepts(size) {
            return Err(CompileError::DimMismatch {
                name: value.name.clone(),
                axis,
                found: size,
                expected: format!("{dim:?}"),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
