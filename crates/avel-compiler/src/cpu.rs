//! CPU execution of the compiler's kernel IR, with a reusable activation arena.

mod attention;
mod elementwise;
mod movement;
mod rows;
mod spatial;

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::rc::Rc;

use crate::kernel::{Kernel, KernelNode};
use crate::plan::{Binding, BufferDesc, BufferRole, TensorDesc};
use crate::view::{View, row_major_strides};
use crate::{CompileError, Dtype, Graph};

pub(crate) type Weights = HashMap<String, Rc<[u32]>>;

/// A validated graph and its immutable weights, shared by all compiled shapes.
pub struct CpuModel {
    graph: Graph,
    weights: Weights,
}

impl CpuModel {
    pub fn new(graph: Graph) -> Result<Self, CompileError> {
        crate::validate(&graph)?;
        if !graph.media.is_empty() {
            return Err(CompileError::CpuUnsupported("media inputs".into()));
        }
        if graph
            .parameters
            .iter()
            .any(|parameter| parameter.update.is_some())
        {
            return Err(CompileError::CpuUnsupported(
                "persistent model state".into(),
            ));
        }
        if let Some(node) = graph
            .nodes
            .iter()
            .find(|node| !node.op.starts_with("aten.") && !node.op.starts_with("avel."))
        {
            return Err(CompileError::CpuUnsupported(format!(
                "custom op `{}`",
                node.op
            )));
        }
        Ok(Self {
            graph,
            weights: HashMap::new(),
        })
    }

    pub fn set_parameter(&mut self, name: &str, bytes: &[u8]) -> Result<(), CompileError> {
        let error = |message| CompileError::CpuParameter {
            name: name.to_owned(),
            message,
        };
        let parameter = self
            .graph
            .parameters
            .iter()
            .find(|p| p.tensor == name)
            .ok_or_else(|| error("unknown tensor"))?;
        if self.weights.contains_key(name) {
            return Err(error("already loaded"));
        }
        let element_size: u64 = match parameter.dtype {
            Dtype::Bool | Dtype::I8 => 1,
            Dtype::F32 | Dtype::I32 | Dtype::I64 => 4,
            Dtype::F16 => return Err(error("unsupported dtype f16")),
        };
        let size = parameter
            .shape
            .iter()
            .try_fold(element_size, |size, &n| size.checked_mul(n));
        if size != Some(bytes.len() as u64) {
            return Err(error("wrong byte count"));
        }
        let words: Rc<[u32]> = match parameter.dtype {
            Dtype::Bool => bytes.iter().map(|&byte| u32::from(byte != 0)).collect(),
            Dtype::I8 => bytes
                .chunks(4)
                .map(|chunk| {
                    let mut word = [0; 4];
                    word[..chunk.len()].copy_from_slice(chunk);
                    u32::from_le_bytes(word)
                })
                .collect(),
            _ => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .map(u32::from_le_bytes)
                .collect(),
        };
        self.weights.insert(name.to_owned(), words);
        Ok(())
    }

    pub fn compile(
        &self,
        shapes: BTreeMap<String, Vec<u64>>,
        bucket: Option<u64>,
    ) -> Result<CpuPlan, CompileError> {
        crate::compile::compile_cpu(&self.graph, shapes, bucket, &self.weights)
    }
}

enum Buffer {
    Arena(Range<usize>),
    Parameter(Rc<[u32]>),
}

struct Memory {
    buffers: Vec<Buffer>,
    words: Vec<u32>,
}

impl Memory {
    fn new(buffers: &[BufferDesc], weights: &Weights) -> Result<Self, CompileError> {
        let mut size = 0usize;
        let buffers = buffers
            .iter()
            .map(|desc| {
                if let BufferRole::Parameter { tensor } = &desc.role {
                    let words = weights
                        .get(tensor)
                        .filter(|words| words.len() as u64 * 4 == desc.size)
                        .ok_or_else(|| CompileError::CpuParameter {
                            name: tensor.clone(),
                            message: "tensor is missing or has the wrong size",
                        })?;
                    return Ok(Buffer::Parameter(Rc::clone(words)));
                }
                let start = size;
                let length = (desc.size / 4) as usize;
                size = size
                    .checked_add(length)
                    .ok_or(CompileError::CpuAllocation { bytes: usize::MAX })?;
                Ok(Buffer::Arena(start..size))
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            buffers,
            words: zeroed(size)?,
        })
    }

    #[inline]
    fn read(&self, buffer: usize) -> &[u32] {
        match &self.buffers[buffer] {
            Buffer::Arena(range) => &self.words[range.clone()],
            Buffer::Parameter(words) => words,
        }
    }

    #[inline]
    fn write(&mut self, buffer: usize) -> &mut [u32] {
        let Buffer::Arena(range) = &self.buffers[buffer] else {
            unreachable!("kernels and inputs only write owned buffers");
        };
        &mut self.words[range.clone()]
    }
}

fn zeroed<T: Copy + Default>(length: usize) -> Result<Vec<T>, CompileError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| CompileError::CpuAllocation {
            bytes: length.saturating_mul(std::mem::size_of::<T>()),
        })?;
    values.resize(length, T::default());
    Ok(values)
}

/// Logical row-major coordinates mapped to a view's physical strides.
struct Index {
    shape: Vec<usize>,
    strides: Vec<usize>,
    logical: Vec<usize>,
    contiguous: bool,
    count: usize,
}

impl Index {
    fn new(shape: &[u64], strides: &[u64]) -> Self {
        let logical = row_major_strides(shape);
        let contiguous = logical
            .iter()
            .zip(shape)
            .zip(strides)
            .all(|((&dense, &dim), &stride)| dim == 1 || dense == stride);
        Self {
            shape: shape.iter().map(|&n| n as usize).collect(),
            strides: strides.iter().map(|&n| n as usize).collect(),
            logical: logical.into_iter().map(|n| n as usize).collect(),
            contiguous,
            count: shape.iter().product::<u64>() as usize,
        }
    }

    #[inline]
    fn at(&self, mut index: usize) -> usize {
        if self.contiguous {
            return index;
        }
        let mut offset = 0;
        for (&size, &stride) in self.shape.iter().zip(&self.strides).rev() {
            offset += index % size * stride;
            index /= size;
        }
        offset
    }

    #[inline]
    fn coordinate(&self, axis: usize, index: usize) -> usize {
        index / self.logical[axis] % self.shape[axis]
    }
}

struct Tensor {
    buffer: usize,
    offset: usize,
    dtype: Dtype,
    index: Index,
}

impl Tensor {
    fn new(view: &View, binding: Binding, dtype: Dtype) -> Self {
        let elements_per_word = if dtype == Dtype::I8 { 4 } else { 1 };
        Self {
            buffer: binding.buffer as usize,
            offset: binding.offset as usize / 4 * elements_per_word + view.offset as usize,
            dtype,
            index: Index::new(&view.shape, &view.strides),
        }
    }

    #[inline]
    fn read(&self, memory: &Memory, offset: usize) -> u32 {
        let data = memory.read(self.buffer);
        let index = self.offset + offset;
        if self.dtype == Dtype::I8 {
            (data[index / 4] >> (index % 4 * 8)) as i8 as i32 as u32
        } else {
            data[index]
        }
    }

    #[inline]
    fn get(&self, memory: &Memory, index: usize) -> u32 {
        self.read(memory, self.index.at(index))
    }

    #[inline]
    fn float(&self, memory: &Memory, offset: usize) -> f32 {
        f32::from_bits(memory.read(self.buffer)[self.offset + offset])
    }

    #[inline]
    fn write(&self, memory: &mut Memory, offset: usize, value: u32) {
        memory.write(self.buffer)[self.offset + offset] = value;
    }

    #[inline]
    fn put(&self, memory: &mut Memory, index: usize, value: u32) {
        self.write(memory, self.index.at(index), value);
    }
}

struct Step {
    kernel: CpuKernel,
    inputs: Vec<Tensor>,
    output: Tensor,
    second: Option<Tensor>,
}

enum CpuKernel {
    Attention(attention::Attention),
    Elementwise(elementwise::Expression),
    Rows(rows::Rows),
    Matmul(spatial::Matmul),
    Convolution(spatial::Conv),
    Pool(spatial::Pool),
    Movement(movement::Movement),
}

impl Step {
    fn new(
        node: KernelNode,
        buffers: &[BufferDesc],
        bind: &impl Fn(usize) -> Binding,
        ops: &HashMap<&String, &String>,
    ) -> Result<Self, CompileError> {
        let tensor = |view: &View| Tensor::new(view, bind(view.buffer), buffers[view.buffer].dtype);
        let inputs: Vec<_> = node.inputs.iter().map(tensor).collect();
        let output = tensor(&node.output);
        let second = node.second.as_ref().map(tensor);
        let unsupported = || {
            let name = &node.ops[0];
            let op = ops.get(name).copied().unwrap_or(name);
            CompileError::CpuUnsupported(format!("operator `{op}` (value `{name}`)"))
        };
        let kernel = match node.kernel {
            Kernel::Attention { scale, causal, .. } => {
                CpuKernel::Attention(attention::Attention::new(scale, causal, &inputs, &output)?)
            }
            Kernel::Rotary => CpuKernel::Movement(movement::Movement::Rotary {
                depth: *output.index.shape.last().unwrap(),
            }),
            Kernel::ElementWise(expr) => CpuKernel::Elementwise(elementwise::Expression::new(
                expr,
                &inputs,
                &node.output.shape,
            )),
            Kernel::Reduction { kind, axes, .. } => {
                CpuKernel::Rows(rows::Rows::reduce(kind, &axes, &inputs[0]))
            }
            Kernel::Row { kind, axes, .. } => {
                CpuKernel::Rows(rows::Rows::row(kind, &axes, &inputs[0], &output))
            }
            Kernel::LayerNorm {
                eps,
                normalized,
                rms,
                weight,
                ..
            } => CpuKernel::Rows(rows::Rows::norm(
                eps,
                normalized.len(),
                rms,
                weight,
                &inputs[0],
                &output,
            )),
            Kernel::Matmul { beta, alpha, .. } => {
                CpuKernel::Matmul(spatial::Matmul::new(beta, alpha, &inputs))
            }
            Kernel::Convolution { convolution, .. } => {
                CpuKernel::Convolution(spatial::Conv::new(&convolution, &inputs, &output)?)
            }
            Kernel::Pool(pool) => {
                CpuKernel::Pool(spatial::Pool::new(Some(&pool), &inputs[0], &output)?)
            }
            Kernel::Movement {
                kind: crate::ops::Movement::AdaptiveAverage,
                ..
            } => CpuKernel::Pool(spatial::Pool::new(None, &inputs[0], &output)?),
            Kernel::Movement { kind, fill } => CpuKernel::Movement(
                movement::Movement::new(kind, fill, &inputs, &output).ok_or_else(unsupported)?,
            ),
            Kernel::Cat { axis, .. } => CpuKernel::Movement(movement::Movement::cat(axis, &inputs)),
            Kernel::Embedding { .. } => CpuKernel::Movement(movement::Movement::Embedding),
            _ => return Err(unsupported()),
        };
        Ok(Self {
            kernel,
            inputs,
            output,
            second,
        })
    }

    fn run(&mut self, memory: &mut Memory) {
        let Self {
            kernel,
            inputs,
            output,
            second,
        } = self;
        match kernel {
            CpuKernel::Attention(attention) => attention.run(memory, inputs, output),
            CpuKernel::Elementwise(expr) => expr.run(memory, inputs, output),
            CpuKernel::Rows(rows) => rows.run(memory, inputs, output, second.as_ref()),
            CpuKernel::Matmul(matmul) => matmul.run(memory, inputs, output),
            CpuKernel::Convolution(conv) => conv.run(memory, inputs, output),
            CpuKernel::Pool(pool) => pool.run(memory, &inputs[0], output, second.as_ref()),
            CpuKernel::Movement(movement) => movement.run(memory, inputs, output),
        }
    }
}

/// A prepared CPU executable. Input and output addresses are stable for its lifetime.
pub struct CpuPlan {
    memory: Memory,
    inputs: Vec<TensorDesc>,
    outputs: Vec<TensorDesc>,
    steps: Vec<Step>,
}

pub(crate) struct Schedule {
    pub inputs: Vec<TensorDesc>,
    pub outputs: Vec<TensorDesc>,
    pub prepare: Vec<KernelNode>,
    pub kernels: Vec<KernelNode>,
}

impl CpuPlan {
    pub(crate) fn new(
        placed: &[BufferDesc],
        buffers: &[BufferDesc],
        bind: impl Fn(usize) -> Binding,
        schedule: Schedule,
        weights: &Weights,
        ops: &HashMap<&String, &String>,
    ) -> Result<Self, CompileError> {
        let Schedule {
            mut inputs,
            mut outputs,
            prepare,
            kernels,
        } = schedule;
        let build = |node| Step::new(node, buffers, &bind, ops);
        let steps = kernels
            .into_iter()
            .map(build)
            .collect::<Result<Vec<_>, _>>()?;
        let mut prepare = prepare
            .into_iter()
            .map(build)
            .collect::<Result<Vec<_>, _>>()?;
        let mut memory = Memory::new(placed, weights)?;
        for step in &mut prepare {
            step.run(&mut memory);
        }
        for desc in inputs.iter_mut().chain(&mut outputs) {
            let binding = bind(desc.buffer as usize);
            debug_assert_eq!(binding.offset, 0);
            desc.buffer = binding.buffer;
        }
        Ok(Self {
            memory,
            inputs,
            outputs,
            steps,
        })
    }

    pub fn inputs(&self) -> impl Iterator<Item = (&TensorDesc, &[u32])> {
        self.inputs
            .iter()
            .map(|desc| (desc, self.memory.read(desc.buffer as usize)))
    }

    pub fn input_mut(&mut self, index: usize) -> &mut [u32] {
        self.memory.write(self.inputs[index].buffer as usize)
    }

    pub fn outputs(&self) -> impl Iterator<Item = (&TensorDesc, &[u32])> {
        self.outputs
            .iter()
            .map(|desc| (desc, self.memory.read(desc.buffer as usize)))
    }

    pub fn run(&mut self) {
        for step in &mut self.steps {
            step.run(&mut self.memory);
        }
    }
}
