//! Lowers kernels to dispatches and assigns their storage in the final plan.

use super::{Builder, Limits};
use crate::codegen::{self, Storage};
use crate::fuse::{fuse, fuse_prologues};
use crate::kernel::{Kernel, KernelNode, Prologue, PrologueKind, ScalarType};
use crate::media::MediaRead;
use crate::memory::{self, Lifetime};
use crate::plan::{Binding, BufferDesc, BufferRole, Dispatch, Shader, TensorDesc};
use crate::{CompileError, Dtype};
use std::collections::{HashMap, HashSet};

pub(super) struct Schedule {
    pub shaders: Vec<Shader>,
    pub buffers: Vec<BufferDesc>,
    pub prepare: Vec<Dispatch>,
    pub dispatches: Vec<Dispatch>,
}

pub(super) fn schedule(
    builder: Builder<'_>,
    enable_f16: bool,
    f16_compute: bool,
    media_reads: &HashMap<usize, MediaRead>,
    inputs: &mut [TensorDesc],
    outputs: &mut [TensorDesc],
) -> Result<Schedule, CompileError> {
    let Builder {
        limits,
        mut buffers,
        mut prepare,
        kernels,
        ..
    } = builder;
    let fusible = |buffer: usize| buffers[buffer].role == BufferRole::Arena;
    let max_bindings = limits.max_storage_buffers_per_shader_stage as usize;
    let kernels = fuse(kernels, fusible, max_bindings);
    let mut kernels = fuse_prologues(kernels, fusible, max_bindings);
    fuse_weight_loads(&mut buffers, &mut prepare, &mut kernels, max_bindings);

    // Outputs are read back as f32, so only intermediates and constants
    // that are not outputs may be stored in f16.
    let f16: Vec<bool> = (0..buffers.len())
        .map(|buffer| {
            enable_f16
                && buffers[buffer].dtype == Dtype::F32
                && matches!(
                    buffers[buffer].role,
                    BufferRole::Arena | BufferRole::Constant
                )
                && !outputs
                    .iter()
                    .any(|output| output.buffer as usize == buffer)
        })
        .collect();
    for (buffer, &half) in buffers.iter_mut().zip(&f16) {
        if half {
            buffer.size = (buffer.size / 2).next_multiple_of(4);
        }
    }

    // The custom ABI is dense f32, even in a plan with f16 intermediates.
    let mut f16 = f16;
    for node in prepare.iter().chain(&kernels) {
        if matches!(node.kernel, Kernel::Scatter(_)) && f16[node.output.buffer] {
            // Additive scatters repeatedly update their destination, so
            // the buffer must preserve f32 precision between updates.
            f16[node.output.buffer] = false;
            buffers[node.output.buffer].size = node.output.elements() * 4;
        }
        if matches!(node.kernel, Kernel::Custom(_)) {
            for view in node.inputs.iter().chain(node.writes()) {
                if f16[view.buffer] {
                    f16[view.buffer] = false;
                    buffers[view.buffer].size = view.shape.iter().product::<u64>() * 4;
                }
            }
        }
    }
    let places = place(&buffers, &kernels, limits)?;
    let storage = |buffer: usize| Storage {
        f16: f16[buffer],
        packed_i8: buffers[buffer].dtype == Dtype::I8,
        media: media_reads.get(&buffer).copied(),
        writable: matches!(
            buffers[buffer].role,
            BufferRole::Arena
                | BufferRole::Constant
                | BufferRole::Output
                | BufferRole::State { .. }
        ),
        ty: ScalarType::of(buffers[buffer].dtype),
    };
    let mut shaders: Vec<Shader> = Vec::new();
    let mut dispatch = |node: &KernelNode| -> Result<Dispatch, CompileError> {
        let (shader, workgroups) = codegen::generate_with_compute(node, &storage, f16_compute);
        // Identical kernels share one shader. A plan has few distinct
        // shaders, and comparing strings checks their lengths first, so a
        // scan is cheaper than hashing every generated module.
        let index = match shaders.iter().position(|known| known.wgsl == shader.wgsl) {
            Some(index) => index as u32,
            None => {
                shaders.push(shader);
                shaders.len() as u32 - 1
            }
        };
        // One binding per distinct buffer, as in the shader: WebGPU rejects
        // aliased writable storage bindings. A buffer the kernel also writes
        // is read through its output binding.
        let mut bindings: Vec<Binding> = Vec::new();
        let mut bound: Vec<usize> = node.writes().map(|view| view.buffer).collect();
        for view in node.reads() {
            if !bound.contains(&view.buffer) {
                bound.push(view.buffer);
                bindings.push(places.binding(view.buffer, &buffers));
            }
        }
        for view in node.writes() {
            bindings.push(places.binding(view.buffer, &buffers));
        }
        if bindings.len() > limits.max_storage_buffers_per_shader_stage as usize
            || workgroups.iter().any(|&size| size == 0 || size > 65535)
        {
            return Err(CompileError::InvalidOp {
                op: node.ops.join(", "),
                message: "dispatch exceeds device binding or workgroup limits".to_owned(),
            });
        }
        Ok(Dispatch {
            shader: index,
            bindings,
            workgroups,
            ops: node.ops.clone(),
        })
    };
    let prepare = prepare
        .iter()
        .map(&mut dispatch)
        .collect::<Result<_, _>>()?;
    let dispatches = kernels
        .iter()
        .map(&mut dispatch)
        .collect::<Result<_, _>>()?;

    for tensor in inputs.iter_mut().chain(outputs) {
        tensor.buffer = places.index[tensor.buffer as usize].expect("inputs and outputs are kept");
    }
    Ok(Schedule {
        shaders,
        buffers: places.buffers,
        prepare,
        dispatches,
    })
}

/// Inlines prepared dequantization into GEMM/conv loads. Keep a prepared
/// f32 copy only for consumers that need it or have no spare binding.
fn fuse_weight_loads(
    buffers: &mut [BufferDesc],
    prepare: &mut Vec<KernelNode>,
    kernels: &mut [KernelNode],
    max_bindings: usize,
) {
    let sources: HashMap<_, _> = prepare
        .iter()
        .filter_map(|node| match &node.kernel {
            Kernel::ElementWise(expr)
                if node
                    .inputs
                    .first()
                    .is_some_and(|view| buffers[view.buffer].dtype == Dtype::I8) =>
            {
                Some((node.output.buffer, (node, expr)))
            }
            _ => None,
        })
        .collect();
    for node in kernels.iter_mut() {
        if !matches!(
            node.kernel,
            Kernel::Matmul { .. } | Kernel::TiledMatmul { .. } | Kernel::Convolution { .. }
        ) || node.prologue(1).is_some()
        {
            continue;
        }
        let weight = node.inputs[1].buffer;
        let Some(&(source, expr)) = sources.get(&weight) else {
            continue;
        };
        let bound: HashSet<_> = node
            .reads()
            .chain(node.writes())
            .filter(|view| view.buffer != weight)
            .chain(&source.inputs)
            .map(|view| view.buffer)
            .collect();
        // Another input/epilogue might also read the decoded weight.
        let others = node.reads().filter(|view| view.buffer == weight).count();
        if bound.len() + usize::from(others > 1) > max_bindings {
            continue;
        }
        node.prologues.resize(node.inputs.len(), None);
        node.prologues[1] = Some(Prologue {
            kind: PrologueKind::ElementWise(expr.clone()),
            inputs: source.inputs.clone(),
            source: source.output.clone(),
        });
    }
    let used: HashSet<_> = prepare
        .iter()
        .chain(kernels.iter())
        .flat_map(KernelNode::reads)
        .map(|view| view.buffer)
        .collect();
    let removed: HashSet<_> = sources
        .keys()
        .copied()
        .filter(|&buffer| buffers[buffer].role == BufferRole::Constant && !used.contains(&buffer))
        .collect();
    prepare.retain(|node| !removed.contains(&node.output.buffer));
    for buffer in removed {
        // Arena buffers without a lifetime are omitted by placement.
        buffers[buffer].role = BufferRole::Arena;
    }
}

/// Where each buffer of the builder ends up in the plan.
pub(super) struct Places {
    /// The plan buffers: every buffer that is not an intermediate, in order,
    /// then the arena if any intermediate is used.
    pub(super) buffers: Vec<BufferDesc>,
    /// Index into `buffers` of each builder buffer, if it is used.
    index: Vec<Option<u32>>,
    /// Offset of each builder buffer in its plan buffer.
    offsets: Vec<u64>,
}

impl Places {
    pub(super) fn binding(&self, buffer: usize, sizes: &[BufferDesc]) -> Binding {
        Binding {
            buffer: self.index[buffer].expect("bound buffers are placed"),
            offset: self.offsets[buffer],
            size: sizes[buffer].size,
        }
    }
}

/// Plans the arena: the intermediates `kernels` use share it, by lifetime.
pub(super) fn place(
    buffers: &[BufferDesc],
    kernels: &[KernelNode],
    limits: &Limits,
) -> Result<Places, CompileError> {
    let mut lifetimes: HashMap<usize, Lifetime> = HashMap::new();
    for (step, node) in kernels.iter().enumerate() {
        for view in node.reads().chain(node.writes()) {
            if buffers[view.buffer].role != BufferRole::Arena {
                continue;
            }
            let lifetime = lifetimes.entry(view.buffer).or_insert(Lifetime {
                size: buffers[view.buffer].size,
                first: step,
                last: step,
            });
            lifetime.last = step;
        }
    }
    let mut intermediates: Vec<usize> = lifetimes.keys().copied().collect();
    intermediates.sort_unstable();
    let alignment = u64::from(limits.min_storage_buffer_offset_alignment).max(4);
    let (offsets, arena) = memory::plan(
        &intermediates
            .iter()
            .map(|buffer| lifetimes[buffer])
            .collect::<Vec<_>>(),
        alignment,
    );
    if arena > limits.max_buffer_size {
        return Err(CompileError::ArenaTooLarge {
            size: arena,
            limit: limits.max_buffer_size,
        });
    }

    let mut places = Places {
        buffers: Vec::new(),
        index: vec![None; buffers.len()],
        offsets: vec![0; buffers.len()],
    };
    for (buffer, desc) in buffers.iter().enumerate() {
        if desc.role != BufferRole::Arena {
            places.index[buffer] = Some(places.buffers.len() as u32);
            places.buffers.push(desc.clone());
        }
    }
    if !intermediates.is_empty() {
        let index = places.buffers.len() as u32;
        places.buffers.push(BufferDesc {
            size: arena,
            role: BufferRole::Arena,
            dtype: Dtype::F32,
        });
        for (&buffer, &offset) in intermediates.iter().zip(&offsets) {
            places.index[buffer] = Some(index);
            places.offsets[buffer] = offset;
        }
    }
    Ok(places)
}
