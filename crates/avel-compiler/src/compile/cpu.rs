use super::*;
use crate::cpu::{CpuPlan, Schedule, Weights};

pub(crate) fn compile_cpu(
    graph: &Graph,
    input_shapes: BTreeMap<String, Vec<u64>>,
    bucket: Option<u64>,
    weights: &Weights,
) -> Result<CpuPlan, CompileError> {
    let options = CompileOptions {
        input_shapes,
        limits: Limits {
            max_buffer_size: i32::MAX as u64,
            max_storage_buffer_binding_size: i32::MAX as u64,
            max_storage_buffers_per_shader_stage: u32::MAX,
            min_storage_buffer_offset_alignment: 16,
        },
        features: Features::default(),
        f16: false,
        f16_compute: false,
        bucket,
        media: BTreeMap::new(),
        custom_kernels: BTreeMap::new(),
        gpu_class: crate::GpuClass::default(),
        tiles: BTreeMap::new(),
    };
    let GraphSetup {
        mut builder,
        inputs,
        symbols,
        ..
    } = setup(graph, &options)?;
    let resolved = resolve_nodes(graph, &builder, &symbols, &options.custom_kernels)?;
    if bucket.is_some() {
        shapes::validate_bucketing(graph, &resolved)?;
    }
    for node in &resolved {
        lower(node, Layout::RowMajor, &mut builder)?;
    }
    let outputs = resolve_outputs(graph, &mut builder)?;
    let Builder {
        buffers,
        prepare,
        kernels,
        ..
    } = builder;

    // Inline element-wise producers without introducing GPU kernel epilogues.
    let elementwise: HashSet<_> = kernels
        .iter()
        .filter(|node| matches!(node.kernel, Kernel::ElementWise(_)))
        .map(|node| node.output.buffer)
        .collect();
    let kernels = crate::fuse::fuse(
        kernels,
        |buffer| buffers[buffer].role == BufferRole::Arena && elementwise.contains(&buffer),
        8,
    );
    let places = schedule::place(&buffers, &kernels, &options.limits)?;
    let bind = |buffer| places.binding(buffer, &buffers);
    let ops = graph
        .nodes
        .iter()
        .flat_map(|node| {
            node.outputs
                .iter()
                .flatten()
                .map(|out| (&out.name, &node.op))
        })
        .collect();
    CpuPlan::new(
        &places.buffers,
        &buffers,
        bind,
        Schedule {
            inputs,
            outputs,
            prepare,
            kernels,
        },
        weights,
        &ops,
    )
}
