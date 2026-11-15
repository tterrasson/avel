use super::*;

/// A sum or mean over `axes` of the input. Serially, one invocation per
/// output element loops over the reduced elements; cooperatively, one
/// workgroup per output element strides over them and combines the lanes.
/// With a split, the workgroups along the third dispatch axis each cover one
/// slice and store a partial result behind the slice axis of the output.
pub(super) fn reduction(
    io: Io,
    axes: &[usize],
    kind: ReduceKind,
    strategy: Strategy,
    split: u64,
) -> (String, [u32; 3]) {
    let node = io.node;
    let input = &node.inputs[0];
    // The output without the slice axis of the partial results.
    let output = match split {
        1 => node.output.clone(),
        _ => node.output.select(0, 0),
    };
    let keepdim = output.shape.len() == input.shape.len();
    let kept: Vec<usize> = (0..input.shape.len())
        .filter(|axis| !axes.contains(axis))
        .collect();
    let base: Vec<(String, u64)> = kept
        .iter()
        .enumerate()
        .map(|(position, &axis)| {
            let out_axis = if keepdim { axis } else { position };
            let coordinate = coordinate("k", output.strides[out_axis], output.shape[out_axis]);
            (coordinate, input.strides[axis])
        })
        .collect();
    let base = linear(input.offset, &base);

    let reduced_shape: Vec<u64> = axes.iter().map(|&axis| input.shape[axis]).collect();
    let count: u64 = reduced_shape.iter().product();
    let offset: Vec<(String, u64)> = axes
        .iter()
        .zip(row_major_strides(&reduced_shape))
        .map(|(&axis, stride)| {
            (
                coordinate("r", stride, input.shape[axis]),
                input.strides[axis],
            )
        })
        .collect();
    let element = io.read(0, &format!("base + {}", linear(0, &offset)));
    // A partial mean is already divided by the full count, so that the
    // second pass only sums.
    let result = match kind {
        ReduceKind::Sum => "acc".to_owned(),
        ReduceKind::Mean => format!("acc / {}", float(count as f32)),
    };
    let rows = output.elements();

    let Strategy::Workgroup { lanes, subgroup } = strategy else {
        assert_eq!(split, 1, "split reductions are cooperative");
        let body = wgsl!(
            "reduction.wgsl",
            base = base,
            count = count,
            element = element,
            store = io.store_k(&result),
        );
        return linear_dispatch(io, rows, &body);
    };
    let store = match split {
        1 => io.store_k(&result),
        _ => io.store(
            &format!("slice * {}u + k", node.output.strides[0]),
            &result,
            &|_| unreachable!("partial results have no epilogue"),
        ),
    };
    let mut io = io;
    io.subgroups = subgroup;
    io.add_helper(&workgroup_sum(lanes, subgroup));
    let wgsl = wgsl!(
        "reduction_workgroup.wgsl",
        declarations = io.declarations(),
        helpers = io.helpers(),
        lanes = lanes,
        entry_point = ENTRY_POINT,
        max_workgroups = MAX_WORKGROUPS,
        rows = rows,
        base = base,
        slice = count.div_ceil(split),
        count = count,
        element = element,
        store = store,
    );
    (wgsl, row_workgroups(rows, split))
}

/// One workgroup per row, up to `MAX_WORKGROUPS` per dispatch axis, and one
/// per slice along the third axis.
pub(super) fn row_workgroups(rows: u64, split: u64) -> [u32; 3] {
    [
        rows.min(MAX_WORKGROUPS) as u32,
        rows.div_ceil(MAX_WORKGROUPS) as u32,
        split as u32,
    ]
}

/// `workgroup_sum(value, lane)`: the sum of `value` over the `lanes`
/// invocations of the workgroup, returned to all of them. Called in uniform
/// control flow. A workgroup that is a single subgroup adds within it;
/// otherwise a tree in workgroup memory halves the active lanes each step.
pub(super) fn workgroup_sum(lanes: u32, subgroup: bool) -> String {
    match subgroup {
        true => wgsl!("workgroup_sum_subgroup.wgsl", lanes = lanes,),
        false => wgsl!("workgroup_sum.wgsl", lanes = lanes, half = lanes / 2,),
    }
}
