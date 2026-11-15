use super::*;

/// Layer norm over the trailing `normalized` axes of the input, one row per
/// outer index: the mean, then the variance of the centered values, then the
/// normalized store, through the epilogue. RMS norm skips the mean. Serially,
/// one invocation covers a row; cooperatively, a workgroup strides over it.
pub(super) fn layer_norm(
    mut io: Io,
    eps: f32,
    normalized: &[u64],
    rms: bool,
    weight: bool,
    strategy: Strategy,
) -> (String, [u32; 3]) {
    let node = io.node;
    let input = &node.inputs[0];
    let output = &node.output;
    let lead = output.shape.len() - normalized.len();
    let rows: u64 = output.shape[..lead].iter().product();
    let depth: u64 = normalized.iter().product();

    let lead_strides = row_major_strides(&output.shape[..lead]);
    let base_terms: Vec<(String, u64)> = (0..lead)
        .map(|axis| {
            (
                coordinate("k", lead_strides[axis], output.shape[axis]),
                input.strides[axis],
            )
        })
        .collect();
    let base = linear(input.offset, &base_terms);
    let norm_strides = row_major_strides(normalized);
    let terms: Vec<_> = normalized
        .iter()
        .zip(norm_strides.iter().copied())
        .enumerate()
        .map(|(axis, (&size, stride))| (coordinate("e", stride, size), input.strides[lead + axis]))
        .collect();
    let read = io.read(0, &format!("base + {}", linear(0, &terms)));

    // The weight and bias are broadcast to the output shape, so they are read
    // at the same element the store writes.
    let element = if output.is_row_major() {
        format!("k * {depth}u + e")
    } else {
        let terms: Vec<_> = (0..output.shape.len())
            .map(|axis| {
                let coord = if axis < lead {
                    coordinate("k", lead_strides[axis], output.shape[axis])
                } else {
                    let reduced = axis - lead;
                    coordinate("e", norm_strides[reduced], normalized[reduced])
                };
                (coord, output.strides[axis])
            })
            .collect();
        linear(output.offset, &terms)
    };
    let normalized_value = match rms {
        true => format!("{read} * rstd"),
        false => format!("({read} - mean) * rstd"),
    };
    let affine = |index: usize| io.read(index, &gather(&element, output, &node.inputs[index]));
    let value = if weight {
        format!("({normalized_value} * {})", affine(1))
    } else {
        normalized_value
    };
    let bias = 1 + usize::from(weight);
    let value = if node.inputs.len() > bias {
        format!("({value} + {})", affine(bias))
    } else {
        value
    };
    let store = io.store(&element, &value, &|view| gather(&element, output, view));
    // Serially, one invocation covers the row; cooperatively, the lanes
    // stride over it and add their sums.
    let (first, step, cooperative) = match strategy {
        Strategy::Serial => ("0u".to_owned(), "1u".to_owned(), false),
        Strategy::Workgroup { lanes, .. } => ("lane".to_owned(), format!("{lanes}u"), true),
    };
    let total = |sum: &str| match cooperative {
        true => format!("workgroup_sum({sum}, lane)"),
        false => sum.to_owned(),
    };
    let mean = match rms {
        true => "let mean = 0.0;".to_owned(),
        false => wgsl!(
            "layer_norm_mean.wgsl",
            first = first,
            depth = depth,
            step = step,
            read = read,
            sum = total("sum"),
            depth_f32 = float(depth as f32),
        ),
    };
    let body = wgsl!(
        "layer_norm.wgsl",
        base = base,
        mean = mean,
        first = first,
        depth = depth,
        step = step,
        read = read,
        sum = total("sq"),
        depth_f32 = float(depth as f32),
        eps = float(eps),
        store = store,
    );
    let Strategy::Workgroup { lanes, subgroup } = strategy else {
        return linear_dispatch(io, rows, &body);
    };
    io.subgroups = subgroup;
    io.add_helper(&super::reduction::workgroup_sum(lanes, subgroup));
    let wgsl = wgsl!(
        "row_workgroup.wgsl",
        declarations = io.declarations(),
        helpers = io.helpers(),
        lanes = lanes,
        entry_point = ENTRY_POINT,
        max_workgroups = MAX_WORKGROUPS,
        rows = rows,
        body = body,
    );
    (wgsl, super::reduction::row_workgroups(rows, 1))
}

/// Fused scaled dot-product attention, flash style: one invocation per query
/// row (outer index, head, query position), an online softmax over the keys
/// and the value accumulator in registers. The output is row-major, so row
/// `k` holds the `value_depth` elements at `k * value_depth + d`.
pub(super) fn attention(
    io: Io,
    scale: f32,
    causal: bool,
    mask: bool,
    cached: bool,
) -> (String, [u32; 3]) {
    let node = io.node;
    let (q, k, v) = (&node.inputs[0], &node.inputs[1], &node.inputs[2]);
    let output = &node.output;
    let rank = output.shape.len();
    let key_seq = k.shape[rank - 2];
    let depth = q.shape[rank - 1];
    let value_depth = output.shape[rank - 1];
    let rows: u64 = output.shape[..rank - 1].iter().product();

    let row_shape = &output.shape[..rank - 1];
    let row_strides = row_major_strides(row_shape);
    let coords: Vec<String> = (0..rank - 1)
        .map(|axis| coordinate("k", row_strides[axis], row_shape[axis]))
        .collect();
    let head = coords[rank - 3].clone();
    let position = coords[rank - 2].clone();

    let q_at = |d: &str| {
        let mut terms: Vec<(String, u64)> = (0..rank - 1)
            .map(|axis| (coords[axis].clone(), q.strides[axis]))
            .collect();
        terms.push((d.to_owned(), q.strides[rank - 1]));
        linear(q.offset, &terms)
    };
    let key_at = |j: &str, d: &str| {
        let mut terms: Vec<(String, u64)> = (0..rank - 3)
            .map(|axis| (coords[axis].clone(), k.strides[axis]))
            .collect();
        terms.push((head.clone(), k.strides[rank - 3]));
        terms.push((j.to_owned(), k.strides[rank - 2]));
        terms.push((d.to_owned(), k.strides[rank - 1]));
        linear(k.offset, &terms)
    };
    let value_at = |j: &str, d: &str| {
        let mut terms: Vec<(String, u64)> = (0..rank - 3)
            .map(|axis| (coords[axis].clone(), v.strides[axis]))
            .collect();
        terms.push((head.clone(), v.strides[rank - 3]));
        terms.push((j.to_owned(), v.strides[rank - 2]));
        terms.push((d.to_owned(), v.strides[rank - 1]));
        linear(v.offset, &terms)
    };

    let bound = if cached {
        let start = super::generation::position(&io, 3, key_seq - q.shape[rank - 2]);
        format!("({start}) + ({position}) + 1u")
    } else if causal {
        format!("min({key_seq}u, ({position}) + 1u)")
    } else {
        format!("{key_seq}u")
    };
    let mask_statement = if mask {
        let mask = &node.inputs[3];
        let mut terms: Vec<(String, u64)> = (0..rank - 3)
            .map(|axis| (coords[axis].clone(), mask.strides[axis]))
            .collect();
        terms.push((head.clone(), mask.strides[rank - 3]));
        terms.push((position.clone(), mask.strides[rank - 2]));
        terms.push(("j".to_owned(), mask.strides[rank - 1]));
        let read = io.read(3, &linear(mask.offset, &terms));
        match io.ty(3) {
            // Skipping masked keys also handles rows with no visible key.
            ScalarType::Bool => format!("if (!{read}) {{ continue; }}"),
            _ => wgsl!("attention_additive_mask.wgsl", read = read,),
        }
    } else {
        String::new()
    };

    let element = format!("k * {value_depth}u + d");
    let store = io.store(&element, "acc[d] / max(l, 1.0e-30)", &|view| {
        gather(&element, output, view)
    });
    let load_query = io.read(0, &q_at("d"));
    let (prepare_query, query_read) = if io
        .node
        .prologue(0)
        .is_some_and(|p| matches!(p.kind, PrologueKind::Rotary))
    {
        (
            format!(
                "var q_row: array<f32, {depth}>;
for (var d = 0u; d < {depth}u; d++) {{ q_row[d] = {load_query}; }}"
            ),
            "q_row[d]".to_owned(),
        )
    } else {
        (String::new(), load_query)
    };
    let body = wgsl!(
        "attention.wgsl",
        lowest = float(LOWEST),
        value_depth = value_depth,
        key_bound = bound,
        depth = depth,
        prepare_query = prepare_query,
        query_read = query_read,
        key_read = io.read(1, &key_at("j", "d")),
        scale = float(scale),
        mask = mask_statement,
        value_read = io.read(2, &value_at("j", "d")),
        store = store,
    );
    linear_dispatch(io, rows, &body)
}
