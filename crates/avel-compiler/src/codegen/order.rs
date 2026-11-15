use super::reduction::row_workgroups;
use super::*;

fn base(input: &View, axis: usize, row: &str) -> String {
    let mut row_shape = input.shape.clone();
    row_shape[axis] = 1;
    let terms: Vec<_> = row_major_strides(&row_shape)
        .into_iter()
        .zip(row_shape)
        .zip(&input.strides)
        .map(|((stride, size), &storage)| (coordinate(row, stride, size), storage))
        .collect();
    linear(input.offset, &terms)
}

fn comparator(io: &Io, largest: bool) -> &'static str {
    let ty = io.ty(0);
    let cmp = match (ty, largest) {
        (ScalarType::Bool, true) => "a && !b",
        (ScalarType::Bool, false) => "!a && b",
        (_, true) => "a > b",
        (_, false) => "a < b",
    };
    let nan = if ty == ScalarType::F32 {
        io.add_helper(EXTREMA);
        let prefer_nan = if largest { "a_nan" } else { "b_nan" };
        format!(
            "let a_nan = avel_is_nan(a); let b_nan = avel_is_nan(b);
            if (a_nan || b_nan) {{ return select({prefer_nan}, ai < bi, a_nan && b_nan); }}"
        )
    } else {
        String::new()
    };
    io.add_helper(&format!(
        "fn before(a: {ty}, ai: u32, b: {ty}, bi: u32) -> bool {{
            if (ai == 0xffffffffu || bi == 0xffffffffu) {{ return ai < bi; }}
            {nan}
            return ({cmp}) || (a == b && ai < bi);
        }}",
        ty = ty.wgsl(),
    ));
    match ty {
        // WGSL workgroup arrays cannot hold bool.
        ScalarType::Bool => "u32",
        _ => ty.wgsl(),
    }
}

pub(super) fn blocks(io: Io, axis: usize, largest: bool, tile: u64) -> (String, [u32; 3]) {
    let input = &io.node.inputs[0];
    let [rows, blocks, keep] = io.output().shape.as_slice() else {
        unreachable!()
    };
    let groups = rows * blocks;
    let ty = comparator(&io, largest);
    let compare = if io.ty(0) == ScalarType::Bool {
        "before(values[i] != 0u, ids[i], values[j] != 0u, ids[j])"
    } else {
        "before(values[i], ids[i], values[j], ids[j])"
    };
    let read = io.read(0, &format!("base + position * {}u", input.strides[axis]));
    let wgsl = wgsl!(
        "order_blocks.wgsl",
        declarations = io.declarations(),
        helpers = io.helpers(),
        entry_point = ENTRY_POINT,
        max_workgroups = MAX_WORKGROUPS,
        groups = groups,
        lanes = WORKGROUP_SIZE,
        tile = tile,
        blocks = blocks,
        keep = keep,
        width = input.shape[axis],
        base = base(input, axis, "row"),
        read = format!("{ty}({read})"),
        ty = ty,
        compare = compare,
        store = io.store_k("i32(ids[i])"),
    );
    (wgsl, row_workgroups(groups, 1))
}

pub(super) fn merge(io: &Io, axis: usize, largest: bool, span: u64) -> String {
    let [_, groups, keep] = io.output().shape.as_slice() else {
        unreachable!()
    };
    let input = &io.node.inputs[0];
    let [_, old_groups, old_keep] = io.node.inputs[1].shape.as_slice() else {
        unreachable!()
    };
    comparator(io, largest);
    let read = |id: &str| io.read(0, &format!("base + {id} * {}u", input.strides[axis]));
    wgsl!(
        "order_merge.wgsl",
        width = input.shape[axis],
        groups = groups,
        keep = keep,
        span = span,
        old_groups = old_groups,
        old_keep = old_keep,
        base = base(input, axis, "row"),
        index_left = io.read(1, "left + mid"),
        index_right = io.read(1, "right + diagonal - mid - 1u"),
        selected_left = io.read(1, "left + low"),
        selected_right = io.read(1, "right + diagonal - low"),
        compare = format!("before({}, a, {}, b)", read("a"), read("b")),
        store = io.store_k("i32(chosen)"),
    )
}

pub(super) fn gather_values(io: &Io, axis: usize) -> String {
    let input = &io.node.inputs[0];
    let output = io.output();
    let mut rows = output.shape.clone();
    rows[axis] = 1;
    let row = linear(
        0,
        &row_major_strides(&rows)
            .into_iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(i, stride)| (format!("o{i}"), stride))
            .collect::<Vec<_>>(),
    );
    let id = io.read(1, &format!("({row}) * {}u + o{axis}", output.shape[axis]));
    let value = io.read(
        0,
        &format!(
            "{} + u32(chosen) * {}u",
            base(input, axis, &format!("({row})")),
            input.strides[axis]
        ),
    );
    format!(
        "{}\nlet chosen = {id};\n{}\n{}",
        output_coordinates(output, "o"),
        io.store_k(&value),
        io.store_second_k("chosen"),
    )
}
