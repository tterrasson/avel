use super::*;

pub(super) fn position(io: &Io, input: usize, max: u64) -> String {
    let value = io.read(input, &format!("{}u", io.node.inputs[input].offset));
    format!("u32(clamp({value}, 0, {max}))")
}

pub(super) fn cache_write(io: &Io, in_place: bool) -> (String, u64) {
    let (cache, patch) = (&io.node.inputs[0], &io.node.inputs[1]);
    let output = io.output();
    let axis = cache.shape.len() - 2;
    let start = position(io, 2, cache.shape[axis] - patch.shape[axis]);
    let coordinates = if in_place {
        View::dense(0, patch.shape.clone(), crate::view::Layout::RowMajor)
    } else {
        output.clone()
    };
    let at = |view: &View, index: &str| {
        let terms: Vec<_> = view
            .strides
            .iter()
            .enumerate()
            .map(|(a, &stride)| {
                (
                    if a == axis {
                        index.to_owned()
                    } else {
                        format!("o{a}")
                    },
                    stride,
                )
            })
            .collect();
        linear(view.offset, &terms)
    };
    let setup = output_coordinates(&coordinates, "o");
    if in_place {
        let dst = at(output, &format!("(o{axis} + start)"));
        let src = io.read(1, &at(patch, &format!("o{axis}")));
        let store = io.store(&dst, &src, &|_| {
            unreachable!("cache writes have no epilogue")
        });
        return (
            format!("{setup}\nlet start = {start};\n{store}"),
            patch.elements(),
        );
    }
    let patch_value = io.read(1, &at(patch, &format!("(o{axis} - start)")));
    let old_value = io.read(0, &gather("k", output, cache));
    let len = patch.shape[axis];
    let store = io.store_k("value");
    let body = format!(
        "{setup}
         let start = {start};
         var value: f32;
         if (o{axis} >= start && o{axis} - start < {len}u) {{
             value = {patch_value};
         }} else {{
             value = {old_value};
         }}
         {store}"
    );
    (body, output.elements())
}

pub(super) fn sample(io: &Io) -> String {
    let (logits, indices) = (&io.node.inputs[0], &io.node.inputs[1]);
    let axis = logits.shape.len() - 1;
    let offset = |view: &View| {
        let terms: Vec<_> = view.strides[..axis]
            .iter()
            .enumerate()
            .map(|(a, &stride)| (format!("o{a}"), stride))
            .collect();
        linear(view.offset, &terms)
    };
    io.add_helper(EXTREMA);
    wgsl!(
        "sample_candidates.wgsl",
        coordinates = output_coordinates(io.output(), "o"),
        width = logits.shape[axis],
        logits_base = offset(logits),
        indices_base = offset(indices),
        seed = io.read(2, &format!("{}u", io.node.inputs[2].offset)),
        first = io.read(0, "base"),
        logit = io.read(0, &format!("base + i * {}u", logits.strides[axis])),
        selected = io.read(
            1,
            &format!("index_base + selected * {}u", indices.strides[axis])
        ),
        store = io.store_k("token"),
    )
}

/// Split-half rotary embedding, with the paired coordinate evaluated in
/// the logical source view even when the consumer reads a strided alias.
pub(super) fn rotary(io: &Io, output: &View, inputs: &[View], first: usize, index: &str) -> String {
    let axis = output.shape.len() - 1;
    let depth = output.shape[axis];
    let half = depth / 2;
    let stride = output.strides[axis];
    let d = coordinate(index, stride, depth);
    let upper = format!("{d} >= {half}u");
    let distance = half * stride;
    let paired = format!("(select({index} + {distance}u, {index} - {distance}u, {upper}))");
    let read = |input: usize, position: &str| {
        io.read(first + input, &gather(position, output, &inputs[input]))
    };
    let x = read(0, index);
    let y = read(0, &paired);
    let cosine = read(1, index);
    let sine = read(2, index);
    format!("({x} * {cosine} + select(-{y}, {y}, {upper}) * {sine})")
}
