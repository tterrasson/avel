use super::*;
use crate::ops::{Movement, PadMode, Padding, Resize, ResizeMode};

pub(super) fn movement_body(io: &Io, kind: &Movement, fill: Option<Scalar>) -> String {
    let node = io.node;
    let input = &node.inputs[0];
    let output = &node.output;
    let mut body = output_coordinates(output, "o");
    let rank = input.shape.len();
    let mut terms: Vec<_> = (0..rank)
        .map(|axis| (format!("o{axis}"), input.strides[axis]))
        .collect();
    let mut guard = Vec::new();

    match kind {
        Movement::Repeat => {
            let lead = output.shape.len() - rank;
            for (axis, (coord, stride)) in terms.iter_mut().enumerate() {
                let size = input.shape[axis];
                let out = axis + lead;
                if size == 1 {
                    *stride = 0;
                } else if size == output.shape[out] {
                    *coord = format!("o{out}");
                } else {
                    *coord = format!("(o{out} % {size}u)");
                }
            }
        }
        Movement::Flip(axes) => {
            for &axis in axes {
                terms[axis].0 = format!("({}u - o{axis})", input.shape[axis] - 1);
            }
        }
        Movement::Roll { shifts, flattened } => {
            let roll = |coord: &str, size: u64, shift: u64| {
                // Branchless wrap without an integer divide.
                format!(
                    "select({coord} - {shift}u, {coord} + {}u, {coord} < {shift}u)",
                    size - shift
                )
            };
            if *flattened {
                let logical = View::dense(0, input.shape.clone(), crate::view::Layout::RowMajor);
                let logical_index = gather("k", output, &logical);
                let rolled = roll(
                    &format!("({logical_index})"),
                    logical.elements(),
                    shifts[0].1,
                );
                writeln!(body, "let rolled = {rolled};").unwrap();
                return body + &io.store_k(&io.read(0, &gather("rolled", &logical, input)));
            }
            for &(axis, shift) in shifts {
                terms[axis].0 = roll(&format!("o{axis}"), input.shape[axis], shift);
            }
        }
        Movement::Pad(pad) => {
            for (axis, &(left, right)) in pad.widths.iter().enumerate() {
                if left == 0 && right == 0 {
                    continue;
                }
                let size = input.shape[axis];
                let name = format!("p{axis}");
                let mapped = pad_position(
                    &mut body,
                    &mut guard,
                    &name,
                    &format!("o{axis}"),
                    size,
                    pad,
                    axis,
                );
                terms[axis].0 = mapped;
            }
        }
        Movement::IndexSelect(axis) | Movement::Gather(axis) => {
            let index = &node.inputs[1];
            let idx = if matches!(kind, Movement::Gather(_)) {
                gather("k", output, index)
            } else if index.shape.is_empty() || output.shape.is_empty() {
                format!("{}u", index.offset)
            } else {
                linear(index.offset, &[(format!("o{axis}"), index.strides[0])])
            };
            let size = input.shape.get(*axis).copied().unwrap_or(1);
            let mapped =
                index_position(&mut body, &mut guard, &io.read(1, &idx), *axis, size, false);
            if let Some(term) = terms.get_mut(*axis) {
                term.0 = mapped;
            }
            // gather([N], scalar) has scalar output, but the indexed term is
            // the only term so it does not reference an output coordinate.
        }
        Movement::Index {
            axes,
            start,
            rank: broadcast_rank,
        } => {
            let mut unindexed = 0;
            for (axis, term) in terms.iter_mut().enumerate() {
                if let Some(position) = axes.iter().position(|&a| a == axis) {
                    let index = node.inputs[position + 1]
                        .broadcast_to(&output.shape[*start..start + broadcast_rank]);
                    let index_terms: Vec<_> = index
                        .strides
                        .iter()
                        .enumerate()
                        .map(|(a, &stride)| (format!("o{}", start + a), stride))
                        .collect();
                    let value = io.read(position + 1, &linear(index.offset, &index_terms));
                    term.0 = index_position(
                        &mut body,
                        &mut guard,
                        &value,
                        axis,
                        input.shape[axis],
                        true,
                    );
                } else {
                    let a = if unindexed < *start {
                        unindexed
                    } else {
                        unindexed + broadcast_rank
                    };
                    term.0 = format!("o{a}");
                    unindexed += 1;
                }
            }
        }
        Movement::Resize(resize) => {
            let store = match resize.mode {
                ResizeMode::Nearest => {
                    return io.store_k(&io.read(0, &nearest_index("k", output, input, resize)));
                }
                ResizeMode::Bilinear { align_corners } => {
                    bilinear(io, resize, align_corners, &mut body)
                }
                ResizeMode::Bicubic { align_corners } => bicubic(io, resize, align_corners),
            };
            return body + &store;
        }
        Movement::AdaptiveAverage => {
            return body + &adaptive_average(io, &terms);
        }
    }
    let value = io.read(0, &linear(input.offset, &terms));
    if guard.is_empty() {
        return body + &io.store_k(&value);
    }
    let default = match fill {
        Some(Scalar::F32(value)) => float(value),
        Some(Scalar::I32(value)) => int(value),
        Some(Scalar::Bool(value)) => value.to_string(),
        None => format!("{}(0)", io.ty(0).wgsl()),
    };
    body + &wgsl!(
        "guarded_read.wgsl",
        default = default,
        valid = guard.join(" && "),
        read = value,
        store = io.store_k("value"),
    )
}

fn index_position(
    body: &mut String,
    guard: &mut Vec<String>,
    value: &str,
    axis: usize,
    size: u64,
    negative: bool,
) -> String {
    writeln!(body, "let idx{axis} = {value};").unwrap();
    let mapped = if negative {
        writeln!(
            body,
            "let idx{axis}_wrapped = select(idx{axis}, idx{axis} + {size}, idx{axis} < 0);"
        )
        .unwrap();
        format!("idx{axis}_wrapped")
    } else {
        format!("idx{axis}")
    };
    guard.push(format!("{mapped} >= 0 && {mapped} < {size}"));
    format!("u32({mapped})")
}

/// Map a coordinate in a padded tensor back to the unpadded input. Constant
/// padding adds a guard so its caller never loads out of bounds.
pub(super) fn pad_position(
    body: &mut String,
    guard: &mut Vec<String>,
    name: &str,
    coord: &str,
    size: u64,
    pad: &Padding,
    axis: usize,
) -> String {
    let (left, right) = pad.widths[axis];
    if left == 0 && right == 0 {
        return coord.to_owned();
    }
    writeln!(body, "let {name} = i32({coord}) - ({});", int(left as i32)).unwrap();
    match pad.mode {
        PadMode::Constant(_) => {
            guard.push(format!("{name} >= 0 && {name} < {size}"));
            format!("u32({name})")
        }
        PadMode::Reflect => format!(
            "u32(select(abs({name}), {} - ({name} - {}), {name} >= {size}))",
            size - 1,
            size - 1
        ),
        PadMode::Replicate => format!("u32(clamp({name}, 0, {}))", size - 1),
        PadMode::Circular => {
            let start = (-left).max(0);
            let end = size as i64 + right.min(0);
            let width = end - start;
            // PyTorch permits at most one wrap around the cropped interval.
            format!(
                "u32(select(select({name}, {name} - {width}, {name} >= {end}), {name} + {width}, {name} < {start}))"
            )
        }
    }
}

pub(super) fn nearest_index(index: &str, output: &View, input: &View, resize: &Resize) -> String {
    let terms: Vec<_> = (0..4)
        .map(|axis| {
            let out = output.shape[axis];
            let size = input.shape[axis];
            let coord = coordinate(index, output.strides[axis], out);
            let coord = if axis < 2 || size == out {
                coord
            } else if size == 1 {
                "0u".to_owned()
            } else if out == 2 * size {
                // The legacy nearest operator has an exact 2x fast path.
                format!("({coord} / 2u)")
            } else {
                // No integer shortcut for other ratios: PyTorch rounds the
                // f32 scale, so `o * (2 / 82)` floors to 0 at `o = 41`.
                let scale = resize
                    .scales
                    .map_or(size as f64 / out as f64, |scales| 1.0 / scales[axis - 2]);
                format!(
                    "min(u32(f32({coord}) * {}), {}u)",
                    float(scale as f32),
                    size - 1
                )
            };
            (coord, input.strides[axis])
        })
        .collect();
    linear(input.offset, &terms)
}

fn bilinear(io: &Io, resize: &Resize, align_corners: bool, body: &mut String) -> String {
    let input = &io.node.inputs[0];
    let output = &io.node.output;
    for (axis, name) in [(2, "y"), (3, "x")] {
        let (size, out) = (input.shape[axis], output.shape[axis]);
        let position = resize_position(resize, axis, size, out, align_corners, false);
        writeln!(
            body,
            "let {name} = min({position}, {});",
            float((size - 1) as f32)
        )
        .unwrap();
        writeln!(body, "let {name}0 = u32({name});").unwrap();
        writeln!(body, "let {name}1 = min({name}0 + 1u, {}u);", size - 1).unwrap();
        writeln!(body, "let d{name} = {name} - f32({name}0);").unwrap();
    }
    let read = |y: &str, x: &str| {
        io.read(
            0,
            &linear(
                input.offset,
                &[
                    ("o0".to_owned(), input.strides[0]),
                    ("o1".to_owned(), input.strides[1]),
                    (y.to_owned(), input.strides[2]),
                    (x.to_owned(), input.strides[3]),
                ],
            ),
        )
    };
    let value = format!(
        "mix(mix({}, {}, dx), mix({}, {}, dx), dy)",
        read("y0", "x0"),
        read("y0", "x1"),
        read("y1", "x0"),
        read("y1", "x1")
    );
    io.store_k(&value)
}

fn resize_position(
    resize: &Resize,
    axis: usize,
    size: u64,
    out: u64,
    align_corners: bool,
    cubic: bool,
) -> String {
    if size == out && !cubic {
        return format!("f32(o{axis})");
    }
    if align_corners {
        let scale = if out == 1 {
            0.0
        } else {
            (size - 1) as f64 / (out - 1) as f64
        };
        return format!("f32(o{axis}) * {}", float(scale as f32));
    }
    let scale = resize
        .scales
        .map_or(size as f64 / out as f64, |s| 1.0 / s[axis - 2]);
    let position = format!("(f32(o{axis}) + 0.5) * {} - 0.5", float(scale as f32));
    if cubic {
        position
    } else {
        format!("max({position}, 0.0)")
    }
}

fn bicubic(io: &Io, resize: &Resize, align_corners: bool) -> String {
    let input = &io.node.inputs[0];
    let output = &io.node.output;
    let position = |axis| {
        resize_position(
            resize,
            axis,
            input.shape[axis],
            output.shape[axis],
            align_corners,
            true,
        )
    };
    let read = io.read(
        0,
        &linear(
            input.offset,
            &[
                ("o0".to_owned(), input.strides[0]),
                ("o1".to_owned(), input.strides[1]),
                ("y".to_owned(), input.strides[2]),
                ("x".to_owned(), input.strides[3]),
            ],
        ),
    );
    io.add_helper(include_str!("wgsl/cubic_weights.wgsl"));
    wgsl!(
        "bicubic.wgsl",
        position_x = position(3),
        position_y = position(2),
        last_x = input.shape[3] - 1,
        last_y = input.shape[2] - 1,
        read = read,
        store = io.store_k("value"),
    )
}

fn adaptive_average(io: &Io, terms: &[(String, u64)]) -> String {
    let input = &io.node.inputs[0];
    let output = &io.node.output;
    let rank = input.shape.len();
    let (h, w) = (rank - 2, rank - 1);
    let window = |axis| {
        super::window::adaptive_bounds(input.shape[axis], output.shape[axis], &format!("o{axis}"))
    };
    let (y_start, y_end) = window(h);
    let (x_start, x_end) = window(w);
    let mut terms = terms.to_vec();
    terms[h].0 = "y".to_owned();
    terms[w].0 = "x".to_owned();
    let read = io.read(0, &linear(input.offset, &terms));
    wgsl!(
        "adaptive_average.wgsl",
        y_start = y_start,
        y_end = y_end,
        x_start = x_start,
        x_end = x_end,
        read = read,
        store = io.store_k("sum / f32((y_end - y_start) * (x_end - x_start))"),
    )
}
