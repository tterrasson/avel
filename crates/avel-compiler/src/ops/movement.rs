use super::*;

#[derive(Debug, Clone, PartialEq)]
pub enum Movement {
    Repeat,
    Flip(Vec<usize>),
    Roll {
        shifts: Vec<(usize, u64)>,
        flattened: bool,
    },
    Pad(Padding),
    IndexSelect(usize),
    Gather(usize),
    /// Indexed input axes, and where their broadcast shape starts in the result.
    Index {
        axes: Vec<usize>,
        start: usize,
        rank: usize,
    },
    Resize(Resize),
    AdaptiveAverage,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Padding {
    /// (left, right) for each input axis, including leading unpadded axes.
    pub widths: Vec<(i64, i64)>,
    pub mode: PadMode,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PadMode {
    Constant(Number),
    Reflect,
    Replicate,
    Circular,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Resize {
    pub mode: ResizeMode,
    /// Explicit scale factors, or None when inferred from the two sizes.
    pub scales: Option<[f64; 2]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeMode {
    Nearest,
    Bilinear { align_corners: bool },
    Bicubic { align_corners: bool },
}

pub(super) fn resolve_movement(
    attributes: &Attributes,
    shapes: &[&[u64]],
) -> Result<(Op, Vec<u64>), String> {
    let input = shapes[0];
    let rank = input.len();
    let (kind, shape) = match attributes.node.op.as_str() {
        "aten.repeat.default" => {
            let repeats = attributes.ints("repeats")?.ok_or("missing repeats")?;
            if repeats.len() < rank || repeats.iter().any(|&n| n < 1) {
                return Err("repeat needs positive repeats for at least all input axes".to_owned());
            }
            let lead = repeats.len() - rank;
            let shape = repeats
                .iter()
                .enumerate()
                .map(|(axis, &n)| {
                    let size = axis.checked_sub(lead).map_or(1, |a| input[a]);
                    size.checked_mul(n as u64)
                        .ok_or_else(|| "repeated shape overflows".to_owned())
                })
                .collect::<Result<_, _>>()?;
            (Movement::Repeat, shape)
        }
        "aten.flip.default" => {
            let axes = attributes.axes("dims", rank)?.ok_or("missing dims")?;
            (Movement::Flip(axes), input.to_vec())
        }
        "aten.roll.default" => {
            let shifts = attributes.ints("shifts")?.ok_or("missing shifts")?;
            let dims = attributes.ints("dims")?.unwrap_or_default();
            let flattened = dims.is_empty();
            let shifts = if flattened {
                let [shift] = shifts.as_slice() else {
                    return Err("flattened roll needs one shift".to_owned());
                };
                vec![(
                    0,
                    shift.rem_euclid(input.iter().product::<u64>() as i64) as u64,
                )]
            } else {
                if shifts.len() != dims.len() {
                    return Err("roll needs one shift per dim".to_owned());
                }
                let mut combined = vec![0u64; rank];
                for (shift, dim) in shifts.into_iter().zip(dims) {
                    let axis = resolve_axis(dim, rank.max(1))?;
                    if rank != 0 {
                        let size = input[axis];
                        combined[axis] =
                            (combined[axis] + shift.rem_euclid(size as i64) as u64) % size;
                    }
                }
                combined
                    .into_iter()
                    .enumerate()
                    .filter(|&(_, shift)| shift != 0)
                    .collect()
            };
            (Movement::Roll { shifts, flattened }, input.to_vec())
        }
        "aten.pad.default" | "aten.constant_pad_nd.default" => resolve_padding(attributes, input)?,
        "aten.index_select.default" | "aten.gather.default" => {
            let axis = resolve_axis(attributes.required_int("dim")?, rank.max(1))?;
            let index = shapes[1];
            if attributes.node.op == "aten.index_select.default" {
                if index.len() > 1 || (rank == 0 && index.iter().product::<u64>() != 1) {
                    return Err("index_select needs a scalar or vector index".to_owned());
                }
                let mut shape = input.to_vec();
                if rank != 0 {
                    shape[axis] = index.first().copied().unwrap_or(1);
                }
                (Movement::IndexSelect(axis), shape)
            } else {
                if attributes.bool("sparse_grad", false)? {
                    return Err("gather with sparse_grad is not supported".to_owned());
                }
                if index.len().max(1) != rank.max(1)
                    || index
                        .iter()
                        .zip(input)
                        .enumerate()
                        .any(|(a, (&size, &limit))| a != axis && size > limit)
                {
                    return Err(
                        "gather index rank and non-gather dimensions must fit the input".to_owned(),
                    );
                }
                (Movement::Gather(axis), index.to_vec())
            }
        }
        "aten.index.Tensor" => resolve_index(&attributes.node.inputs[1..], shapes)?,
        "aten.upsample_nearest2d.vec"
        | "aten.upsample_bilinear2d.vec"
        | "aten.upsample_bicubic2d.vec" => resolve_resize(attributes, input)?,
        "aten._adaptive_avg_pool2d.default" => (
            Movement::AdaptiveAverage,
            super::spatial::adaptive_pool_shape(attributes, input)?,
        ),
        _ => unreachable!(),
    };
    Ok((Op::Movement(kind), shape))
}

fn resolve_padding(attributes: &Attributes, input: &[u64]) -> Result<(Movement, Vec<u64>), String> {
    let pad = attributes.ints("pad")?.ok_or("missing pad")?;
    if pad.len() % 2 != 0 || pad.len() / 2 > input.len() {
        return Err("pad needs a pair per padded axis".to_owned());
    }
    let mode = match attributes.get("mode") {
        None => "constant",
        Some(Attr::Str(mode)) => mode.as_str(),
        _ => return Err("invalid padding mode".to_owned()),
    };
    let value = attributes.number("value")?.unwrap_or(Number::Int(0));
    let mode = match mode {
        "constant" => PadMode::Constant(value),
        "reflect" => PadMode::Reflect,
        "replicate" => PadMode::Replicate,
        "circular" => PadMode::Circular,
        _ => return Err(format!("unsupported padding mode {mode}")),
    };
    if !matches!(mode, PadMode::Constant(_)) {
        let spatial = pad.len() / 2;
        if !(1..=3).contains(&spatial) || !(spatial + 1..=spatial + 2).contains(&input.len()) {
            return Err(
                "non-constant padding needs one to three spatial axes and optional batch"
                    .to_owned(),
            );
        }
        if value.as_f64() != 0.0 {
            return Err("only constant padding takes a nonzero value".to_owned());
        }
    }
    let mut widths = vec![(0, 0); input.len()];
    let mut shape = input.to_vec();
    for (i, pair) in pad.as_chunks::<2>().0.iter().enumerate() {
        let axis = input.len() - i - 1;
        let size = input[axis] as i64;
        let (left, right) = (pair[0], pair[1]);
        if (mode == PadMode::Reflect && (left >= size || right >= size))
            || (mode == PadMode::Circular && (left > size || right > size))
        {
            return Err("reflect/circular pad is larger than its input".to_owned());
        }
        let out = size
            .checked_add(left)
            .and_then(|n| n.checked_add(right))
            .filter(|&n| n > 0 && n <= i64::from(i32::MAX))
            .ok_or("invalid padded dimension")?;
        // Shader coordinates subtract left before applying the boundary rule.
        if left <= i64::from(i32::MIN)
            || left > i64::from(i32::MAX)
            || out - 1 - left > i64::from(i32::MAX)
            || (mode == PadMode::Circular
                && left.max(right).max(0) > size + left.min(0) + right.min(0))
        {
            return Err("padding exceeds the supported coordinate range".to_owned());
        }
        widths[axis] = (left, right);
        shape[axis] = out as u64;
    }
    Ok((Movement::Pad(Padding { widths, mode }), shape))
}

pub(super) fn resolve_index(
    slots: &[Option<String>],
    shapes: &[&[u64]],
) -> Result<(Movement, Vec<u64>), String> {
    let input = shapes[0];
    if slots.len() > input.len() || slots.is_empty() {
        return Err(
            "index.Tensor needs a nonempty list with at most one index per axis".to_owned(),
        );
    }
    let axes: Vec<_> = slots
        .iter()
        .enumerate()
        .filter_map(|(axis, index)| index.as_ref().map(|_| axis))
        .collect();
    let Some(&first) = axes.first() else {
        return Err("index.Tensor needs an integer index tensor".to_owned());
    };
    let adjacent = axes.windows(2).all(|pair| pair[1] == pair[0] + 1);
    let start = if adjacent { first } else { 0 };
    let mut shape: Vec<_> = input
        .iter()
        .enumerate()
        .filter_map(|(a, &size)| (!axes.contains(&a)).then_some(size))
        .collect();
    let index_shape = broadcast_all(&shapes[1..])?;
    let rank = index_shape.len();
    shape.splice(start..start, index_shape);
    Ok((Movement::Index { axes, start, rank }, shape))
}

fn resolve_resize(attributes: &Attributes, input: &[u64]) -> Result<(Movement, Vec<u64>), String> {
    if input.len() != 4 {
        return Err("upsample2d expects [batch, channels, height, width]".to_owned());
    }
    let output = attributes.ints("output_size")?;
    let scales = match attributes.get("scale_factors") {
        None | Some(Attr::Null) => None,
        Some(Attr::List(items)) if items.len() == 2 => {
            let scale = |value: &Attr| match value {
                Attr::Float(x) if x.is_finite() && *x > 0.0 => Ok(*x),
                Attr::Int(x) if *x > 0 => Ok(*x as f64),
                _ => Err("invalid scale factor"),
            };
            Some([scale(&items[0])?, scale(&items[1])?])
        }
        _ => return Err("scale_factors must contain two positive numbers".to_owned()),
    };
    let spatial = match (output.as_deref(), scales) {
        (Some(&[h, w]), None) if h > 0 && w > 0 => [h as u64, w as u64],
        (None, Some([h, w])) => [(input[2] as f64 * h) as u64, (input[3] as f64 * w) as u64],
        _ => return Err("provide either output_size or scale_factors for two axes".to_owned()),
    };
    let align_corners = attributes.bool("align_corners", false)?;
    let mode = match attributes.node.op.as_str() {
        "aten.upsample_nearest2d.vec" => ResizeMode::Nearest,
        "aten.upsample_bilinear2d.vec" => ResizeMode::Bilinear { align_corners },
        _ => ResizeMode::Bicubic { align_corners },
    };
    let mut shape = input[..2].to_vec();
    shape.extend(spatial);
    Ok((Movement::Resize(Resize { mode, scales }), shape))
}
