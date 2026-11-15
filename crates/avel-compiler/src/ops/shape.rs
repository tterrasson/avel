use super::*;

pub(super) fn resolve_permute(
    attributes: &Attributes,
    shape: &[u64],
) -> Result<(Op, Vec<u64>), String> {
    let dims = attributes
        .axes("dims", shape.len())?
        .ok_or("missing `dims`")?;
    if dims.len() != shape.len() {
        return Err(format!("`dims` {dims:?} does not match shape {shape:?}"));
    }
    let output = dims.iter().map(|&axis| shape[axis]).collect();
    Ok((Op::View(ViewOp::Permute(dims)), output))
}

pub(super) fn resolve_reduce(
    attributes: &Attributes,
    kind: ReduceKind,
    shape: &[u64],
) -> Result<(Op, Vec<u64>), String> {
    // No axis at all means every axis, as in PyTorch.
    let axes = match attributes.axes("dim", shape.len())? {
        Some(axes) if !axes.is_empty() => axes,
        _ => (0..shape.len()).collect(),
    };
    let keepdim = attributes.bool("keepdim", false)?;
    let output = shape
        .iter()
        .enumerate()
        .filter_map(|(axis, &size)| match axes.contains(&axis) {
            false => Some(size),
            true => keepdim.then_some(1),
        })
        .collect();
    Ok((Op::Reduce { kind, axes }, output))
}

/// Reads a target shape where one entry may be -1, inferred from `elements`.
fn target_shape(sizes: &[i64], elements: u64) -> Result<Vec<u64>, String> {
    let inferred = sizes.iter().filter(|&&size| size == -1).count();
    if inferred > 1 || sizes.iter().any(|&size| size < -1) {
        return Err(format!("invalid size {sizes:?}"));
    }
    let known: u64 = sizes
        .iter()
        .filter(|&&size| size >= 0)
        .map(|&size| size as u64)
        .try_fold(1u64, |n, size| n.checked_mul(size))
        .ok_or("view shape overflows")?;
    if inferred != 0 && known == 0 {
        return Err("cannot infer a size from an empty shape".to_owned());
    }
    let shape: Vec<u64> = sizes
        .iter()
        .map(|&size| match size {
            -1 if known > 0 => elements / known,
            _ => size as u64,
        })
        .collect();
    if shape.iter().try_fold(1u64, |n, &size| n.checked_mul(size)) != Some(elements) {
        return Err(format!("size {sizes:?} does not have {elements} elements"));
    }
    Ok(shape)
}

pub(super) fn resolve_view(
    attributes: &Attributes,
    shape: &[u64],
) -> Result<(Op, Vec<u64>), String> {
    let sizes = attributes.ints("size")?.ok_or("missing `size`")?;
    let output = target_shape(&sizes, shape.iter().product())?;
    Ok((Op::View(ViewOp::Reshape), output))
}

pub(super) fn resolve_expand(
    attributes: &Attributes,
    shape: &[u64],
) -> Result<(Op, Vec<u64>), String> {
    let sizes = attributes.ints("size")?.ok_or("missing `size`")?;
    if sizes.len() < shape.len() {
        return Err(format!("cannot expand {shape:?} to {sizes:?}"));
    }
    let lead = sizes.len() - shape.len();
    let output = sizes
        .iter()
        .enumerate()
        .map(|(axis, &size)| match (axis.checked_sub(lead), size) {
            (Some(own), -1) => Ok(shape[own]),
            (_, size) if size < 0 => Err(format!("invalid size {sizes:?}")),
            (Some(own), size) if shape[own] != 1 && shape[own] != size as u64 => {
                Err(format!("cannot expand {shape:?} to {sizes:?}"))
            }
            (_, size) => Ok(size as u64),
        })
        .collect::<Result<_, _>>()?;
    Ok((Op::View(ViewOp::Expand), output))
}

pub(super) fn resolve_slice(
    attributes: &Attributes,
    shape: &[u64],
) -> Result<(Op, Vec<u64>), String> {
    let axis = resolve_axis(attributes.int("dim")?.unwrap_or(0), shape.len())?;
    let size = shape[axis] as i64;
    let step = attributes.int("step")?.unwrap_or(1);
    if step < 1 {
        return Err(format!("step {step} must be positive"));
    }
    // Out-of-range bounds are clamped, as in PyTorch.
    let clamp = |bound: i64| {
        let bound = if bound < 0 { bound + size } else { bound };
        bound.clamp(0, size)
    };
    let start = clamp(attributes.int("start")?.unwrap_or(0));
    let end = clamp(attributes.int("end")?.unwrap_or(size)).max(start);
    let mut output = shape.to_vec();
    output[axis] = ((end - start) as u64).div_ceil(step as u64);
    let op = Op::View(ViewOp::Slice {
        axis,
        start: start as u64,
        step: step as u64,
    });
    Ok((op, output))
}

pub(super) fn resolve_select(
    attributes: &Attributes,
    shape: &[u64],
) -> Result<(Op, Vec<u64>), String> {
    let axis = resolve_axis(attributes.required_int("dim")?, shape.len())?;
    let index = attributes.required_int("index")?;
    let size = shape[axis] as i64;
    let resolved = if index < 0 { index + size } else { index };
    if !(0..size).contains(&resolved) {
        return Err(format!("index {index} is out of range for size {size}"));
    }
    let mut output = shape.to_vec();
    output.remove(axis);
    let op = Op::View(ViewOp::Select {
        axis,
        index: resolved as u64,
    });
    Ok((op, output))
}

pub(super) fn resolve_squeeze(
    attributes: &Attributes,
    shape: &[u64],
) -> Result<(Op, Vec<u64>), String> {
    // A scalar has no axis to squeeze, but accepts axis 0 and -1.
    let mut axes = attributes
        .axes("dim", shape.len().max(1))?
        .ok_or("missing `dim`")?;
    axes.retain(|&axis| shape.get(axis) == Some(&1));
    axes.sort_unstable();
    let output = shape
        .iter()
        .enumerate()
        .filter(|(axis, _)| !axes.contains(axis))
        .map(|(_, &size)| size)
        .collect();
    Ok((Op::View(ViewOp::Squeeze(axes)), output))
}

/// `split_with_sizes`: consecutive slices along `dim`, which `chunk` and
/// `split` also export to.
pub(super) fn resolve_split(
    attributes: &Attributes,
    shape: &[u64],
) -> Result<(Op, Vec<Vec<u64>>), String> {
    let axis = resolve_axis(attributes.int("dim")?.unwrap_or(0), shape.len())?;
    let sizes = attributes
        .ints("split_sizes")?
        .ok_or("missing `split_sizes`")?;
    let total = sizes
        .iter()
        .try_fold(0i64, |sum, &size| sum.checked_add(size));
    if sizes.iter().any(|&size| size <= 0) || total != Some(shape[axis] as i64) {
        return Err(format!(
            "sizes {sizes:?} do not split axis {axis} of {shape:?}"
        ));
    }
    let sizes: Vec<u64> = sizes.into_iter().map(|size| size as u64).collect();
    let outputs = sizes
        .iter()
        .map(|&size| {
            let mut output = shape.to_vec();
            output[axis] = size;
            output
        })
        .collect();
    Ok((Op::View(ViewOp::Split { axis, sizes }), outputs))
}
