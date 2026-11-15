use super::*;

/// `i64` values are stored narrowed to 32 bits.
fn fits_int(value: f64) -> bool {
    (i32::MIN as f64..=i32::MAX as f64).contains(&value)
}

/// `_to_copy`: a dtype conversion. Layout, device and memory format are
/// irrelevant to a plan.
pub(super) fn resolve_cast(
    attributes: &Attributes,
    shape: &[u64],
    dtype: Dtype,
) -> Result<(Op, Vec<u64>), String> {
    let target = attributes.dtype("dtype")?.unwrap_or(dtype);
    Ok((Op::Cast(target), shape.to_vec()))
}

/// `arange(start, end, step)`. Without a `dtype`, integer bounds give an
/// `i64` tensor and float ones an `f32` tensor, as in PyTorch.
pub(super) fn resolve_arange(attributes: &Attributes) -> Result<(Op, Vec<u64>), String> {
    let start = attributes.number("start")?.unwrap_or(Number::Int(0));
    let end = attributes.required_number("end")?;
    let step = attributes.number("step")?.unwrap_or(Number::Int(1));
    let float = [start, end, step]
        .iter()
        .any(|number| matches!(number, Number::Float(_)));
    let dtype = match attributes.dtype("dtype")? {
        Some(dtype) => dtype,
        None if float => Dtype::F32,
        None => Dtype::I64,
    };
    if dtype == Dtype::Bool {
        return Err("a bool `arange` is not supported".to_owned());
    }
    let (first, last, increment) = (start.as_f64(), end.as_f64(), step.as_f64());
    if increment == 0.0 || (last - first) * increment < 0.0 {
        return Err(format!("invalid range {first}..{last} by {increment}"));
    }
    let count = ((last - first) / increment).ceil();
    // The exclusive end may be outside the stored range. Check the first
    // and final generated values after integer conversion instead.
    let final_value = first.trunc() + increment.trunc() * (count - 1.0).max(0.0);
    if dtype.is_int() && (!fits_int(first.trunc()) || !fits_int(final_value)) {
        return Err(format!("range {first}..{last} does not fit in 32 bits"));
    }
    let op = Op::Arange { start, step, dtype };
    Ok((op, vec![count as u64]))
}

/// `full(size, value)`, `full_like(input, value)` and `scalar_tensor(value)`.
/// `like` is the shape and dtype of the input of `full_like`.
pub(super) fn resolve_full(
    attributes: &Attributes,
    like: Option<(&[u64], Dtype)>,
) -> Result<(Op, Vec<u64>), String> {
    let (value, shape) = match like {
        Some((shape, _)) => (attributes.required_number("fill_value")?, shape.to_vec()),
        None if attributes.get("size").is_some() => {
            let size = attributes.ints("size")?.unwrap_or_default();
            let shape = size
                .into_iter()
                .map(|size| u64::try_from(size).map_err(|_| format!("invalid size {size}")))
                .collect::<Result<_, _>>()?;
            (attributes.required_number("fill_value")?, shape)
        }
        // `scalar_tensor` defaults to the float dtype, whatever the scalar.
        None => {
            let value = match attributes.get("s") {
                Some(Attr::Str(value)) if value == "-inf" => f64::NEG_INFINITY,
                Some(Attr::Str(value)) if value == "inf" => f64::INFINITY,
                Some(Attr::Str(value)) if value == "nan" => f64::NAN,
                _ => attributes.required_number("s")?.as_f64(),
            };
            (Number::Float(value), Vec::new())
        }
    };
    let dtype = match (attributes.dtype("dtype")?, like) {
        (Some(dtype), _) => dtype,
        (None, Some((_, dtype))) => dtype,
        (None, None) => value.default_dtype(),
    };
    if !dtype.is_float() && !value.as_f64().is_finite() {
        return Err("non-finite scalar tensors require a float dtype".to_owned());
    }
    if dtype.is_int() && !fits_int(value.as_f64().trunc()) {
        return Err(format!("{} does not fit in 32 bits", value.as_f64()));
    }
    Ok((Op::Full { value, dtype }, shape))
}
