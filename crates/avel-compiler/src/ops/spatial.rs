use super::*;

pub(super) fn resolve_convolution(
    attributes: &Attributes,
    shapes: &[&[u64]],
    bias: bool,
) -> Result<(Op, Vec<u64>), String> {
    let (input, weight) = (shapes[0], shapes[1]);
    let transposed = attributes.bool("transposed", false)?;
    let spatial = input.len().saturating_sub(2);
    if !(1..=3).contains(&spatial) || weight.len() != input.len() {
        return Err(format!(
            "expected [batch, channels, ...] input and weight with 1 to 3 spatial axes, got {input:?} and {weight:?}"
        ));
    }
    let groups = attributes.int("groups")?.unwrap_or(1);
    let groups = u64::try_from(groups)
        .ok()
        .filter(|&n| n > 0)
        .ok_or("groups must be positive")?;
    let (batch, channels) = (input[0], input[1]);
    let (matched, out_channels) = if transposed {
        (
            weight[0] == channels,
            weight[1]
                .checked_mul(groups)
                .ok_or("output channels overflow")?,
        )
    } else {
        (weight[1].checked_mul(groups) == Some(channels), weight[0])
    };
    if !matched || channels % groups != 0 || out_channels % groups != 0 {
        return Err(format!(
            "weight {weight:?} does not match input {input:?} with {groups} groups"
        ));
    }
    if bias && shapes[2] != [out_channels] {
        return Err(format!(
            "bias has shape {:?}, expected [{out_channels}]",
            shapes[2]
        ));
    }

    let convolution = Convolution {
        stride: attributes.spatial("stride", spatial, &vec![1; spatial])?,
        padding: attributes.spatial("padding", spatial, &vec![0; spatial])?,
        dilation: attributes.spatial("dilation", spatial, &vec![1; spatial])?,
        groups,
        bias,
        transposed,
        input_pad: None,
    };
    if convolution.stride.contains(&0) || convolution.dilation.contains(&0) {
        return Err("stride and dilation must be positive".to_owned());
    }
    let output_padding = attributes.spatial("output_padding", spatial, &vec![0; spatial])?;
    let mut shape = vec![batch, out_channels];
    for axis in 0..spatial {
        let size = i128::from(input[2 + axis]);
        let pad = i128::from(convolution.padding[axis]);
        let stride = i128::from(convolution.stride[axis]);
        let dilation = i128::from(convolution.dilation[axis]);
        let extra = i128::from(output_padding[axis]);
        let window = dilation * i128::from(weight[2 + axis] - 1) + 1;
        let out = if transposed {
            if extra >= stride.max(dilation) {
                return Err("output_padding must be smaller than stride or dilation".to_owned());
            }
            (size - 1) * stride - 2 * pad + window + extra
        } else {
            if extra != 0 {
                return Err("output_padding requires a transposed convolution".to_owned());
            }
            if size + 2 * pad < window {
                return Err(format!("kernel {weight:?} is larger than input {input:?}"));
            }
            (size + 2 * pad - window) / stride + 1
        };
        // Bounds for the shader's specialized coordinate arithmetic.
        let max_coordinate = if transposed {
            out - 1 + pad
        } else {
            (out - 1) * stride + window - 1
        };
        if out <= 0
            || out > i128::from(u32::MAX)
            || [stride, dilation, window - 1, max_coordinate]
                .into_iter()
                .any(|v| v > i128::from(u32::MAX))
            || (!transposed
                && pad != 0
                && (pad > i128::from(i32::MAX) || max_coordinate > i128::from(i32::MAX)))
        {
            return Err("convolution spatial coordinates exceed the supported range".to_owned());
        }
        shape.push(out as u64);
    }
    Ok((Op::Convolution(convolution), shape))
}

pub(super) fn resolve_pool(
    attributes: &Attributes,
    input: &[u64],
    spatial: usize,
    max: bool,
) -> Result<(Op, Vec<u64>), String> {
    if !(spatial + 1..=spatial + 2).contains(&input.len()) {
        return Err(format!(
            "input has shape {input:?}, expected {spatial} spatial axes"
        ));
    }
    let kernel = attributes.spatial("kernel_size", spatial, &[])?;
    if kernel.len() != spatial || kernel.contains(&0) {
        return Err("invalid `kernel_size`".to_owned());
    }
    let stride = attributes.spatial("stride", spatial, &kernel)?;
    let padding = attributes.spatial("padding", spatial, &vec![0; spatial])?;
    let dilation = match max {
        true => attributes.spatial("dilation", spatial, &vec![1; spatial])?,
        false => vec![1; spatial],
    };
    let ceil_mode = attributes.bool("ceil_mode", false)?;
    let kind = match max {
        true => PoolKind::Max,
        false => PoolKind::Avg {
            count_include_pad: attributes.bool("count_include_pad", true)?,
            divisor: match attributes.int("divisor_override")? {
                Some(divisor) if divisor <= 0 => {
                    return Err(format!("invalid `divisor_override` {divisor}"));
                }
                divisor => divisor.map(|divisor| divisor as u64),
            },
        },
    };
    if stride.contains(&0) || dilation.contains(&0) {
        return Err("stride and dilation must be positive".to_owned());
    }

    let lead = input.len() - spatial;
    let mut shape = input[..lead].to_vec();
    for axis in 0..spatial {
        let (size, pad) = (input[lead + axis], padding[axis]);
        let window = dilation[axis]
            .checked_mul(kernel[axis] - 1)
            .and_then(|n| n.checked_add(1))
            .ok_or("pool window overflows")?;
        let padded = pad
            .checked_mul(2)
            .and_then(|n| size.checked_add(n))
            .ok_or("pool padding overflows")?;
        let tail = if ceil_mode { stride[axis] - 1 } else { 0 };
        let span = padded
            .checked_add(tail)
            .and_then(|size| size.checked_sub(window));
        let Some(span) = span.filter(|_| pad <= kernel[axis] / 2) else {
            return Err(format!(
                "invalid window {kernel:?} with padding {padding:?}"
            ));
        };
        // The last window must start inside the input or the left padding.
        let mut out = span / stride[axis] + 1;
        if ceil_mode && (out - 1).saturating_mul(stride[axis]) >= size + pad {
            out -= 1;
        }
        let last =
            (out.saturating_sub(1) as u128) * (stride[axis] as u128) + u128::from(window - 1);
        if [padded, stride[axis], dilation[axis]]
            .iter()
            .any(|&n| n > i32::MAX as u64)
            || last > i32::MAX as u128
        {
            return Err("pool window coordinates exceed the supported range".to_owned());
        }
        shape.push(out);
    }
    let pool = Pool {
        kind,
        kernel,
        stride,
        padding,
        dilation,
    };
    Ok((Op::Pool(pool), shape))
}

pub(super) fn adaptive_pool_shape(
    attributes: &Attributes,
    input: &[u64],
) -> Result<Vec<u64>, String> {
    let rank = input.len();
    if !(3..=4).contains(&rank) {
        return Err("adaptive pooling needs 2 spatial axes".to_owned());
    }
    let spatial = attributes.spatial("output_size", 2, &[])?;
    if spatial.len() != 2 || spatial.contains(&0) {
        return Err("expected a positive output_size".to_owned());
    }
    for (&size, &out) in input[rank - 2..].iter().zip(&spatial) {
        if size != out && size.saturating_mul(out) > u32::MAX as u64 {
            return Err("adaptive pooling window arithmetic exceeds 32 bits".to_owned());
        }
    }
    Ok(input[..rank - 2].iter().copied().chain(spatial).collect())
}

pub(super) fn resolve_adaptive_max_pool(
    attributes: &Attributes,
    input: &[u64],
) -> Result<(Op, Vec<u64>), String> {
    let shape = adaptive_pool_shape(attributes, input)?;
    let pool = Pool {
        kind: PoolKind::AdaptiveMax,
        kernel: Vec::new(),
        stride: Vec::new(),
        padding: Vec::new(),
        dilation: Vec::new(),
    };
    Ok((Op::Pool(pool), shape))
}
