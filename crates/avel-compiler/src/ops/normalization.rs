use super::*;

pub(super) fn resolve_layer_norm(
    attributes: &Attributes,
    shapes: &[&[u64]],
    weight: bool,
    bias: bool,
) -> Result<(Op, Vec<u64>), String> {
    let normalized = normalized_shape(attributes, shapes, weight, bias)?;
    let eps = attributes.required_scalar("eps")?;
    let op = Op::LayerNorm {
        eps,
        weight,
        bias,
        normalized,
        rms: false,
    };
    Ok((op, shapes[0].to_vec()))
}

/// `rms_norm`, preserved by the exporter. Without `eps`, PyTorch uses the
/// machine epsilon of the input dtype.
pub(super) fn resolve_rms_norm(
    attributes: &Attributes,
    shapes: &[&[u64]],
    weight: bool,
) -> Result<(Op, Vec<u64>), String> {
    let normalized = normalized_shape(attributes, shapes, weight, false)?;
    let eps = attributes.scalar("eps")?.unwrap_or(f32::EPSILON);
    let op = Op::LayerNorm {
        eps,
        weight,
        bias: false,
        normalized,
        rms: true,
    };
    Ok((op, shapes[0].to_vec()))
}

/// Reads `normalized_shape` and checks the input and affine shapes against it.
fn normalized_shape(
    attributes: &Attributes,
    shapes: &[&[u64]],
    weight: bool,
    bias: bool,
) -> Result<Vec<u64>, String> {
    let input = shapes[0];
    let normalized: Vec<u64> = attributes
        .ints("normalized_shape")?
        .ok_or("missing `normalized_shape`")?
        .into_iter()
        .map(|size| {
            u64::try_from(size).map_err(|_| format!("invalid normalized shape size {size}"))
        })
        .collect::<Result<_, _>>()?;
    if normalized.is_empty() || normalized.contains(&0) {
        return Err(format!("invalid `normalized_shape` {normalized:?}"));
    }
    let lead = input.len().checked_sub(normalized.len()).ok_or_else(|| {
        format!("input has shape {input:?}, cannot normalize over {normalized:?}")
    })?;
    if input[lead..] != normalized[..] {
        return Err(format!(
            "`normalized_shape` {normalized:?} does not match the trailing dimensions of {input:?}"
        ));
    }
    let affine = [
        weight.then(|| (shapes[1], "weight")),
        bias.then(|| (shapes[1 + usize::from(weight)], "bias")),
    ];
    for (shape, name) in affine.into_iter().flatten() {
        if shape != &normalized[..] {
            return Err(format!(
                "{name} has shape {shape:?}, expected {normalized:?}"
            ));
        }
    }
    Ok(normalized)
}

pub(super) fn resolve_attention(
    attributes: &Attributes,
    shapes: &[&[u64]],
    dtypes: &[Dtype],
    mask: bool,
) -> Result<(Op, Vec<u64>), String> {
    let (query, key, value) = (shapes[0], shapes[1], shapes[2]);
    if query.len() < 3 || query.len() != key.len() || key.len() != value.len() {
        return Err(format!(
            "query, key and value must have the same rank of at least 3, got {query:?} {key:?} {value:?}"
        ));
    }
    let rank = query.len();
    let depth = query[rank - 1];
    let heads = query[rank - 3];
    let query_seq = query[rank - 2];
    let key_seq = key[rank - 2];
    if key[rank - 1] != depth {
        return Err("query and key depths differ".to_owned());
    }
    if key[rank - 3] != heads || value[rank - 3] != heads {
        return Err("head counts differ (`enable_gqa` is not supported)".to_owned());
    }
    if value[rank - 2] != key_seq {
        return Err("key and value sequence lengths differ".to_owned());
    }
    let mut outer = broadcast(&query[..rank - 3], &key[..rank - 3])
        .ok_or_else(|| format!("cannot broadcast {query:?} with {key:?}"))?;
    outer = broadcast(&outer, &value[..rank - 3])
        .ok_or_else(|| format!("cannot broadcast {outer:?} with {value:?}"))?;

    let mask_kind = match mask {
        true => {
            let target: Vec<u64> = outer
                .iter()
                .copied()
                .chain([heads, query_seq, key_seq])
                .collect();
            if !broadcasts_to(shapes[3], &target) {
                return Err(format!(
                    "cannot broadcast `attn_mask` {:?} to {target:?}",
                    shapes[3]
                ));
            }
            Some(match dtypes[3] {
                Dtype::Bool => MaskKind::Select,
                _ => MaskKind::Additive,
            })
        }
        false => None,
    };
    if attributes.scalar("dropout_p")?.unwrap_or(0.0) != 0.0 {
        return Err("dropout is not supported at inference".to_owned());
    }
    if attributes.bool("enable_gqa", false)? {
        return Err("`enable_gqa` is not supported".to_owned());
    }
    let causal = attributes.bool("is_causal", false)?;
    if causal && mask {
        return Err("`is_causal` with an explicit mask is not supported".to_owned());
    }
    let scale = attributes
        .scalar("scale")?
        .unwrap_or_else(|| 1.0 / (depth as f32).sqrt());
    let mut output = outer;
    output.extend([heads, query_seq, value[rank - 1]]);
    Ok((
        Op::Attention {
            scale,
            causal,
            mask: mask_kind,
        },
        output,
    ))
}

/// Group norm and training batch norm without running statistics. The latter
/// normalizes channels across batch and space, so we transpose N and C to
/// represent it as one group per channel, with one logical sample.
pub(super) fn resolve_group_norm(
    attributes: &Attributes,
    shapes: &[&[u64]],
    weight: bool,
    bias: bool,
) -> Result<(Op, Vec<u64>), String> {
    let input = shapes[0];
    if input.len() < 2 {
        return Err("expected input with batch and channel axes".to_owned());
    }
    let channels = input[1];
    let native = attributes.node.op == "aten.native_group_norm.default";
    let groups = if native {
        let spatial: u64 = input[2..].iter().product();
        for (attr, expected) in [("N", input[0]), ("C", channels), ("HxW", spatial)] {
            if attributes.required_int(attr)? != expected as i64 {
                return Err(format!("`{attr}` does not match the input shape"));
            }
        }
        attributes.required_int("group")?
    } else {
        if !attributes.bool("training", false)? {
            return Err("batch normalization without statistics requires training=true".to_owned());
        }
        channels as i64
    };
    if groups <= 0 || !channels.is_multiple_of(groups as u64) {
        return Err(format!(
            "{channels} channels cannot be split into {groups} groups"
        ));
    }
    if (weight && shapes[1] != [channels])
        || (bias && shapes[1 + usize::from(weight)] != [channels])
    {
        return Err(format!("weight and bias must have shape [{channels}]"));
    }
    let eps = attributes.required_scalar("eps")?;
    if eps < 0.0 {
        return Err("eps must be non-negative".to_owned());
    }
    Ok((
        Op::GroupNorm {
            eps,
            groups: groups as u64,
            weight,
            bias,
            across_batch: !native,
        },
        input.to_vec(),
    ))
}
