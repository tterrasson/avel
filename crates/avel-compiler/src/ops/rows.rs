use super::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RowKind {
    Softmax { log: bool },
    Extrema { max: bool, output: Extremum },
    Variance { correction: f32 },
    Scan { dtype: Option<Dtype> },
    Product { dtype: Option<Dtype> },
    Any,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extremum {
    Value,
    Index,
    Both,
}

impl RowKind {
    pub fn keeps_extent(self) -> bool {
        matches!(self, Self::Softmax { .. } | Self::Scan { .. })
    }
}

pub(super) fn resolve_row(
    attributes: &Attributes,
    input: &[u64],
) -> Result<(Op, Vec<Vec<u64>>), String> {
    let op = attributes.node.op.as_str();
    let max = matches!(
        op,
        "aten.amax.default" | "aten.max.dim" | "aten.argmax.default"
    );
    let kind = match op {
        "aten._softmax.default" | "aten._log_softmax.default" => {
            if attributes.bool("half_to_float", false)? {
                return Err("half_to_float requires an f16 input dtype".to_owned());
            }
            RowKind::Softmax {
                log: op == "aten._log_softmax.default",
            }
        }
        "aten.cumsum.default" => RowKind::Scan {
            dtype: attributes.dtype("dtype")?,
        },
        "aten.prod.dim_int" => RowKind::Product {
            dtype: attributes.dtype("dtype")?,
        },
        "aten.any.dim" => RowKind::Any,
        "aten.var.correction" => RowKind::Variance {
            correction: attributes.scalar("correction")?.unwrap_or(1.0),
        },
        "aten.max.dim" | "aten.min.dim" => RowKind::Extrema {
            max,
            output: Extremum::Both,
        },
        "aten.argmax.default" | "aten.argmin.default" => RowKind::Extrema {
            max,
            output: Extremum::Index,
        },
        _ => RowKind::Extrema {
            max,
            output: Extremum::Value,
        },
    };
    let single_axis = kind.keeps_extent()
        || matches!(kind, RowKind::Product { .. } | RowKind::Any)
        || matches!(
            kind,
            RowKind::Extrema {
                output: Extremum::Both,
                ..
            }
        );
    let dims = if single_axis {
        vec![attributes.required_int("dim")?]
    } else {
        attributes.ints("dim")?.unwrap_or_default()
    };
    // PyTorch wraps -1 and 0 on a scalar as if it had a single unit axis.
    let mut axes = Vec::new();
    for dim in dims {
        let axis = resolve_axis(dim, input.len().max(1))?;
        if axes.contains(&axis) {
            return Err(format!("axis {dim} is repeated"));
        }
        axes.push(axis);
    }
    if axes.is_empty() {
        axes.extend(0..input.len());
    } else if input.is_empty() {
        axes.clear();
    }
    axes.sort_unstable();

    let keepdim = attributes.bool("keepdim", false)?;
    let shape = if kind.keeps_extent() {
        input.to_vec()
    } else {
        input
            .iter()
            .enumerate()
            .filter_map(|(axis, &size)| {
                if axes.contains(&axis) {
                    keepdim.then_some(1)
                } else {
                    Some(size)
                }
            })
            .collect()
    };
    let shapes = if matches!(
        kind,
        RowKind::Extrema {
            output: Extremum::Both,
            ..
        }
    ) {
        vec![shape.clone(), shape]
    } else {
        vec![shape]
    };
    Ok((Op::Row { kind, axes }, shapes))
}
