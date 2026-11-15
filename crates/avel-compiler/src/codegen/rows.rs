use super::reduction::{row_workgroups, workgroup_sum};
use super::*;
use crate::ops::{Extremum, RowKind};

/// The row and within-row indices follow logical axis order, independently
/// of the input and output layouts. This also defines flattened argmax ties.
struct Rows {
    kept: Vec<usize>,
    axes: Vec<usize>,
    shape: Vec<u64>,
    depth: u64,
    count: u64,
}

impl Rows {
    fn new(shape: &[u64], axes: &[usize]) -> Self {
        let kept = (0..shape.len()).filter(|a| !axes.contains(a)).collect();
        let depth = axes.iter().map(|&a| shape[a]).product();
        Self {
            kept,
            axes: axes.to_vec(),
            shape: shape.to_vec(),
            depth,
            count: shape.iter().product::<u64>() / depth,
        }
    }

    fn coords(&self, axes: &[usize], index: &str) -> impl Iterator<Item = (usize, String)> {
        let shape: Vec<_> = axes.iter().map(|&a| self.shape[a]).collect();
        let strides = row_major_strides(&shape);
        axes.iter().copied().zip(
            shape
                .into_iter()
                .zip(strides)
                .map(move |(size, stride)| coordinate(index, stride, size)),
        )
    }

    fn base(&self, view: &View) -> String {
        let terms: Vec<_> = self
            .coords(&self.kept, "k")
            .enumerate()
            .map(|(i, (axis, coord))| {
                let axis = if view.shape.len() == self.shape.len() {
                    axis
                } else {
                    i
                };
                (coord, view.strides[axis])
            })
            .collect();
        linear(view.offset, &terms)
    }

    fn position(&self, view: &View) -> String {
        let terms: Vec<_> = self
            .coords(&self.axes, "r")
            .map(|(axis, coord)| (coord, view.strides[axis]))
            .collect();
        linear(0, &terms)
    }
}

pub(super) fn row_kernel(
    io: Io,
    axes: &[usize],
    kind: RowKind,
    strategy: Strategy,
) -> (String, [u32; 3]) {
    let node = io.node;
    let input = &node.inputs[0];
    let output = &node.output;
    let rows = Rows::new(&input.shape, axes);
    let depth = rows.depth;
    let base = rows.base(input);
    let read = io.read(0, &format!("base + {}", rows.position(input)));
    let out_base = rows.base(output);
    let out_pos = || format!("out_base + {}", rows.position(output));
    let store_row =
        |value: &str| io.store(&out_pos(), value, &|view| gather(&out_pos(), output, view));
    let (first, step, lanes) = match strategy {
        Strategy::Serial => ("0u", 1, None),
        Strategy::Workgroup { lanes, .. } => ("lane", lanes, Some(lanes)),
    };

    let body = match kind {
        RowKind::Softmax { log } => {
            io.add_helper(FLOAT_BITS);
            io.add_helper(EXTREMA);
            let (maximum, sum) = if let Some(lanes) = lanes {
                io.add_helper(&tree(lanes, "f32", "avel_max"));
                io.add_helper(&workgroup_sum(lanes, false));
                ("row_merge(maximum, lane)", "workgroup_sum(sum, lane)")
            } else {
                ("maximum", "sum")
            };
            let result = if log {
                format!("({read} - row_maximum) - log_sum")
            } else {
                format!("exp({read} - row_maximum) * inv_sum")
            };
            wgsl!(
                "softmax.wgsl",
                base = base,
                out_base = out_base,
                depth = depth,
                first = first,
                step = step,
                read = read,
                maximum = maximum,
                sum = sum,
                finish = if log {
                    "let log_sum = log(row_sum);"
                } else {
                    "let inv_sum = 1.0 / row_sum;"
                },
                result = result,
                store = store_row("result"),
            )
        }
        RowKind::Extrema {
            max,
            output: result,
        } => {
            let ty = io.ty(0);
            let prefer = match (ty, max) {
                (ScalarType::Bool, true) => "b.value && !a.value",
                (ScalarType::Bool, false) => "!b.value && a.value",
                (_, true) => "b.value > a.value",
                (_, false) => "b.value < a.value",
            };
            let nan = if ty == ScalarType::F32 {
                io.add_helper(EXTREMA);
                "if (avel_is_nan(a.value) || avel_is_nan(b.value)) {
                    if (!avel_is_nan(a.value) || (avel_is_nan(b.value) && b.index < a.index)) { return b; }
                    return a;
                }"
            } else {
                ""
            };
            io.add_helper(&wgsl!(
                "row_extrema.wgsl",
                ty = ty.wgsl(),
                prefer = prefer,
                nan = nan,
            ));
            let zero = match ty {
                ScalarType::F32 => "0.0",
                ScalarType::I32 => "0",
                ScalarType::Bool => "false",
            };
            let value = if result == Extremum::Index {
                "i32(total.index)"
            } else {
                "total.value"
            };
            let second = io.store_second_k("i32(total.index)");
            let update = format!("acc = combine(acc, RowState({read}, r));");
            aggregate(
                &io,
                lanes,
                first,
                step,
                depth,
                &base,
                &out_base,
                &format!("RowState({zero}, 0xffffffffu)"),
                &update,
                value,
                &second,
            )
        }
        RowKind::Variance { correction } => {
            io.add_helper(include_str!("wgsl/welford.wgsl"));
            let update = format!("acc = welford_add(acc, {read});");
            let divisor = (depth as f64 - f64::from(correction)).max(0.0) as f32;
            let result = if divisor == 0.0 {
                io.add_helper(FLOAT_BITS);
                io.add_helper(EXTREMA);
                "select(avel_bits(0x7f800000u), avel_bits(0x7fc00000u), total.m2 == 0.0 || avel_is_nan(total.m2))".to_owned()
            } else {
                format!("total.m2 / {}", float(divisor))
            };
            aggregate(
                &io,
                lanes,
                first,
                step,
                depth,
                &base,
                &out_base,
                "RowState(0.0, 0.0, 0.0)",
                &update,
                &result,
                "",
            )
        }
        RowKind::Any | RowKind::Product { .. } => {
            let ty = match kind {
                RowKind::Product { dtype: Some(dtype) } => ScalarType::of(dtype),
                RowKind::Any => ScalarType::Bool,
                _ => unreachable!("lowering fixes the accumulator dtype"),
            };
            let (identity, operator) = match (kind, ty) {
                (RowKind::Any, _) => ("false", "||"),
                (_, ScalarType::Bool) => ("true", "&&"),
                (_, ScalarType::F32) => ("1.0", "*"),
                (_, ScalarType::I32) => ("1", "*"),
            };
            let read = match (ty, io.ty(0)) {
                (ScalarType::Bool, ScalarType::F32) => format!("({read} != 0.0)"),
                (ScalarType::Bool, ScalarType::I32) => format!("({read} != 0)"),
                _ => format!("{}({read})", ty.wgsl()),
            };
            io.add_helper(&format!(
                "alias RowState = {ty};
fn combine(a: RowState, b: RowState) -> RowState {{ return a {operator} b; }}",
                ty = ty.wgsl(),
            ));
            aggregate(
                &io,
                lanes,
                first,
                step,
                depth,
                &base,
                &out_base,
                identity,
                &format!("acc = combine(acc, {read});"),
                "total",
                "",
            )
        }
        RowKind::Scan { dtype } => {
            let ty = ScalarType::of(dtype.expect("lowering fixes the accumulator dtype"));
            let read = format!("{}({read})", ty.wgsl());
            match lanes {
                None => wgsl!(
                    "scan_serial.wgsl",
                    base = base,
                    out_base = out_base,
                    depth = depth,
                    zero = if ty == ScalarType::F32 { "0.0" } else { "0" },
                    read = read,
                    store = store_row("prefix"),
                ),
                Some(lanes) => {
                    io.add_helper(&wgsl!("scan_tree.wgsl", lanes = lanes, ty = ty.wgsl(),));
                    wgsl!(
                        "scan_workgroup.wgsl",
                        base = base,
                        out_base = out_base,
                        depth = depth,
                        lanes = lanes,
                        ty = ty.wgsl(),
                        read = read,
                        store = store_row("prefix"),
                    )
                }
            }
        }
    };
    match lanes {
        None => linear_dispatch(io, rows.count, &body),
        Some(lanes) => {
            let wgsl = wgsl!(
                "row_workgroup.wgsl",
                declarations = io.declarations(),
                helpers = io.helpers(),
                lanes = lanes,
                entry_point = ENTRY_POINT,
                max_workgroups = MAX_WORKGROUPS,
                rows = rows.count,
                body = body,
            );
            (wgsl, row_workgroups(rows.count, 1))
        }
    }
}

fn tree(lanes: u32, ty: &str, combine: &str) -> String {
    wgsl!(
        "row_tree.wgsl",
        lanes = lanes,
        half = lanes / 2,
        ty = ty,
        combine = combine,
    )
}

#[allow(clippy::too_many_arguments)]
fn aggregate(
    io: &Io,
    lanes: Option<u32>,
    first: &str,
    step: u32,
    depth: u64,
    base: &str,
    out_index: &str,
    identity: &str,
    update: &str,
    result: &str,
    second: &str,
) -> String {
    if let Some(lanes) = lanes {
        io.add_helper(&tree(lanes, "RowState", "combine"));
    }
    wgsl!(
        "row_aggregate.wgsl",
        first = first,
        step = step,
        depth = depth,
        base = base,
        out_index = out_index,
        identity = identity,
        update = update,
        total = if lanes.is_some() {
            "row_merge(acc, lane)"
        } else {
            "acc"
        },
        owner = if lanes.is_some() {
            "lane == 0u"
        } else {
            "true"
        },
        store = io.store_k(result),
        second = second,
    )
}
