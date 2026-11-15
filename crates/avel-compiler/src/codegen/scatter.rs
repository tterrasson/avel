use super::*;
use crate::ops::Scatter;

/// Each invocation owns a disjoint slice of the destination. It initializes
/// that slice and applies updates in index order, without atomics or races.
pub(super) fn scatter(io: &Io, spec: &Scatter) -> (String, u64) {
    let input = &io.node.inputs[0];
    let output = io.output();
    let (axes, accumulate) = match spec {
        Scatter::Along { axis, accumulate } => (std::slice::from_ref(axis), *accumulate),
        Scatter::Advanced {
            axes, accumulate, ..
        } => (axes.as_slice(), *accumulate),
    };
    let (sliced, kept): (Vec<_>, Vec<_>) = (0..output.shape.len()).partition(|a| axes.contains(a));
    let extents = |dims: &[usize]| dims.iter().map(|&a| output.shape[a]).collect::<Vec<_>>();
    let mut code = String::new();
    let row = extents(&kept);
    for ((&axis, stride), &size) in kept.iter().zip(row_major_strides(&row)).zip(&row) {
        writeln!(code, "let o{axis} = {};", coordinate("k", stride, size)).unwrap();
    }
    let shape = extents(&sliced);
    let size = shape.iter().product::<u64>();
    let coords = |axis: usize| format!("o{axis}");
    let element_at = |view: &View| at(view, (0..view.shape.len()).map(coords));
    writeln!(code, "for (var r = 0u; r < {size}u; r++) {{").unwrap();
    for ((&axis, stride), &size) in sliced.iter().zip(row_major_strides(&shape)).zip(&shape) {
        writeln!(code, "let o{axis} = {};", coordinate("r", stride, size)).unwrap();
    }
    let initial = io.read(0, &element_at(input));
    writeln!(
        code,
        "{}\n}}",
        io.store(&element_at(output), &initial, &|_| unreachable!())
    )
    .unwrap();
    let indices = &io.node.inputs[1];
    let mut guard = Vec::new();
    let src = match spec {
        Scatter::Along { .. } if input.shape.is_empty() => {
            guard.push(format!("{} == 0", io.read(1, &element_at(indices))));
            io.read(2, &element_at(&io.node.inputs[2]))
        }
        Scatter::Along { axis, .. } => {
            for &a in &kept {
                if indices.shape[a] != input.shape[a] {
                    writeln!(code, "if (o{a} >= {}u) {{ return; }}", indices.shape[a]).unwrap();
                }
            }
            writeln!(
                code,
                "for (var r = 0u; r < {}u; r++) {{",
                indices.shape[*axis]
            )
            .unwrap();
            writeln!(code, "let o{axis} = r;").unwrap();
            let index = io.read(1, &element_at(indices));
            let src = io.read(2, &element_at(&io.node.inputs[2]));
            writeln!(code, "let target{axis} = {index};").unwrap();
            guard.push(format!(
                "target{axis} >= 0 && u32(target{axis}) < {}u",
                input.shape[*axis]
            ));
            src
        }
        Scatter::Advanced {
            axes,
            start,
            rank,
            selection,
            ..
        } => {
            let index_shape = &selection[*start..*start + *rank];
            let size = index_shape.iter().product::<u64>();
            writeln!(code, "for (var r = 0u; r < {size}u; r++) {{").unwrap();
            let index_coords: Vec<_> = row_major_strides(index_shape)
                .into_iter()
                .zip(index_shape)
                .map(|(stride, &size)| coordinate("r", stride, size))
                .collect();
            for (i, &axis) in axes.iter().enumerate() {
                let index_view = &io.node.inputs[1 + i];
                let at = at(index_view, index_coords.iter().cloned());
                let index = io.read(1 + i, &at);
                let size = input.shape[axis];
                writeln!(code, "let index{axis} = {index};\nlet target{axis} = select(index{axis}, index{axis} + {size}, index{axis} < 0);").unwrap();
                guard.push(format!("target{axis} >= 0 && u32(target{axis}) < {size}u"));
            }
            let values = io.node.inputs.len() - 1;
            let mut coords: Vec<_> = kept.iter().map(|&a| coords(a)).collect();
            coords.splice(*start..*start, index_coords);
            io.read(values, &at(&io.node.inputs[values], coords.into_iter()))
        }
    };
    let destination = at(
        output,
        (0..output.shape.len()).map(|a| {
            if axes.contains(&a) {
                format!("u32(target{a})")
            } else {
                coords(a)
            }
        }),
    );
    let value = if accumulate {
        let old = io.output.load("dst", &destination);
        let op = if io.output.ty == ScalarType::Bool {
            "||"
        } else {
            "+"
        };
        format!("{old} {op} {src}")
    } else {
        src
    };
    writeln!(
        code,
        "if ({}) {{ {} }}",
        guard.join(" && "),
        io.store(&destination, &value, &|_| unreachable!())
    )
    .unwrap();
    if !input.shape.is_empty() {
        code.push('}');
    }
    (code, row.iter().product())
}

fn at(view: &View, coords: impl Iterator<Item = String>) -> String {
    let terms: Vec<_> = coords.zip(view.strides.iter().copied()).collect();
    linear(view.offset, &terms)
}
