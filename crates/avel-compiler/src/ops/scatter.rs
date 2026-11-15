use super::*;

#[derive(Debug, Clone, PartialEq)]
pub enum Scatter {
    Along {
        axis: usize,
        accumulate: bool,
    },
    Advanced {
        axes: Vec<usize>,
        start: usize,
        rank: usize,
        selection: Vec<u64>,
        accumulate: bool,
    },
}

pub(super) fn resolve_scatter(
    attrs: &Attributes,
    shapes: &[&[u64]],
) -> Result<(Op, Vec<u64>), String> {
    let (input, index, src) = (shapes[0], shapes[1], shapes[2]);
    let axis = resolve_axis(attrs.required_int("dim")?, input.len().max(1))?;
    if index.len() != input.len()
        || src.len() != input.len()
        || index
            .iter()
            .enumerate()
            .any(|(a, &size)| size > src[a] || (a != axis && size > input[a]))
    {
        return Err(
            "scatter index must fit the source and non-scatter axes of the destination".into(),
        );
    }
    let accumulate = attrs.node.op == "aten.scatter_add.default";
    Ok((
        Op::Scatter(Scatter::Along { axis, accumulate }),
        input.to_vec(),
    ))
}

pub(super) fn resolve_index_put(
    attrs: &Attributes,
    shapes: &[&[u64]],
) -> Result<(Op, Vec<u64>), String> {
    let last = attrs.node.inputs.len() - 1;
    let (index, selection) =
        movement::resolve_index(&attrs.node.inputs[1..last], &shapes[..shapes.len() - 1])?;
    let Movement::Index { axes, start, rank } = index else {
        unreachable!()
    };
    if !broadcasts_to(shapes[shapes.len() - 1], &selection) {
        return Err("index_put values do not broadcast to the indexed shape".into());
    }
    checked_elements(&selection)?;
    let accumulate = attrs.bool("accumulate", false)?;
    Ok((
        Op::Scatter(Scatter::Advanced {
            axes,
            start,
            rank,
            selection,
            accumulate,
        }),
        shapes[0].to_vec(),
    ))
}
