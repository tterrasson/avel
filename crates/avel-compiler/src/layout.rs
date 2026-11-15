//! Picks the memory layout of each computed tensor, per subgraph.
//!
//! Ops that read and write through strides (element-wise, convolution, pooling,
//! normalization, reductions that keep their rank) do not care about layout,
//! so tensors connected through them form a subgraph that shares one. A
//! subgraph with a convolution goes channels-last, which lets implicit GEMM
//! read neighbouring channels together; everything else stays row-major.
//! Transpositions only appear where a reshape cannot be expressed as a view.

use std::collections::HashMap;

use crate::ops::Op;
use crate::view::Layout;

/// A resolved node, as seen by the layout pass.
pub struct Producer<'a> {
    pub op: &'a Op,
    pub inputs: &'a [&'a str],
    pub output: &'a str,
    pub rank: usize,
}

fn layout_free(op: &Op, rank: usize, input_ranks: impl Iterator<Item = usize>) -> bool {
    match op {
        Op::Unary(_)
        | Op::Binary { .. }
        | Op::Power { .. }
        | Op::Cast(_)
        | Op::Compare { .. }
        | Op::Where
        | Op::Bitwise(_)
        | Op::Arange { .. }
        | Op::Full { .. }
        | Op::BatchNorm { .. }
        | Op::GroupNorm { .. }
        | Op::Pool(_)
        | Op::Movement(_)
        | Op::Cat { .. } => true,
        Op::Convolution(_) => true,
        Op::Reduce { .. } | Op::Row { .. } => input_ranks.into_iter().all(|input| input == rank),
        // These write one row of output per invocation, so their output stays
        // row-major: layer norm, attention and embedding.
        Op::LayerNorm { .. } | Op::Attention { .. } | Op::Embedding { .. } => false,
        Op::Matmul { .. }
        | Op::View(_)
        | Op::Custom(_)
        | Op::Dequantize(_)
        | Op::Order { .. }
        | Op::Scatter(_)
        | Op::Rotary
        | Op::CacheWrite
        | Op::CachedAttention { .. }
        | Op::SampleCandidates => false,
    }
}

/// Returns the layout of every node output, in node order.
pub fn choose(nodes: &[Producer], rank_of: impl Fn(&str) -> usize) -> Vec<Layout> {
    let index: HashMap<&str, usize> = nodes
        .iter()
        .enumerate()
        .map(|(position, node)| (node.output, position))
        .collect();
    let free: Vec<bool> = nodes
        .iter()
        .map(|node| {
            node.rank >= 3
                && layout_free(
                    node.op,
                    node.rank,
                    node.inputs.iter().map(|name| rank_of(name)),
                )
        })
        .collect();

    let mut parent: Vec<usize> = (0..nodes.len()).collect();
    fn root(parent: &mut [usize], mut node: usize) -> usize {
        while parent[node] != node {
            parent[node] = parent[parent[node]];
            node = parent[node];
        }
        node
    }
    for (position, node) in nodes.iter().enumerate() {
        if !free[position] {
            continue;
        }
        for input in node.inputs {
            if let Some(&producer) = index.get(input)
                && free[producer]
                && nodes[producer].rank == node.rank
            {
                let (a, b) = (root(&mut parent, position), root(&mut parent, producer));
                parent[a] = b;
            }
        }
    }

    let mut convolutions = vec![false; nodes.len()];
    for (position, node) in nodes.iter().enumerate() {
        if free[position] && matches!(node.op, Op::Convolution(_)) {
            let component = root(&mut parent, position);
            convolutions[component] = true;
        }
    }
    nodes
        .iter()
        .enumerate()
        .map(
            |(position, _)| match free[position] && convolutions[root(&mut parent, position)] {
                true => Layout::ChannelsLast,
                false => Layout::RowMajor,
            },
        )
        .collect()
}
