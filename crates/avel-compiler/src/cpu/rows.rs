use super::*;
use crate::ops::{Extremum, ReduceKind, RowKind};

enum Operation {
    Reduce(ReduceKind),
    Row(RowKind),
    Normalize { eps: f32, rms: bool, weight: bool },
}

pub(super) struct Rows {
    op: Operation,
    outer: Index,
    inner: Index,
    output_axes: Option<(Index, Index)>,
}

impl Rows {
    fn new(op: Operation, axes: &[usize], input: &Tensor, output: Option<&Tensor>) -> Self {
        let kept: Vec<_> = (0..input.index.shape.len())
            .filter(|axis| !axes.contains(axis))
            .collect();
        let project = |tensor: &Tensor, axes: &[usize]| {
            let shape: Vec<_> = axes.iter().map(|&i| tensor.index.shape[i] as u64).collect();
            let strides: Vec<_> = axes
                .iter()
                .map(|&i| tensor.index.strides[i] as u64)
                .collect();
            Index::new(&shape, &strides)
        };
        Self {
            op,
            outer: project(input, &kept),
            inner: project(input, axes),
            output_axes: output.map(|tensor| (project(tensor, &kept), project(tensor, axes))),
        }
    }

    pub(super) fn reduce(kind: ReduceKind, axes: &[usize], input: &Tensor) -> Self {
        Self::new(Operation::Reduce(kind), axes, input, None)
    }

    pub(super) fn row(kind: RowKind, axes: &[usize], input: &Tensor, output: &Tensor) -> Self {
        Self::new(
            Operation::Row(kind),
            axes,
            input,
            kind.keeps_extent().then_some(output),
        )
    }

    pub(super) fn norm(
        eps: f32,
        rank: usize,
        rms: bool,
        weight: bool,
        input: &Tensor,
        output: &Tensor,
    ) -> Self {
        let axes: Vec<_> = (input.index.shape.len() - rank..input.index.shape.len()).collect();
        Self::new(
            Operation::Normalize { eps, rms, weight },
            &axes,
            input,
            Some(output),
        )
    }

    pub(super) fn run(
        &self,
        memory: &mut Memory,
        inputs: &[Tensor],
        output: &Tensor,
        second: Option<&Tensor>,
    ) {
        let input = &inputs[0];
        let depth = self.inner.count;
        for row in 0..self.outer.count {
            let base = self.outer.at(row);
            let read = |memory: &Memory, i| input.read(memory, base + self.inner.at(i));
            let float = |memory: &Memory, i| f32::from_bits(read(memory, i));
            let store_at = |i| {
                let (outer, inner) = self.output_axes.as_ref().expect("row has element outputs");
                outer.at(row) + inner.at(i)
            };
            let result = match self.op {
                Operation::Reduce(kind) => {
                    let sum = (0..depth).map(|i| float(memory, i)).sum::<f32>();
                    match kind {
                        ReduceKind::Sum => sum.to_bits(),
                        ReduceKind::Mean => (sum / depth as f32).to_bits(),
                    }
                }
                Operation::Row(RowKind::Softmax { log }) => {
                    let max = (0..depth)
                        .map(|i| float(memory, i))
                        .fold(f32::NEG_INFINITY, f32::max);
                    let sum = (0..depth)
                        .map(|i| (float(memory, i) - max).exp())
                        .sum::<f32>();
                    let ln = if log { sum.ln() } else { 0.0 };
                    for i in 0..depth {
                        let centered = float(memory, i) - max;
                        let value = if log {
                            centered - ln
                        } else {
                            centered.exp() / sum
                        };
                        output.write(memory, store_at(i), value.to_bits());
                    }
                    continue;
                }
                Operation::Row(RowKind::Extrema { max, output: kind }) => {
                    let mut best = read(memory, 0);
                    let mut index = 0;
                    for i in 1..depth {
                        let value = read(memory, i);
                        if better(value, best, input.dtype, max) {
                            best = value;
                            index = i;
                        }
                    }
                    if let Some(second) = second {
                        second.put(memory, row, index as u32);
                    }
                    match kind {
                        Extremum::Index => index as u32,
                        _ if input.dtype == Dtype::Bool => u32::from(best != 0),
                        _ => best,
                    }
                }
                Operation::Row(RowKind::Variance { correction }) => {
                    let mean = (0..depth).map(|i| float(memory, i)).sum::<f32>() / depth as f32;
                    let sum = (0..depth)
                        .map(|i| {
                            let delta = float(memory, i) - mean;
                            delta * delta
                        })
                        .sum::<f32>();
                    (sum / (depth as f32 - correction).max(0.0)).to_bits()
                }
                Operation::Row(RowKind::Any) => {
                    u32::from((0..depth).any(|i| truth(read(memory, i), input.dtype)))
                }
                Operation::Row(RowKind::Product { .. }) => {
                    let mut result = match output.dtype {
                        Dtype::F32 => 1.0f32.to_bits(),
                        _ => 1,
                    };
                    for i in 0..depth {
                        let value = convert(read(memory, i), input.dtype, output.dtype);
                        result = match output.dtype {
                            Dtype::F32 => {
                                (f32::from_bits(result) * f32::from_bits(value)).to_bits()
                            }
                            _ => result.wrapping_mul(value),
                        };
                    }
                    result
                }
                Operation::Row(RowKind::Scan { .. }) => {
                    let mut sum = 0u32;
                    for i in 0..depth {
                        let value = convert(read(memory, i), input.dtype, output.dtype);
                        sum = match output.dtype {
                            Dtype::F32 => (f32::from_bits(sum) + f32::from_bits(value)).to_bits(),
                            _ => sum.wrapping_add(value),
                        };
                        output.write(memory, store_at(i), sum);
                    }
                    continue;
                }
                Operation::Normalize { eps, rms, weight } => {
                    let mean = if rms {
                        0.0
                    } else {
                        (0..depth).map(|i| float(memory, i)).sum::<f32>() / depth as f32
                    };
                    let variance = (0..depth)
                        .map(|i| {
                            let delta = float(memory, i) - mean;
                            delta * delta
                        })
                        .sum::<f32>()
                        / depth as f32;
                    let scale = (variance + eps).sqrt().recip();
                    for i in 0..depth {
                        let index = row * depth + i;
                        let mut value = (float(memory, i) - mean) * scale;
                        if weight {
                            value *= f32::from_bits(inputs[1].get(memory, index));
                        }
                        if let Some(bias) = inputs.get(1 + usize::from(weight)) {
                            value += f32::from_bits(bias.get(memory, index));
                        }
                        output.write(memory, store_at(i), value.to_bits());
                    }
                    continue;
                }
            };
            output.put(memory, row, result);
        }
    }
}

pub(super) fn better(value: u32, best: u32, dtype: Dtype, max: bool) -> bool {
    match dtype {
        Dtype::F32 => {
            let (value, best) = (f32::from_bits(value), f32::from_bits(best));
            !best.is_nan() && (value.is_nan() || if max { value > best } else { value < best })
        }
        Dtype::Bool => (value != 0) != (best != 0) && (value != 0) == max,
        _ if max => (value as i32) > best as i32,
        _ => (value as i32) < best as i32,
    }
}

fn truth(word: u32, dtype: Dtype) -> bool {
    if dtype == Dtype::F32 {
        f32::from_bits(word) != 0.0
    } else {
        word != 0
    }
}

fn convert(word: u32, from: Dtype, to: Dtype) -> u32 {
    match (from, to) {
        (_, Dtype::Bool) => u32::from(truth(word, from)),
        (Dtype::F32, Dtype::F32) => word,
        (Dtype::F32, _) => f32::from_bits(word) as i32 as u32,
        (Dtype::Bool, Dtype::F32) => f32::from(u8::from(word != 0)).to_bits(),
        (_, Dtype::F32) => (word as i32 as f32).to_bits(),
        (Dtype::Bool, _) => u32::from(word != 0),
        _ => word,
    }
}
