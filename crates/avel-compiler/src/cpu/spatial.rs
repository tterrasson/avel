use super::*;
use crate::ops::{Convolution, Pool as PoolSpec, PoolKind};

pub(super) struct Matmul {
    alpha: f32,
    beta: f32,
    inner: usize,
}

impl Matmul {
    pub(super) fn new(beta: f32, alpha: f32, inputs: &[Tensor]) -> Self {
        Self {
            alpha,
            beta,
            inner: *inputs[0].index.shape.last().unwrap(),
        }
    }

    pub(super) fn run(&self, memory: &mut Memory, inputs: &[Tensor], output: &Tensor) {
        let (lhs, rhs) = (&inputs[0], &inputs[1]);
        let rank = output.index.shape.len();
        let m = output.index.shape[rank - 2];
        let n = output.index.shape[rank - 1];
        let (lhs_row, lhs_inner) = (lhs.index.strides[rank - 2], lhs.index.strides[rank - 1]);
        let (rhs_inner, rhs_col) = (rhs.index.strides[rank - 2], rhs.index.strides[rank - 1]);
        for batch in 0..output.index.count / (m * n) {
            let lhs_base = lhs.offset
                + if rank == 3 {
                    batch * lhs.index.strides[0]
                } else {
                    0
                };
            let rhs_base = rhs.offset
                + if rank == 3 {
                    batch * rhs.index.strides[0]
                } else {
                    0
                };
            for row in 0..m {
                for col in 0..n {
                    let a = &memory.read(lhs.buffer)[lhs_base + row * lhs_row..];
                    let b = &memory.read(rhs.buffer)[rhs_base + col * rhs_col..];
                    let product = if lhs_inner == 1 && rhs_inner == 1 {
                        dot(
                            a[..self.inner].iter().copied().map(f32::from_bits),
                            &b[..self.inner],
                        )
                    } else {
                        (0..self.inner)
                            .map(|k| {
                                f32::from_bits(a[k * lhs_inner]) * f32::from_bits(b[k * rhs_inner])
                            })
                            .sum()
                    };
                    let index = (batch * m + row) * n + col;
                    let mut result = self.alpha * product;
                    if let Some(bias) = inputs.get(2).filter(|_| self.beta != 0.0) {
                        result += self.beta * f32::from_bits(bias.get(memory, index));
                    }
                    output.put(memory, index, result.to_bits());
                }
            }
        }
    }
}

/// Independent accumulators keep the contiguous inner loop from serializing on
/// one sum. Both fully connected layers and convolution reuse this loop.
#[inline]
pub(super) fn dot(values: impl Iterator<Item = f32>, weights: &[u32]) -> f32 {
    let mut sums = [0.0; 4];
    let mut products = values.zip(weights).map(|(x, &w)| x * f32::from_bits(w));
    for _ in 0..weights.len() / sums.len() {
        for sum in &mut sums {
            *sum += products.next().unwrap();
        }
    }
    sums.into_iter().sum::<f32>() + products.sum::<f32>()
}

/// The valid input offset for each (output coordinate, filter coordinate).
/// This table is separable across the spatial axes, even for transpose.
struct ConvAxis {
    kernel: usize,
    positions: Vec<Option<usize>>,
}

pub(super) struct Conv {
    axes: Vec<ConvAxis>,
    output_spatial: Vec<usize>,
    spatial_offsets: Vec<Option<usize>>,
    patch: Vec<f32>,
    filter: Index,
    groups: usize,
    transposed: bool,
}

impl Conv {
    pub(super) fn new(
        conv: &Convolution,
        inputs: &[Tensor],
        output: &Tensor,
    ) -> Result<Self, CompileError> {
        let (image, weights) = (&inputs[0].index, &inputs[1].index);
        let groups = conv.groups as usize;
        let channels = image.shape[1] / groups;
        let mut axes = Vec::with_capacity(conv.stride.len());
        for axis in 0..conv.stride.len() {
            let kernel = weights.shape[axis + 2];
            let count = output.index.shape[axis + 2]
                .checked_mul(kernel)
                .ok_or(CompileError::CpuAllocation { bytes: usize::MAX })?;
            let mut positions = zeroed(count)?;
            for (o, window) in positions.chunks_exact_mut(kernel).enumerate() {
                for (k, slot) in window.iter_mut().enumerate() {
                    let stride = conv.stride[axis] as i64;
                    let dilation = conv.dilation[axis] as i64;
                    let padding = conv.padding[axis] as i64;
                    let input = if conv.transposed {
                        let position = o as i64 + padding - k as i64 * dilation;
                        if position % stride != 0 {
                            continue;
                        }
                        position / stride
                    } else {
                        o as i64 * stride + k as i64 * dilation - padding
                    };
                    if (0..image.shape[axis + 2] as i64).contains(&input) {
                        *slot = Some(input as usize * image.strides[axis + 2]);
                    }
                }
            }
            axes.push(ConvAxis { kernel, positions });
        }
        let mut shape = vec![channels as u64];
        shape.extend(weights.shape[2..].iter().map(|&x| x as u64));
        let mut strides = vec![weights.strides[usize::from(!conv.transposed)] as u64];
        strides.extend(weights.strides[2..].iter().map(|&x| x as u64));
        let filter = Index::new(&shape, &strides);
        Ok(Self {
            axes,
            output_spatial: output.index.shape[2..].to_vec(),
            spatial_offsets: zeroed(filter.count / channels)?,
            patch: zeroed(filter.count)?,
            filter,
            groups,
            transposed: conv.transposed,
        })
    }

    pub(super) fn run(&mut self, memory: &mut Memory, inputs: &[Tensor], output: &Tensor) {
        let (image, weights) = (&inputs[0], &inputs[1]);
        let channels = image.index.shape[1] / self.groups;
        let filters = output.index.shape[1] / self.groups;
        let plane = self.output_spatial.iter().product::<usize>();
        let weight_row = weights.index.strides[usize::from(self.transposed)];
        for position in 0..plane {
            for (k, offset) in self.spatial_offsets.iter_mut().enumerate() {
                let mut out = position;
                let mut filter = k;
                *offset = Some(0);
                for (axis, &size) in self.axes.iter().zip(&self.output_spatial).rev() {
                    let index = out % size * axis.kernel + filter % axis.kernel;
                    *offset = offset.zip(axis.positions[index]).map(|(a, b)| a + b);
                    out /= size;
                    filter /= axis.kernel;
                }
            }
            for batch in 0..output.index.shape[0] {
                for group in 0..self.groups {
                    let data = memory.read(image.buffer);
                    let base = image.offset
                        + batch * image.index.strides[0]
                        + group * channels * image.index.strides[1];
                    for (channel, patch) in self
                        .patch
                        .chunks_exact_mut(self.spatial_offsets.len())
                        .enumerate()
                    {
                        let start = base + channel * image.index.strides[1];
                        for (value, &offset) in patch.iter_mut().zip(&self.spatial_offsets) {
                            *value =
                                offset.map_or(0.0, |offset| f32::from_bits(data[start + offset]));
                        }
                    }
                    for filter in 0..filters {
                        let channel = group * filters + filter;
                        let base = if self.transposed {
                            group * channels * weights.index.strides[0] + filter * weight_row
                        } else {
                            channel * weight_row
                        };
                        let data = &memory.read(weights.buffer)[weights.offset + base..];
                        let mut sum = if self.filter.contiguous {
                            dot(self.patch.iter().copied(), &data[..self.filter.count])
                        } else {
                            self.patch
                                .iter()
                                .enumerate()
                                .map(|(i, &x)| x * f32::from_bits(data[self.filter.at(i)]))
                                .sum()
                        };
                        if let Some(bias) = inputs.get(2) {
                            sum += f32::from_bits(bias.get(memory, channel));
                        }
                        output.put(
                            memory,
                            (batch * output.index.shape[1] + channel) * plane + position,
                            sum.to_bits(),
                        );
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Bounds {
    start: usize,
    count: usize,
    padded_count: usize,
}

struct PoolAxis {
    step: usize,
    bounds: Vec<Bounds>,
}

pub(super) struct Pool {
    kind: PoolKind,
    axes: Vec<PoolAxis>,
    outer: Index,
    spatial: Vec<usize>,
    logical: Vec<usize>,
}

impl Pool {
    pub(super) fn new(
        spec: Option<&PoolSpec>,
        input: &Tensor,
        output: &Tensor,
    ) -> Result<Self, CompileError> {
        let kind = spec.map_or(
            PoolKind::Avg {
                count_include_pad: false,
                divisor: None,
            },
            |p| p.kind,
        );
        let regular = spec.filter(|p| p.kind != PoolKind::AdaptiveMax);
        let rank = regular.map_or(2, |p| p.kernel.len());
        let lead = input.index.shape.len() - rank;
        let spatial = input.index.shape[lead..].to_vec();
        let mut axes = Vec::with_capacity(rank);
        for (axis, &size) in spatial.iter().enumerate() {
            let target = output.index.shape[lead + axis];
            let step = regular.map_or(1, |p| p.dilation[axis] as usize);
            let mut bounds = zeroed(target)?;
            for (o, bounds) in bounds.iter_mut().enumerate() {
                *bounds = if let Some(p) = regular {
                    let kernel = p.kernel[axis] as i64;
                    let pad = p.padding[axis] as i64;
                    let start = (o as i64 * p.stride[axis] as i64) - pad;
                    let step = step as i64;
                    let first = ((-start + step - 1) / step).clamp(0, kernel);
                    let end = ((size as i64 - start + step - 1) / step).clamp(first, kernel);
                    let padded_end =
                        ((size as i64 + pad - start + step - 1) / step).clamp(0, kernel);
                    Bounds {
                        start: (start + first * step).max(0) as usize,
                        count: (end - first) as usize,
                        padded_count: padded_end as usize,
                    }
                } else {
                    let start = (o as u64 * size as u64 / target as u64) as usize;
                    let end = ((o as u64 + 1) * size as u64).div_ceil(target as u64) as usize;
                    Bounds {
                        start,
                        count: end - start,
                        padded_count: end - start,
                    }
                };
            }
            axes.push(PoolAxis { step, bounds });
        }
        let shape: Vec<_> = input.index.shape.iter().map(|&n| n as u64).collect();
        let strides: Vec<_> = input.index.strides[..lead]
            .iter()
            .map(|&n| n as u64)
            .collect();
        Ok(Self {
            kind,
            axes,
            outer: Index::new(&shape[..lead], &strides),
            spatial,
            logical: row_major_strides(&shape[lead..])
                .into_iter()
                .map(|n| n as usize)
                .collect(),
        })
    }

    pub(super) fn run(
        &self,
        memory: &mut Memory,
        input: &Tensor,
        output: &Tensor,
        second: Option<&Tensor>,
    ) {
        let lead = self.outer.shape.len();
        let plane = output.index.count / self.outer.count;
        for outer in 0..self.outer.count {
            let base = self.outer.at(outer);
            for position in 0..plane {
                let mut remainder = position;
                let mut bounds = [Bounds::default(); 3];
                for axis in (0..self.axes.len()).rev() {
                    let axis_bounds = &self.axes[axis].bounds;
                    bounds[axis] = axis_bounds[remainder % axis_bounds.len()];
                    remainder /= axis_bounds.len();
                }
                let bounds = &bounds[..self.spatial.len()];
                let count = bounds.iter().map(|window| window.count).product();
                let average = matches!(self.kind, PoolKind::Avg { .. });
                let mut acc = if average { 0.0 } else { f32::NEG_INFINITY };
                let mut best = 0;
                for k in 0..count {
                    let mut index = k;
                    let mut offset = 0;
                    let mut logical = 0;
                    for (axis, window) in bounds.iter().enumerate().rev() {
                        let coordinate = window.start + index % window.count * self.axes[axis].step;
                        index /= window.count;
                        offset += coordinate * input.index.strides[lead + axis];
                        logical += coordinate * self.logical[axis];
                    }
                    let x = input.float(memory, base + offset);
                    if average {
                        acc += x;
                    } else if k == 0 || (!acc.is_nan() && (x > acc || x.is_nan())) {
                        acc = x;
                        best = logical;
                    }
                }
                if let PoolKind::Avg {
                    count_include_pad,
                    divisor,
                } = self.kind
                {
                    let divisor = divisor.unwrap_or_else(|| {
                        if count_include_pad {
                            bounds
                                .iter()
                                .map(|window| window.padded_count as u64)
                                .product()
                        } else {
                            count as u64
                        }
                    });
                    acc /= divisor as f32;
                }
                let index = outer * plane + position;
                output.put(memory, index, acc.to_bits());
                if let Some(second) = second {
                    second.put(memory, index, best as u32);
                }
            }
        }
    }
}
