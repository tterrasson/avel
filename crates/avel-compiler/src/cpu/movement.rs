use super::*;
use crate::kernel::Scalar;
use crate::ops::{Movement as Op, PadMode, ResizeMode};

pub(super) enum Movement {
    Remap {
        axes: Vec<Axis>,
        fill: u32,
    },
    Roll(usize),
    Indexed {
        axes: Vec<IndexedAxis>,
        count: usize,
        stride: usize,
        negative: bool,
    },
    Resize([Vec<Sample>; 2], bool),
    Cat {
        axis: usize,
        ends: Vec<usize>,
    },
    Embedding,
    Rotary {
        depth: usize,
    },
}

pub(super) enum Axis {
    Coordinate(usize),
    Repeat {
        axis: usize,
        size: usize,
    },
    Flip {
        axis: usize,
        size: usize,
    },
    Roll {
        axis: usize,
        shift: usize,
        size: usize,
    },
    Pad {
        axis: usize,
        left: i64,
        right: i64,
        size: i64,
        mode: PadMode,
    },
}

pub(super) enum IndexedAxis {
    Coordinate(usize),
    Read { input: usize, index: Index },
}

#[derive(Clone, Copy)]
pub(super) struct Sample {
    offsets: [usize; 4],
    weights: [f32; 4],
    count: usize,
}

impl Movement {
    pub(super) fn new(
        kind: Op,
        fill: Option<Scalar>,
        inputs: &[Tensor],
        output: &Tensor,
    ) -> Option<Self> {
        let input = &inputs[0].index;
        let shape = &output.index.shape;
        let rank = input.shape.len();
        let coordinates = || (0..rank).map(Axis::Coordinate).collect::<Vec<_>>();
        let remap = |axes| Self::Remap {
            axes,
            fill: fill.map_or(0, elementwise::word),
        };
        Some(match kind {
            Op::Repeat => {
                let lead = shape.len() - rank;
                remap(
                    input
                        .shape
                        .iter()
                        .enumerate()
                        .map(|(axis, &size)| Axis::Repeat {
                            axis: axis + lead,
                            size,
                        })
                        .collect(),
                )
            }
            Op::Flip(flips) => {
                let mut axes = coordinates();
                for axis in flips {
                    axes[axis] = Axis::Flip {
                        axis,
                        size: input.shape[axis],
                    };
                }
                remap(axes)
            }
            Op::Roll {
                shifts,
                flattened: true,
            } => Self::Roll(shifts[0].1 as usize),
            Op::Roll {
                shifts,
                flattened: false,
            } => {
                let mut axes = coordinates();
                for (axis, shift) in shifts {
                    axes[axis] = Axis::Roll {
                        axis,
                        shift: shift as usize,
                        size: input.shape[axis],
                    };
                }
                remap(axes)
            }
            Op::Pad(pad) => remap(
                pad.widths
                    .into_iter()
                    .enumerate()
                    .map(|(axis, (left, right))| {
                        if left == 0 && right == 0 {
                            Axis::Coordinate(axis)
                        } else {
                            Axis::Pad {
                                axis,
                                left,
                                right,
                                size: input.shape[axis] as i64,
                                mode: pad.mode,
                            }
                        }
                    })
                    .collect(),
            ),
            Op::IndexSelect(axis) | Op::Gather(axis) => {
                let index = &inputs[1].index;
                let stride = if matches!(kind, Op::IndexSelect(_)) {
                    output.index.logical.get(axis).copied().unwrap_or(1)
                } else {
                    1
                };
                let mut axes: Vec<_> = (0..rank).map(IndexedAxis::Coordinate).collect();
                let index = Index::new(&as_dims(&index.shape), &as_dims(&index.strides));
                let count = index.count;
                if rank > 0 {
                    axes[axis] = IndexedAxis::Read { input: 1, index };
                }
                Self::Indexed {
                    axes,
                    count,
                    stride,
                    negative: false,
                }
            }
            Op::Index {
                axes,
                start,
                rank: broadcast_rank,
            } => {
                let target = &shape[start..start + broadcast_rank];
                let mut next = 0;
                let axes = (0..rank)
                    .map(|axis| {
                        if let Some(i) = axes.iter().position(|&indexed| indexed == axis) {
                            let index = &inputs[i + 1].index;
                            let lead = target.len() - index.shape.len();
                            let strides = target
                                .iter()
                                .enumerate()
                                .map(|(a, _)| {
                                    a.checked_sub(lead).map_or(0, |a| {
                                        if index.shape[a] == 1 {
                                            0
                                        } else {
                                            index.strides[a] as u64
                                        }
                                    })
                                })
                                .collect::<Vec<_>>();
                            IndexedAxis::Read {
                                input: i + 1,
                                index: Index::new(&as_dims(target), &strides),
                            }
                        } else {
                            let axis = next + if next < start { 0 } else { broadcast_rank };
                            next += 1;
                            IndexedAxis::Coordinate(axis)
                        }
                    })
                    .collect();
                Self::Indexed {
                    axes,
                    count: target.iter().product(),
                    stride: shape[start + broadcast_rank..].iter().product(),
                    negative: true,
                }
            }
            Op::Resize(resize) => {
                let [y, x] = std::array::from_fn(|i| {
                    let axis = i + 2;
                    let size = input.shape[axis];
                    let out = shape[axis];
                    let scale = resize
                        .scales
                        .map_or(size as f64 / out as f64, |scales| 1.0 / scales[i])
                        as f32;
                    (0..out)
                        .map(|coord| {
                            Sample::new(resize.mode, scale, coord, size, out, input.strides[axis])
                        })
                        .collect()
                });
                Self::Resize([y, x], resize.mode == ResizeMode::Nearest)
            }
            Op::AdaptiveAverage => return None,
        })
    }

    pub(super) fn cat(axis: usize, inputs: &[Tensor]) -> Self {
        let mut end = 0;
        let ends = inputs
            .iter()
            .map(|input| {
                end += input.index.shape[axis];
                end
            })
            .collect();
        Self::Cat { axis, ends }
    }

    pub(super) fn run(&self, memory: &mut Memory, inputs: &[Tensor], output: &Tensor) {
        let input = &inputs[0];
        for linear in 0..output.index.count {
            let coord = |axis| output.index.coordinate(axis, linear);
            let word = match self {
                Self::Remap { axes, fill } => {
                    let offset = axes.iter().zip(&input.index.strides).try_fold(
                        0,
                        |offset, (axis, stride)| {
                            axis.map(&coord).map(|position| offset + position * stride)
                        },
                    );
                    offset.map_or(*fill, |offset| input.read(memory, offset))
                }
                Self::Roll(shift) => input.get(
                    memory,
                    (linear + output.index.count - shift) % output.index.count,
                ),
                Self::Indexed {
                    axes,
                    count,
                    stride,
                    negative,
                } => {
                    let logical = linear / stride % count;
                    let offset = axes.iter().enumerate().try_fold(0, |offset, (a, axis)| {
                        let position = match axis {
                            IndexedAxis::Coordinate(axis) => coord(*axis),
                            IndexedAxis::Read {
                                input: source,
                                index,
                            } => {
                                let mut i =
                                    inputs[*source].read(memory, index.at(logical)) as i32 as i64;
                                let size = input.index.shape[a] as i64;
                                if *negative && i < 0 {
                                    i += size;
                                }
                                if !(0..size).contains(&i) {
                                    return None;
                                }
                                i as usize
                            }
                        };
                        Some(offset + position * input.index.strides[a])
                    });
                    offset.map_or(0, |offset| input.read(memory, offset))
                }
                Self::Resize([rows, columns], nearest) => {
                    let y = &rows[coord(2)];
                    let x = &columns[coord(3)];
                    let base =
                        coord(0) * input.index.strides[0] + coord(1) * input.index.strides[1];
                    if *nearest {
                        input.read(memory, base + y.offsets[0] + x.offsets[0])
                    } else {
                        let mut sum = 0.0;
                        for row in 0..y.count {
                            let mut line = 0.0;
                            for col in 0..x.count {
                                line += x.weights[col]
                                    * input.float(memory, base + y.offsets[row] + x.offsets[col]);
                            }
                            sum += y.weights[row] * line;
                        }
                        sum.to_bits()
                    }
                }
                Self::Cat { axis, ends } => {
                    let position = coord(*axis);
                    let i = ends.partition_point(|&end| end <= position);
                    let start = if i == 0 { 0 } else { ends[i - 1] };
                    let input = &inputs[i];
                    let offset = input
                        .index
                        .strides
                        .iter()
                        .enumerate()
                        .map(|(a, stride)| {
                            (if a == *axis {
                                position - start
                            } else {
                                coord(a)
                            }) * stride
                        })
                        .sum();
                    input.read(memory, offset)
                }
                Self::Embedding => {
                    let width = input.index.shape[1];
                    let row = inputs[1].get(memory, linear / width) as usize;
                    if row < input.index.shape[0] {
                        input.read(
                            memory,
                            row * input.index.strides[0] + linear % width * input.index.strides[1],
                        )
                    } else {
                        0
                    }
                }
                Self::Rotary { depth } => {
                    let half = depth / 2;
                    let upper = linear % depth >= half;
                    let pair = if upper { linear - half } else { linear + half };
                    let x = f32::from_bits(input.get(memory, linear));
                    let y = f32::from_bits(input.get(memory, pair));
                    let cosine = f32::from_bits(inputs[1].get(memory, linear));
                    let sine = f32::from_bits(inputs[2].get(memory, linear));
                    (x * cosine + (if upper { y } else { -y }) * sine).to_bits()
                }
            };
            output.put(
                memory,
                linear,
                if output.dtype == Dtype::Bool {
                    u32::from(word != 0)
                } else {
                    word
                },
            );
        }
    }
}

impl Axis {
    fn map(&self, coord: &impl Fn(usize) -> usize) -> Option<usize> {
        Some(match *self {
            Self::Coordinate(axis) => coord(axis),
            Self::Repeat { axis, size } => coord(axis) % size,
            Self::Flip { axis, size } => size - 1 - coord(axis),
            Self::Roll { axis, shift, size } => (coord(axis) + size - shift) % size,
            Self::Pad {
                axis,
                left,
                right,
                size,
                mode,
            } => {
                let position = coord(axis) as i64 - left;
                (match mode {
                    PadMode::Constant(_) if !(0..size).contains(&position) => return None,
                    PadMode::Constant(_) => position,
                    PadMode::Reflect if position >= size => 2 * (size - 1) - position,
                    PadMode::Reflect => position.abs(),
                    PadMode::Replicate => position.clamp(0, size - 1),
                    PadMode::Circular => {
                        let start = (-left).max(0);
                        let width = size + right.min(0) - start;
                        start + (position - start).rem_euclid(width)
                    }
                }) as usize
            }
        })
    }
}

impl Sample {
    fn new(
        mode: ResizeMode,
        scale: f32,
        coord: usize,
        size: usize,
        out: usize,
        stride: usize,
    ) -> Self {
        let offset = |i: i64| i.clamp(0, size as i64 - 1) as usize * stride;
        if mode == ResizeMode::Nearest {
            let pos = if out == size {
                coord
            } else if out == size * 2 {
                coord / 2
            } else {
                (coord as f32 * scale) as usize
            };
            return Self {
                offsets: [offset(pos as i64), 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
                count: 1,
            };
        }
        let align = matches!(
            mode,
            ResizeMode::Bilinear {
                align_corners: true
            } | ResizeMode::Bicubic {
                align_corners: true
            }
        );
        let source = if out == size && matches!(mode, ResizeMode::Bilinear { .. }) {
            coord as f32
        } else if align {
            if out == 1 {
                0.0
            } else {
                (size - 1) as f32 / (out - 1) as f32 * coord as f32
            }
        } else {
            scale * (coord as f32 + 0.5) - 0.5
        };
        if matches!(mode, ResizeMode::Bilinear { .. }) {
            let source = source.max(0.0);
            let base = source.floor() as i64;
            let t = source - base as f32;
            Self {
                offsets: [offset(base), offset(base + 1), 0, 0],
                weights: [1.0 - t, t, 0.0, 0.0],
                count: 2,
            }
        } else {
            let base = source.floor() as i64;
            let taps = [base - 1, base, base + 1, base + 2];
            Self {
                offsets: taps.map(offset),
                weights: taps.map(|i| {
                    let x = (source - i as f32).abs();
                    let a = -0.75;
                    if x <= 1.0 {
                        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
                    } else {
                        ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
                    }
                }),
                count: 4,
            }
        }
    }
}

fn as_dims(dims: &[usize]) -> Vec<u64> {
    dims.iter().map(|&n| n as u64).collect()
}
