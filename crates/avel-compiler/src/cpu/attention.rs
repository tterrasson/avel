use super::*;

pub(super) struct Attention {
    scale: f32,
    causal: bool,
    batches: Vec<Index>,
    query: Vec<f32>,
    scores: Vec<f32>,
    values: Vec<f32>,
}

impl Attention {
    pub(super) fn new(
        scale: f32,
        causal: bool,
        inputs: &[Tensor],
        output: &Tensor,
    ) -> Result<Self, CompileError> {
        let rank = output.index.shape.len();
        let shape: Vec<_> = output.index.shape[..rank - 2]
            .iter()
            .map(|&n| n as u64)
            .collect();
        let batches = inputs
            .iter()
            .map(|tensor| {
                let strides: Vec<_> = tensor.index.strides[..rank - 2]
                    .iter()
                    .map(|&n| n as u64)
                    .collect();
                Index::new(&shape, &strides)
            })
            .collect();
        Ok(Self {
            scale,
            causal,
            batches,
            query: zeroed(inputs[0].index.shape[rank - 1])?,
            scores: zeroed(inputs[1].index.shape[rank - 2])?,
            values: zeroed(output.index.shape[rank - 1])?,
        })
    }

    pub(super) fn run(&mut self, memory: &mut Memory, inputs: &[Tensor], output: &Tensor) {
        let (query, key, value) = (&inputs[0], &inputs[1], &inputs[2]);
        let rank = output.index.shape.len();
        let sequence = output.index.shape[rank - 2];
        let depth = self.query.len();
        let width = self.values.len();
        let (token, feature) = (rank - 2, rank - 1);
        for batch in 0..self.batches[0].count {
            let key_base = self.batches[1].at(batch);
            let value_base = self.batches[2].at(batch);
            for row in 0..sequence {
                let query_base = self.batches[0].at(batch) + row * query.index.strides[token];
                for (i, x) in self.query.iter_mut().enumerate() {
                    *x = query.float(memory, query_base + i * query.index.strides[feature]);
                }
                let length = if self.causal {
                    self.scores.len().min(row + 1)
                } else {
                    self.scores.len()
                };
                let mut max = f32::NEG_INFINITY;
                for (col, score) in self.scores[..length].iter_mut().enumerate() {
                    let mut add = 0.0;
                    if let Some(mask) = inputs.get(3) {
                        let offset = self.batches[3].at(batch)
                            + row * mask.index.strides[token]
                            + col * mask.index.strides[feature];
                        let word = mask.read(memory, offset);
                        if mask.dtype == Dtype::Bool {
                            if word == 0 {
                                *score = f32::NEG_INFINITY;
                                continue;
                            }
                        } else {
                            add = f32::from_bits(word);
                            if add == f32::NEG_INFINITY {
                                *score = f32::NEG_INFINITY;
                                continue;
                            }
                        }
                    }
                    let base = key_base + col * key.index.strides[token];
                    let sum = if key.index.strides[feature] == 1 {
                        let start = key.offset + base;
                        spatial::dot(
                            self.query.iter().copied(),
                            &memory.read(key.buffer)[start..start + depth],
                        )
                    } else {
                        self.query
                            .iter()
                            .enumerate()
                            .map(|(i, x)| {
                                x * key.float(memory, base + i * key.index.strides[feature])
                            })
                            .sum()
                    };
                    *score = sum * self.scale + add;
                    max = max.max(*score);
                }
                self.values.fill(0.0);
                let mut total = 0.0;
                for (col, &score) in self.scores[..length].iter().enumerate() {
                    if score == f32::NEG_INFINITY {
                        continue;
                    }
                    let probability = (score - max).exp();
                    let base = value_base + col * value.index.strides[token];
                    for (i, acc) in self.values.iter_mut().enumerate() {
                        *acc += probability
                            * value.float(memory, base + i * value.index.strides[feature]);
                    }
                    total += probability;
                }
                let inverse = if total == 0.0 { 0.0 } else { total.recip() };
                let start = (batch * sequence + row) * width;
                for (i, &acc) in self.values.iter().enumerate() {
                    output.put(memory, start + i, (acc * inverse).to_bits());
                }
            }
        }
    }
}
