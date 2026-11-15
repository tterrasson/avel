//! Strided views over buffers, so that layout ops never copy.
//!
//! Every kernel reads its inputs through a view and writes a dense output, so
//! transpose, reshape, slice, expand and permute only change strides and offset.

/// The memory order of a dense tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// Last axis innermost, as PyTorch stores contiguous tensors.
    RowMajor,
    /// Axis 1 (channels) innermost, then the spatial axes, then the batch.
    ChannelsLast,
}

/// A tensor read from `buffer`: element `[i, j, ...]` lives at
/// `offset + i * strides[0] + j * strides[1] + ...`, counted in elements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct View {
    pub buffer: usize,
    pub shape: Vec<u64>,
    pub strides: Vec<u64>,
    pub offset: u64,
}

pub fn row_major_strides(shape: &[u64]) -> Vec<u64> {
    let mut strides = vec![1; shape.len()];
    for axis in (1..shape.len()).rev() {
        strides[axis - 1] = strides[axis] * shape[axis];
    }
    strides
}

impl View {
    /// A tensor that fills `buffer` in `layout`. Channels-last only applies
    /// from rank 3, where there is a spatial axis to move channels after.
    pub fn dense(buffer: usize, shape: Vec<u64>, layout: Layout) -> Self {
        match layout {
            Layout::ChannelsLast if shape.len() >= 3 => {
                let mut order: Vec<usize> = (0..shape.len()).filter(|&axis| axis != 1).collect();
                order.push(1);
                Self::dense_in_order(buffer, shape, &order)
            }
            _ => {
                let order: Vec<usize> = (0..shape.len()).collect();
                Self::dense_in_order(buffer, shape, &order)
            }
        }
    }

    /// A tensor that fills `buffer` with its axes laid out in `order`, from
    /// the outermost to the innermost.
    pub fn dense_in_order(buffer: usize, shape: Vec<u64>, order: &[usize]) -> Self {
        let mut strides = vec![0; shape.len()];
        let mut stride = 1;
        for &axis in order.iter().rev() {
            strides[axis] = stride;
            stride *= shape[axis];
        }
        Self {
            buffer,
            shape,
            strides,
            offset: 0,
        }
    }

    pub fn elements(&self) -> u64 {
        self.shape.iter().product()
    }

    /// Whether the view reads its elements in row-major order, from offset 0
    /// and without gaps. Size-1 axes have no meaningful stride.
    pub fn is_row_major(&self) -> bool {
        self.offset == 0
            && self
                .shape
                .iter()
                .zip(&self.strides)
                .zip(row_major_strides(&self.shape))
                .all(|((&size, &stride), expected)| size == 1 || stride == expected)
    }

    pub fn permute(&self, dims: &[usize]) -> Self {
        Self {
            buffer: self.buffer,
            shape: dims.iter().map(|&axis| self.shape[axis]).collect(),
            strides: dims.iter().map(|&axis| self.strides[axis]).collect(),
            offset: self.offset,
        }
    }

    /// Reinterprets the view as `shape`, as `torch.Tensor.view` does. Returns
    /// `None` when the strides cannot express it, and the data must be copied.
    pub fn reshape(&self, shape: &[u64]) -> Option<Self> {
        if shape.iter().product::<u64>() != self.elements() {
            return None;
        }
        let (old_shape, old_strides) = (&self.shape, &self.strides);
        let mut strides = vec![1; shape.len()];
        // Walks both shapes from the innermost axis, splitting each chunk of
        // old axes that are contiguous among themselves. Mirrors `at::computeStride`.
        let mut view_axis = shape.len();
        let mut base = old_strides.last().copied().unwrap_or(1);
        let mut old_elements = 1;
        let mut view_elements = 1;
        for axis in (0..old_shape.len()).rev() {
            old_elements *= old_shape[axis];
            let chunk_ends = axis == 0
                || (old_shape[axis - 1] != 1 && old_strides[axis - 1] != old_elements * base);
            if !chunk_ends {
                continue;
            }
            while view_axis > 0 && (view_elements < old_elements || shape[view_axis - 1] == 1) {
                strides[view_axis - 1] = view_elements * base;
                view_elements *= shape[view_axis - 1];
                view_axis -= 1;
            }
            if view_elements != old_elements {
                return None;
            }
            if axis > 0 {
                base = old_strides[axis - 1];
                old_elements = 1;
                view_elements = 1;
            }
        }
        if view_axis != 0 && shape[..view_axis].iter().any(|&size| size != 1) {
            return None;
        }
        Some(Self {
            buffer: self.buffer,
            shape: shape.to_vec(),
            strides,
            offset: self.offset,
        })
    }

    /// Broadcasts the view to `shape`, aligning trailing axes as PyTorch does.
    /// The caller checks that the shapes are compatible.
    pub fn broadcast_to(&self, shape: &[u64]) -> Self {
        let lead = shape.len() - self.shape.len();
        let strides = shape
            .iter()
            .enumerate()
            .map(|(axis, &size)| match axis.checked_sub(lead) {
                Some(own) if self.shape[own] == size => self.strides[own],
                _ => 0,
            })
            .collect();
        Self {
            buffer: self.buffer,
            shape: shape.to_vec(),
            strides,
            offset: self.offset,
        }
    }

    /// Keeps `len` elements of `axis`, from `start` by `step`.
    pub fn slice(&self, axis: usize, start: u64, len: u64, step: u64) -> Self {
        let mut view = self.clone();
        view.offset += start * self.strides[axis];
        view.shape[axis] = len;
        view.strides[axis] *= step;
        view
    }

    /// Fixes `axis` at `index`, removing it.
    pub fn select(&self, axis: usize, index: u64) -> Self {
        let mut view = self.clone();
        view.offset += index * self.strides[axis];
        view.shape.remove(axis);
        view.strides.remove(axis);
        view
    }

    pub fn unsqueeze(&self, axis: usize) -> Self {
        let mut view = self.clone();
        view.shape.insert(axis, 1);
        view.strides.insert(axis, 0);
        view
    }

    /// Removes the size-1 `axes`, sorted in ascending order.
    pub fn squeeze(&self, axes: &[usize]) -> Self {
        let mut view = self.clone();
        for &axis in axes.iter().rev() {
            view.shape.remove(axis);
            view.strides.remove(axis);
        }
        view
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contiguous(shape: &[u64]) -> View {
        View::dense(0, shape.to_vec(), Layout::RowMajor)
    }

    #[test]
    fn lays_out_dense_tensors() {
        assert_eq!(contiguous(&[2, 3, 4]).strides, [12, 4, 1]);
        let nhwc = View::dense(0, vec![2, 3, 4, 5], Layout::ChannelsLast);
        assert_eq!(nhwc.strides, [60, 1, 15, 3]);
        let rank_2 = View::dense(0, vec![2, 3], Layout::ChannelsLast);
        assert_eq!(rank_2.strides, [3, 1]);
    }

    #[test]
    fn tells_row_major_views() {
        assert!(contiguous(&[2, 3]).is_row_major());
        assert!(!contiguous(&[2, 3]).permute(&[1, 0]).is_row_major());
        assert!(View::dense(0, vec![1, 8, 1, 1], Layout::ChannelsLast).is_row_major());
        assert!(!contiguous(&[4, 3]).slice(0, 1, 2, 1).is_row_major());
    }

    #[test]
    fn reshapes_contiguous_chunks() {
        let view = contiguous(&[2, 3, 4]);
        assert_eq!(view.reshape(&[6, 4]).unwrap().strides, [4, 1]);
        assert_eq!(view.reshape(&[2, 12]).unwrap().strides, [12, 1]);
        assert_eq!(view.reshape(&[24]).unwrap().strides, [1]);
        assert_eq!(view.reshape(&[1, 2, 1, 12, 1]).unwrap().strides[1], 12);

        // Splitting the outer axis of a transposed tensor is fine, merging is not.
        let transposed = view.permute(&[2, 1, 0]);
        assert_eq!(
            transposed.reshape(&[2, 2, 3, 2]).unwrap().strides,
            [2, 1, 4, 12]
        );
        assert_eq!(transposed.reshape(&[24]), None);
        assert_eq!(view.reshape(&[5, 5]), None);
    }

    #[test]
    fn reshapes_channels_last_tensors_without_spatial_extent() {
        let pooled = View::dense(0, vec![1, 8, 1, 1], Layout::ChannelsLast);
        assert_eq!(pooled.reshape(&[1, 8]).unwrap().strides, [8, 1]);
        let image = View::dense(0, vec![1, 8, 2, 2], Layout::ChannelsLast);
        assert_eq!(image.reshape(&[1, 32]), None);
    }

    #[test]
    fn broadcasts_slices_and_selects() {
        let view = contiguous(&[3, 1]);
        assert_eq!(view.broadcast_to(&[2, 3, 4]).strides, [0, 1, 0]);

        let sliced = contiguous(&[4, 6]).slice(1, 1, 3, 2);
        assert_eq!((sliced.offset, sliced.strides.clone()), (1, vec![6, 2]));
        let selected = sliced.select(0, 2);
        assert_eq!((selected.offset, selected.shape), (13, vec![3]));
    }

    #[test]
    fn adds_and_removes_unit_axes() {
        let view = contiguous(&[2, 3]).unsqueeze(1);
        assert_eq!(view.shape, [2, 1, 3]);
        assert_eq!(view.squeeze(&[1]), contiguous(&[2, 3]));
    }
}
