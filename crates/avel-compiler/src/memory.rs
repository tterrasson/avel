//! Places activations in a single arena, reusing memory between tensors whose
//! lifetimes do not overlap.
//!
//! A tensor lives from the dispatch that writes it to the last one that reads
//! it. Tensors are placed largest first, each at the lowest offset that does
//! not overlap a tensor already placed and alive at the same time. A kernel's
//! inputs are alive while it writes its output, so they never share memory.

/// An activation to place: its size in bytes and the dispatches that write
/// (`first`) and last read (`last`) it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lifetime {
    pub size: u64,
    pub first: usize,
    pub last: usize,
}

/// Returns the offset of each tensor, aligned to `alignment`, and the arena size.
pub fn plan(tensors: &[Lifetime], alignment: u64) -> (Vec<u64>, u64) {
    let mut order: Vec<usize> = (0..tensors.len()).collect();
    // Largest first; ties in definition order, so plans are deterministic.
    order.sort_by_key(|&index| (std::cmp::Reverse(tensors[index].size), index));

    let mut offsets = vec![0; tensors.len()];
    let mut taken: Vec<(u64, u64)> = Vec::new();
    let mut arena = 0;
    for (position, &index) in order.iter().enumerate() {
        let tensor = tensors[index];
        taken.clear();
        taken.extend(
            order[..position]
                .iter()
                .filter(|&&other| {
                    let other = tensors[other];
                    other.first <= tensor.last && tensor.first <= other.last
                })
                .map(|&other| (offsets[other], offsets[other] + tensors[other].size)),
        );
        taken.sort_unstable();

        let mut offset = 0;
        for &(start, end) in &taken {
            if offset + tensor.size <= start {
                break;
            }
            offset = offset.max(end.next_multiple_of(alignment));
        }
        offsets[index] = offset;
        arena = arena.max(offset + tensor.size);
    }
    (offsets, arena)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lifetime(size: u64, first: usize, last: usize) -> Lifetime {
        Lifetime { size, first, last }
    }

    fn overlap(a: (u64, u64), b: (u64, u64)) -> bool {
        a.0 < b.1 && b.0 < a.1
    }

    #[test]
    fn reuses_memory_of_dead_tensors() {
        // A chain: each tensor is read by the next dispatch only.
        let chain: Vec<Lifetime> = (0..6).map(|step| lifetime(1000, step, step + 1)).collect();
        let (offsets, size) = plan(&chain, 256);
        assert_eq!(size, 1024 + 1000);
        for pair in offsets.windows(2) {
            assert_ne!(pair[0], pair[1]);
        }
    }

    #[test]
    fn never_overlaps_tensors_alive_together() {
        let tensors = [
            lifetime(512, 0, 5),
            lifetime(300, 1, 2),
            lifetime(1000, 2, 3),
            lifetime(100, 3, 4),
            lifetime(800, 4, 5),
            lifetime(64, 5, 5),
        ];
        let (offsets, size) = plan(&tensors, 256);
        for (i, a) in tensors.iter().enumerate() {
            assert_eq!(offsets[i] % 256, 0);
            assert!(offsets[i] + a.size <= size);
            for (j, b) in tensors.iter().enumerate().skip(i + 1) {
                let alive = a.first <= b.last && b.first <= a.last;
                let range = |index: usize, t: &Lifetime| (offsets[index], offsets[index] + t.size);
                assert!(!alive || !overlap(range(i, a), range(j, b)), "{i} and {j}");
            }
        }
        assert!(size < tensors.iter().map(|tensor| tensor.size).sum());
    }
}
