//! Greedy fusion of element-wise kernels.
//!
//! Two rewrites, applied until neither matches:
//!
//! - An element-wise kernel is inlined into the element-wise kernel that reads
//!   its output, which then reads the producer's inputs directly.
//! - An element-wise kernel becomes the epilogue of the kernel that produced
//!   one of its inputs (matmul, convolution, pooling, reduction), which then
//!   stores the element-wise result instead of its own.
//!
//! Both only apply to an intermediate tensor read by a single kernel, at the
//! coordinates it was written at: fusion never recomputes an element, and
//! never needs to translate coordinates between the two kernels.
//!
//! A second pass, [`fuse_prologues`], then inlines the element-wise kernels
//! left into the loads of the other kernel reading them. That one translates
//! coordinates, and may recompute an element each time it is loaded.

use crate::kernel::{Epilogue, Expr, Kernel, KernelNode, Prologue, PrologueKind};
use crate::ops::{Movement, Resize, ResizeMode};
use crate::view::View;

/// Fuses `kernels`, in execution order. `fusible(buffer)` tells whether the
/// buffer is an intermediate, which fusion may drop; `max_bindings` bounds the
/// buffers a fused kernel may bind, its output included.
///
/// Consumers are visited in order, and each one fuses with its producers until
/// none fits. A fusion only changes the kernels at the two positions involved,
/// and the readers of buffers they share, so it never makes an earlier
/// consumer fusible: one pass gives the same result as restarting the search
/// from the first kernel after every fusion.
pub fn fuse(
    kernels: Vec<KernelNode>,
    fusible: impl Fn(usize) -> bool,
    max_bindings: usize,
) -> Vec<KernelNode> {
    let mut kernels: Vec<Option<KernelNode>> = kernels.into_iter().map(Some).collect();
    let mut graph = Uses::new(&kernels);
    for consumer in 0..kernels.len() {
        while let Some(producer) = candidate(&kernels, &graph, consumer, &fusible, max_bindings) {
            let (Some(first), Some(second)) = (kernels[producer].take(), kernels[consumer].take())
            else {
                unreachable!("`candidate` only returns live kernels");
            };
            graph.remove(producer, &first);
            graph.remove(consumer, &second);
            let (position, fused) = match first.kernel {
                Kernel::ElementWise(_) => (consumer, inline(first, second)),
                _ => (producer, absorb(first, second)),
            };
            graph.add(position, &fused);
            kernels[position] = Some(fused);
            if position != consumer {
                break;
            }
        }
    }
    kernels.into_iter().flatten().collect()
}

/// Which live kernel writes each buffer, and which ones read it, by kernel
/// index in ascending order.
struct Uses {
    producers: Vec<Option<usize>>,
    readers: Vec<Vec<usize>>,
}

impl Uses {
    fn new(kernels: &[Option<KernelNode>]) -> Self {
        let buffers = kernels
            .iter()
            .flatten()
            .flat_map(|node| node.reads().chain(node.writes()))
            .map(|view| view.buffer + 1)
            .max()
            .unwrap_or(0);
        let mut uses = Self {
            producers: vec![None; buffers],
            readers: vec![Vec::new(); buffers],
        };
        for (index, node) in kernels.iter().enumerate() {
            if let Some(node) = node {
                uses.add(index, node);
            }
        }
        uses
    }

    fn add(&mut self, index: usize, node: &KernelNode) {
        for view in node.writes() {
            let latest = self.producers[view.buffer].map_or(index, |other| other.max(index));
            self.producers[view.buffer] = Some(latest);
        }
        for view in node.reads() {
            let list = &mut self.readers[view.buffer];
            if let Err(position) = list.binary_search(&index) {
                list.insert(position, index);
            }
        }
    }

    fn remove(&mut self, index: usize, node: &KernelNode) {
        for view in node.writes() {
            if self.producers[view.buffer] == Some(index) {
                self.producers[view.buffer] = None;
            }
        }
        for view in node.reads() {
            self.readers[view.buffer].retain(|&reader| reader != index);
        }
    }
}

/// The producer `consumer` fuses with next, if any.
fn candidate(
    kernels: &[Option<KernelNode>],
    uses: &Uses,
    consumer: usize,
    fusible: &impl Fn(usize) -> bool,
    max_bindings: usize,
) -> Option<usize> {
    let Some(
        node @ KernelNode {
            kernel: Kernel::ElementWise(_),
            ..
        },
    ) = &kernels[consumer]
    else {
        return None;
    };
    // The latest producer among the inputs: an epilogue moves the consumer up
    // to it, after every other input is ready.
    let mut candidates: Vec<(usize, usize)> = node
        .inputs
        .iter()
        .filter_map(|view| Some((uses.producers[view.buffer]?, view.buffer)))
        .collect();
    candidates.sort_unstable();
    candidates.dedup();
    let latest = candidates.last().map(|&(producer, _)| producer);

    let Kernel::ElementWise(expr) = &node.kernel else {
        unreachable!("matched above");
    };
    // A buffer other kernels read before this one writes it, such as state
    // rewritten at the end of a run, must not be written any earlier.
    let read_before = uses.readers[node.output.buffer]
        .iter()
        .any(|&reader| reader < consumer);
    for &(producer, buffer) in candidates.iter().rev() {
        let source = kernels[producer].as_ref().expect("producers are live");
        let single_reader = uses.readers[buffer] == [consumer];
        let aligned = node
            .inputs
            .iter()
            .filter(|view| view.buffer == buffer)
            .all(|view| same_coordinates(view, &source.output));
        if !(producer < consumer
            && fusible(buffer)
            && single_reader
            && aligned
            && buffer == source.output.buffer)
        {
            continue;
        }
        // An epilogue cannot read another output of its own producer: it is
        // still being written, and may be read through different coordinates.
        if source
            .second
            .as_ref()
            .is_some_and(|second| node.inputs.iter().any(|view| view.buffer == second.buffer))
        {
            continue;
        }
        let fits = match &source.kernel {
            Kernel::Custom(_) | Kernel::CacheWrite { .. } | Kernel::Scatter(_) => continue,
            // The fused kernel may read the buffer it writes, as an update of
            // state does, but only element by element.
            Kernel::ElementWise(_) => source
                .inputs
                .iter()
                .filter(|view| view.buffer == node.output.buffer)
                .all(|view| same_coordinates(view, &node.output)),
            _ => Some(producer) == latest && !read_before && !expr.uses_index(),
        } && reads_unchanged(kernels, uses, source, producer, consumer)
            && fused_bindings(source, node, buffer) <= max_bindings;
        if fits {
            return Some(producer);
        }
    }
    None
}

/// Moving a read to its consumer must not cross a write of the old value,
/// including snapshots that separate simultaneous state updates.
fn reads_unchanged(
    kernels: &[Option<KernelNode>],
    uses: &Uses,
    source: &KernelNode,
    from: usize,
    to: usize,
) -> bool {
    source.reads().all(|read| {
        // Inputs/parameters have no writer; SSA intermediates were written
        // before the source. Only mutable buffers need the interval scan.
        if uses.producers[read.buffer].is_none_or(|last| last <= from) {
            return true;
        }
        kernels[from + 1..to]
            .iter()
            .flatten()
            .all(|node| node.writes().all(|write| write.buffer != read.buffer))
    })
}

/// The buffers the fusion of `consumer` into `producer` through `buffer`
/// binds, its output included, without building the fused kernel. Views of
/// the same buffer share one binding.
fn fused_bindings(producer: &KernelNode, consumer: &KernelNode, buffer: usize) -> usize {
    let mut bound: Vec<usize> = Vec::new();
    let consumed = consumer.inputs.iter().filter(|view| view.buffer != buffer);
    for view in producer
        .reads()
        .chain(consumed)
        .chain(producer.second.iter())
        .chain([&consumer.output])
    {
        if !bound.contains(&view.buffer) {
            bound.push(view.buffer);
        }
    }
    bound.len()
}

/// Whether `view` reads a tensor at the coordinates `output` wrote it at.
fn same_coordinates(view: &View, output: &View) -> bool {
    view.offset == output.offset
        && view.shape == output.shape
        && view
            .shape
            .iter()
            .zip(view.strides.iter().zip(&output.strides))
            .all(|(&size, (a, b))| size == 1 || a == b)
}

/// Inlines the element-wise `producer` into the element-wise `consumer`.
fn inline(producer: KernelNode, consumer: KernelNode) -> KernelNode {
    let (Kernel::ElementWise(inner), Kernel::ElementWise(outer)) =
        (&producer.kernel, &consumer.kernel)
    else {
        unreachable!("only element-wise kernels are inlined");
    };
    let buffer = producer.output.buffer;
    let mut inputs = Vec::new();
    let kept: Vec<Option<usize>> = consumer
        .inputs
        .iter()
        .map(|view| (view.buffer != buffer).then(|| push(&mut inputs, view)))
        .collect();
    let moved: Vec<usize> = producer
        .inputs
        .iter()
        .map(|view| push(&mut inputs, view))
        .collect();
    let inner = inner.substitute(&|index| Expr::Input(moved[index]), &Expr::Acc);
    let expr = outer.substitute(
        &|index| kept[index].map_or_else(|| inner.clone(), Expr::Input),
        &Expr::Acc,
    );
    KernelNode {
        kernel: Kernel::ElementWise(expr),
        inputs,
        output: consumer.output,
        second: None,
        epilogue: None,
        prologues: Vec::new(),
        ops: [producer.ops, consumer.ops].concat(),
    }
}

/// Makes the element-wise `consumer` the epilogue of `producer`.
fn absorb(producer: KernelNode, consumer: KernelNode) -> KernelNode {
    let Kernel::ElementWise(outer) = &consumer.kernel else {
        unreachable!("only element-wise kernels become epilogues");
    };
    let buffer = producer.output.buffer;
    let Epilogue {
        expr: previous,
        mut inputs,
    } = producer.epilogue.unwrap_or(Epilogue {
        expr: Expr::Acc,
        inputs: Vec::new(),
    });
    let kept: Vec<Option<usize>> = consumer
        .inputs
        .iter()
        .map(|view| (view.buffer != buffer).then(|| push(&mut inputs, view)))
        .collect();
    let expr = outer.substitute(
        &|index| kept[index].map_or_else(|| Expr::Acc, Expr::Input),
        &Expr::Acc,
    );
    KernelNode {
        kernel: producer.kernel,
        inputs: producer.inputs,
        output: consumer.output,
        second: producer.second,
        epilogue: Some(Epilogue {
            expr: expr.substitute(&Expr::Input, &previous),
            inputs,
        }),
        prologues: producer.prologues,
        ops: [producer.ops, consumer.ops].concat(),
    }
}

/// Inlines element-wise kernels into the loads of the one other kernel that
/// reads their output, as prologues, under the same conditions as the first
/// pass: an intermediate tensor, a single reader and the binding limit. The
/// consumer recomputes the expression on every load, so kernels that load an
/// element many times, such as a GEMM or a window, only take cheap ones.
pub fn fuse_prologues(
    kernels: Vec<KernelNode>,
    fusible: impl Fn(usize) -> bool,
    max_bindings: usize,
) -> Vec<KernelNode> {
    let mut kernels: Vec<Option<KernelNode>> = kernels.into_iter().map(Some).collect();
    let mut uses = Uses::new(&kernels);
    for consumer in 0..kernels.len() {
        while let Some((producer, input)) =
            prologue_candidate(&kernels, &uses, consumer, &fusible, max_bindings)
        {
            let (Some(source), Some(mut node)) =
                (kernels[producer].take(), kernels[consumer].take())
            else {
                unreachable!("`prologue_candidate` only returns live kernels");
            };
            uses.remove(producer, &source);
            uses.remove(consumer, &node);
            let kind = match source.kernel {
                Kernel::ElementWise(expr) => PrologueKind::ElementWise(expr),
                Kernel::Rotary => PrologueKind::Rotary,
                Kernel::Movement {
                    kind: Movement::Resize(resize),
                    ..
                } => PrologueKind::Nearest(resize),
                _ => unreachable!("only element-wise kernels and nearest resize become prologues"),
            };
            node.prologues.resize(node.inputs.len(), None);
            node.prologues[input] = Some(Prologue {
                kind,
                inputs: source.inputs,
                source: source.output,
            });
            node.ops = [source.ops, node.ops].concat();
            uses.add(consumer, &node);
            kernels[consumer] = Some(node);
        }
    }
    kernels.into_iter().flatten().collect()
}

/// The element-wise producer `consumer` takes as a prologue next, with the
/// input it computes.
fn prologue_candidate(
    kernels: &[Option<KernelNode>],
    uses: &Uses,
    consumer: usize,
    fusible: &impl Fn(usize) -> bool,
    max_bindings: usize,
) -> Option<(usize, usize)> {
    let node = kernels[consumer].as_ref()?;
    // Kernels that load every element once may recompute anything.
    let any_cost = match node.kernel {
        Kernel::Custom(_) | Kernel::CacheWrite { .. } | Kernel::Scatter(_) => return None,
        Kernel::Reduction { .. } | Kernel::Cat { .. } => true,
        _ => false,
    };
    (0..node.inputs.len()).find_map(|input| {
        let view = &node.inputs[input];
        let producer = uses.producers[view.buffer]?;
        let source = kernels[producer].as_ref()?;
        match &source.kernel {
            Kernel::ElementWise(expr)
                if !matches!(node.kernel, Kernel::ElementWise(_))
                    && source.prologues.is_empty()
                    && !expr.uses_index()
                    && (any_cost || expr.is_cheap()) => {}
            Kernel::Rotary
                if matches!(
                    node.kernel,
                    Kernel::Attention { .. } | Kernel::CachedAttention { .. }
                ) && source.epilogue.is_none()
                    && source.prologues.is_empty()
                    && (input == 0
                        || (input == 1 && node.output.shape[node.output.shape.len() - 2] == 1)) => {
            }
            Kernel::Movement {
                kind:
                    Movement::Resize(Resize {
                        mode: ResizeMode::Nearest,
                        ..
                    }),
                ..
            } if source.epilogue.is_none() && source.prologues.is_empty() => {}
            _ => return None,
        }
        let read_once = node
            .reads()
            .filter(|other| other.buffer == view.buffer)
            .count()
            == 1;
        let dense = source.output.offset == 0 && source.second.is_none();
        let fits = node.prologue(input).is_none()
            && producer < consumer
            && reads_unchanged(kernels, uses, source, producer, consumer)
            && fusible(view.buffer)
            && uses.readers[view.buffer] == [consumer]
            && read_once
            && dense;
        let mut bound: Vec<usize> = Vec::new();
        let reads = node.reads().filter(|other| other.buffer != view.buffer);
        for other in reads.chain(&source.inputs) {
            if !bound.contains(&other.buffer) {
                bound.push(other.buffer);
            }
        }
        let outputs = node.writes().count();
        (fits && bound.len() + outputs <= max_bindings).then_some((producer, input))
    })
}

/// Adds `view` to `inputs` unless it is already there, and returns its index.
fn push(inputs: &mut Vec<View>, view: &View) -> usize {
    match inputs.iter().position(|existing| existing == view) {
        Some(index) => index,
        None => {
            inputs.push(view.clone());
            inputs.len() - 1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::BinaryFn;
    use crate::ops::{Pool, PoolKind};
    use crate::view::Layout;

    fn dense(buffer: usize, shape: &[u64]) -> View {
        View::dense(buffer, shape.to_vec(), Layout::RowMajor)
    }

    fn node(kernel: Kernel, inputs: Vec<View>, output: View, op: &str) -> KernelNode {
        KernelNode::new(kernel, inputs, output, op)
    }

    fn add(a: View, b: View, output: View, op: &str) -> KernelNode {
        let expr = Expr::binary(BinaryFn::Add, Expr::Input(0), Expr::Input(1));
        node(Kernel::ElementWise(expr), vec![a, b], output, op)
    }

    fn relu(input: View, output: View, op: &str) -> KernelNode {
        let expr = Expr::binary(BinaryFn::Max, Expr::Input(0), Expr::f32(0.0));
        node(Kernel::ElementWise(expr), vec![input], output, op)
    }

    fn pool(input: View, output: View) -> KernelNode {
        let pool = Pool {
            kind: PoolKind::Max,
            kernel: vec![2],
            stride: vec![2],
            padding: vec![0],
            dilation: vec![1],
        };
        node(Kernel::Pool(pool), vec![input], output, "pool")
    }

    /// Buffer 0 is the input and 9 the output; the others are intermediates.
    fn fused(kernels: Vec<KernelNode>) -> Vec<KernelNode> {
        fuse(kernels, |buffer| buffer != 0 && buffer != 9, 8)
    }

    #[test]
    fn inlines_element_wise_chains() {
        let kernels = fused(vec![
            add(dense(0, &[4]), dense(0, &[4]), dense(1, &[4]), "a"),
            relu(dense(1, &[4]), dense(2, &[4]), "b"),
            add(dense(2, &[4]), dense(0, &[4]), dense(9, &[4]), "c"),
        ]);
        assert_eq!(kernels.len(), 1);
        let [kernel] = &kernels[..] else { panic!() };
        assert_eq!(kernel.inputs, [dense(0, &[4])]);
        assert_eq!(kernel.ops, ["a", "b", "c"]);
        assert_eq!(kernel.output.buffer, 9);
        let Kernel::ElementWise(expr) = &kernel.kernel else {
            panic!()
        };
        let x = || Expr::Input(0);
        let sum = Expr::binary(BinaryFn::Add, x(), x());
        let relu = Expr::binary(BinaryFn::Max, sum, Expr::f32(0.0));
        assert_eq!(*expr, Expr::binary(BinaryFn::Add, relu, x()));
    }

    #[test]
    fn appends_epilogues_to_other_kernels() {
        let kernels = fused(vec![
            pool(dense(0, &[8]), dense(1, &[4])),
            relu(dense(1, &[4]), dense(2, &[4]), "relu"),
            add(dense(2, &[4]), dense(3, &[4]), dense(9, &[4]), "add"),
        ]);
        let [kernel] = &kernels[..] else {
            panic!("{kernels:?}")
        };
        assert_eq!(kernel.ops, ["pool", "relu", "add"]);
        assert_eq!(kernel.output.buffer, 9);
        let epilogue = kernel.epilogue.as_ref().unwrap();
        assert_eq!(epilogue.inputs, [dense(3, &[4])]);
        let relu = Expr::binary(BinaryFn::Max, Expr::Acc, Expr::f32(0.0));
        assert_eq!(
            epilogue.expr,
            Expr::binary(BinaryFn::Add, relu, Expr::Input(0))
        );
    }

    #[test]
    fn keeps_tensors_read_twice_or_through_other_coordinates() {
        // `1` has two readers, and `2` is read transposed.
        let kernels = fused(vec![
            relu(dense(0, &[4]), dense(1, &[4]), "a"),
            relu(dense(1, &[4]), dense(2, &[2, 2]), "b"),
            add(dense(1, &[4]), dense(1, &[4]), dense(3, &[4]), "c"),
            relu(dense(2, &[2, 2]).permute(&[1, 0]), dense(9, &[2, 2]), "d"),
        ]);
        assert_eq!(kernels.len(), 4);
    }

    #[test]
    fn keeps_outputs_and_respects_the_binding_limit() {
        let chain = || {
            vec![
                relu(dense(0, &[4]), dense(9, &[4]), "a"),
                relu(dense(9, &[4]), dense(1, &[4]), "b"),
            ]
        };
        assert_eq!(fused(chain()).len(), 2);

        let kernels = vec![
            add(dense(0, &[4]), dense(3, &[4]), dense(1, &[4]), "a"),
            add(dense(1, &[4]), dense(4, &[4]), dense(9, &[4]), "b"),
        ];
        assert_eq!(fuse(kernels.clone(), |buffer| buffer == 1, 3).len(), 2);
        assert_eq!(fuse(kernels, |buffer| buffer == 1, 4).len(), 1);
    }

    #[test]
    fn only_moves_a_consumer_up_to_its_latest_producer() {
        // `add` reads the outputs of both pools: it can only become the
        // epilogue of the second, when both are ready.
        let kernels = fused(vec![
            pool(dense(0, &[8]), dense(1, &[4])),
            pool(dense(0, &[8]), dense(2, &[4])),
            add(dense(1, &[4]), dense(2, &[4]), dense(9, &[4]), "add"),
        ]);
        assert_eq!(kernels.len(), 2);
        assert_eq!(kernels[1].ops, ["pool", "add"]);
        assert_eq!(
            kernels[1].epilogue.as_ref().unwrap().inputs,
            [dense(1, &[4])]
        );
    }

    /// The buffers a kernel binds, its output included.
    fn bindings(node: &KernelNode) -> usize {
        let mut bound: Vec<usize> = Vec::new();
        for view in node.reads() {
            if !bound.contains(&view.buffer) {
                bound.push(view.buffer);
            }
        }
        bound.len() + 1
    }

    #[test]
    fn counts_fused_bindings_without_building_the_kernel() {
        let producer = add(dense(0, &[4]), dense(3, &[4]), dense(1, &[4]), "a");
        let consumer = add(dense(1, &[4]), dense(3, &[4]), dense(2, &[4]), "b");
        let expected = bindings(&inline(producer.clone(), consumer.clone()));
        assert_eq!(fused_bindings(&producer, &consumer, 1), expected);

        let mut producer = pool(dense(0, &[8]), dense(1, &[4]));
        producer.epilogue = Some(Epilogue {
            expr: Expr::binary(BinaryFn::Add, Expr::Acc, Expr::Input(0)),
            inputs: vec![dense(4, &[4])],
        });
        let consumer = add(dense(1, &[4]), dense(5, &[4]), dense(2, &[4]), "b");
        let expected = bindings(&absorb(producer.clone(), consumer.clone()));
        assert_eq!(fused_bindings(&producer, &consumer, 1), expected);
        assert_eq!(expected, 4);
    }
}
