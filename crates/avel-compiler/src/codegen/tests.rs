use super::*;
#[allow(unused_imports)]
use crate::kernel::Scalar;
use crate::kernel::{Epilogue, conv_path};
use crate::ops::{Convolution, Pool};
use crate::view::Layout;

fn dense(shape: &[u64]) -> View {
    View::dense(0, shape.to_vec(), Layout::RowMajor)
}

/// Validates a generated shader with naga, as CI does for every variant.
pub(crate) fn validate_wgsl(wgsl: &str) {
    // Naga parses subgroup operations, but not yet the directive browsers
    // require for them: drop it, the `SUBGROUP` capability checks the rest.
    let wgsl = &wgsl.replace("enable subgroups;", "");
    let module = naga::front::wgsl::parse_str(wgsl)
        .unwrap_or_else(|error| panic!("{}\n{wgsl}", error.emit_to_string(wgsl)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::SHADER_FLOAT16
            | naga::valid::Capabilities::TEXTURE_EXTERNAL
            | naga::valid::Capabilities::SUBGROUP,
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("{error:?}\n{wgsl}"));
}

fn check(wgsl: &str) {
    validate_wgsl(wgsl);
}

/// Generates and validates the kernel with every buffer in f32, then
/// again with an epilogue and every buffer but the first in f16.
fn validate(kernel: Kernel, inputs: Vec<View>, output: View) -> String {
    // Kernels never write a buffer they read, but for state updates.
    let output = match inputs.iter().any(|view| view.buffer == output.buffer) {
        true => View {
            buffer: 9,
            ..output
        },
        false => output,
    };
    let mut node = KernelNode {
        kernel,
        inputs,
        output,
        epilogue: None,
        second: None,
        prologues: Vec::new(),
        ops: Vec::new(),
    };
    let (shader, _) = generate(&node, &|_| Storage::default());
    check(&shader.wgsl);

    if !matches!(node.kernel, Kernel::ElementWise(_)) {
        let residual = View::dense(7, node.output.shape.clone(), Layout::RowMajor);
        let expr = Expr::binary(
            BinaryFn::Add,
            Expr::unary(UnaryFn::Gelu, Expr::Acc),
            Expr::Input(0),
        );
        node.epilogue = Some(Epilogue {
            expr,
            inputs: vec![residual],
        });
    }
    let first = node.inputs[0].buffer;
    let storage = |buffer| Storage {
        f16: buffer != first,
        packed_i8: false,
        ty: ScalarType::F32,
        writable: true,
        media: None,
    };
    let (mixed, _) = generate(&node, &storage);
    check(&mixed.wgsl);
    shader.wgsl
}

fn element_wise(expr: Expr, inputs: Vec<View>, output: View) -> String {
    validate(Kernel::ElementWise(expr), inputs, output)
}

#[test]
fn element_wise_shaders_are_valid() {
    let x = || Box::new(Expr::Input(0));
    for function in [
        UnaryFn::Gelu,
        UnaryFn::GeluTanh,
        UnaryFn::Sigmoid,
        UnaryFn::InverseSqrt,
    ] {
        element_wise(
            Expr::Unary(function, x()),
            vec![dense(&[2, 3])],
            dense(&[2, 3]),
        );
    }
    for function in [
        BinaryFn::Add,
        BinaryFn::Sub,
        BinaryFn::Mul,
        BinaryFn::Div,
        BinaryFn::Max,
        BinaryFn::Min,
    ] {
        let expr = Expr::binary(function, Expr::Input(0), Expr::f32(-6.0));
        element_wise(expr, vec![dense(&[3])], dense(&[3]));
    }
}

#[test]
fn element_wise_shaders_read_through_views() {
    let output = View::dense(1, vec![2, 3, 4, 5], Layout::ChannelsLast);
    let broadcast = dense(&[3, 1, 1]).broadcast_to(&output.shape);
    let permuted = dense(&[5, 4, 3, 2]).permute(&[3, 2, 1, 0]);
    let expr = Expr::binary(BinaryFn::Add, Expr::Input(0), Expr::Input(1));
    let wgsl = element_wise(expr, vec![broadcast, permuted], output);
    assert!(
        wgsl.contains("in0[(k % 3u) * 1u") || wgsl.contains("in0[k % 3u]"),
        "{wgsl}"
    );

    // A copy between identical layouts reads the same index it writes.
    let copy = element_wise(Expr::Input(0), vec![dense(&[4, 6])], dense(&[4, 6]));
    assert!(copy.contains("dst[k] = in0[k];"), "{copy}");
    let sliced = dense(&[4, 6]).slice(1, 1, 2, 2);
    let copy = element_wise(Expr::Input(0), vec![sliced], dense(&[4, 2]));
    assert!(copy.contains("+ 1u]"), "{copy}");
}

#[test]
fn matmul_shaders_are_valid_for_every_bias_shape() {
    let kernel = Kernel::Matmul {
        beta: 1.0,
        alpha: 0.5,
        bias: true,
    };
    for bias in [&[][..], &[1], &[4], &[3, 4], &[1, 4], &[3, 1]] {
        let bias = dense(bias).broadcast_to(&[3, 4]);
        let rhs = dense(&[4, 2]).permute(&[1, 0]);
        validate(
            kernel.clone(),
            vec![dense(&[3, 2]), rhs, bias],
            dense(&[3, 4]),
        );
    }
}

#[test]
fn reduction_shaders_are_valid() {
    for kind in [ReduceKind::Sum, ReduceKind::Mean] {
        let reduce = |axes: Vec<usize>| Kernel::Reduction {
            kind,
            axes,
            strategy: Strategy::Serial,
            split: 1,
        };
        validate(reduce(vec![1]), vec![dense(&[2, 3, 4])], dense(&[2, 4]));
        validate(reduce(vec![0, 1]), vec![dense(&[2, 3])], dense(&[]));
        validate(reduce(vec![1]), vec![dense(&[2, 3])], dense(&[2, 1]));
        let nhwc = View::dense(0, vec![1, 8, 4, 4], Layout::ChannelsLast);
        let pooled = View::dense(1, vec![1, 8, 1, 1], Layout::ChannelsLast);
        validate(reduce(vec![3, 2]), vec![nhwc], pooled);
    }
}

fn pool(kind: PoolKind, padding: u64) -> Kernel {
    Kernel::Pool(Pool {
        kind,
        kernel: vec![3, 3],
        stride: vec![2, 2],
        padding: vec![padding; 2],
        dilation: vec![1, 1],
    })
}

#[test]
fn pool_shaders_are_valid() {
    let kinds = [
        PoolKind::Max,
        PoolKind::Avg {
            count_include_pad: true,
            divisor: None,
        },
        PoolKind::Avg {
            count_include_pad: false,
            divisor: None,
        },
        PoolKind::Avg {
            count_include_pad: false,
            divisor: Some(4),
        },
    ];
    for kind in kinds {
        for padding in [0, 1] {
            let input = View::dense(0, vec![1, 4, 9, 9], Layout::ChannelsLast);
            validate(
                pool(kind, padding),
                vec![input],
                dense(&[1, 4, 4 + padding, 4 + padding]),
            );
        }
        validate(pool(kind, 0), vec![dense(&[4, 9, 9])], dense(&[4, 4, 4]));
    }
}

struct ConvCase {
    input: &'static [u64],
    weight: &'static [u64],
    groups: u64,
    padding: u64,
    stride: u64,
    output: &'static [u64],
    path: ConvPath,
}

impl ConvCase {
    fn kernel(&self, bias: bool) -> (Kernel, Vec<View>) {
        let spatial = self.input.len() - 2;
        let convolution = Convolution {
            stride: vec![self.stride; spatial],
            padding: vec![self.padding; spatial],
            dilation: vec![1; spatial],
            groups: self.groups,
            bias,
            transposed: false,
            input_pad: None,
        };
        let path = conv_path(&convolution, self.input, self.weight, self.output);
        assert_eq!(path, self.path);
        let mut inputs = vec![dense(self.input), dense(self.weight)];
        if bias {
            inputs.push(dense(&self.weight[..1]));
        }
        let kernel = Kernel::Convolution {
            convolution,
            path,
            tiling: Tiling::default(),
        };
        (kernel, inputs)
    }
}

#[test]
fn convolution_shaders_are_valid_on_every_path() {
    let cases = [
        ConvCase {
            input: &[1, 3, 8, 8],
            weight: &[4, 3, 3, 3],
            groups: 1,
            padding: 1,
            stride: 2,
            output: &[1, 4, 4, 4],
            path: ConvPath::Gemm,
        },
        ConvCase {
            input: &[2, 4, 8, 8],
            weight: &[6, 2, 3, 3],
            groups: 2,
            padding: 0,
            stride: 1,
            output: &[2, 6, 6, 6],
            path: ConvPath::Gemm,
        },
        ConvCase {
            input: &[1, 4, 8, 8],
            weight: &[8, 4, 1, 1],
            groups: 1,
            padding: 0,
            stride: 1,
            output: &[1, 8, 8, 8],
            path: ConvPath::Pointwise,
        },
        ConvCase {
            input: &[1, 4, 8, 8],
            weight: &[8, 1, 3, 3],
            groups: 4,
            padding: 1,
            stride: 1,
            output: &[1, 8, 8, 8],
            path: ConvPath::Depthwise,
        },
        ConvCase {
            input: &[2, 3, 10],
            weight: &[5, 3, 3],
            groups: 1,
            padding: 1,
            stride: 1,
            output: &[2, 5, 10],
            path: ConvPath::Gemm,
        },
        ConvCase {
            input: &[1, 2, 4, 4, 4],
            weight: &[3, 2, 3, 3, 3],
            groups: 1,
            padding: 1,
            stride: 1,
            output: &[1, 3, 4, 4, 4],
            path: ConvPath::Gemm,
        },
    ];
    for case in cases {
        for bias in [false, true] {
            let (kernel, mut inputs) = case.kernel(bias);
            validate(kernel.clone(), inputs.clone(), dense(case.output));
            // The same convolution between channels-last tensors.
            inputs[0] = View::dense(0, case.input.to_vec(), Layout::ChannelsLast);
            let output = View::dense(1, case.output.to_vec(), Layout::ChannelsLast);
            validate(kernel, inputs, output);
        }
    }
}

#[test]
fn gemm_convolutions_are_valid_for_every_tiling() {
    let tiling = |rows, columns, split| Tiling {
        rows,
        columns,
        split,
    };
    for tiling in [
        tiling(4, 4, 1),
        tiling(1, 2, 1),
        tiling(4, 1, 1),
        tiling(2, 4, 4),
    ] {
        let convolution = Convolution {
            stride: vec![1, 1],
            padding: vec![1, 1],
            dilation: vec![1, 1],
            groups: 1,
            bias: false,
            transposed: false,
            input_pad: None,
        };
        // Partial sums gain a leading axis, one per slice of the reduction.
        let mut shape = vec![1, 32, 7, 7];
        if tiling.split > 1 {
            shape.insert(0, tiling.split);
        }
        let node = KernelNode {
            kernel: Kernel::Convolution {
                convolution,
                path: ConvPath::Gemm,
                tiling,
            },
            inputs: vec![
                View::dense(0, vec![1, 64, 7, 7], Layout::ChannelsLast),
                dense(&[32, 64, 3, 3]),
            ],
            output: View::dense(2, shape, Layout::RowMajor),
            epilogue: None,
            second: None,
            prologues: Vec::new(),
            ops: Vec::new(),
        };
        let (shader, workgroups) = generate(&node, &|_| Storage::default());
        check(&shader.wgsl);
        let last = format!("acc{}_{}", tiling.rows - 1, tiling.columns - 1);
        assert!(shader.wgsl.contains(&last), "{}", shader.wgsl);
        let tiles = |size: u64, block: u64| size.div_ceil(TILE * block) as u32;
        let expected = [
            tiles(49, tiling.rows),
            tiles(32, tiling.columns) * tiling.split as u32,
            1,
        ];
        assert_eq!(workgroups, expected);
    }
}

#[test]
fn epilogues_read_the_computed_value_as_a_whole() {
    let mut inputs = vec![dense(&[1, 4, 8, 8]), dense(&[4, 1, 3, 3]), dense(&[4])];
    inputs[0].buffer = 1;
    let convolution = Convolution {
        stride: vec![1, 1],
        padding: vec![1, 1],
        dilation: vec![1, 1],
        groups: 4,
        bias: true,
        transposed: false,
        input_pad: None,
    };
    let node = KernelNode {
        kernel: Kernel::Convolution {
            convolution,
            path: ConvPath::Depthwise,
            tiling: Tiling::default(),
        },
        inputs,
        output: View::dense(9, vec![1, 4, 8, 8], Layout::RowMajor),
        epilogue: Some(Epilogue {
            expr: Expr::binary(BinaryFn::Mul, Expr::Acc, Expr::f32(2.0)),
            inputs: Vec::new(),
        }),
        second: None,
        prologues: Vec::new(),
        ops: Vec::new(),
    };
    let (shader, _) = generate(&node, &|_| Storage::default());
    // The weight and the bias share buffer 0, so they share one binding.
    assert!(
        shader.wgsl.contains("dst[k] = ((acc + weight[o1]) * 2.0);"),
        "{}",
        shader.wgsl
    );
}

#[test]
fn layer_norm_shaders_are_valid() {
    for count in 1..=3 {
        let kernel = Kernel::LayerNorm {
            eps: 1e-5,
            normalized: vec![64],
            rms: false,
            weight: count >= 2,
            strategy: Strategy::Serial,
        };
        let mut inputs = vec![dense(&[2, 17, 64])];
        if count >= 2 {
            inputs.push(dense(&[64]));
        }
        if count == 3 {
            inputs.push(dense(&[64]));
        }
        validate(kernel, inputs, dense(&[2, 17, 64]));
    }
    // Normalized over two trailing axes, read from a channels-last view.
    let kernel = Kernel::LayerNorm {
        eps: 1e-3,
        normalized: vec![4, 8],
        rms: false,
        weight: false,
        strategy: Strategy::Serial,
    };
    let input = View::dense(0, vec![2, 3, 4, 8], Layout::ChannelsLast);
    validate(kernel, vec![input], dense(&[2, 3, 4, 8]));
}

#[test]
fn attention_shaders_are_valid() {
    let q = || dense(&[2, 4, 17, 16]);
    for (causal, mask) in [(false, false), (true, false), (false, true)] {
        let kernel = Kernel::Attention {
            scale: 0.25,
            causal,
            mask,
        };
        let mut inputs = vec![q(), q(), dense(&[2, 4, 17, 32])];
        if mask {
            inputs.push(dense(&[2, 1, 17, 17]));
        }
        let output = dense(&[2, 4, 17, 32]);
        validate(kernel, inputs, output);
    }
    // Rank-3 inputs (no batch), value depth different from the query depth.
    let kernel = Kernel::Attention {
        scale: 0.5,
        causal: true,
        mask: false,
    };
    validate(
        kernel,
        vec![dense(&[4, 9, 8]), dense(&[4, 9, 8]), dense(&[4, 9, 4])],
        dense(&[4, 9, 4]),
    );
}

#[test]
fn attention_reads_bool_masks_as_u32() {
    let node = KernelNode {
        kernel: Kernel::Attention {
            scale: 0.25,
            causal: false,
            mask: true,
        },
        inputs: vec![
            dense(&[1, 2, 5, 8]),
            dense(&[1, 2, 5, 8]),
            dense(&[1, 2, 5, 8]),
            View::dense(5, vec![1, 1, 5, 5], Layout::RowMajor),
        ],
        output: View::dense(9, vec![1, 2, 5, 8], Layout::RowMajor),
        epilogue: None,
        second: None,
        prologues: Vec::new(),
        ops: Vec::new(),
    };
    let (shader, workgroups) = generate(&node, &|buffer| Storage {
        ty: match buffer {
            5 => ScalarType::Bool,
            _ => ScalarType::F32,
        },
        ..Storage::default()
    });
    check(&shader.wgsl);
    assert!(shader.wgsl.contains("mask: array<u32>"), "{}", shader.wgsl);
    assert!(shader.wgsl.contains("!= 0u"), "{}", shader.wgsl);
    // One invocation per query row: 1 batch × 2 heads × 5 queries.
    assert_eq!(workgroups, [1, 1, 1]);
    // The lowest score is a literal f32 can represent when parsed as a decimal.
    assert!(shader.wgsl.contains("(-3.4e38)"), "{}", shader.wgsl);
    assert!(!shader.wgsl.contains("3.4028235"), "{}", shader.wgsl);
}

#[test]
fn views_of_one_buffer_share_a_binding() {
    // q, k and v are different views of the same buffer: one binding,
    // as WebGPU rejects aliased writable storage bindings.
    let q = View::dense(0, vec![3, 1, 2, 5, 8], Layout::RowMajor);
    let k = q.clone().select(0, 1);
    let v = q.clone().select(0, 2);
    let node = KernelNode {
        kernel: Kernel::Attention {
            scale: 0.25,
            causal: false,
            mask: false,
        },
        inputs: vec![q, k, v],
        output: View::dense(9, vec![1, 2, 5, 8], Layout::RowMajor),
        epilogue: None,
        second: None,
        prologues: Vec::new(),
        ops: Vec::new(),
    };
    let (shader, _) = generate(&node, &|_| Storage::default());
    check(&shader.wgsl);
    assert!(
        shader.wgsl.contains("@binding(0) var<storage, read> query"),
        "{}",
        shader.wgsl
    );
    assert!(
        shader
            .wgsl
            .contains("@binding(1) var<storage, read_write> dst"),
        "{}",
        shader.wgsl
    );
    assert!(!shader.wgsl.contains("> key"), "{}", shader.wgsl);
    assert!(!shader.wgsl.contains("> value"), "{}", shader.wgsl);
    // The view offsets stay in the read expressions.
    assert!(shader.wgsl.contains("+ 80u"), "{}", shader.wgsl);
    assert!(shader.wgsl.contains("+ 160u"), "{}", shader.wgsl);
}

#[test]
fn embedding_shaders_are_valid() {
    for padding in [None, Some(3)] {
        let node = KernelNode {
            kernel: Kernel::Embedding { padding },
            inputs: vec![
                dense(&[128, 64]),
                View::dense(3, vec![2, 17], Layout::RowMajor),
            ],
            output: View::dense(9, vec![2, 17, 64], Layout::RowMajor),
            epilogue: None,
            second: None,
            prologues: Vec::new(),
            ops: Vec::new(),
        };
        let (shader, _) = generate(&node, &|buffer| Storage {
            ty: match buffer {
                3 => ScalarType::I32,
                _ => ScalarType::F32,
            },
            ..Storage::default()
        });
        check(&shader.wgsl);
        assert!(
            shader.wgsl.contains("indices: array<u32>"),
            "{}",
            shader.wgsl
        );
        assert!(!shader.wgsl.contains("select("), "{}", shader.wgsl);
    }
}

#[test]
fn cat_shaders_are_valid() {
    let kernel = Kernel::Cat {
        axis: 1,
        sizes: vec![1, 4],
    };
    validate(
        kernel,
        vec![dense(&[2, 1, 8]), dense(&[2, 4, 8])],
        dense(&[2, 5, 8]),
    );
    // Three inputs along axis 0, into a channels-last output.
    let kernel = Kernel::Cat {
        axis: 0,
        sizes: vec![2, 3, 1],
    };
    let output = View::dense(9, vec![6, 4, 8], Layout::ChannelsLast);
    validate(
        kernel,
        vec![dense(&[2, 4, 8]), dense(&[3, 4, 8]), dense(&[1, 4, 8])],
        output,
    );
}

#[test]
fn computes_coordinates() {
    assert_eq!(coordinate("k", 3, 2), "(k / 3u) % 2u");
    assert_eq!(coordinate("k", 1, 3), "k % 3u");
    assert_eq!(coordinate("k", 7, 1), "0u");
    assert_eq!(linear(0, &[]), "0u");
    assert_eq!(
        linear(5, &[("a".to_owned(), 0), ("b".to_owned(), 1)]),
        "b + 5u"
    );
}

#[test]
fn splits_dispatches_over_the_workgroup_limit() {
    assert_eq!(workgroups(1), [1, 1, 1]);
    assert_eq!(workgroups(65), [2, 1, 1]);
    assert_eq!(
        workgroups(WORKGROUP_SIZE * MAX_WORKGROUPS + 1),
        [65535, 2, 1]
    );
}

#[test]
fn single_input_concatenation_generates_valid_wgsl() {
    validate(
        Kernel::Cat {
            axis: 1,
            sizes: vec![4],
        },
        vec![dense(&[2, 4, 8])],
        dense(&[2, 4, 8]),
    );
}

fn storage(ty: ScalarType) -> Storage {
    Storage {
        ty,
        ..Storage::default()
    }
}

/// An element-wise kernel over typed buffers: buffer `i` holds `types[i]`,
/// the output the last one.
fn typed(expr: Expr, types: &[ScalarType]) -> String {
    let shape = [2, 3];
    let inputs = (0..types.len() - 1)
        .map(|buffer| View::dense(buffer, shape.to_vec(), Layout::RowMajor))
        .collect();
    let output = View::dense(types.len() - 1, shape.to_vec(), Layout::RowMajor);
    let node = KernelNode::new(Kernel::ElementWise(expr), inputs, output, "typed");
    let (shader, _) = generate(&node, &|buffer| storage(types[buffer]));
    check(&shader.wgsl);
    shader.wgsl
}

#[test]
fn typed_expressions_load_and_store_their_storage() {
    use ScalarType::{Bool, F32, I32};
    let (x, y) = (|| Expr::Input(0), || Expr::Input(1));
    // `where(x > 0, x, 0.1 * x)`: a bool in the middle of an f32 chain.
    let leaky = Expr::select(
        Expr::binary(BinaryFn::Gt, x(), Expr::f32(0.0)),
        x(),
        Expr::binary(BinaryFn::Mul, Expr::f32(0.1), x()),
    );
    let wgsl = typed(leaky, &[F32, F32]);
    assert!(wgsl.contains("select("), "{wgsl}");

    // Integer arithmetic and comparisons, stored as u32 words.
    let wgsl = typed(
        Expr::binary(BinaryFn::Add, x(), Expr::Const(Scalar::I32(-3))),
        &[I32, I32],
    );
    assert!(wgsl.contains("bitcast<i32>(in0[k])"), "{wgsl}");
    assert!(wgsl.contains("dst[k] = bitcast<u32>("), "{wgsl}");
    let wgsl = typed(Expr::binary(BinaryFn::Le, x(), y()), &[I32, I32, Bool]);
    assert!(wgsl.contains("select(0u, 1u,"), "{wgsl}");

    // Logic on bools, bitwise ops on integers, casts both ways.
    let logic = Expr::unary(
        UnaryFn::Not,
        Expr::binary(BinaryFn::Or, x(), Expr::binary(BinaryFn::Ne, x(), y())),
    );
    let wgsl = typed(logic, &[Bool, Bool, Bool]);
    assert!(wgsl.contains("(in0[k] != 0u)"), "{wgsl}");
    typed(
        Expr::binary(BinaryFn::Xor, x(), Expr::binary(BinaryFn::And, x(), y())),
        &[I32, I32, I32],
    );
    typed(Expr::cast(F32, x()), &[Bool, F32]);
    typed(Expr::cast(I32, x()), &[F32, I32]);
    typed(Expr::cast(Bool, x()), &[F32, Bool]);
    typed(Expr::Const(Scalar::I32(i32::MIN)), &[I32]);

    // f16 storage still computes in f32.
    let node = KernelNode::new(
        Kernel::ElementWise(Expr::cast(I32, x())),
        vec![dense(&[4])],
        View::dense(1, vec![4], Layout::RowMajor),
        "half",
    );
    let (shader, _) = generate(&node, &|buffer| match buffer {
        0 => Storage {
            f16: true,
            ..Storage::default()
        },
        _ => storage(I32),
    });
    check(&shader.wgsl);
}

#[test]
fn generators_read_output_coordinates() {
    // `arange` along the second axis of a [3, 5] tensor, without inputs.
    let expr = Expr::binary(
        BinaryFn::Add,
        Expr::Const(Scalar::I32(2)),
        Expr::binary(BinaryFn::Mul, Expr::Const(Scalar::I32(3)), Expr::Index(1)),
    );
    let node = KernelNode::new(
        Kernel::ElementWise(expr),
        Vec::new(),
        dense(&[3, 5]),
        "range",
    );
    let (shader, workgroups) = generate(&node, &|_| storage(ScalarType::I32));
    check(&shader.wgsl);
    assert!(shader.wgsl.contains("i32(k % 5u)"), "{}", shader.wgsl);
    assert_eq!(workgroups, [1, 1, 1]);
}

#[test]
fn batched_matmul_shaders_are_valid() {
    // `[2, 3, M, K] @ [2, 3, K, N]`, the right operand broadcast over the
    // first batch axis, with and without a bias, tiled or not.
    let lhs = dense(&[2, 3, 5, 7]);
    let rhs = View::dense(1, vec![3, 7, 4], Layout::RowMajor).broadcast_to(&[2, 3, 7, 4]);
    let output = View::dense(2, vec![2, 3, 5, 4], Layout::RowMajor);
    for bias in [false, true] {
        let mut inputs = vec![lhs.clone(), rhs.clone()];
        if bias {
            inputs.push(View::dense(3, vec![4], Layout::RowMajor).broadcast_to(&output.shape));
        }
        let matmul = Kernel::Matmul {
            beta: 0.5,
            alpha: 2.0,
            bias,
        };
        validate(matmul, inputs.clone(), output.clone());
        let tiled = Kernel::TiledMatmul {
            beta: 0.5,
            alpha: 2.0,
            bias,
            tile: 8,
        };
        let node = KernelNode::new(tiled, inputs, output.clone(), "bmm");
        let (shader, workgroups) = generate(&node, &|_| Storage::default());
        check(&shader.wgsl);
        assert_eq!(workgroups, [1, 1, 6]);
        assert!(
            shader.wgsl.contains("let b1 = (group.z % 3u)")
                || shader.wgsl.contains("let b1 = group.z % 3u"),
            "{}",
            shader.wgsl
        );
    }
}

#[test]
fn cooperative_reductions_are_valid() {
    let input = dense(&[3, 4096]);
    for subgroup in [false, true] {
        let lanes = if subgroup { 32 } else { 64 };
        let strategy = Strategy::Workgroup { lanes, subgroup };
        for kind in [ReduceKind::Sum, ReduceKind::Mean] {
            let kernel = Kernel::Reduction {
                kind,
                axes: vec![1],
                strategy,
                split: 1,
            };
            let wgsl = validate(
                kernel,
                vec![input.clone()],
                View::dense(1, vec![3], Layout::RowMajor),
            );
            assert!(wgsl.contains("workgroup_sum(acc, lane)"), "{wgsl}");
            assert_eq!(wgsl.contains("subgroupAdd"), subgroup, "{wgsl}");
        }
        // Split in 4 slices: partial results behind a slice axis.
        let kernel = Kernel::Reduction {
            kind: ReduceKind::Mean,
            axes: vec![1],
            strategy,
            split: 4,
        };
        let node = KernelNode::new(
            kernel,
            vec![input.clone()],
            View::dense(1, vec![4, 3], Layout::RowMajor),
            "split",
        );
        let (shader, workgroups) = generate(&node, &|_| Storage::default());
        check(&shader.wgsl);
        assert_eq!(workgroups, [3, 1, 4]);
        assert!(
            shader.wgsl.contains("dst[slice * 3u + k]"),
            "{}",
            shader.wgsl
        );

        for rms in [false, true] {
            let kernel = Kernel::LayerNorm {
                eps: 1e-6,
                normalized: vec![768],
                rms,
                weight: true,
                strategy,
            };
            let wgsl = validate(
                kernel,
                vec![
                    dense(&[2, 768]),
                    View::dense(1, vec![768], Layout::RowMajor).broadcast_to(&[2, 768]),
                ],
                View::dense(2, vec![2, 768], Layout::RowMajor),
            );
            assert_eq!(wgsl.contains("let mean = 0.0;"), rms, "{wgsl}");
        }
    }
    // Serial RMS norm.
    let kernel = Kernel::LayerNorm {
        eps: 1e-6,
        normalized: vec![8],
        rms: true,
        weight: false,
        strategy: Strategy::Serial,
    };
    validate(
        kernel,
        vec![dense(&[4, 8])],
        View::dense(1, vec![4, 8], Layout::RowMajor),
    );
}

#[test]
fn max_pools_write_their_indices() {
    let mut node = KernelNode::new(
        pool(PoolKind::Max, 1),
        vec![dense(&[1, 2, 7, 7])],
        View::dense(1, vec![1, 2, 4, 4], Layout::ChannelsLast),
        "pool",
    );
    node.second = Some(View::dense(2, vec![1, 2, 4, 4], Layout::ChannelsLast));
    let (shader, _) = generate(&node, &|buffer| match buffer {
        2 => storage(ScalarType::I32),
        _ => Storage::default(),
    });
    check(&shader.wgsl);
    assert!(
        shader
            .wgsl
            .contains("var<storage, read_write> dst1: array<u32>"),
        "{}",
        shader.wgsl
    );
    assert!(
        shader.wgsl.contains("index = i32(i0 * 7u + i1)"),
        "{}",
        shader.wgsl
    );
}

#[test]
fn prologues_compute_inputs_where_they_are_loaded() {
    // relu(x) transposed, then summed: the relu is evaluated at the
    // coordinates each load of the reduction has in its dense output.
    let source = View::dense(5, vec![4, 6], Layout::RowMajor);
    let mut node = KernelNode::new(
        Kernel::Reduction {
            kind: ReduceKind::Sum,
            axes: vec![1],
            strategy: Strategy::Serial,
            split: 1,
        },
        vec![source.permute(&[1, 0])],
        View::dense(2, vec![6], Layout::RowMajor),
        "sum",
    );
    node.prologues = vec![Some(crate::kernel::Prologue {
        kind: crate::kernel::PrologueKind::ElementWise(Expr::binary(
            BinaryFn::Max,
            Expr::Input(0),
            Expr::f32(0.0),
        )),
        inputs: vec![View::dense(0, vec![4, 6], Layout::ChannelsLast)],
        source,
    })];
    let (shader, _) = generate(&node, &|_| Storage::default());
    check(&shader.wgsl);
    assert!(shader.wgsl.contains("max(pro0["), "{}", shader.wgsl);
    assert!(!shader.wgsl.contains("src"), "{}", shader.wgsl);

    // A scaled left operand of a tiled matmul.
    let lhs = View::dense(5, vec![4, 8], Layout::RowMajor);
    let mut node = KernelNode::new(
        Kernel::TiledMatmul {
            beta: 0.0,
            alpha: 1.0,
            bias: false,
            tile: 16,
        },
        vec![lhs.clone(), View::dense(1, vec![8, 3], Layout::RowMajor)],
        View::dense(2, vec![4, 3], Layout::RowMajor),
        "mm",
    );
    node.prologues = vec![
        Some(crate::kernel::Prologue {
            kind: crate::kernel::PrologueKind::ElementWise(Expr::binary(
                BinaryFn::Mul,
                Expr::Input(0),
                Expr::f32(0.125),
            )),
            inputs: vec![View::dense(0, vec![4, 8], Layout::RowMajor)],
            source: lhs,
        }),
        None,
    ];
    let (shader, _) = generate(&node, &|_| Storage::default());
    check(&shader.wgsl);
    assert!(shader.wgsl.contains("* 0.125"), "{}", shader.wgsl);
}

#[test]
fn kernels_read_the_buffer_they_update_through_their_output() {
    // A state update `state = 0.9 * state + x`, element by element.
    let state = View::dense(3, vec![8], Layout::RowMajor);
    let expr = Expr::binary(
        BinaryFn::Add,
        Expr::binary(BinaryFn::Mul, Expr::f32(0.9), Expr::Input(0)),
        Expr::Input(1),
    );
    let node = KernelNode::new(
        Kernel::ElementWise(expr),
        vec![state.clone(), dense(&[8])],
        state,
        "update",
    );
    let (shader, _) = generate(&node, &|_| Storage::default());
    check(&shader.wgsl);
    assert!(
        shader.wgsl.contains("dst[k] = ((0.9 * dst[k]) + in1[k]);"),
        "{}",
        shader.wgsl
    );
    assert!(
        shader
            .wgsl
            .contains("@binding(1) var<storage, read_write> dst"),
        "{}",
        shader.wgsl
    );
}

#[test]
fn validates_order_and_generation_kernels() {
    let view = |buffer, shape: &[u64]| View::dense(buffer, shape.to_vec(), Layout::RowMajor);
    let storage = |buffer| Storage {
        ty: if matches!(buffer, 1 | 4) {
            ScalarType::I32
        } else {
            ScalarType::F32
        },
        ..Storage::default()
    };
    let source = view(0, &[3, 2189]);
    let indices = view(1, &[3, 3, 32]);
    let merged = view(4, &[3, 2, 32]);
    let kernels = [
        KernelNode::new(
            Kernel::OrderBlocks {
                axis: 1,
                largest: true,
                tile: 1024,
            },
            vec![source.clone()],
            indices.clone(),
            "blocks",
        ),
        KernelNode::new(
            Kernel::OrderMerge {
                axis: 1,
                largest: true,
                span: 1024,
            },
            vec![source, indices],
            merged,
            "merge",
        ),
        KernelNode::new(
            Kernel::CacheWrite { in_place: true },
            vec![view(0, &[2, 8, 4]), view(3, &[2, 1, 4]), view(1, &[])],
            view(0, &[2, 8, 4]),
            "write",
        ),
        KernelNode::new(
            Kernel::CacheWrite { in_place: false },
            vec![view(0, &[2, 8, 4]), view(3, &[2, 1, 4]), view(1, &[])],
            view(2, &[2, 8, 4]),
            "write",
        ),
        KernelNode::new(
            Kernel::CachedAttention { scale: 0.5 },
            vec![
                view(0, &[2, 1, 4]),
                view(2, &[2, 8, 4]),
                view(3, &[2, 8, 5]),
                view(1, &[]),
            ],
            view(5, &[2, 1, 5]),
            "attend",
        ),
        KernelNode::new(
            Kernel::SampleCandidates,
            vec![view(0, &[3, 32]), view(1, &[3, 32]), view(4, &[1])],
            view(4, &[3]),
            "sample",
        ),
        KernelNode::new(
            Kernel::OrderGather { axis: 1 },
            vec![view(0, &[3, 8]), view(1, &[3, 4])],
            view(2, &[3, 4]),
            "gather",
        ),
    ];
    for node in kernels {
        let (shader, _) = generate(&node, &storage);
        check(&shader.wgsl);
    }
}
