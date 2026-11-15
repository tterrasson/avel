use super::*;
use crate::ops::Padding;

/// Loops over a window of `kernel` positions on the trailing `spatial` axes,
/// binding `i{axis}` to each input position and skipping the padding. `inner`
/// runs once per position inside the input.
struct Window<'a> {
    input: &'a View,
    lead: usize,
    kernel: &'a [u64],
    stride: &'a [u64],
    padding: &'a [u64],
    dilation: &'a [u64],
    input_pad: Option<&'a Padding>,
}

impl Window<'_> {
    fn loops(&self, inner: &str, output: &[u64]) -> String {
        let mut inner = inner.to_owned();
        // Wrap from the innermost axis outward, preserving window order.
        for axis in (0..self.kernel.len()).rev() {
            let size = self.input.shape[self.lead + axis];
            let (kernel, stride, padding, dilation) = (
                self.kernel[axis],
                self.stride[axis],
                self.padding[axis],
                self.dilation[axis],
            );
            let out_axis = self.lead + axis;
            let position = format!("o{out_axis} * {stride}u + w{axis} * {dilation}u");
            let position = if let Some(pad) = self.input_pad {
                let (left, right) = pad.widths[out_axis];
                let padded_size = (size as i64 + left + right) as u64;
                let padded = format!("padded{axis}");
                let mut code = format!("let {padded} = i32({position}) - {padding};\n");
                let mut guard = Vec::new();
                if padding != 0 {
                    guard.push(format!("{padded} >= 0 && {padded} < {padded_size}"));
                }
                let index = super::movement::pad_position(
                    &mut code,
                    &mut guard,
                    &format!("p{axis}"),
                    &padded,
                    size,
                    pad,
                    out_axis,
                );
                if !guard.is_empty() {
                    writeln!(code, "if (!({})) {{ continue; }}", guard.join(" && ")).unwrap();
                }
                writeln!(code, "let i{axis} = u32({index});").unwrap();
                code
            } else if padding == 0
                && (output[out_axis] - 1) * stride + (kernel - 1) * dilation < size
            {
                format!("let i{axis} = {position};")
            } else {
                wgsl!(
                    "window/padded_position.wgsl",
                    axis = axis,
                    position = position,
                    padding = padding,
                    size = size,
                )
            };
            inner = wgsl!(
                "window/axis.wgsl",
                axis = axis,
                kernel = kernel,
                position = position,
                inner = inner,
            );
        }
        inner
    }

    /// Offset of the input element at `i0, i1, ...` from `base`.
    fn position(&self) -> String {
        let terms: Vec<(String, u64)> = (0..self.kernel.len())
            .map(|axis| (format!("i{axis}"), self.input.strides[self.lead + axis]))
            .collect();
        linear(0, &terms)
    }
}

pub(super) fn pool_body(io: &Io, pool: &Pool) -> String {
    let node = io.node;
    let input = &node.inputs[0];
    let adaptive = pool.kind == PoolKind::AdaptiveMax;
    let spatial_rank = if adaptive { 2 } else { pool.kernel.len() };
    let lead = input.shape.len() - spatial_rank;
    let window = Window {
        input,
        lead,
        kernel: &pool.kernel,
        stride: &pool.stride,
        padding: &pool.padding,
        dilation: &pool.dilation,
        input_pad: None,
    };
    let base: Vec<(String, u64)> = (0..lead)
        .map(|axis| (format!("o{axis}"), input.strides[axis]))
        .collect();
    let spatial = &input.shape[lead..];
    let element = io.read(
        0,
        &format!(
            "base + {}",
            linear(
                0,
                &(0..spatial_rank)
                    .map(|axis| (format!("i{axis}"), input.strides[lead + axis]))
                    .collect::<Vec<_>>()
            )
        ),
    );
    if matches!(pool.kind, PoolKind::Max | PoolKind::AdaptiveMax) {
        io.add_helper(EXTREMA);
    }
    let (setup, step, finalize, value) = match pool.kind {
        // Starts from the first element rather than -inf, which WGSL cannot
        // spell as a constant.
        PoolKind::Max | PoolKind::AdaptiveMax if node.second.is_none() => (
            "var first = true;",
            wgsl!("window/max_step.wgsl", element = element,),
            String::new(),
            "acc".to_owned(),
        ),
        // With indices, the first maximum in window order wins, as in
        // PyTorch: the index of an element in its flattened input plane.
        PoolKind::Max | PoolKind::AdaptiveMax => {
            let terms: Vec<(String, u64)> = row_major_strides(spatial)
                .into_iter()
                .enumerate()
                .map(|(axis, stride)| (format!("i{axis}"), stride))
                .collect();
            (
                "var first = true;\nvar index = 0i;",
                wgsl!(
                    "window/max_index_step.wgsl",
                    element = element,
                    index = linear(0, &terms),
                ),
                io.store_second_k("index"),
                "acc".to_owned(),
            )
        }
        PoolKind::Avg {
            count_include_pad,
            divisor,
        } => {
            let (finalize, divisor) = match divisor {
                Some(divisor) => (String::new(), float(divisor as f32)),
                None => (
                    average_count(input, pool, lead, count_include_pad),
                    "count".to_owned(),
                ),
            };
            (
                "",
                format!("acc += {element};"),
                finalize,
                format!("acc / {divisor}"),
            )
        }
    };
    let loops = if adaptive {
        (0..spatial_rank).rev().fold(step, |inner, axis| {
            let out_axis = lead + axis;
            let (start, end) = adaptive_bounds(
                input.shape[out_axis],
                node.output.shape[out_axis],
                &format!("o{out_axis}"),
            );
            format!("for (var i{axis} = {start}; i{axis} < {end}; i{axis}++) {{\n{inner}\n}}")
        })
    } else {
        window.loops(&step, &node.output.shape)
    };
    wgsl!(
        "window/accumulate.wgsl",
        coordinates = output_coordinates(io.output(), "o"),
        base = linear(input.offset, &base),
        setup = setup,
        window = loops,
        finalize = finalize,
        store = io.store_k(&value),
    )
}

/// PyTorch clips each window to the padded input, then to the input when
/// padding is excluded from the denominator.
fn average_count(input: &View, pool: &Pool, lead: usize, count_include_pad: bool) -> String {
    let mut code = String::new();
    let mut factors = Vec::new();
    for axis in 0..pool.kernel.len() {
        let out_axis = lead + axis;
        let size = input.shape[out_axis];
        let (kernel, stride, padding) = (pool.kernel[axis], pool.stride[axis], pool.padding[axis]);
        wgsl!(
            to = code,
            "window/average_bounds.wgsl",
            axis = axis,
            out_axis = out_axis,
            stride = stride,
            padding = padding,
            kernel = kernel,
            padded_size = size + padding,
        );
        factors.push(if count_include_pad {
            format!("(end{axis} - start{axis})")
        } else {
            format!("(min(end{axis}, {size}) - max(start{axis}, 0))")
        });
    }
    write!(code, "let count = f32({});", factors.join(" * ")).unwrap();
    code
}

pub(super) fn depthwise_body(io: &Io, convolution: &Convolution) -> String {
    let node = io.node;
    let (input, weight) = (&node.inputs[0], &node.inputs[1]);
    let multiplier = weight.shape[0] / input.shape[1];
    let window = Window {
        input,
        lead: 2,
        kernel: &weight.shape[2..],
        stride: &convolution.stride,
        padding: &convolution.padding,
        dilation: &convolution.dilation,
        input_pad: convolution.input_pad.as_ref(),
    };
    let channel = match multiplier {
        1 => "o1".to_owned(),
        _ => format!("o1 / {multiplier}u"),
    };
    let base = linear(
        input.offset,
        &[
            ("o0".to_owned(), input.strides[0]),
            (channel, input.strides[1]),
        ],
    );
    let kernel_terms: Vec<(String, u64)> = (0..weight.shape.len() - 2)
        .map(|axis| (format!("w{axis}"), weight.strides[2 + axis]))
        .collect();
    let mut kernel_base = vec![("o1".to_owned(), weight.strides[0])];
    kernel_base.extend(kernel_terms);
    let inner = format!(
        "acc += {} * {};",
        io.read(0, &format!("base + {}", window.position())),
        io.read(1, &linear(weight.offset, &kernel_base)),
    );
    let value = match node.inputs.get(2) {
        Some(bias) => format!(
            "acc + {}",
            io.read(
                2,
                &linear(bias.offset, &[("o1".to_owned(), bias.strides[0])])
            )
        ),
        None => "acc".to_owned(),
    };
    wgsl!(
        "window/accumulate.wgsl",
        coordinates = output_coordinates(io.output(), "o"),
        base = base,
        setup = "",
        window = window.loops(&inner, &node.output.shape),
        finalize = "",
        store = io.store_k(&value),
    )
}

pub(super) fn adaptive_bounds(size: u64, out: u64, coord: &str) -> (String, String) {
    if size == out {
        (coord.to_owned(), format!("{coord} + 1u"))
    } else {
        (
            format!("{coord} * {size}u / {out}u"),
            format!("((({coord} + 1u) * {size}u - 1u) / {out}u) + 1u"),
        )
    }
}
