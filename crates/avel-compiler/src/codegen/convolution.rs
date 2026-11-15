use super::*;

/// Implicit GEMM: output pixels are rows (`m`), output channels of a group are
/// columns, and the reduction runs over input channels times window positions
/// (`r`). A 16×16 workgroup computes a tile of `16 * tm` rows by `16 * tn`
/// columns, each invocation `tm × tn` outputs, staging 16-wide slices of both
/// operands in workgroup memory. Invocations load the same rows on every slice,
/// so they split them into pixel coordinates once. With a split reduction,
/// workgroups each cover one part of it and store partial sums.
pub(super) fn gemm_convolution(
    io: Io,
    convolution: &Convolution,
    path: ConvPath,
    tiling: Tiling,
) -> (String, [u32; 3]) {
    let node = io.node;
    let inputs = &node.inputs;
    let output = node.output.clone();
    let (input, weight) = (&inputs[0], &inputs[1]);
    let spatial = input.shape.len() - 2;
    let groups = convolution.groups;
    let group_channels = input.shape[1] / groups;
    let group_outputs = if convolution.transposed {
        weight.shape[1]
    } else {
        weight.shape[0] / groups
    };
    let disjoint = path == ConvPath::TransposeDisjoint;
    let kernel = &weight.shape[2..];
    let window_size: u64 = kernel.iter().product();
    let reduction = group_channels * if disjoint { 1 } else { window_size };
    let columns = group_outputs * if disjoint { window_size } else { 1 };
    // Partial sums have a leading axis, one entry per slice of the reduction.
    let lead = usize::from(tiling.split > 1);
    let out_spatial = if disjoint {
        &input.shape[2..]
    } else {
        &output.shape[lead + 2..]
    };
    let pixels: u64 = out_spatial.iter().product();
    let rows = output.shape[lead] * pixels;
    let slice = reduction.div_ceil(tiling.split).next_multiple_of(TILE);

    let (tm, tn) = (tiling.rows, tiling.columns);
    let (bm, bn) = (TILE * tm, TILE * tn);
    let column_tiles = columns.div_ceil(bn);
    let tiles = groups * column_tiles;

    // Splits row `m{i}` into the image `n{i}` and output position `q{axis}_{i}`.
    let mut setup = String::new();
    for i in 0..tm {
        wgsl!(
            to = setup,
            "convolution/row.wgsl",
            i = i,
            offset = TILE * i,
            pixels = pixels,
        );
        for (axis, (&size, stride)) in out_spatial
            .iter()
            .zip(row_major_strides(out_spatial))
            .enumerate()
        {
            let coord = coordinate(&format!("p{i}"), stride, size);
            writeln!(setup, "let q{axis}_{i} = {coord};").unwrap();
        }
        for j in 0..tn {
            writeln!(setup, "var acc{i}_{j} = 0.0;").unwrap();
        }
    }

    // Splits `r{suffix}` into the channel `c{suffix}` and window position
    // `w{axis}{suffix}`. With channels innermost in the input, they are
    // innermost in `r` too, so neighbouring invocations read neighbouring
    // elements.
    let channels_inner = input.strides[1] == 1 && spatial > 0;
    let split = |suffix: &str| {
        if path == ConvPath::Pointwise || disjoint {
            return format!("let c{suffix} = r{suffix};");
        }
        let (channel, position) = match channels_inner {
            true => (
                format!("r{suffix} % {group_channels}u"),
                format!("r{suffix} / {group_channels}u"),
            ),
            false => (
                format!("r{suffix} / {window_size}u"),
                format!("r{suffix} % {window_size}u"),
            ),
        };
        let mut code = format!("let c{suffix} = {channel};\nlet window{suffix} = {position};\n");
        for (axis, (&size, stride)) in kernel.iter().zip(row_major_strides(kernel)).enumerate() {
            let coord = coordinate(&format!("window{suffix}"), stride, size);
            writeln!(code, "let w{axis}{suffix} = {coord};").unwrap();
        }
        code
    };

    // Loads row `i` of the slice into `tile_a`, at column `local.x`.
    let mut load_a = String::new();
    for i in 0..tm {
        let mut checks = vec![format!("m{i} < {rows}u"), format!("ra < {reduction}u")];
        let mut code = String::new();
        let mut terms = vec![(format!("n{i}"), input.strides[0])];
        match path {
            ConvPath::Pointwise | ConvPath::TransposeDisjoint => {
                terms.push((format!("(g * {group_channels}u + ca)"), input.strides[1]));
                terms.extend(
                    (0..spatial).map(|axis| (format!("q{axis}_{i}"), input.strides[2 + axis])),
                );
            }
            _ => {
                terms.push((format!("(g * {group_channels}u + ca)"), input.strides[1]));
                for axis in 0..spatial {
                    let size = input.shape[2 + axis];
                    let (stride, padding, dilation) = (
                        convolution.stride[axis],
                        convolution.padding[axis],
                        convolution.dilation[axis],
                    );
                    let coordinate = if convolution.transposed {
                        writeln!(code, "let numerator{axis} = q{axis}_{i} + {padding}u;").unwrap();
                        writeln!(code, "let kernel{axis} = w{axis}a * {dilation}u;").unwrap();
                        writeln!(code, "let delta{axis} = numerator{axis} - kernel{axis};")
                            .unwrap();
                        writeln!(code, "let i{axis} = delta{axis} / {stride}u;").unwrap();
                        checks.push(format!(
                            "numerator{axis} >= kernel{axis} && i{axis} < {size}u"
                        ));
                        if stride != 1 {
                            checks.push(format!("delta{axis} % {stride}u == 0u"));
                        }
                        format!("i{axis}")
                    } else {
                        let position = format!("q{axis}_{i} * {stride}u + w{axis}a * {dilation}u");
                        let padded_size = convolution.input_pad.as_ref().map_or(size, |pad| {
                            let (left, right) = pad.widths[2 + axis];
                            (size as i64 + left + right) as u64
                        });
                        if padding == 0 {
                            writeln!(code, "let i{axis} = {position};").unwrap();
                        } else {
                            writeln!(code, "let h{axis} = i32({position}) - {padding};").unwrap();
                            writeln!(code, "let i{axis} = u32(h{axis});").unwrap();
                            checks.push(format!("h{axis} >= 0 && h{axis} < {padded_size}"));
                        }
                        if let Some(pad) = &convolution.input_pad {
                            super::movement::pad_position(
                                &mut code,
                                &mut checks,
                                &format!("p{axis}"),
                                &format!("i{axis}"),
                                size,
                                pad,
                                2 + axis,
                            )
                        } else {
                            format!("i{axis}")
                        }
                    };
                    terms.push((coordinate, input.strides[2 + axis]));
                }
            }
        }
        wgsl!(
            to = load_a,
            "convolution/load_a.wgsl",
            positions = code,
            checks = checks.join(" && "),
            ty = io.operand_type(),
            element = io.operand(&io.read(0, &linear(input.offset, &terms))),
            bm = bm,
            offset = TILE * i,
        );
    }

    // Loads column `j` of the slice into `tile_b`, at row `local.y`.
    let mut load_b = String::new();
    for j in 0..tn {
        let mut terms = if convolution.transposed {
            let out = if disjoint {
                format!("(column / {window_size}u)")
            } else {
                "column".to_owned()
            };
            vec![
                (format!("(g * {group_channels}u + cb)"), weight.strides[0]),
                (out, weight.strides[1]),
            ]
        } else {
            vec![
                (
                    format!("(g * {group_outputs}u + column)"),
                    weight.strides[0],
                ),
                ("cb".to_owned(), weight.strides[1]),
            ]
        };
        if disjoint {
            for (axis, (&size, stride)) in kernel.iter().zip(row_major_strides(kernel)).enumerate()
            {
                terms.push((coordinate("column", stride, size), weight.strides[2 + axis]));
            }
        } else if path != ConvPath::Pointwise {
            terms.extend((0..spatial).map(|axis| (format!("w{axis}b"), weight.strides[2 + axis])));
        }
        wgsl!(
            to = load_b,
            "convolution/load_b.wgsl",
            offset = TILE * j,
            reduction = reduction,
            group_outputs = columns,
            ty = io.operand_type(),
            element = io.operand(&io.read(1, &linear(weight.offset, &terms))),
            bn = bn,
        );
    }

    let mut multiply = String::new();
    for i in 0..tm {
        writeln!(
            multiply,
            "let a{i} = tile_a[k * {bm}u + local.y + {}u];",
            TILE * i
        )
        .unwrap();
    }
    for j in 0..tn {
        writeln!(
            multiply,
            "let b{j} = tile_b[k * {bn}u + local.x + {}u];",
            TILE * j
        )
        .unwrap();
    }
    for i in 0..tm {
        for j in 0..tn {
            let product = io.accumulate_product(&format!("a{i}"), &format!("b{j}"));
            writeln!(multiply, "acc{i}_{j} += {product};").unwrap();
        }
    }

    let coordinates = |view: &View| {
        let lead = view.shape.len() - spatial - 2;
        let mut terms: Vec<(String, u64)> = (0..lead)
            .map(|_| ("slice".to_owned(), view.strides[0]))
            .collect();
        terms.push(("n".to_owned(), view.strides[lead]));
        terms.push(("channel".to_owned(), view.strides[lead + 1]));
        terms.extend((0..spatial).map(|axis| (format!("q{axis}"), view.strides[lead + 2 + axis])));
        linear(view.offset, &terms)
    };
    let mut stores = String::new();
    for i in 0..tm {
        for j in 0..tn {
            let value = match inputs.get(2) {
                Some(bias) => format!(
                    "acc{i}_{j} + {}",
                    io.read(
                        2,
                        &linear(bias.offset, &[("channel".to_owned(), bias.strides[0])])
                    )
                ),
                None => format!("acc{i}_{j}"),
            };
            let store = io.store(&coordinates(&output), &value, &coordinates);
            let mut position = String::new();
            let window_strides = row_major_strides(kernel);
            for axis in 0..spatial {
                if disjoint {
                    let size = kernel[axis];
                    let local = coordinate("column", window_strides[axis], size);
                    writeln!(position, "let q{axis} = q{axis}_{i} * {size}u + {local};").unwrap();
                } else {
                    writeln!(position, "let q{axis} = q{axis}_{i};").unwrap();
                }
            }
            let out_column = if disjoint {
                format!("column / {window_size}u")
            } else {
                "column".to_owned()
            };
            wgsl!(
                to = stores,
                "convolution/store.wgsl",
                i = i,
                offset = TILE * j,
                rows = rows,
                columns = columns,
                channel = format!("g * {group_outputs}u + {out_column}"),
                position = position,
                store = store,
            );
        }
    }

    let wgsl = wgsl!(
        "convolution/gemm.wgsl",
        declarations = io.declarations(),
        helpers = io.helpers(),
        ty = io.operand_type(),
        area_a = TILE * bm,
        area_b = TILE * bn,
        tile = TILE,
        entry_point = ENTRY_POINT,
        max_workgroups = MAX_WORKGROUPS,
        bm = bm,
        bn = bn,
        tiles = tiles,
        column_tiles = column_tiles,
        slice = slice,
        reduction = reduction,
        setup = setup,
        split_a = split("a"),
        load_a = load_a,
        split_b = split("b"),
        load_b = load_b,
        multiply = multiply,
        stores = stores,
    );

    let row_tiles = rows.div_ceil(bm);
    let workgroups = [
        row_tiles.min(MAX_WORKGROUPS) as u32,
        (tiling.split * tiles) as u32,
        row_tiles.div_ceil(MAX_WORKGROUPS) as u32,
    ];
    (wgsl, workgroups)
}
