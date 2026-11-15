use super::*;

/// The batch axes of a matmul output: `b{axis}` from the flattened batch
/// index `batch`, and the index of an element of `view` at batch coordinates,
/// row `row` and column `column`.
struct Batch<'a> {
    shape: &'a [u64],
}

impl Batch<'_> {
    fn new(output: &View) -> Batch<'_> {
        Batch {
            shape: &output.shape[..output.shape.len() - 2],
        }
    }

    fn count(&self) -> u64 {
        self.shape.iter().product()
    }

    /// Declares `b{axis}` for each batch axis, from `index`.
    fn coordinates(&self, index: &str) -> String {
        let strides = row_major_strides(self.shape);
        let mut lines = String::new();
        for (axis, (&size, stride)) in self.shape.iter().zip(strides).enumerate() {
            writeln!(lines, "let b{axis} = {};", coordinate(index, stride, size)).unwrap();
        }
        lines
    }

    fn at(&self, view: &View, row: &str, column: &str) -> String {
        let rank = view.shape.len();
        let mut terms: Vec<(String, u64)> = (0..self.shape.len())
            .map(|axis| (format!("b{axis}"), view.strides[axis]))
            .collect();
        terms.push((row.to_owned(), view.strides[rank - 2]));
        terms.push((column.to_owned(), view.strides[rank - 1]));
        linear(view.offset, &terms)
    }
}

/// `beta * bias + alpha * sum`, leaving out what is constant.
fn combine(io: &Io, bias: bool, beta: f32, alpha: f32, at: &dyn Fn(&View) -> String) -> String {
    let product = match alpha {
        1.0 => "sum".to_owned(),
        _ => format!("{} * sum", float(alpha)),
    };
    match bias {
        true => {
            let bias = io.read(2, &at(&io.node.inputs[2]));
            format!("{} * {bias} + {product}", float(beta))
        }
        false => product,
    }
}

/// A matmul staged through workgroup memory: each workgroup computes a
/// square tile of one batch entry, the batch being the third dispatch axis.
pub(super) fn tiled_matmul(
    io: Io,
    beta: f32,
    alpha: f32,
    bias: bool,
    tile: u32,
) -> (String, [u32; 3]) {
    let node = io.node;
    let (lhs, rhs, output) = (&node.inputs[0], &node.inputs[1], &node.output);
    let rank = output.shape.len();
    let (m, n, inner) = (
        output.shape[rank - 2],
        output.shape[rank - 1],
        lhs.shape[rank - 1],
    );
    let batch = Batch::new(output);
    let columns = n.div_ceil(u64::from(tile));
    let groups = columns * m.div_ceil(u64::from(tile));
    let gx = groups.min(MAX_WORKGROUPS);
    let at = |view: &View| batch.at(view, "row", "col");
    let value = combine(&io, bias, beta, alpha, &at);
    let store = io.store(&at(output), &value, &at);
    let wgsl = wgsl!(
        "tiled_matmul.wgsl",
        declarations = io.declarations(),
        helpers = io.helpers(),
        entry_point = ENTRY_POINT,
        n = n,
        m = m,
        inner = inner,
        area = tile * tile,
        tile = tile,
        gx = gx,
        groups = groups,
        columns = columns,
        batch = batch.coordinates("group.z"),
        ty = io.operand_type(),
        lhs = io.operand(&io.read(0, &batch.at(lhs, "row", "ak"))),
        rhs = io.operand(&io.read(1, &batch.at(rhs, "bk", "col"))),
        product = io.accumulate_product(
            &format!("a[local.y * {tile}u + k]"),
            &format!("b[k * {tile}u + local.x]"),
        ),
        store = store,
    );
    let workgroups = [gx as u32, groups.div_ceil(gx) as u32, batch.count() as u32];
    (wgsl, workgroups)
}

/// One invocation per output element, looping over the inner dimension.
pub(super) fn matmul_body(io: &Io, beta: f32, alpha: f32, bias: bool) -> String {
    let node = io.node;
    let (lhs, rhs, output) = (&node.inputs[0], &node.inputs[1], &node.output);
    let rank = output.shape.len();
    let batch = Batch::new(output);
    let row = format!("o{}", rank - 2);
    let column = format!("o{}", rank - 1);
    let mut coordinates = output_coordinates(output, "o");
    for axis in 0..rank - 2 {
        writeln!(coordinates, "let b{axis} = o{axis};").unwrap();
    }
    let value = combine(io, bias, beta, alpha, &|view| batch.at(view, &row, &column));
    wgsl!(
        "matmul.wgsl",
        coordinates = coordinates,
        inner = lhs.shape[rank - 1],
        product = io.accumulate_product(
            &io.operand(&io.read(0, &batch.at(lhs, &row, "i"))),
            &io.operand(&io.read(1, &batch.at(rhs, "i", &column))),
        ),
        store = io.store_k(&value),
    )
}
