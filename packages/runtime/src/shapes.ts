import type { Graph } from "../wasm/avel_wasm.js";
import { AvelError } from "./errors.js";

/** Validate before cache lookup or rounding, so a cache hit cannot hide invalid inputs. */
export function validateShapes(
  graph: Graph,
  shapes: Record<string, number[]>,
  bucket: number,
): void {
  if (!Number.isSafeInteger(bucket) || bucket < 0)
    throw new AvelError("bucket must be a nonnegative safe integer");
  const symbols = new Map<string, number>();
  for (const name of Object.keys(shapes)) {
    if (!graph.inputs.some((input) => input.name === name))
      throw new AvelError(`unknown input \`${name}\``);
  }
  for (const input of graph.inputs) {
    const shape = shapes[input.name];
    if (!shape) throw new AvelError(`missing shape for input \`${input.name}\``);
    if (shape.length !== input.shape.length)
      throw new AvelError(
        `input \`${input.name}\` has rank ${shape.length}, expected ${input.shape.length}`,
      );
    input.shape.forEach((dim, axis) => {
      const size = shape[axis]!;
      if (
        !Number.isSafeInteger(size) ||
        size < 1 ||
        size > 0xffffffff ||
        (typeof dim === "number"
          ? dim !== size
          : (dim.min !== undefined && size < dim.min) || (dim.max !== undefined && size > dim.max))
      ) {
        throw new AvelError(
          `input \`${input.name}\` has size ${size} at dimension ${axis}, expected ${JSON.stringify(dim)}`,
        );
      }
      if (typeof dim === "object") {
        const previous = symbols.get(dim.name);
        if (previous !== undefined && previous !== size)
          throw new AvelError(`symbol \`${dim.name}\` has inconsistent sizes across the inputs`);
        symbols.set(dim.name, size);
      }
    });
  }
}

export function product(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

/** Pad each contiguous block along the one variable axis of an input. */
export function pad<T extends Float32Array | Int32Array | Uint32Array>(
  input: T,
  sourceBlock: number,
  targetBlock: number,
  out: T,
): T {
  out.fill(0);
  // Avoid subarray views and coordinate/stride arrays on the submission path.
  for (
    let source = 0, target = 0;
    source < input.length;
    source += sourceBlock, target += targetBlock
  ) {
    for (let i = 0; i < sourceBlock; i++) out[target + i] = input[source + i]!;
  }
  return out;
}

/** Copies a row-major output, trimming any trailing padded coordinates. */
export function trim<T extends Float32Array | Int32Array | Uint32Array>(
  source: T,
  from: number[],
  to: number[],
): T {
  if (to.length !== from.length) throw new AvelError("invalid trimmed output rank");
  let lastCut = -1;
  let block = 1;
  for (let axis = 0; axis < to.length; axis++) {
    const size = to[axis]!;
    if (!Number.isSafeInteger(size) || size < 1 || size > from[axis]!)
      throw new AvelError("invalid trimmed output shape");
    if (size !== from[axis]) {
      lastCut = axis;
      block = 1;
    }
    block *= size;
  }
  if (lastCut < 0) return source.slice() as T;

  // Everything from the last shortened axis inward is a contiguous block.
  // Resolve the source coordinates once per block, not for every element.
  const sourceStride = (block / to[lastCut]!) * from[lastCut]!;
  const out = new (source.constructor as new (length: number) => T)(product(to));
  for (let row = 0, target = 0; target < out.length; row++, target += block) {
    let remainder = row;
    let offset = 0;
    let stride = sourceStride;
    for (let axis = lastCut - 1; axis >= 0 && remainder > 0; axis--) {
      const coordinate = remainder % to[axis]!;
      remainder = (remainder - coordinate) / to[axis]!;
      offset += coordinate * stride;
      stride *= from[axis]!;
    }
    for (let i = 0; i < block; i++) out[target + i] = source[offset + i]!;
  }
  return out;
}

/** Evaluates a symbolic size expression such as `s27` or `2*s27`, as the graph names them. */
export function evalSymbol(expr: string, symbols: Record<string, number>): number | undefined {
  let total = 0;
  for (const sumTerm of expr.split("+")) {
    let productTerm = 1;
    for (const rawFactor of sumTerm.split("*")) {
      const factor = rawFactor.trim();
      const literal = Number(factor);
      const value = Number.isInteger(literal) && factor !== "" ? literal : symbols[factor];
      if (value === undefined) {
        return undefined;
      }
      productTerm *= value;
    }
    total += productTerm;
  }
  return total;
}

/**
 * The shapes a plan is compiled for: the given ones, with every symbolic input
 * dimension snapped up to a multiple of `bucket`, capped by its declared
 * maximum. Mirrors `effective_shapes` in the compiler.
 */
export function snapShapes(
  graph: Graph,
  inputShapes: Record<string, number[]>,
  bucket: number,
): [string, number[]][] {
  return Object.entries(inputShapes)
    .map(([name, shape]) => {
      const input = graph.inputs.find((candidate) => candidate.name === name);
      if (!bucket || !input) {
        return [name, shape] as [string, number[]];
      }
      return [
        name,
        shape.map((size, axis) => {
          const dim = input.shape[axis];
          if (typeof dim !== "object") {
            return size;
          }
          const snapped = Math.ceil(size / bucket) * bucket;
          return dim.max !== undefined && snapped > dim.max ? dim.max : snapped;
        }),
      ] as [string, number[]];
    })
    .sort(([a], [b]) => (a < b ? -1 : 1));
}
