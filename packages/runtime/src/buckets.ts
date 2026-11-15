import type { Graph } from "../wasm/avel_wasm.js";
import { AvelError } from "./errors.js";
import type { RunInput } from "./plan-inputs.js";
import { evalSymbol, product } from "./shapes.js";

interface Padding {
  compiled: number;
  minimum: number;
  fixed: number;
  inner: number;
  block: number;
}

interface SymbolAxis {
  name: string;
  index: number;
  compiled: number;
  inferred: boolean;
}

/** Resolves the actual input lengths and shared symbols of a bucketed run. */
export class BucketInputs {
  private readonly bindings: { name: string; padding: Padding | undefined }[];
  private readonly symbols: SymbolAxis[] = [];
  private readonly checks: { first: SymbolAxis; next: SymbolAxis }[] = [];

  constructor(
    graph: Graph,
    inputs: readonly { name: string; shape: number[] }[],
    isMedia: (index: number) => boolean,
  ) {
    const first = new Map<string, SymbolAxis>();
    this.bindings = inputs.map(({ name, shape }, index) => {
      let padding: Padding | undefined;
      const dims = graph.inputs[index]!.shape;
      const axes = dims.flatMap((dim, axis) => (typeof dim === "object" ? [axis] : []));
      for (const axis of axes) {
        const dim = dims[axis]!;
        if (typeof dim !== "object") continue;
        const compiled = shape[axis]!;
        const inferred = !isMedia(index) && axes.length === 1;
        if (inferred) {
          const count = product(shape);
          const inner = product(shape.slice(axis + 1));
          padding = {
            compiled,
            minimum: Math.max(1, dim.min ?? 1),
            fixed: count / compiled,
            inner,
            block: inner * compiled,
          };
        }
        const symbol = { name: dim.name, index, compiled, inferred };
        const previous = first.get(dim.name);
        if (!previous) {
          first.set(dim.name, symbol);
          this.symbols.push(symbol);
        } else if (previous.inferred || inferred) {
          this.checks.push({ first: previous, next: symbol });
        }
      }
      return { name, padding };
    });
  }

  padding(index: number): Padding | undefined {
    return this.bindings[index]!.padding;
  }

  dimensions(): Uint32Array | undefined {
    return this.bindings.some((binding) => binding.padding)
      ? new Uint32Array(this.bindings.length)
      : undefined;
  }

  /** Actual symbols, only when a CPU input was shorter than the compiled shape. */
  outputSymbols(dimensions: Uint32Array | undefined): Record<string, number> | undefined {
    if (!dimensions) return undefined;
    let padded = false;
    for (const symbol of this.symbols) {
      if (symbolSize(symbol, dimensions) !== symbol.compiled) {
        padded = true;
        break;
      }
    }
    if (!padded) return undefined;
    const values: Record<string, number> = Object.create(null);
    for (const symbol of this.symbols) values[symbol.name] = symbolSize(symbol, dimensions);
    return values;
  }

  captureDimensions(inputs: Record<string, RunInput>, dimensions: Uint32Array): void {
    for (let i = 0; i < this.bindings.length; i++) {
      const { name, padding } = this.bindings[i]!;
      if (!padding) continue;
      const input = Object.hasOwn(inputs, name) ? inputs[name] : undefined;
      const length = ArrayBuffer.isView(input) ? input.length / padding.fixed : padding.compiled;
      // An invalid length is reported by the normal byte count check.
      dimensions[i] =
        Number.isInteger(length) && length >= padding.minimum && length <= padding.compiled
          ? length
          : padding.compiled;
    }
    for (const { first, next } of this.checks) {
      if (symbolSize(first, dimensions) !== symbolSize(next, dimensions))
        throw new AvelError(`symbol \`${first.name}\` has inconsistent sizes across the inputs`);
    }
  }
}

function symbolSize(symbol: SymbolAxis, dimensions: Uint32Array): number {
  return symbol.inferred ? dimensions[symbol.index]! : symbol.compiled;
}

/** Output shape at the actual symbols of this run, when it needs trimming. */
export function requestedOutputShape(
  graph: Graph,
  index: number,
  planShape: number[],
  symbols: Record<string, number>,
): number[] | undefined {
  const dims = graph.outputs[index]!.shape;
  if (dims.length !== planShape.length) return undefined;
  const shape = dims.map((dim, axis) =>
    typeof dim === "number" ? dim : (evalSymbol(dim.name, symbols) ?? planShape[axis]!),
  );
  return shape.every((size, axis) => size === planShape[axis]) ? undefined : shape;
}
