import {
  inferOutputs,
  resolveAttributes,
  type Attr,
  type CustomKernel,
  type Graph,
  type Node,
  type ShapeInput,
} from "../wasm/avel_wasm.js";
import { AvelError } from "./errors.js";

/**
 * Attributes as custom ops see them: symbolic integers are resolved to the
 * concrete sizes the plan is compiled for.
 */
export type CustomAttributes = Readonly<Record<string, Attr>>;

export interface CustomOp {
  /** One WGSL identifier per tensor argument, in schema order. */
  inputs: readonly string[];
  /** Pure, concrete shape inference, run only while compiling. Output is f32. */
  shape(inputs: readonly ShapeInput[], attributes: CustomAttributes): number[];
  /**
   * Group 0 declarations are generated. The `main` entry point must use all
   * bindings and may only write `output`.
   */
  kernel(context: {
    inputs: readonly ShapeInput[];
    shape: readonly number[];
    attributes: CustomAttributes;
  }): {
    wgsl: string;
    workgroups: [number, number, number];
  };
}

function isBuiltin(name: string): boolean {
  return name.startsWith("aten.") || name.startsWith("avel.");
}

export function validateRegistration(name: string, op: CustomOp): void {
  if (!/^[A-Za-z_]\w*\.[A-Za-z_]\w*\.[A-Za-z_]\w*$/.test(name) || isBuiltin(name)) {
    throw new AvelError("custom op names must use namespace.op.overload outside aten and avel");
  }
  if (
    new Set(op.inputs).size !== op.inputs.length ||
    op.inputs.some((s) => !/^[A-Za-z_]\w*$/.test(s) || s === "output")
  ) {
    throw new AvelError("custom input bindings must be unique WGSL identifiers other than output");
  }
}

/** The attributes of a custom node, with symbolic integers resolved. */
function customAttributes(node: Node, symbols: Record<string, number>): CustomAttributes {
  return resolveAttributes({ attributes: node.attributes ?? {}, symbols }).attributes;
}

/** `inputs` holds the tensors present, in schema order. */
export function inferNodeOutputs(
  node: Node,
  inputs: ShapeInput[],
  symbols: Record<string, number>,
  registry: ReadonlyMap<string, CustomOp>,
  attributes?: CustomAttributes,
): ShapeInput[] {
  if (isBuiltin(node.op)) return inferOutputs({ node, inputs, symbols });
  const op = registry.get(node.op);
  if (!op) throw new AvelError(`custom op \`${node.op}\` is not registered`);
  if (
    node.inputs.length !== op.inputs.length ||
    node.inputs.includes(null) ||
    inputs.some((v) => v.dtype !== "f32")
  ) {
    throw new AvelError(`custom op \`${node.op}\` expects ${op.inputs.length} f32 inputs`);
  }
  const shape = op.shape(inputs, attributes ?? customAttributes(node, symbols));
  if (shape.some((n) => !Number.isSafeInteger(n) || n < 1 || n > 0xffffffff))
    throw new AvelError(`invalid shape for ${node.op}`);
  return [{ dtype: "f32", shape: [...shape] }];
}

/** Resolve intermediate shapes through the same Rust inference used by compile. */
export function customKernels(
  graph: Graph,
  shapes: Record<string, number[]>,
  registry: ReadonlyMap<string, CustomOp>,
): Record<string, CustomKernel> {
  if (!graph.nodes.some((n) => !isBuiltin(n.op))) return {};
  const values = new Map<string, ShapeInput>();
  const symbols: Record<string, number> = Object.create(null);
  for (const input of graph.inputs) {
    const shape = shapes[input.name];
    if (!shape) throw new AvelError(`missing shape for ${input.name}`);
    values.set(input.name, { dtype: input.dtype, shape });
    input.shape.forEach((dim, i) => {
      if (typeof dim === "object") symbols[dim.name] = shape[i]!;
    });
  }
  for (const p of graph.parameters) values.set(p.name, { dtype: p.dtype, shape: p.shape });
  const kernels: Record<string, CustomKernel> = Object.create(null);
  for (const node of graph.nodes) {
    const inputs = node.inputs.flatMap((name) => {
      if (name === null) return [];
      const value = values.get(name);
      if (!value) throw new AvelError(`unknown value ${name}`);
      return [value];
    });
    const custom = !isBuiltin(node.op);
    const attributes = custom ? customAttributes(node, symbols) : undefined;
    const inferred = inferNodeOutputs(node, inputs, symbols, registry, attributes);
    for (const [index, output] of node.outputs.entries()) {
      if (!output) continue;
      const info = inferred[index];
      if (!info) throw new AvelError(`output ${index} is not supported for ${node.op}`);
      values.set(output.name, info);
    }
    if (custom) {
      const output = node.outputs[0];
      if (!output) throw new AvelError(`missing output for ${node.op}`);
      const { shape } = inferred[0]!;
      const op = registry.get(node.op)!;
      const result = op.kernel({ inputs, shape, attributes: attributes! });
      const declarations = op.inputs.map(
        (name, i) => `@group(0) @binding(${i}) var<storage, read_write> ${name}: array<f32>;`,
      );
      declarations.push(
        `@group(0) @binding(${inputs.length}) var<storage, read_write> output: array<f32>;`,
      );
      kernels[output.name] = {
        shape,
        wgsl: `${declarations.join("\n")}\n${result.wgsl}`,
        workgroups: result.workgroups,
      };
    }
  }
  return kernels;
}
