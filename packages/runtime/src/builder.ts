import type { Attr, Dtype, Graph, Node, ShapeInput, ValueInfo } from "../wasm/avel_wasm.js";
import { AvelError } from "./errors.js";
import type { Model } from "./model.js";
import type { CpuModel } from "./cpu.js";

/** A value owned by one explicit graph builder. Shapes are concrete. */
export interface GraphValue extends ShapeInput {
  readonly name: string;
}

export class GraphBuilder<M extends Model | CpuModel = Model | CpuModel> {
  private readonly graph: Graph = {
    formatVersion: 1,
    opset: "core-aten/1",
    inputs: [],
    outputs: [],
    nodes: [],
    parameters: [],
    media: {},
    metadata: {},
    f16: false,
  };
  private readonly values = new Map<string, GraphValue>();
  private readonly parameters = new Map<string, Float32Array>();
  private serial = 0;
  constructor(
    private readonly infer: (node: Node, inputs: ShapeInput[]) => ShapeInput[],
    private readonly createModel: (
      graph: Graph,
      parameters: ReadonlyMap<string, Float32Array>,
    ) => M,
  ) {}

  input(name: string, shape: number[], dtype: Dtype = "f32"): GraphValue {
    const value = this.define(name, shape, dtype);
    this.graph.inputs.push(value);
    return value;
  }

  parameter(name: string, shape: number[], data: Float32Array): GraphValue {
    if (data.length !== shape.reduce((a, b) => a * b, 1))
      throw new AvelError(`wrong parameter size: ${name}`);
    const value = this.define(name, shape, "f32");
    this.parameters.set(name, data.slice());
    this.graph.parameters.push({ ...value, tensor: name });
    return value;
  }

  /**
   * Adds a supported ATen op or a registered custom op, inferring its output.
   * `inputs` has one slot per tensor argument, `null` for an optional one left out.
   */
  op(
    op: string,
    inputs: readonly (GraphValue | null)[],
    attributes: Record<string, Attr> = {},
  ): GraphValue {
    const present = inputs.filter((value): value is GraphValue => value !== null);
    for (const value of present) this.owns(value);
    let name: string;
    do {
      name = `v${this.serial++}`;
    } while (this.values.has(name));
    const node: Node = { op, inputs: inputs.map((v) => v?.name ?? null), outputs: [], attributes };
    const { shape, dtype } = this.infer(node, present)[0]!;
    const value = this.define(name, shape, dtype);
    node.outputs = [value];
    this.graph.nodes.push(node);
    return value;
  }

  relu(x: GraphValue): GraphValue {
    return this.op("aten.relu.default", [x]);
  }
  add(a: GraphValue, b: GraphValue): GraphValue {
    return this.op("aten.add.Tensor", [a, b], { alpha: 1 });
  }
  mul(a: GraphValue, b: GraphValue): GraphValue {
    return this.op("aten.mul.Tensor", [a, b]);
  }
  reshape(x: GraphValue, shape: number[]): GraphValue {
    return this.op("aten.view.default", [x], { size: shape });
  }
  matmul(a: GraphValue, b: GraphValue, bias: GraphValue): GraphValue {
    return this.op("aten.addmm.default", [bias, a, b], { alpha: 1, beta: 1 });
  }
  conv2d(
    x: GraphValue,
    w: GraphValue,
    bias?: GraphValue,
    options: { stride?: number[]; padding?: number[]; dilation?: number[]; groups?: number } = {},
  ): GraphValue {
    return this.op("aten.convolution.default", [x, w, bias ?? null], {
      stride: options.stride ?? [1, 1],
      padding: options.padding ?? [0, 0],
      dilation: options.dilation ?? [1, 1],
      groups: options.groups ?? 1,
      transposed: false,
      output_padding: [0, 0],
    });
  }

  /** Takes a snapshot; later builder edits do not change the model. */
  build(outputs: readonly GraphValue[], metadata: Record<string, unknown> = {}): M {
    if (!outputs.length) throw new AvelError("a graph needs an output");
    outputs.forEach((v) => this.owns(v));
    const graph = structuredClone({ ...this.graph, outputs: outputs as ValueInfo[], metadata });
    return this.createModel(graph, this.parameters);
  }

  private owns(value: GraphValue): void {
    if (this.values.get(value.name) !== value)
      throw new AvelError("value belongs to another graph");
  }
  private define(name: string, shape: number[], dtype: Dtype): GraphValue {
    if (!name || this.values.has(name))
      throw new AvelError(`duplicate or empty value name: ${name}`);
    if (shape.some((n) => !Number.isSafeInteger(n) || n < 1))
      throw new AvelError(`invalid shape: ${name}`);
    const value = Object.freeze({
      name,
      dtype,
      shape: Object.freeze([...shape]) as unknown as number[],
    });
    this.values.set(name, value);
    return value;
  }
}
