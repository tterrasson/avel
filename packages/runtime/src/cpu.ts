import type { CpuExecutable, CpuGraph, CpuTensor } from "../wasm/avel_cpu_wasm.js";
import type { Graph } from "../wasm/avel_wasm.js";
import { GraphBuilder } from "./builder.js";
import { BucketInputs, requestedOutputShape } from "./buckets.js";
import { fetchGraph, resolveWasmSource, type WasmSource } from "./compiler.js";
import { AvelError } from "./errors.js";
import { deepFreeze, type CompileSettings } from "./model.js";
import type { RunInput, RunOptions, TensorOutput } from "./plan.js";
import { narrow } from "./plan-inputs.js";
import { readParameters } from "./safetensors.js";
import { pad, snapShapes, trim, validateShapes } from "./shapes.js";

type CpuModule = typeof import("../wasm/avel_cpu_wasm.js");
interface CpuWasm {
  module: CpuModule;
  memory: WebAssembly.Memory;
}

let ready: Promise<CpuWasm> | undefined;

async function loadCpu(source?: WasmSource): Promise<CpuWasm> {
  ready ??= (async () => {
    const module = await import("../wasm/avel_cpu_wasm.js");
    const wasm = await resolveWasmSource(
      source,
      new URL("../wasm/avel_cpu_wasm_bg.wasm", import.meta.url),
    );
    const { memory } = await module.default(
      wasm === undefined ? undefined : { module_or_path: wasm },
    );
    return { module, memory };
  })().catch((error: unknown) => {
    ready = undefined;
    throw error;
  });
  return ready;
}

export class CpuRuntime {
  readonly backend = "cpu" as const;

  private constructor(private readonly wasm: CpuWasm) {}

  static async create(source?: WasmSource): Promise<CpuRuntime> {
    return new CpuRuntime(await loadCpu(source));
  }

  graph(): GraphBuilder<CpuModel> {
    return new GraphBuilder(
      (node, inputs) => this.wasm.module.inferOutputs({ node, inputs, symbols: {} }),
      (graph, weights) => {
        const loaded = new this.wasm.module.CpuGraph(graph);
        try {
          for (const [name, data] of weights)
            loaded.setParameter(
              name,
              new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
            );
          return new CpuModel(graph, loaded, this.wasm.memory);
        } catch (error) {
          loaded.free();
          throw error;
        }
      },
    );
  }

  async load(url: string): Promise<CpuModel> {
    const base = url.endsWith("/") ? url : `${url}/`;
    const graph = await fetchGraph(base);
    const loaded = new this.wasm.module.CpuGraph(graph);
    try {
      await readParameters(`${base}model.safetensors`, graph.parameters, ({ tensor }, bytes) =>
        loaded.setParameter(tensor, bytes),
      );
      return new CpuModel(graph, loaded, this.wasm.memory);
    } catch (error) {
      loaded.free();
      throw error;
    }
  }
}

export class CpuModel {
  readonly backend = "cpu" as const;
  readonly graph: Readonly<Graph>;
  private readonly plans = new Map<string, CpuExecutionPlan>();
  private loaded?: CpuGraph;

  constructor(
    graph: Graph,
    loaded: CpuGraph,
    private readonly memory: WebAssembly.Memory,
  ) {
    this.graph = deepFreeze(graph);
    this.loaded = loaded;
  }

  async compile(
    inputShapes: Record<string, number[]>,
    settings: CompileSettings = {},
  ): Promise<CpuExecutionPlan> {
    if (!this.loaded) throw new AvelError("model is destroyed");
    if (settings.media && Object.keys(settings.media).length)
      throw new AvelError("CPU backend does not support media inputs");
    if (settings.f16 || settings.f16Compute)
      throw new AvelError("CPU backend does not support f16 execution");
    const bucket = settings.bucket ?? 0;
    validateShapes(this.graph, inputShapes, bucket);
    const key = JSON.stringify([bucket, snapShapes(this.graph, inputShapes, bucket)]);
    const cached = this.plans.get(key);
    if (cached && !cached.destroyed) return cached;
    const execution = this.loaded.compile({ inputShapes, bucket: bucket || undefined });
    try {
      const plan = new CpuExecutionPlan(execution, this.memory, this.graph, bucket !== 0);
      this.plans.set(key, plan);
      return plan;
    } catch (error) {
      execution.free();
      throw error;
    }
  }

  resetState(): void {
    if (!this.loaded) throw new AvelError("model is destroyed");
  }

  async destroy(): Promise<void> {
    for (const plan of this.plans.values()) plan.destroy();
    this.plans.clear();
    this.loaded?.free();
    this.loaded = undefined;
  }
}

type TensorView = Float32Array<ArrayBuffer> | Int32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>;

function view(tensor: CpuTensor, memory: ArrayBuffer): TensorView {
  const Type =
    tensor.dtype === "f32" ? Float32Array : tensor.dtype === "bool" ? Uint32Array : Int32Array;
  return new Type(memory, tensor.offset, tensor.length);
}

export class CpuExecutionPlan {
  readonly backend = "cpu" as const;
  private readonly inputInfo: CpuTensor[];
  private readonly outputInfo: CpuTensor[];
  private readonly inputViews: TensorView[] = [];
  private readonly outputViews: TensorView[] = [];
  private readonly bucket?: BucketInputs;
  private readonly dimensions?: Uint32Array;
  private execution?: CpuExecutable;
  private buffer?: ArrayBuffer;

  constructor(
    execution: CpuExecutable,
    private readonly memory: WebAssembly.Memory,
    private readonly graph: Graph,
    bucketed: boolean,
  ) {
    this.execution = execution;
    const { inputs, outputs } = execution.info();
    this.inputInfo = inputs;
    this.outputInfo = outputs;
    if (bucketed) {
      this.bucket = new BucketInputs(graph, inputs, () => false);
      this.dimensions = this.bucket.dimensions();
    }
    this.refreshViews();
  }

  get destroyed(): boolean {
    return this.execution === undefined;
  }

  async run(
    inputs: Record<string, RunInput>,
    options?: RunOptions,
  ): Promise<Record<string, TensorOutput>> {
    if (!this.execution) throw new AvelError("execution plan is destroyed");
    if (options?.keepOnGpu) throw new AvelError("CPU backend does not support GPU outputs");
    this.refreshViews();
    if (this.dimensions) this.bucket!.captureDimensions(inputs, this.dimensions);
    for (let i = 0; i < this.inputInfo.length; i++) {
      const { name, dtype, length } = this.inputInfo[i]!;
      const target = this.inputViews[i]!;
      const input = Object.hasOwn(inputs, name) ? inputs[name] : undefined;
      if (input === undefined) throw new AvelError(`missing input \`${name}\``);
      const needsNarrowing = dtype === "i64" && input instanceof BigInt64Array;
      if (!(input instanceof target.constructor) && !needsNarrowing)
        throw new AvelError(
          `CPU input \`${name}\` must be a ${target.constructor.name}${dtype === "i64" ? " or a BigInt64Array" : ""}`,
        );
      const data = input as TensorView | BigInt64Array;
      const layout = this.bucket?.padding(i);
      const actual = this.dimensions?.[i];
      const padded = layout && actual !== undefined && actual !== layout.compiled;
      const expected = padded ? actual * layout.fixed : length;
      if (data.length !== expected)
        throw new AvelError(`input \`${name}\` has ${data.length} elements, expected ${expected}`);
      const block = padded ? actual * layout.inner : length;
      const stride = padded ? layout.block : length;
      if (needsNarrowing) {
        narrow(name, input, target as Int32Array, block, stride);
      } else if (padded) {
        pad(data as TensorView, block, stride, target);
      } else {
        target.set(data as TensorView);
      }
    }
    this.execution.run();
    const symbols = this.bucket?.outputSymbols(this.dimensions);
    const outputs: Record<string, TensorOutput> = Object.create(null);
    for (let i = 0; i < this.outputInfo.length; i++) {
      const { name, dtype, shape } = this.outputInfo[i]!;
      const source = this.outputViews[i]!;
      const wanted = symbols && requestedOutputShape(this.graph, i, shape, symbols);
      const values = wanted ? trim(source, shape, wanted) : source;
      outputs[name] =
        dtype === "i64" ? BigInt64Array.from(values, BigInt) : wanted ? values : values.slice();
    }
    return outputs;
  }

  destroy(): void {
    this.execution?.free();
    this.execution = undefined;
    this.inputViews.length = 0;
    this.outputViews.length = 0;
    this.buffer = undefined;
  }

  private refreshViews(): void {
    // Compiling another shape or loading weights can grow the shared wasm memory.
    const buffer = this.memory.buffer;
    if (this.buffer === buffer) return;
    this.buffer = buffer;
    for (let i = 0; i < this.inputInfo.length; i++)
      this.inputViews[i] = view(this.inputInfo[i]!, buffer);
    for (let i = 0; i < this.outputInfo.length; i++)
      this.outputViews[i] = view(this.outputInfo[i]!, buffer);
  }
}
