import type { Features, Graph, Limits, GpuClass } from "../wasm/avel_wasm.js";
import { fetchGraph, loadCompiler, loadGraph, type WasmSource } from "./compiler.js";
import type { CpuRuntime } from "./cpu.js";
import { AvelError } from "./errors.js";
import { Model } from "./model.js";
import { loadParameters } from "./safetensors.js";
import { GraphBuilder } from "./builder.js";
import { inferNodeOutputs, type CustomOp, validateRegistration } from "./custom.js";
import { gpuClass, IndexedDbTuningCache, type TuningCache } from "./tuning.js";

export type RuntimeBackend = "webgpu" | "cpu";

export interface RuntimeOptions {
  /** Select WebGPU when a device can be acquired, otherwise use the CPU. */
  backend?: RuntimeBackend | "auto";
  /** Overrides the class inferred from adapter information. */
  gpuClass?: GpuClass;
  /** Tile persistence, defaulting to IndexedDB. */
  tuningCache?: TuningCache;
  /** Opt in to tile measurement at compile time; requests timestamp-query. */
  autotune?: boolean;
  /** Device to run on. Requested from `navigator.gpu` when omitted. */
  device?: GPUDevice;
  /** Store activations in f16 when the device supports `shader-f16` and the model was validated in f16. */
  preferF16?: boolean;
  /** Request `timestamp-query`, so plans can be profiled. Ignored when `device` is given. */
  profiling?: boolean;
  /** Overrides where the wasm compiler is loaded from. */
  wasm?: WasmSource;
  /** Overrides where the CPU compiler and executor are loaded from. */
  cpuWasm?: WasmSource;
}

export type Runtime = WebGpuRuntime | CpuRuntime;

export const Runtime = { create };

function create(options: RuntimeOptions & { backend: "cpu"; device?: never }): Promise<CpuRuntime>;
function create(
  options: RuntimeOptions &
    ({ backend: "webgpu" } | { device: GPUDevice; backend?: "auto" | "webgpu" }),
): Promise<WebGpuRuntime>;
function create(options?: RuntimeOptions): Promise<Runtime>;
async function create(options: RuntimeOptions = {}): Promise<Runtime> {
  const { backend = "auto", device: supplied } = options;
  if (backend !== "auto" && backend !== "webgpu" && backend !== "cpu")
    throw new AvelError(`unknown runtime backend \`${backend}\``);
  if (backend === "cpu" && supplied)
    throw new AvelError("a WebGPU device cannot be used with the CPU backend");
  let device = supplied;
  if (!device && backend !== "cpu") {
    try {
      device = await requestDevice((options.profiling ?? false) || (options.autotune ?? false));
    } catch (error) {
      if (backend === "webgpu") throw error;
    }
  }
  if (!device) {
    const { CpuRuntime } = await import("./cpu.js");
    return CpuRuntime.create(options.cpuWasm);
  }
  try {
    await loadCompiler(options.wasm);
    return new WebGpuRuntime(
      device,
      readLimits(device),
      {
        shaderF16: device.features.has("shader-f16"),
        subgroups: device.features.has("subgroups" as GPUFeatureName),
        subgroupMinSize: subgroupMinSize(device),
      },
      options.preferF16 ?? false,
      options.gpuClass ?? gpuClass(device.adapterInfo),
      options.autotune ?? false,
      options.tuningCache ?? new IndexedDbTuningCache(),
    );
  } catch (error) {
    if (!supplied) device.destroy();
    throw error;
  }
}

export class WebGpuRuntime {
  readonly backend = "webgpu" as const;
  private readonly registry = new Map<string, CustomOp>();
  private revision = 0;
  get kernelVersion(): number {
    return this.revision;
  }
  get customOps(): ReadonlyMap<string, CustomOp> {
    return this.registry;
  }
  /** Register a pure custom op before compiling models that use it. */
  registerOp(name: string, op: CustomOp): this {
    validateRegistration(name, op);
    this.registry.set(name, Object.freeze({ ...op, inputs: Object.freeze([...op.inputs]) }));
    this.revision++;
    return this;
  }
  /** Start a concrete explicit graph using this runtime and its custom registry. */
  graph(): GraphBuilder<Model> {
    return new GraphBuilder(
      (node, inputs) => inferNodeOutputs(node, inputs, {}, this.customOps),
      (graph, parameters) => this.buildModel(graph, parameters),
    );
  }

  constructor(
    readonly device: GPUDevice,
    readonly limits: Limits,
    readonly features: Features,
    readonly preferF16: boolean,
    readonly gpuClass: GpuClass,
    readonly autotune: boolean,
    readonly tuningCache: TuningCache,
  ) {}

  private buildModel(graph: Graph, weights: ReadonlyMap<string, Float32Array>): Model {
    const loaded = loadGraph(graph);
    const parameters = new Map<string, GPUBuffer>();
    try {
      for (const [name, data] of weights) {
        const buffer = this.device.createBuffer({
          label: name,
          size: data.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
          mappedAtCreation: true,
        });
        parameters.set(name, buffer);
        new Float32Array(buffer.getMappedRange()).set(data);
        buffer.unmap();
      }
      return new Model(this, graph, loaded, parameters);
    } catch (error) {
      loaded.free();
      for (const buffer of parameters.values()) buffer.destroy();
      throw error;
    }
  }

  /** Loads a model directory containing `model.graph.json` and `model.safetensors`. */
  async load(url: string): Promise<Model> {
    const base = url.endsWith("/") ? url : `${url}/`;
    const graph = await fetchGraph(base);
    // Validated before any weight is downloaded.
    const loaded = loadGraph(graph);
    let parameters: Map<string, GPUBuffer> | undefined;
    try {
      parameters = await loadParameters(this.device, `${base}model.safetensors`, graph.parameters);
      return new Model(this, graph, loaded, parameters);
    } catch (error) {
      for (const buffer of parameters?.values() ?? []) buffer.destroy();
      loaded.free();
      throw error;
    }
  }
}

async function requestDevice(profiling: boolean): Promise<GPUDevice> {
  const adapter = await globalThis.navigator?.gpu?.requestAdapter();
  if (!adapter) {
    throw new AvelError("WebGPU is not available");
  }
  const wanted: GPUFeatureName[] = [
    "shader-f16",
    "subgroups" as GPUFeatureName,
    ...(profiling ? ["timestamp-query" as const] : []),
  ];
  return adapter.requestDevice({
    requiredFeatures: wanted.filter((feature) => adapter.features.has(feature)),
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
    },
  });
}

/** The smallest subgroup size of the adapter, 0 when unknown. */
function subgroupMinSize(device: GPUDevice): number {
  const info = device.adapterInfo as GPUAdapterInfo & { subgroupMinSize?: number };
  return info?.subgroupMinSize ?? 0;
}

function readLimits(device: GPUDevice): Limits {
  return {
    maxBufferSize: device.limits.maxBufferSize,
    maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
    maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage,
    minStorageBufferOffsetAlignment: device.limits.minStorageBufferOffsetAlignment,
  };
}
