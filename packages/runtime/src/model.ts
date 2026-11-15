import type { CompileOptions, Graph, LoadedGraph, MediaInput } from "../wasm/avel_wasm.js";
import { ExecutionPlan } from "./plan.js";
import type { WebGpuRuntime } from "./runtime.js";
import { customKernels } from "./custom.js";
import { tune } from "./tuning.js";
import { snapShapes, validateShapes } from "./shapes.js";
import { AvelError } from "./errors.js";

export interface CompileSettings {
  /** Measure tile variants at compilation. Needs timestamp-query on the device. */
  autotune?: boolean;
  /**
   * Store activations in f16. Needs `shader-f16` and a graph the exporter
   * validated in f16. Defaults to on when both hold and the runtime prefers f16.
   */
  f16?: boolean;
  /** Use f16 GEMM products with f32 accumulation on a device with shader-f16. */
  f16Compute?: boolean;
  /** Media inputs, overriding graph.media by input name. */
  media?: Record<string, MediaInput>;
  /**
   * Snap every symbolic input dimension up to a multiple of this, capped by
   * its declared maximum, so variable-length sequences share one plan per
   * bucket instead of recompiling per length. Inputs are padded up to the
   * bucket at run time, and outputs trimmed back to the requested shape.
   */
  bucket?: number;
}

export class Model {
  readonly backend = "webgpu" as const;
  private readonly plans = new Map<string, Promise<ExecutionPlan>>();
  private destroyed = false;
  /** The compiler's copy of the graph, freed by `destroy`. */
  private loaded: LoadedGraph | undefined;
  /** Frozen: the compiler keeps its own copy, which later edits would not reach. */
  readonly graph: Readonly<Graph>;

  /** Copies of the initial state, by tensor name, for {@link resetState}. */
  private readonly initialState = new Map<string, GPUBuffer>();

  /** Takes ownership of `loaded` and `parameters`. */
  constructor(
    private readonly runtime: WebGpuRuntime,
    graph: Graph,
    loaded: LoadedGraph,
    private readonly parameters: Map<string, GPUBuffer>,
  ) {
    this.graph = deepFreeze(graph);
    this.loaded = loaded;
    const state = graph.parameters.filter((parameter) => parameter.update);
    try {
      if (state.length) {
        const { device } = runtime;
        const encoder = device.createCommandEncoder();
        for (const { tensor } of state) {
          if (this.initialState.has(tensor)) continue;
          const buffer = parameters.get(tensor)!;
          const copy = device.createBuffer({
            label: `initial ${tensor}`,
            size: buffer.size,
            usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
          });
          this.initialState.set(tensor, copy);
          encoder.copyBufferToBuffer(buffer, 0, copy, 0, buffer.size);
        }
        device.queue.submit([encoder.finish()]);
      }
    } catch (error) {
      for (const buffer of this.initialState.values()) buffer.destroy();
      throw error;
    }
  }

  /**
   * Restores the persistent state (buffers the model mutates) to its initial
   * value. Every plan of the model shares that state; runs already submitted
   * finish first, so await pending runs before resetting.
   */
  resetState(): void {
    if (this.destroyed) throw new AvelError("model is destroyed");
    if (!this.initialState.size) return;
    const { device } = this.runtime;
    const encoder = device.createCommandEncoder();
    for (const [tensor, initial] of this.initialState)
      encoder.copyBufferToBuffer(initial, 0, this.parameters.get(tensor)!, 0, initial.size);
    device.queue.submit([encoder.finish()]);
  }

  /** Returns the plan for these input shapes, compiling it on first use. */
  async compile(
    inputShapes: Record<string, number[]>,
    settings: CompileSettings = {},
  ): Promise<ExecutionPlan> {
    const { features } = this.runtime;
    if (this.destroyed) throw new AvelError("model is destroyed");
    const f16 =
      settings.f16 ?? (this.runtime.preferF16 && features.shaderF16 && (this.graph.f16 ?? false));
    const bucket = settings.bucket ?? 0;
    validateShapes(this.graph, inputShapes, bucket);
    // Snapshot caller-owned settings before any asynchronous tuning or pipeline creation.
    inputShapes = structuredClone(inputShapes);
    settings = structuredClone(settings);
    // Plans are cached by the shapes they are compiled for: with a bucket,
    // every length in the same bucket shares one plan.
    const key = JSON.stringify([
      snapShapes(this.graph, inputShapes, bucket),
      f16,
      settings.f16Compute ?? false,
      bucket,
      settings.media,
      settings.autotune ?? this.runtime.autotune,
      this.runtime.kernelVersion,
    ]);
    let plan = this.plans.get(key);
    if (!plan) {
      plan = this.create(inputShapes, settings, f16, bucket);
      this.plans.set(key, plan);
      void plan.catch(() => {
        if (this.plans.get(key) === plan) this.plans.delete(key);
      });
    }
    const execution = await plan;
    if (this.destroyed) throw new AvelError("model is destroyed");
    if (execution.destroyed) {
      if (this.plans.get(key) === plan) this.plans.delete(key);
      return this.compile(inputShapes, settings);
    }
    return execution;
  }

  /** Release cached plans and shared parameter buffers. */
  async destroy(): Promise<void> {
    this.destroyed = true;
    const plans = await Promise.allSettled(this.plans.values());
    this.plans.clear();
    await Promise.all(
      plans.map((result) => (result.status === "fulfilled" ? result.value.destroy() : undefined)),
    );
    for (const buffer of this.parameters.values()) buffer.destroy();
    this.parameters.clear();
    for (const buffer of this.initialState.values()) buffer.destroy();
    this.initialState.clear();
    // Every plan has settled, so no compilation still reads the graph.
    this.loaded?.free();
    this.loaded = undefined;
  }

  private async create(
    inputShapes: Record<string, number[]>,
    settings: CompileSettings,
    f16: boolean,
    bucket: number,
  ): Promise<ExecutionPlan> {
    const { device, limits, features } = this.runtime;
    // Plans are registered before `destroy` can run, and it waits for them.
    const loaded = this.loaded!;
    const effective = Object.fromEntries(snapShapes(this.graph, inputShapes, bucket));
    const kernels = customKernels(this.graph, effective, this.runtime.customOps);
    if (bucket && Object.keys(kernels).length) {
      const requested = customKernels(this.graph, inputShapes, this.runtime.customOps);
      for (const [name, kernel] of Object.entries(kernels))
        kernel.requestedShape = requested[name]!.shape;
    }
    const options: CompileOptions = {
      inputShapes,
      limits,
      features,
      f16,
      f16Compute: settings.f16Compute ?? false,
      bucket: bucket || undefined,
      media: settings.media ?? {},
      customKernels: kernels,
      gpuClass: this.runtime.gpuClass,
      tiles: {},
    };
    let plan = loaded.compile(options);
    if (settings.autotune ?? this.runtime.autotune)
      plan = await tune(
        device,
        this.graph,
        loaded,
        options,
        this.parameters,
        plan,
        this.runtime.tuningCache,
      );
    return ExecutionPlan.create(device, this.graph, plan, this.parameters);
  }
}

export function deepFreeze<T>(value: T): T {
  if (value && typeof value === "object" && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const item of Object.values(value)) deepFreeze(item);
  }
  return value;
}
