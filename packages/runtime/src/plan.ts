import type { Dispatch, Graph, Plan } from "../wasm/avel_wasm.js";
import { AvelError } from "./errors.js";
import { MediaBinding } from "./media.js";
import { PlanInputs, type RunInput } from "./plan-inputs.js";
import { Profile } from "./profile.js";
import { requestedOutputShape } from "./buckets.js";
import { trim } from "./shapes.js";

export type { RunInput } from "./plan-inputs.js";
/**
 * A Float32Array for f32 outputs, an Int32Array for i32, a Uint32Array for
 * bool and a BigInt64Array for i64 (stored in 32 bits on the GPU, widened on
 * readback), or the plan's GPU buffer with `keepOnGpu`.
 */
export type TensorOutput = Float32Array | Int32Array | Uint32Array | BigInt64Array;
export type RunOutput = TensorOutput | GPUBuffer;
type GpuConsumer = (
  encoder: GPUCommandEncoder,
  outputs: Readonly<Record<string, GPUBuffer>>,
) => void;

export interface RunOptions {
  /**
   * Leave outputs on the GPU instead of reading them back. The returned buffers
   * belong to the plan: treat them as read-only, valid until the next run.
   */
  keepOnGpu?: boolean;
}

/** Staging buffers in the ring: a run only waits for the readback three runs before it. */
const STAGING_SLOTS = 3;

interface Step {
  pipeline: GPUComputePipeline;
  bindGroup?: GPUBindGroup;
  descriptor: GPUBindGroupDescriptor;
  external: { binding: number; media: MediaBinding }[];
  workgroups: [number, number, number];
}

interface Slot {
  buffer: GPUBuffer;
  dimensions: Uint32Array | undefined;
  /** Settles when the latest readback through this buffer is done. */
  idle: Promise<unknown>;
}

function ignore(): void {}

export class ExecutionPlan {
  readonly backend = "webgpu" as const;
  /** Settles when the latest run is submitted. Runs share their output buffers, so they submit in call order. */
  private submitted: Promise<unknown> = Promise.resolve();
  private next = 0;
  private pendingSubmissions = 0;
  private submittingGpu = false;
  private readonly externalSteps: Step[];
  private readonly commands: GPUCommandBuffer[] = [];
  private disposed = false;
  get destroyed(): boolean {
    return this.disposed;
  }
  private destruction?: Promise<void>;
  private readonly gpuOutputs: Record<string, GPUBuffer> = Object.create(null);

  private constructor(
    readonly device: GPUDevice,
    readonly plan: Plan,
    private readonly graph: Graph,
    private readonly inputBindings: PlanInputs,
    private readonly buffers: (GPUBuffer | undefined)[],
    private readonly steps: Step[],
    private readonly ring: Slot[],
    private readonly stagingOffsets: number[],
    private readonly media: Map<number, MediaBinding>,
    private readonly owned: GPUBuffer[],
  ) {
    this.externalSteps = steps.filter((step) => step.external.length > 0);
    for (const { name, buffer } of plan.outputs) {
      this.gpuOutputs[name] = buffers[buffer]!;
    }
    Object.freeze(this.gpuOutputs);
  }

  /** Creates every GPU resource of `plan`. `parameters` are shared, keyed by tensor name. */
  static async create(
    device: GPUDevice,
    graph: Graph,
    plan: Plan,
    parameters: Map<string, GPUBuffer>,
  ): Promise<ExecutionPlan> {
    const media = new Map<number, MediaBinding>();
    const owned: GPUBuffer[] = [];
    try {
      for (const input of plan.inputs) {
        const config = Object.hasOwn(plan.media, input.name) ? plan.media[input.name] : undefined;
        if (config) media.set(input.buffer, new MediaBinding(device, config));
      }
      const buffers = plan.buffers.map(({ size, role }, index) => {
        if (media.has(index)) return undefined;
        // State is shared by every plan of the model, like parameters.
        if (role.kind === "parameter" || role.kind === "state") {
          const buffer = parameters.get(role.tensor);
          if (!buffer) {
            throw new AvelError(`tensor \`${role.tensor}\` is not loaded`);
          }
          return buffer;
        }
        const copies = {
          // Inputs and constants may be returned unchanged, and so read back.
          input: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
          output: GPUBufferUsage.COPY_SRC,
          constant: GPUBufferUsage.COPY_SRC,
          arena: 0,
        };
        const buffer = device.createBuffer({
          label: `${role.kind} ${index}`,
          size,
          usage: GPUBufferUsage.STORAGE | copies[role.kind],
        });
        owned.push(buffer);
        return buffer;
      });

      const pipelines = await Promise.all(
        plan.shaders.map(async ({ wgsl, entryPoint }, index) => {
          const module = device.createShaderModule({ code: wgsl });
          try {
            return await device.createComputePipelineAsync({
              layout: "auto",
              compute: { module, entryPoint },
            });
          } catch (error) {
            const info = await module.getCompilationInfo?.().catch(() => undefined);
            const messages = info?.messages
              .filter((message) => message.type === "error")
              .map((message) => `${message.lineNum}:${message.linePos} ${message.message}`);
            if (messages?.length) {
              throw new AvelError(`shader ${index} failed to compile\n${messages.join("\n")}`);
            }
            throw error;
          }
        }),
      );
      const step = ({ shader, bindings, workgroups }: Dispatch): Step => {
        const pipeline = pipelines[shader]!;
        const external: Step["external"] = [];
        const entries: GPUBindGroupEntry[] = bindings.map(({ buffer, offset, size }, binding) => {
          const source = media.get(buffer);
          if (source?.config.source.kind === "external") {
            external.push({ binding, media: source });
            return { binding, resource: undefined as unknown as GPUExternalTexture };
          }
          return { binding, resource: source?.view ?? { buffer: buffers[buffer]!, offset, size } };
        });
        const descriptor = { layout: pipeline.getBindGroupLayout(0), entries };
        const bindGroup = external.length ? undefined : device.createBindGroup(descriptor);
        return { pipeline, bindGroup, workgroups, descriptor, external };
      };
      const steps = await checkedCreation(device, "could not create plan bind groups", () => {
        const steps = plan.dispatches.map(step);
        // Constants only depend on parameters: compute them once, here.
        if (plan.prepare.length > 0) {
          const encoder = device.createCommandEncoder();
          encode(encoder, plan.prepare.map(step));
          device.queue.submit([encoder.finish()]);
        }
        return steps;
      });

      let stagingSize = 0;
      const stagingOffsets = plan.outputs.map((output) => {
        const offset = stagingSize;
        stagingSize += plan.buffers[output.buffer]!.size;
        return offset;
      });
      if (stagingSize > device.limits.maxBufferSize) {
        throw new AvelError("combined outputs exceed the device readback buffer limit");
      }
      const inputBindings = new PlanInputs(device, graph, plan, buffers, media);
      const ring = Array.from({ length: STAGING_SLOTS }, (_, slot) => {
        const buffer = device.createBuffer({
          label: `staging ${slot}`,
          size: stagingSize,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        owned.push(buffer);
        return { buffer, dimensions: inputBindings.readbackDimensions(), idle: Promise.resolve() };
      });

      return new ExecutionPlan(
        device,
        plan,
        graph,
        inputBindings,
        buffers,
        steps,
        ring,
        stagingOffsets,
        media,
        owned,
      );
    } catch (error) {
      for (const buffer of owned) buffer.destroy();
      for (const binding of media.values()) binding.destroy();
      throw error;
    }
  }

  /** Reject new runs, then release owned resources once pending readbacks settle. */
  destroy(): Promise<void> {
    if (this.destruction) return this.destruction;
    this.disposed = true;
    return (this.destruction = Promise.all([
      this.submitted,
      ...this.ring.map((slot) => slot.idle),
    ]).then(() => {
      for (const buffer of this.owned) buffer.destroy();
      for (const binding of this.media.values()) binding.destroy();
      this.inputBindings.clear();
      this.owned.length = 0;
      this.buffers.length = 0;
      this.steps.length = 0;
      this.externalSteps.length = 0;
      this.ring.length = 0;
      this.media.clear();
    }));
  }

  private assertActive(): void {
    if (this.disposed) throw new AvelError("execution plan is destroyed");
  }

  /** Whether {@link profile} is available: the device needs the `timestamp-query` feature. */
  get profiling(): boolean {
    return this.device.features.has("timestamp-query");
  }

  /**
   * Runs the plan once. Runs submit in call order, and a run's readback does
   * not hold back the next one. Inputs must stay unchanged until the returned
   * promise resolves, since submission can wait for a staging slot. A `GPUBuffer` input needs `COPY_SRC`
   * usage and is copied into the plan. Media inputs must stay open until
   * submission (await the returned promise before closing a VideoFrame).
   */
  run(inputs: Record<string, RunInput>, options?: RunOptions): Promise<Record<string, RunOutput>> {
    if (this.disposed) return Promise.reject(new AvelError("execution plan is destroyed"));
    this.pendingSubmissions++;
    if (options?.keepOnGpu) {
      const submission = this.submitted.then(() => {
        try {
          this.submit(inputs);
        } finally {
          this.pendingSubmissions--;
        }
      });
      this.submitted = submission.catch(ignore);
      return submission.then(() => this.gpuOutputs);
    }
    const slot = this.ring[this.next]!;
    this.next = (this.next + 1) % this.ring.length;
    const idle = slot.idle;
    const submission = this.submitted
      .then(() => idle)
      .then(() => {
        try {
          this.submit(inputs, slot);
        } finally {
          this.pendingSubmissions--;
        }
      });
    this.submitted = submission.catch(ignore);
    const result = submission.then(() => this.read(slot));
    slot.idle = result.catch(ignore);
    return result;
  }

  /**
   * Submits immediately, without readback or a GPU wait. `consume` encodes a
   * renderer or downstream model into this same submit, before outputs can be
   * overwritten. Returned buffers belong to the plan, valid until its next run.
   * A VideoFrame may be closed as soon as this method returns.
   */
  runGpu(
    inputs: Record<string, RunInput>,
    consume?: GpuConsumer,
  ): Readonly<Record<string, GPUBuffer>> {
    this.assertActive();
    if (this.submittingGpu) throw new AvelError("the plan is already submitting a GPU run");
    if (this.pendingSubmissions)
      throw new AvelError("a run is waiting to submit; await it before runGpu");
    this.submittingGpu = true;
    try {
      this.submit(inputs, undefined, consume);
      return this.gpuOutputs;
    } finally {
      this.submittingGpu = false;
    }
  }

  private bindExternal(): void {
    for (const step of this.externalSteps) {
      // External textures expire: import and bind in the task that submits.
      for (const { binding, media } of step.external)
        (step.descriptor.entries as GPUBindGroupEntry[])[binding]!.resource = media.external!;
      step.bindGroup = this.device.createBindGroup(step.descriptor);
    }
  }

  /**
   * Runs the plan once with one timestamp pair per dispatch, and returns how
   * long each took on the GPU. For analysis only: every dispatch gets its own
   * compute pass, which adds a little overhead.
   */
  profile(inputs: Record<string, RunInput>): Promise<Profile> {
    if (this.disposed) return Promise.reject(new AvelError("execution plan is destroyed"));
    if (!this.profiling)
      return Promise.reject(new AvelError("profiling needs the `timestamp-query` feature"));
    this.pendingSubmissions++;
    const result = this.submitted
      .then(() => this.measure(inputs))
      .finally(() => {
        this.pendingSubmissions--;
      });
    this.submitted = result.catch(ignore);
    return result;
  }

  private async measure(inputs: Record<string, RunInput>): Promise<Profile> {
    const { device, steps } = this;
    if (!steps.length) {
      this.submit(inputs);
      return new Profile([], new BigInt64Array(0));
    }
    const count = steps.length * 2;
    const querySet = device.createQuerySet({ type: "timestamp", count });
    let resolved: GPUBuffer | undefined;
    let readback: GPUBuffer | undefined;
    try {
      resolved = device.createBuffer({
        size: count * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      readback = device.createBuffer({
        size: count * 8,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

      const encoder = device.createCommandEncoder();
      this.inputBindings.write(encoder, inputs);
      this.bindExternal();
      for (let i = 0; i < steps.length; i++) {
        const { pipeline, bindGroup, workgroups } = steps[i]!;
        const pass = encoder.beginComputePass({
          timestampWrites: {
            querySet,
            beginningOfPassWriteIndex: 2 * i,
            endOfPassWriteIndex: 2 * i + 1,
          },
        });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup!);
        pass.dispatchWorkgroups(workgroups[0], workgroups[1], workgroups[2]);
        pass.end();
      }
      encoder.resolveQuerySet(querySet, 0, count, resolved, 0);
      encoder.copyBufferToBuffer(resolved, 0, readback, 0, count * 8);
      device.queue.submit([encoder.finish()]);

      await readback.mapAsync(GPUMapMode.READ);
      const timestamps = new BigInt64Array(readback.getMappedRange().slice(0));
      return new Profile(this.plan.dispatches, timestamps);
    } finally {
      readback?.unmap();
      querySet.destroy();
      resolved?.destroy();
      readback?.destroy();
    }
  }

  /** Encodes one run and its optional output consumer or readback copy. */
  private submit(inputs: Record<string, RunInput>, slot?: Slot, consume?: GpuConsumer): void {
    const { device, plan, buffers, steps, stagingOffsets } = this;
    const encoder = device.createCommandEncoder();
    this.inputBindings.write(encoder, inputs, slot?.dimensions);
    this.bindExternal();
    encode(encoder, steps);
    consume?.(encoder, this.gpuOutputs);
    if (slot) {
      for (let i = 0; i < plan.outputs.length; i++) {
        const { buffer } = plan.outputs[i]!;
        encoder.copyBufferToBuffer(
          buffers[buffer]!,
          0,
          slot.buffer,
          stagingOffsets[i]!,
          plan.buffers[buffer]!.size,
        );
      }
    }
    this.commands[0] = encoder.finish();
    device.queue.submit(this.commands);
  }

  private async read({ buffer: staging, dimensions }: Slot): Promise<Record<string, RunOutput>> {
    const { plan, stagingOffsets } = this;
    await staging.mapAsync(GPUMapMode.READ);
    try {
      const mapped = staging.getMappedRange();
      const symbols = this.inputBindings.outputSymbols(dimensions);
      const outputs: Record<string, RunOutput> = Object.create(null);
      for (let i = 0; i < plan.outputs.length; i++) {
        const { name, buffer } = plan.outputs[i]!;
        const start = stagingOffsets[i]!;
        const size = plan.buffers[buffer]!.size;
        const planShape = plan.outputs[i]!.shape;
        const wanted = symbols && requestedOutputShape(this.graph, i, planShape, symbols);
        const dtype = plan.outputs[i]!.dtype;
        const array =
          dtype === "i32" || dtype === "i64"
            ? Int32Array
            : dtype === "bool"
              ? Uint32Array
              : Float32Array;
        const source = new array(mapped, start, size / 4);
        const values = wanted ? trim(source, planShape, wanted) : source.slice();
        outputs[name] = dtype === "i64" ? BigInt64Array.from(values as Int32Array, BigInt) : values;
      }
      return outputs;
    } finally {
      staging.unmap();
    }
  }
}

function encode(encoder: GPUCommandEncoder, steps: Step[]): void {
  if (!steps.length) return;
  const pass = encoder.beginComputePass();
  for (let i = 0; i < steps.length; i++) {
    const { pipeline, bindGroup, workgroups } = steps[i]!;
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup!);
    pass.dispatchWorkgroups(workgroups[0], workgroups[1], workgroups[2]);
  }
  pass.end();
}

/** WebGPU can return invalid objects and report the error only via a scope. */
async function checkedCreation<T>(device: GPUDevice, label: string, create: () => T): Promise<T> {
  device.pushErrorScope("validation");
  try {
    return create();
  } finally {
    // Pop synchronously before awaiting so concurrent compiles cannot nest scopes.
    const error = await device.popErrorScope();
    if (error) throw new AvelError(`${label}: ${error.message}`);
  }
}
