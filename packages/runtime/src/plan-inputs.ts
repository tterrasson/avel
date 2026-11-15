import type { Graph, Plan } from "../wasm/avel_wasm.js";
import { BucketInputs } from "./buckets.js";
import { AvelError } from "./errors.js";
import type { MediaBinding, MediaInputSource } from "./media.js";
import { pad, product } from "./shapes.js";

type CpuInput = Float32Array | Int32Array | Uint32Array;
/**
 * A Float32Array for f32 inputs, an Int32Array for i32, a Uint32Array for
 * bool, and a BigInt64Array or an Int32Array for i64: the GPU stores i64 in
 * 32 bits, so larger values are rejected.
 */
export type RunInput = CpuInput | BigInt64Array | GPUBuffer | MediaInputSource;

interface Padding {
  compiled: number;
  minimum: number;
  fixed: number;
  inner: number;
  block: number;
  scratch: CpuInput;
}

interface InputBinding {
  name: string;
  array: typeof Float32Array | typeof Int32Array | typeof Uint32Array;
  media: MediaBinding | undefined;
  buffer: GPUBuffer | undefined;
  bytes: number;
  padding: Padding | undefined;
  /** For i64 inputs: where BigInt64Array values are narrowed to 32 bits. */
  narrowed: Int32Array | undefined;
}

/** Prepares uploads and bucket scratch space once for the frozen input layout. */
export class PlanInputs {
  private readonly bindings: InputBinding[];
  private readonly bucket: BucketInputs | undefined;
  private readonly dimensions: Uint32Array | undefined;

  constructor(
    private readonly device: GPUDevice,
    graph: Graph,
    plan: Plan,
    buffers: (GPUBuffer | undefined)[],
    media: Map<number, MediaBinding>,
  ) {
    this.bucket = plan.requested
      ? new BucketInputs(graph, plan.inputs, (index) => media.has(plan.inputs[index]!.buffer))
      : undefined;
    this.bindings = plan.inputs.map(({ name, shape, dtype, buffer }, index) => {
      const array =
        dtype === "f32"
          ? Float32Array
          : dtype === "i32" || dtype === "i64"
            ? Int32Array
            : Uint32Array;
      const image = media.get(buffer);
      const layout = this.bucket?.padding(index);
      const padding = layout && { ...layout, scratch: new array(product(shape)) };
      return {
        name,
        array,
        media: image,
        buffer: buffers[buffer],
        bytes: plan.buffers[buffer]!.size,
        padding,
        narrowed: dtype === "i64" ? new Int32Array(product(shape)) : undefined,
      };
    });
    this.dimensions = this.bucket?.dimensions();
  }

  /** A readback slot owns its dimensions until its mapped output is copied. */
  readbackDimensions(): Uint32Array | undefined {
    return this.dimensions && new Uint32Array(this.dimensions.length);
  }

  clear(): void {
    this.bindings.length = 0;
  }

  write(
    encoder: GPUCommandEncoder,
    inputs: Record<string, RunInput>,
    dimensions = this.dimensions,
  ): void {
    if (dimensions) this.bucket!.captureDimensions(inputs, dimensions);
    for (let i = 0; i < this.bindings.length; i++) {
      const { name, array, media, buffer, bytes, padding, narrowed } = this.bindings[i]!;
      const input = Object.hasOwn(inputs, name) ? inputs[name] : undefined;
      if (input === undefined) throw new AvelError(`missing input \`${name}\``);
      if (media) {
        media.write(input);
      } else if (narrowed && input instanceof BigInt64Array) {
        const length = dimensions?.[i];
        const padded = padding && length !== undefined && length !== padding.compiled;
        const expected = padded ? length * padding.fixed : narrowed.length;
        if (input.length !== expected)
          throw new AvelError(
            `input \`${name}\` has ${input.length} elements, expected ${expected}`,
          );
        narrow(
          name,
          input,
          narrowed,
          padded ? length * padding.inner : input.length,
          padded ? padding.block : input.length,
        );
        this.device.queue.writeBuffer(buffer!, 0, narrowed);
      } else if (ArrayBuffer.isView(input)) {
        if (!(input instanceof array))
          throw new AvelError(`input \`${name}\` must be a ${array.name} or a GPUBuffer`);
        const length = dimensions?.[i];
        const data =
          padding && length !== undefined && length !== padding.compiled
            ? pad(input, length * padding.inner, padding.block, padding.scratch)
            : input;
        if (data.byteLength !== bytes)
          throw new AvelError(`input \`${name}\` has ${input.byteLength} bytes, expected ${bytes}`);
        this.device.queue.writeBuffer(buffer!, 0, data);
      } else {
        if (!input || typeof input !== "object" || !("size" in input && "usage" in input))
          throw new AvelError(
            `input \`${name}\` requires a tensor; compile with media settings for image sources`,
          );
        if (input.size !== bytes)
          throw new AvelError(`input \`${name}\` has ${input.size} bytes, expected ${bytes}`);
        if (!(input.usage & GPUBufferUsage.COPY_SRC))
          throw new AvelError(`input \`${name}\` needs COPY_SRC usage`);
        encoder.copyBufferToBuffer(input, 0, buffer!, 0, bytes);
      }
    }
  }

  outputSymbols(dimensions: Uint32Array | undefined): Record<string, number> | undefined {
    return this.bucket?.outputSymbols(dimensions);
  }
}

/** Narrows and pads into preallocated storage without creating typed-array views. */
export function narrow(
  name: string,
  input: BigInt64Array,
  scratch: Int32Array,
  sourceBlock: number,
  targetBlock: number,
): void {
  if (sourceBlock !== targetBlock) scratch.fill(0);
  for (
    let source = 0, target = 0;
    source < input.length;
    source += sourceBlock, target += targetBlock
  ) {
    for (let i = 0; i < sourceBlock; i++) {
      const value = input[source + i]!;
      if (value < -2147483648n || value > 2147483647n)
        throw new AvelError(`input \`${name}\` holds an int64 value outside the 32-bit range`);
      scratch[target + i] = Number(value);
    }
  }
}
