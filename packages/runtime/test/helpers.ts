import { afterEach, beforeEach, mock } from "bun:test";
import { type Model, Runtime, type RuntimeOptions, type WebGpuRuntime } from "../src/index.js";

export const IDENTITY = new URL("../../../tests/fixtures/identity.graph.json", import.meta.url);
export const MLP = new URL("../../../tests/fixtures/mlp.graph.json", import.meta.url);
const WASM = new URL("../wasm/avel_wasm_bg.wasm", import.meta.url);

export const MAP_READ = 1;
export const COPY_SRC = 4;
export const COPY_DST = 8;
export const STORAGE = 128;
export const QUERY_RESOLVE = 512;

export const MLP_TENSORS = { "0.weight": [8, 4], "0.bias": [8], "2.weight": [2, 8], "2.bias": [2] };

/** A `ValueInfo` literal for inline test graphs. */
export function valueInfo(name: string, dtype = "f32", shape: unknown[] = []) {
  return { name, dtype, shape };
}

/** A fresh zeroed input for the [3, 4] f32 `input` of mlp.graph.json. */
export function mlpInput(): Float32Array {
  return new Float32Array(12);
}

export class FakeBuffer {
  readonly bytes: Uint8Array;
  readonly size: number;
  readonly usage: number;
  readonly label: string;
  mapped: ArrayBuffer | undefined;

  constructor(descriptor: GPUBufferDescriptor) {
    this.size = descriptor.size;
    this.usage = descriptor.usage;
    this.label = descriptor.label ?? "";
    this.bytes = new Uint8Array(descriptor.size);
  }

  getMappedRange() {
    this.mapped = new ArrayBuffer(this.size);
    new Uint8Array(this.mapped).set(this.bytes);
    return this.mapped;
  }

  async mapAsync() {
    if (this.mapped) {
      throw new Error("buffer is already mapped");
    }
    await Promise.resolve();
  }

  destroyed = false;
  destroy() {
    this.destroyed = true;
  }

  unmap() {
    if (this.mapped) {
      this.bytes.set(new Uint8Array(this.mapped));
      this.mapped = undefined;
    }
  }
}

/**
 * Records GPU calls and applies copies at submit. Timed passes start 20 ns
 * apart and take 10 ns by default; tests can supply timings per shader.
 */
export function fakeDevice(features: string[] = [], durationNs = (_wgsl: string) => 10) {
  const calls: string[] = [];
  const buffers: FakeBuffer[] = [];
  const device = {
    features: new Set<string>(features),
    pushErrorScope() {},
    async popErrorScope() {
      return null;
    },
    limits: {
      maxBufferSize: 1 << 30,
      maxStorageBufferBindingSize: 1 << 30,
      maxComputeWorkgroupStorageSize: 32 << 10,
      maxStorageBuffersPerShaderStage: 10,
      minStorageBufferOffsetAlignment: 256,
    },
    createQuerySet: ({ count }: { count: number }) => ({
      timestamps: new BigInt64Array(count),
      destroy() {},
    }),
    createBuffer(descriptor: GPUBufferDescriptor) {
      const buffer = new FakeBuffer(descriptor);
      buffers.push(buffer);
      return buffer;
    },
    createShaderModule: ({ code }: { code: string }) => ({ code }),
    createComputePipelineAsync: async ({ compute }: { compute: { module: { code: string } } }) => ({
      code: compute.module.code,
      getBindGroupLayout: () => ({}),
    }),
    createBindGroup: (descriptor: GPUBindGroupDescriptor) => descriptor,
    createCommandEncoder() {
      const copies: [FakeBuffer, number, FakeBuffer, number, number][] = [];
      return {
        copyBufferToBuffer(
          src: FakeBuffer,
          srcOffset: number,
          dst: FakeBuffer,
          dstOffset: number,
          size: number,
        ) {
          calls.push(`copy ${src.label} -> ${dst.label}`);
          copies.push([src, srcOffset, dst, dstOffset, size]);
        },
        resolveQuerySet(
          querySet: { timestamps: BigInt64Array },
          first: number,
          count: number,
          dst: FakeBuffer,
        ) {
          const timestamps = querySet.timestamps.slice(first, first + count);
          dst.bytes.set(new Uint8Array(timestamps.buffer));
        },
        beginComputePass: (descriptor?: GPUComputePassDescriptor) => ({
          setPipeline(pipeline: { code: string }) {
            const writes = descriptor?.timestampWrites;
            calls.push(writes ? "timed pass" : "setPipeline");
            if (!writes) return;
            const { timestamps } = writes.querySet as unknown as { timestamps: BigInt64Array };
            const start = 10 * writes.beginningOfPassWriteIndex!;
            timestamps[writes.beginningOfPassWriteIndex!] = BigInt(start);
            timestamps[writes.endOfPassWriteIndex!] = BigInt(start + durationNs(pipeline.code));
          },
          setBindGroup: () => calls.push("setBindGroup"),
          dispatchWorkgroups: (...workgroups: number[]) => calls.push(`dispatch ${workgroups}`),
          end: () => {},
        }),
        finish: () => copies,
      };
    },
    queue: {
      async onSubmittedWorkDone() {},
      writeBuffer(buffer: FakeBuffer, offset: number, data: ArrayBufferView) {
        calls.push(`write ${buffer.label}`);
        buffer.bytes.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), offset);
      },
      submit(commands: [FakeBuffer, number, FakeBuffer, number, number][][]) {
        calls.push("submit");
        for (const [src, srcOffset, dst, dstOffset, size] of commands.flat()) {
          dst.bytes.set(src.bytes.subarray(srcOffset, srcOffset + size), dstOffset);
        }
      },
    },
  };
  return { device: device as unknown as GPUDevice, calls, buffers };
}

export function safetensors(
  tensors: Record<string, number[]>,
  dtype = "F32",
): Uint8Array<ArrayBuffer> {
  let offset = 0;
  const header: Record<string, unknown> = { __metadata__: { format: "pt" } };
  const values: number[] = [];
  for (const [name, shape] of Object.entries(tensors)) {
    const size = shape.reduce((a, b) => a * b, 1);
    header[name] = { dtype, shape, data_offsets: [offset, offset + size * 4] };
    offset += size * 4;
    values.push(...Array.from({ length: size }, (_, i) => values.length + i));
  }
  return safetensorsFile(header, new Uint8Array(new Float32Array(values).buffer));
}

export function safetensorsFile(header: object, data: Uint8Array): Uint8Array<ArrayBuffer> {
  const json = new TextEncoder().encode(JSON.stringify(header));
  const file = new Uint8Array(8 + json.length + data.length);
  new DataView(file.buffer).setBigUint64(0, BigInt(json.length), true);
  file.set(json, 8);
  file.set(data, 8 + json.length);
  return file;
}

/** Serves `graph` and `weights`, answering `Range` requests unless `ranges` is false. */
export function serve(graph: URL | object, weights = new Uint8Array(0), ranges = true) {
  const requests: string[] = [];
  stubGlobal(
    "fetch",
    mock(async (url: string, init?: RequestInit) => {
      if (url.endsWith("model.graph.json")) {
        return graph instanceof URL ? new Response(Bun.file(graph)) : Response.json(graph);
      }
      const range = new Headers(init?.headers).get("Range") ?? "";
      requests.push(range);
      const match = /^bytes=(\d+)-(\d+)$/.exec(range);
      if (!match) throw new Error(`expected a Range request, got ${JSON.stringify(range)}`);
      const [, start, end] = match;
      return ranges
        ? new Response(weights.slice(Number(start), Number(end) + 1), { status: 206 })
        : new Response(weights);
    }),
  );
  return requests;
}

const mlpWeights = safetensors(MLP_TENSORS);

export function serveMlp(ranges = true) {
  return serve(MLP, mlpWeights, ranges);
}

let wasm: Promise<ArrayBuffer> | undefined;

export async function runtime(
  device: GPUDevice,
  options: Omit<RuntimeOptions, "backend" | "device" | "wasm"> = {},
) {
  return Runtime.create({
    ...options,
    device,
    wasm: await (wasm ??= Bun.file(WASM).arrayBuffer()),
  });
}

const models = new Set<Model>();
const globals = new Map<string, PropertyDescriptor | undefined>();

export function stubGlobal(name: string, value: unknown) {
  if (!globals.has(name)) globals.set(name, Object.getOwnPropertyDescriptor(globalThis, name));
  Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
}

export function ownModel(model: Model): Model {
  models.add(model);
  return model;
}

export async function loadModel(rt: WebGpuRuntime, url: string): Promise<Model> {
  return ownModel(await rt.load(url));
}

export function setupTests(): void {
  beforeEach(() => {
    stubGlobal("GPUBufferUsage", { MAP_READ, COPY_SRC, COPY_DST, STORAGE, QUERY_RESOLVE });
    stubGlobal("GPUMapMode", { READ: 1 });
    stubGlobal("GPUTextureUsage", { COPY_DST: 2, TEXTURE_BINDING: 4, RENDER_ATTACHMENT: 16 });
    serve(IDENTITY);
  });

  afterEach(async () => {
    try {
      await Promise.all([...models].map((model) => model.destroy()));
    } finally {
      models.clear();
      for (const [name, descriptor] of globals) {
        if (descriptor) Object.defineProperty(globalThis, name, descriptor);
        else Reflect.deleteProperty(globalThis, name);
      }
      globals.clear();
    }
  });
}

export async function loadIdentity() {
  const fake = fakeDevice();
  return { ...fake, model: await loadModel(await runtime(fake.device), "/models/identity") };
}

export async function compileMlp(fake = fakeDevice()) {
  serveMlp();
  const model = await loadModel(await runtime(fake.device), "/models/mlp");
  return { ...fake, model, plan: await model.compile({ input: [3, 4] }) };
}
