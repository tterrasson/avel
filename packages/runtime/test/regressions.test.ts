import { expect, mock, test } from "bun:test";
import type { Parameter } from "../wasm/avel_wasm.js";
import { loadParameters } from "../src/safetensors.js";
import { trim } from "../src/shapes.js";
import {
  FakeBuffer,
  IDENTITY,
  MLP,
  MLP_TENSORS,
  fakeDevice,
  loadIdentity,
  loadModel,
  mlpInput,
  ownModel,
  runtime,
  safetensors,
  safetensorsFile,
  serve,
  serveMlp,
  setupTests,
  stubGlobal,
  valueInfo,
} from "./helpers.js";

setupTests();

test("custom ops infer dynamic secondary split outputs even when the first is unused", async () => {
  const value = (name: string, width: number) =>
    valueInfo(name, "f32", [{ name: "s", min: 1, max: 32 }, width]);
  serve({
    formatVersion: 1,
    opset: "core-aten/1",
    parameters: [],
    inputs: [value("x", 4)],
    outputs: [value("y", 2)],
    nodes: [
      {
        op: "aten.split_with_sizes.default",
        inputs: ["x"],
        outputs: [null, value("chunk", 2)],
        attributes: { split_sizes: [2, 2], dim: 1 },
      },
      { op: "test.copy.default", inputs: ["chunk"], outputs: [value("y", 2)] },
    ],
  });
  const rt = await runtime(fakeDevice().device);
  rt.registerOp("test.copy.default", {
    inputs: ["x"],
    shape: (inputs) => inputs[0]!.shape,
    kernel: () => ({
      wgsl: "@compute @workgroup_size(1) fn main() { output[0] = x[0]; }",
      workgroups: [1, 1, 1],
    }),
  });
  const model = await loadModel(rt, "/custom-split");
  const plan = await model.compile({ x: [3, 4] }, { bucket: 8 });
  expect(plan.plan.outputs[0]!.shape).toEqual([8, 2]);
  expect(plan.plan.requested!.outputs).toEqual([[3, 2]]);
});

test("i64 uploads pad an inner axis and reject values outside their storage range", async () => {
  const value = { name: "x", dtype: "i64", shape: [2, { name: "s", min: 1, max: 16 }, 3] };
  serve({
    formatVersion: 1,
    opset: "core-aten/1",
    parameters: [],
    nodes: [],
    inputs: [value],
    outputs: [value],
  });
  const model = await loadModel(await runtime(fakeDevice().device), "/i64");
  const plan = await model.compile({ x: [2, 3, 3] }, { bucket: 8 });
  for (const length of [3, 8, 1, 5]) {
    const x = BigInt64Array.from({ length: 2 * length * 3 }, (_, i) => BigInt(i - 20));
    x[0] = -2147483648n;
    x[x.length - 1] = 2147483647n;
    expect((await plan.run({ x })).x).toEqual(x);
  }
  await expect(plan.run({ x: new BigInt64Array(18).fill(2147483648n) })).rejects.toThrow(
    "32-bit range",
  );
  await expect(plan.run({ x: new BigInt64Array(17) })).rejects.toThrow("elements");
});

test("failed state snapshot allocation releases weights and earlier snapshots", async () => {
  serve(
    {
      formatVersion: 1,
      opset: "core-aten/1",
      nodes: [],
      inputs: [{ name: "x", dtype: "f32", shape: [3] }],
      outputs: [{ name: "x", dtype: "f32", shape: [3] }],
      parameters: ["a", "b"].map((name) => ({
        name,
        tensor: name,
        dtype: "f32",
        shape: [3],
        update: "x",
      })),
    },
    safetensors({ a: [3], b: [3] }),
  );
  const fake = fakeDevice();
  const create = fake.device.createBuffer.bind(fake.device);
  fake.device.createBuffer = (descriptor) => {
    if (descriptor.label === "initial b") throw new Error("snapshot allocation failed");
    return create(descriptor);
  };
  await expect(loadModel(await runtime(fake.device), "/state")).rejects.toThrow(
    "snapshot allocation failed",
  );
  expect(fake.buffers).toHaveLength(3);
  expect(fake.buffers.every((buffer) => buffer.destroyed)).toBe(true);
});

test("a GPU consumer cannot overwrite its plan's input with a nested run", async () => {
  const { model } = await loadIdentity();
  const plan = await model.compile({ x: [1, 4] });
  const first = new Float32Array([1, 2, 3, 4]);
  const second = new Float32Array([5, 6, 7, 8]);
  const outputs = plan.runGpu({ x: first }, () => {
    expect(() => plan.runGpu({ x: second })).toThrow("already submitting");
  });
  expect(new Float32Array((outputs.x as unknown as FakeBuffer).bytes.buffer)).toEqual(first);
  expect(() =>
    plan.runGpu({ x: first }, () => {
      throw new Error("consumer failed");
    }),
  ).toThrow("consumer failed");
  expect(() => plan.runGpu({ x: second })).not.toThrow();
});

test("overlapping bucketed readbacks use the input lengths of their own submission", async () => {
  const { model } = await loadIdentity();
  const plan = await model.compile({ x: [17, 4] }, { bucket: 16 });
  const inputs = [17, 32, 21, 30, 18, 31, 22].map((length) =>
    Float32Array.from({ length: length * 4 }, (_, i) => length + i),
  );
  const outputs = await Promise.all(inputs.map((x) => plan.run({ x })));
  expect(outputs.map((output) => output.x)).toEqual(inputs);
});

test("bucketed GPU runs reject empty input even if the declared minimum is zero", async () => {
  const graph = await Bun.file(IDENTITY).json();
  graph.inputs[0].shape[0].min = 0;
  graph.outputs[0].shape[0].min = 0;
  serve(graph);
  const fake = fakeDevice();
  const model = await loadModel(await runtime(fake.device), "/identity");
  const plan = await model.compile({ x: [3, 4] }, { bucket: 4 });
  expect(() => plan.runGpu({ x: new Float32Array(0) })).toThrow("has 0 bytes");
  expect(fake.calls).not.toContain("submit");
});

test("trims more than one axis while retaining complete inner rows", () => {
  const source = Float32Array.from({ length: 24 }, (_, i) => i);
  expect(trim(source, [2, 3, 4], [2, 2, 3])).toEqual(
    new Float32Array([0, 1, 2, 4, 5, 6, 12, 13, 14, 16, 17, 18]),
  );
  expect(trim(source, [2, 3, 4], [2, 2, 4])).toEqual(
    new Float32Array([...source.slice(0, 8), ...source.slice(12, 20)]),
  );
});

test("validates requested shapes even when their bucket is cached", async () => {
  const { model } = await loadIdentity();
  await model.compile({ x: [60, 4] }, { bucket: 16 });
  for (const shape of [[65, 4], [0, 4], [60, 5], [60], [NaN, 4], [1.5, 4]]) {
    await expect(model.compile({ x: shape }, { bucket: 16 })).rejects.toThrow();
  }
  for (const bucket of [-1, 1.5, NaN, Infinity]) {
    await expect(model.compile({ x: [2, 4] }, { bucket })).rejects.toThrow("bucket");
  }
});

test("explicit f16 settings use device support independently of the default preference", async () => {
  serve({
    formatVersion: 1,
    opset: "core-aten/1",
    inputs: [{ name: "x", dtype: "f32", shape: [4] }],
    outputs: [{ name: "x", dtype: "f32", shape: [4] }],
    nodes: [],
    parameters: [],
    f16: true,
  });
  const rt = await runtime(fakeDevice(["shader-f16"]).device);
  const model = await loadModel(rt, "/f16");
  await expect(model.compile({ x: [4] }, { f16: true })).resolves.toBeDefined();
});

test("identity profiling avoids empty query sets and still submits input data", async () => {
  const fake = fakeDevice(["timestamp-query"]);
  const rt = await runtime(fake.device);
  const plan = await (await loadModel(rt, "/identity")).compile({ x: [1, 4] });
  fake.device.createQuerySet = () => {
    throw new Error("must not create an empty query set");
  };
  expect((await plan.profile({ x: new Float32Array(4) })).totalNs).toBe(0);
  expect(fake.calls).toContain("submit");
});

test("profiling reserves its place in the submission queue", async () => {
  const fake = fakeDevice(["timestamp-query"]);
  const plan = await (
    await loadModel(await runtime(fake.device), "/identity")
  ).compile({ x: [1, 4] });
  const profile = plan.profile({ x: new Float32Array(4).fill(1) });
  expect(() => plan.runGpu({ x: new Float32Array(4) })).toThrow("waiting to submit");
  const result = plan.run({ x: new Float32Array(4).fill(2) });
  await profile;
  expect(await result).toEqual({ x: new Float32Array(4).fill(2) });
});

test("destroy waits for pending readbacks and destroys plans and shared weights", async () => {
  serveMlp();
  const fake = fakeDevice();
  const model = await loadModel(await runtime(fake.device), "/mlp");
  const plan = await model.compile({ input: [3, 4] });
  const run = plan.run({ input: mlpInput() });
  const destruction = model.destroy();
  await run;
  await destruction;
  expect(fake.buffers.every((buffer) => buffer.destroyed)).toBe(true);
  await expect(model.compile({ input: [3, 4] })).rejects.toThrow("destroyed");
  await expect(plan.run({ input: mlpInput() })).rejects.toThrow("destroyed");
  expect(() => plan.runGpu({ input: mlpInput() })).toThrow("destroyed");
  await plan.destroy();
});

test("recompiles a destroyed cached plan", async () => {
  const { model } = await loadIdentity();
  const plan = await model.compile({ x: [1, 4] });
  await plan.destroy();
  const replacement = await model.compile({ x: [1, 4] });
  expect(replacement).not.toBe(plan);
  expect(await replacement.run({ x: new Float32Array(4).fill(3) })).toEqual({
    x: new Float32Array(4).fill(3),
  });
});

test("failed pipeline creation releases plan buffers and allows a retry with the same weights", async () => {
  serveMlp();
  const fake = fakeDevice();
  const model = await loadModel(await runtime(fake.device), "/mlp");
  const weights = [...fake.buffers];
  expect(weights).toHaveLength(4);
  const createPipeline = fake.device.createComputePipelineAsync;
  fake.device.createComputePipelineAsync = async () => {
    throw new Error("invalid WGSL");
  };
  await expect(model.compile({ input: [3, 4] })).rejects.toThrow("invalid WGSL");
  const failedAllocations = fake.buffers.slice(weights.length);
  expect(failedAllocations.length).toBeGreaterThan(0);
  expect(failedAllocations.every((b) => b.destroyed)).toBe(true);
  expect(weights.every((b) => !b.destroyed)).toBe(true);

  fake.device.createComputePipelineAsync = createPipeline;
  const retried = await model.compile({ input: [3, 4] });
  expect(await retried.run({ input: mlpInput() })).toEqual({
    relu: new Float32Array(6),
  });
  expect(weights.every((b) => !b.destroyed)).toBe(true);
});

test("failed weight downloads release successful uploads", async () => {
  serve(MLP, safetensors(MLP_TENSORS).slice(0, -4));
  const fake = fakeDevice();
  await expect(loadModel(await runtime(fake.device), "/mlp")).rejects.toThrow("truncated");
  expect(fake.buffers.length).toBeGreaterThan(0);
  expect(fake.buffers.every((buffer) => buffer.destroyed)).toBe(true);
});

test("rejects offsets that disagree with tensor shape before creating a buffer", async () => {
  const file = safetensorsFile(
    { w: { dtype: "F32", shape: [2], data_offsets: [0, 4] } },
    new Uint8Array(8),
  );
  serve({}, file);
  const fake = fakeDevice();
  await expect(
    loadParameters(fake.device, "/weights", [{ name: "w", tensor: "w", dtype: "f32", shape: [2] }]),
  ).rejects.toThrow("invalid data offsets");
  expect(fake.buffers).toHaveLength(0);
});

test("expands packed boolean weights into WebGPU u32 storage and deduplicates tensors", async () => {
  const file = safetensorsFile(
    { mask: { dtype: "BOOL", shape: [3], data_offsets: [0, 3] } },
    Uint8Array.of(1, 0, 1),
  );
  serve({}, file);
  const fake = fakeDevice();
  const parameters = await loadParameters(fake.device, "/weights", [
    { name: "a", tensor: "mask", dtype: "bool", shape: [3] },
    { name: "b", tensor: "mask", dtype: "bool", shape: [3] },
  ]);
  try {
    expect(parameters.size).toBe(1);
    expect(fake.buffers).toHaveLength(1);
    expect(new Uint32Array(fake.buffers[0]!.bytes.buffer)).toEqual(new Uint32Array([1, 0, 1]));
  } finally {
    for (const buffer of parameters.values()) buffer.destroy();
  }
});

test("builder snapshots own independent parameter buffers", async () => {
  const fake = fakeDevice();
  const rt = await runtime(fake.device);
  const builder = rt.graph();
  const weight = builder.parameter("w", [4], new Float32Array([1, 2, 3, 4]));
  const first = ownModel(builder.build([weight]));
  const second = ownModel(builder.build([weight]));
  await first.destroy();
  const plan = await second.compile({});
  expect(await plan.run({})).toEqual({ w: new Float32Array([1, 2, 3, 4]) });
});

test("tensor names do not access Object.prototype", async () => {
  const rt = await runtime(fakeDevice().device);
  const builder = rt.graph();
  const x = builder.input("__proto__", [4]);
  const plan = await ownModel(builder.build([x])).compile(Object.fromEntries([["__proto__", [4]]]));
  const input = new Float32Array([1, 2, 3, 4]);
  const result = await plan.run(Object.fromEntries([["__proto__", input]]));
  expect(Object.keys(result)).toEqual(["__proto__"]);
  expect(result.__proto__).toEqual(input);
});

test("bucket padding on an inner axis is cleared and reused across runs", async () => {
  serve({
    formatVersion: 1,
    opset: "core-aten/1",
    inputs: [{ name: "x", dtype: "f32", shape: [2, { name: "s", min: 1, max: 8 }, 2] }],
    outputs: [{ name: "x", dtype: "f32", shape: [2, { name: "s" }, 2] }],
    nodes: [],
    parameters: [],
  });
  const fake = fakeDevice();
  const plan = await (
    await loadModel(await runtime(fake.device), "/inner")
  ).compile({ x: [2, 3, 2] }, { bucket: 4 });
  for (const length of [3, 1, 2]) {
    const input = Float32Array.from({ length: 4 * length }, (_, i) => i + 1);
    expect((await plan.run({ x: input })).x).toEqual(input);
    const written = new Float32Array(fake.buffers.find((b) => b.label === "input 0")!.bytes.buffer);
    expect(written.slice(length * 2, 8)).toEqual(new Float32Array(8 - length * 2));
    expect(written.slice(8 + length * 2)).toEqual(new Float32Array(8 - length * 2));
  }
});

test("image textures belong to their execution plan", async () => {
  serve({
    formatVersion: 1,
    opset: "core-aten/1",
    inputs: [{ name: "x", dtype: "f32", shape: [1, 3, 2, 2] }],
    outputs: [{ name: "x", dtype: "f32", shape: [1, 3, 2, 2] }],
    nodes: [],
    parameters: [],
  });
  const fake = fakeDevice();
  let destroyed = false;
  fake.device.createTexture = () =>
    ({
      createView: () => ({}),
      destroy: () => {
        destroyed = true;
      },
    }) as unknown as GPUTexture;
  const plan = await (
    await loadModel(await runtime(fake.device), "/image")
  ).compile(
    { x: [1, 3, 2, 2] },
    { media: { x: { source: { kind: "image", width: 2, height: 2 } } } },
  );
  await plan.destroy();
  expect(destroyed).toBe(true);
});

test("bucketed runs validate shared symbolic sizes before writing any input", async () => {
  const dim = { name: "s", min: 1, max: 8 };
  serve({
    formatVersion: 1,
    opset: "core-aten/1",
    inputs: [
      { name: "a", dtype: "f32", shape: [dim, 2] },
      { name: "b", dtype: "f32", shape: [dim, 2] },
    ],
    outputs: [{ name: "a", dtype: "f32", shape: [dim, 2] }],
    nodes: [],
    parameters: [],
  });
  const fake = fakeDevice();
  const model = await loadModel(await runtime(fake.device), "/shared");
  const plan = await model.compile({ a: [3, 2], b: [3, 2] }, { bucket: 4 });
  await expect(plan.run({ a: new Float32Array(6), b: new Float32Array(4) })).rejects.toThrow(
    "inconsistent sizes",
  );
  expect(fake.calls.filter((call) => call.startsWith("write"))).toHaveLength(0);
  expect(() => plan.runGpu({ a: new Float32Array(6), b: new Float32Array(4) })).toThrow(
    "inconsistent sizes",
  );
});

/** A safetensors file whose first 8 bytes declare `length`, whatever the header holds. */
function fileWithHeaderLength(length: bigint, header: string, data = new Uint8Array(0)) {
  const json = new TextEncoder().encode(header);
  const file = new Uint8Array(8 + json.length + data.length);
  new DataView(file.buffer).setBigUint64(0, length, true);
  file.set(json, 8);
  file.set(data, 8 + json.length);
  return file;
}

test.each([
  ["a file shorter than 8 bytes", new Uint8Array(4), "too short"],
  ["a header length below 2", fileWithHeaderLength(1n, "{}"), "invalid header length"],
  ["an oversized header length", fileWithHeaderLength(100_000_001n, "{}"), "invalid header length"],
  ["a truncated header", fileWithHeaderLength(1000n, "{}"), "truncated header"],
  ["a non-object header", fileWithHeaderLength(5n, "[1,2]"), "invalid header"],
])("rejects %s", async (_, file, message) => {
  serve({}, file);
  const fake = fakeDevice();
  const parameter: Parameter = { name: "w", tensor: "w", dtype: "f32", shape: [2] };
  await expect(loadParameters(fake.device, "/weights", [parameter])).rejects.toThrow(message);
});

test("rejects a weight server that fails the header request or answers a range without 206", async () => {
  const fake = fakeDevice();
  const parameter: Parameter = { name: "w", tensor: "w", dtype: "f32", shape: [2] };

  stubGlobal(
    "fetch",
    mock(async () => new Response("gone", { status: 404 })),
  );
  await expect(loadParameters(fake.device, "/weights", [parameter])).rejects.toThrow("HTTP 404");

  const file = safetensors({ w: [2] });
  stubGlobal(
    "fetch",
    mock(async (_url: string, init?: RequestInit) => {
      const range = new Headers(init?.headers).get("Range") ?? "";
      // The header probe is a range too; only the follow-up header fetch gets a plain 200.
      return range === "bytes=0-7"
        ? new Response(file.slice(0, 8), { status: 206 })
        : new Response(file.slice(8), { status: 200 });
    }),
  );
  await expect(loadParameters(fake.device, "/weights", [parameter])).rejects.toThrow("HTTP 200");
});

test("rejects reserved, unknown and oversized tensors", async () => {
  const fake = fakeDevice();
  const file = safetensors({ w: [3] });
  serve({}, file);
  await expect(
    loadParameters(fake.device, "/weights", [
      { name: "meta", tensor: "__metadata__", dtype: "f32", shape: [1] },
    ]),
  ).rejects.toThrow("no tensor `__metadata__`");
  await expect(
    loadParameters(fake.device, "/weights", [
      { name: "w", tensor: "w", dtype: "f64", shape: [3] } as unknown as Parameter,
    ]),
  ).rejects.toThrow("expected f64");

  (fake.device.limits as { maxBufferSize: number }).maxBufferSize = 8;
  await expect(
    loadParameters(fake.device, "/weights", [{ name: "w", tensor: "w", dtype: "f32", shape: [3] }]),
  ).rejects.toThrow("exceeds the device buffer limit");
});

test("rejects parameters that describe the same tensor differently", async () => {
  serve({}, safetensors({ w: [3] }));
  const fake = fakeDevice();
  await expect(
    loadParameters(fake.device, "/weights", [
      { name: "a", tensor: "w", dtype: "f32", shape: [3] },
      { name: "b", tensor: "w", dtype: "f32", shape: [2] },
    ]),
  ).rejects.toThrow("conflicting descriptions for tensor `w`");
});

test("resetState restores the initial state and refuses once destroyed", async () => {
  // A model without state resets nothing.
  const plain = await loadIdentity();
  plain.calls.length = 0;
  plain.model.resetState();
  expect(plain.calls).toEqual([]);

  serve(
    {
      formatVersion: 1,
      opset: "core-aten/1",
      nodes: [],
      inputs: [valueInfo("x", "f32", [3])],
      outputs: [valueInfo("x", "f32", [3])],
      parameters: ["a", "b"].map((name) => ({
        name,
        tensor: name,
        dtype: "f32",
        shape: [3],
        update: "x",
      })),
    },
    safetensors({ a: [3], b: [3] }),
  );
  const fake = fakeDevice();
  const model = await loadModel(await runtime(fake.device), "/state");
  fake.calls.length = 0;
  model.resetState();
  expect(fake.calls).toEqual(["copy initial a -> a", "copy initial b -> b", "submit"]);

  await model.destroy();
  expect(() => model.resetState()).toThrow("model is destroyed");
});

test("profiling allocation failures release the query set and earlier buffers", async () => {
  serveMlp();
  const fake = fakeDevice(["timestamp-query"]);
  const plan = await (
    await loadModel(await runtime(fake.device), "/mlp")
  ).compile({ input: [3, 4] });
  let queryDestroyed = false;
  fake.device.createQuerySet = () =>
    ({
      destroy: () => {
        queryDestroyed = true;
      },
    }) as GPUQuerySet;
  const create = fake.device.createBuffer.bind(fake.device);
  let allocations = 0;
  fake.device.createBuffer = (descriptor) => {
    if (allocations++ === 1) throw new Error("readback allocation failed");
    return create(descriptor);
  };
  await expect(plan.profile({ input: mlpInput() })).rejects.toThrow("readback allocation failed");
  expect(queryDestroyed).toBe(true);
  expect(fake.buffers.at(-1)!.destroyed).toBe(true);
  expect(() => plan.runGpu({ input: mlpInput() })).not.toThrow();
  await plan.destroy();
});
