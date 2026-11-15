import { expect, mock, spyOn, test } from "bun:test";
import { AvelError, ExecutionPlan, Runtime } from "../src/index.js";
import { LoadedGraph } from "../wasm/avel_wasm.js";
import {
  COPY_DST,
  COPY_SRC,
  STORAGE,
  MAP_READ,
  FakeBuffer,
  MLP,
  MLP_TENSORS,
  compileMlp,
  fakeDevice,
  loadIdentity,
  loadModel,
  mlpInput,
  ownModel,
  runtime,
  safetensors,
  serve,
  serveMlp,
  setupTests,
  stubGlobal,
  valueInfo,
} from "./helpers.js";

setupTests();

test("compiles a graph through the wasm compiler", async () => {
  const { model } = await loadIdentity();
  const { plan } = await model.compile({ x: [2, 4] });
  expect(plan.outputs).toEqual([{ name: "x", dtype: "f32", shape: [2, 4], buffer: 0 }]);
  expect(plan.buffers).toEqual([{ size: 32, role: { kind: "input" }, dtype: "f32" }]);
  expect(plan.dispatches).toEqual([]);
});

test("caches plans by shape", async () => {
  const { model } = await loadIdentity();
  expect(await model.compile({ x: [2, 4] })).toBe(await model.compile({ x: [2, 4] }));
  expect(await model.compile({ x: [3, 4] })).not.toBe(await model.compile({ x: [2, 4] }));
});

test("does not fetch weights for a model without parameters", async () => {
  await loadIdentity();
  expect(globalThis.fetch).toHaveBeenCalledTimes(1);
});

test("loads each parameter by byte range", async () => {
  const requests = serve(MLP, safetensors(MLP_TENSORS));
  const { device, buffers } = fakeDevice();
  await loadModel(await runtime(device), "/models/mlp");

  expect(requests).toHaveLength(2 + 4);
  const bias = buffers.find((buffer) => buffer.label === "0.bias")!;
  expect(bias.usage & STORAGE).toBe(STORAGE);
  expect([...new Float32Array(bias.bytes.buffer)]).toEqual(
    Array.from({ length: 8 }, (_, i) => 32 + i),
  );
});

test("reads the whole file when the server ignores ranges", async () => {
  const requests = serve(MLP, safetensors(MLP_TENSORS), false);
  const { device, buffers } = fakeDevice();
  await loadModel(await runtime(device), "/models/mlp");

  expect(requests).toHaveLength(1);
  const bias = buffers.find((buffer) => buffer.label === "2.bias")!;
  expect([...new Float32Array(bias.bytes.buffer)]).toEqual([56, 57]);
});

test("shares parameters between plans", async () => {
  serveMlp();
  const { device, buffers } = fakeDevice();
  const model = await loadModel(await runtime(device), "/models/mlp");
  await model.compile({ input: [3, 4] });
  await model.compile({ input: [5, 4] });
  expect(buffers.filter((buffer) => buffer.label === "0.weight")).toHaveLength(1);
});

test.each([
  ["a missing tensor", { "0.weight": [8, 4] }, "F32", "no tensor `0.bias`"],
  [
    "a wrong shape",
    { ...MLP_TENSORS, "0.weight": [4, 8] },
    "F32",
    "has shape [4,8], expected [8,4]",
  ],
  ["a wrong dtype", MLP_TENSORS, "F64", "has dtype F64, expected f32"],
])("rejects weights with %s", async (_, tensors, dtype, message) => {
  serve(MLP, safetensors(tensors, dtype));
  const { device } = fakeDevice();
  await expect(loadModel(await runtime(device), "/models/mlp")).rejects.toThrow(message);
});

test("allocates buffers and creates pipelines and bind groups at compile time", async () => {
  const fake = fakeDevice();
  const pipeline = spyOn(fake.device, "createComputePipelineAsync");
  const bindGroup = spyOn(fake.device, "createBindGroup");
  const { plan, buffers } = await compileMlp(fake);
  const usage = (label: string) => buffers.find((buffer) => buffer.label.startsWith(label))!.usage;
  const allocations = buffers.length;
  expect(pipeline).toHaveBeenCalledTimes(2);
  expect(bindGroup).toHaveBeenCalledTimes(2);
  await plan.run({ input: mlpInput() });
  expect(pipeline).toHaveBeenCalledTimes(2);
  expect(bindGroup).toHaveBeenCalledTimes(2);
  expect(buffers).toHaveLength(allocations);
  expect(usage("input")).toBe(STORAGE | COPY_DST | COPY_SRC);
  expect(usage("output")).toBe(STORAGE | COPY_SRC);
  expect(usage("arena")).toBe(STORAGE);
  expect(
    buffers.filter((buffer) => buffer.label.startsWith("staging")).map(({ usage }) => usage),
  ).toEqual([COPY_DST | MAP_READ, COPY_DST | MAP_READ, COPY_DST | MAP_READ]);
});

test("writes inputs, runs the frozen dispatch list and submits once", async () => {
  const { plan, calls, buffers } = await compileMlp();
  const encoded = () => calls.filter((call) => call !== "setPipeline" && call !== "setBindGroup");

  const expected = plan.plan.dispatches.map(({ workgroups }) => `dispatch ${workgroups}`);
  const outputBuffer = buffers.find((buffer) => buffer.label.startsWith("output"))!;
  // The fake queue copies real bytes but does not execute WGSL.
  const values = new Float32Array([1, 2, 3, 4, 5, 6]);
  new Float32Array(outputBuffer.bytes.buffer).set(values);

  const input = Float32Array.from({ length: 12 }, (_, i) => i + 1);
  expect(await plan.run({ input })).toEqual({ relu: values });
  const written = buffers.find((buffer) => buffer.label === "input 0")!;
  expect(new Float32Array(written.bytes.buffer)).toEqual(input);
  expect(encoded()).toEqual([
    "write input 0",
    ...expected,
    `copy ${outputBuffer.label} -> staging 0`,
    "submit",
  ]);
});

test("computes constants once, when the plan is created", async () => {
  const value = (name: string) => valueInfo(name, "f32", [3]);
  serve(
    {
      formatVersion: 1,
      opset: "core-aten/1",
      f16: false,
      inputs: [value("x")],
      outputs: [value("out")],
      nodes: [
        {
          op: "aten.mul.Tensor",
          inputs: ["w"],
          outputs: [value("scaled")],
          attributes: { other: 2 },
        },
        {
          op: "aten.add.Tensor",
          inputs: ["x", "scaled"],
          outputs: [value("out")],
          attributes: { alpha: 1 },
        },
      ],
      parameters: [{ name: "w", tensor: "w", dtype: "f32", shape: [3] }],
    },
    safetensors({ w: [3] }),
  );
  const { device, calls, buffers } = fakeDevice();
  const model = await loadModel(await runtime(device), "/models/folded");
  const plan = await model.compile({ x: [3] });

  expect(plan.plan.prepare).toHaveLength(1);
  expect(buffers.find((buffer) => buffer.label.startsWith("constant"))!.usage).toBe(
    STORAGE | COPY_SRC,
  );
  expect(calls.filter((call) => call.startsWith("dispatch") || call === "submit")).toEqual([
    "dispatch 1,1,1",
    "submit",
  ]);

  calls.length = 0;
  await plan.run({ x: new Float32Array(3) }, { keepOnGpu: true });
  expect(calls.filter((call) => call.startsWith("dispatch"))).toHaveLength(1);
});

test("returns an input forwarded as output", async () => {
  const { model, buffers } = await loadIdentity();
  const plan = await model.compile({ x: [1, 4] });
  const input = new Float32Array([1, 2, 3, 4]);
  expect(await plan.run({ x: input })).toEqual({ x: input });
  expect(buffers.find((buffer) => buffer.label === "staging 0")!.mapped).toBeUndefined();
});

test("copies GPUBuffer inputs into the plan", async () => {
  const { model, device, calls } = await loadIdentity();
  const plan = await model.compile({ x: [1, 4] });
  const input = device.createBuffer({ label: "user", size: 16, usage: COPY_SRC | STORAGE });
  try {
    new Float32Array((input as unknown as FakeBuffer).bytes.buffer).set([1, 2, 3, 4]);
    expect(await plan.run({ x: input })).toEqual({ x: new Float32Array([1, 2, 3, 4]) });
    expect(calls).toContain("copy user -> input 0");
    expect(calls).not.toContain("write input 0");
    await plan.destroy();
    expect((input as unknown as FakeBuffer).destroyed).toBe(false);
  } finally {
    input.destroy();
  }
});

test("keeps outputs on the GPU", async () => {
  const { plan, calls, buffers } = await compileMlp();
  const { relu } = await plan.run({ input: mlpInput() }, { keepOnGpu: true });

  expect(relu).toBe(
    buffers.find((buffer) => buffer.label.startsWith("output")) as unknown as GPUBuffer,
  );
  expect(calls.filter((call) => call.includes("staging"))).toEqual([]);
  expect(calls.filter((call) => call === "submit")).toHaveLength(1);
});

test("submits a run before the previous readback completes", async () => {
  const { model, calls, buffers } = await loadIdentity();
  const plan = await model.compile({ x: [1, 4] });
  // Each staging buffer stays busy until its readback is released.
  const blocked = buffers
    .filter(({ label }) => label.startsWith("staging"))
    .map((buffer) => {
      const gate = Promise.withResolvers<void>();
      const map = buffer.mapAsync.bind(buffer);
      buffer.mapAsync = async () => {
        await gate.promise;
        await map();
      };
      return gate;
    });
  const copies = () => calls.filter((call) => call.startsWith("copy"));
  const inputs = [1, 2, 3, 4].map((value) => new Float32Array(4).fill(value));
  const runs = inputs.map((x) => plan.run({ x }));
  const finished = Promise.allSettled(runs);
  try {
    await Bun.sleep(0);
    expect(copies()).toEqual([
      "copy input 0 -> staging 0",
      "copy input 0 -> staging 1",
      "copy input 0 -> staging 2",
    ]);
    blocked[0]!.resolve();
    await Bun.sleep(0);
    expect(copies()).toHaveLength(4);
    expect(copies().at(-1)).toBe("copy input 0 -> staging 0");
  } finally {
    for (const gate of blocked) gate.resolve();
    await finished;
  }
  expect(await Promise.all(runs)).toEqual(inputs.map((x) => ({ x })));
});

test("profiles each dispatch with timestamp queries", async () => {
  serveMlp();
  const fake = fakeDevice(["timestamp-query"]);
  const model = await loadModel(await runtime(fake.device), "/models/mlp");
  const plan = await model.compile({ input: [3, 4] });
  expect(plan.profiling).toBe(true);

  const profile = await plan.profile({ input: mlpInput() });
  expect(fake.calls.filter((call) => call === "timed pass")).toHaveLength(2);
  expect(profile.dispatches.map(({ startNs, durationNs }) => [startNs, durationNs])).toEqual([
    [0, 10],
    [20, 10],
  ]);
  expect(profile.totalNs).toBe(20);
  const trace = profile.trace("mlp");
  const events = trace.traceEvents.filter((event) => (event as { ph: string }).ph === "X");
  expect(events.map((event) => (event as { name: string }).name)).toEqual(
    plan.plan.dispatches.map(({ ops }) => ops.join(" + ")),
  );
  expect(events[1]).toMatchObject({ ts: 0.02, dur: 0.01, args: { index: 1 } });
});

test("refuses to profile without timestamp queries", async () => {
  const { plan } = await compileMlp();
  expect(plan.profiling).toBe(false);
  await expect(plan.profile({ input: mlpInput() })).rejects.toThrow("timestamp-query");
});

test("keeps working after a rejected run", async () => {
  const { model } = await loadIdentity();
  const plan = await model.compile({ x: [1, 4] });
  await expect(plan.run({})).rejects.toBeInstanceOf(AvelError);
  expect(await plan.run({ x: new Float32Array(4) })).toEqual({ x: new Float32Array(4) });
});

test("pads bucketed inputs and trims outputs to the requested shape", async () => {
  const { model, buffers } = await loadIdentity();
  const plan = await model.compile({ x: [17, 4] }, { bucket: 16 });
  // The plan is compiled for the bucket: [17, 4] snaps to [32, 4].
  expect(plan.plan.inputs[0]!.shape).toEqual([32, 4]);
  expect(plan.plan.buffers[plan.plan.inputs[0]!.buffer]!.size).toBe(32 * 4 * 4);
  expect(plan.plan.requested).toEqual({ inputs: [[17, 4]], outputs: [[17, 4]] });

  const input = new Float32Array(17 * 4).map((_, i) => i);
  const outputs = await plan.run({ x: input });
  // The input buffer holds the padded [32, 4] tensor...
  const written = new Float32Array(
    (buffers.find((buffer) => buffer.label === "input 0") as FakeBuffer).bytes.buffer,
  );
  expect(written.length).toBe(32 * 4);
  expect([...written.slice(0, 17 * 4)]).toEqual([...input]);
  expect([...written.slice(17 * 4)]).toEqual(new Array(15 * 4).fill(0));
  // ...and the output is read back trimmed to [17, 4].
  expect(outputs.x).toBeInstanceOf(Float32Array);
  expect([...(outputs.x as Float32Array)]).toEqual([...input]);
});

test("shares one plan between the lengths of a bucket", async () => {
  const { model } = await loadIdentity();
  const a = await model.compile({ x: [17, 4] }, { bucket: 16 });
  const b = await model.compile({ x: [20, 4] }, { bucket: 16 });
  const c = await model.compile({ x: [40, 4] }, { bucket: 16 });
  expect(a).toBe(b);
  expect(a).not.toBe(c);
  // Without a bucket, every length compiles its own plan.
  expect(await model.compile({ x: [17, 4] })).not.toBe(await model.compile({ x: [20, 4] }));
});

test("accepts i32 inputs for embedding indices", async () => {
  const value = valueInfo;
  serve(
    {
      formatVersion: 1,
      opset: "core-aten/1",
      f16: false,
      inputs: [value("ids", "i32", [2, 17])],
      outputs: [value("emb", "f32", [2, 17, 64])],
      nodes: [
        {
          op: "aten.embedding.default",
          inputs: ["weight", "ids"],
          outputs: [value("emb", "f32", [2, 17, 64])],
          attributes: { padding_idx: -1, scale_grad_by_freq: false, sparse: false },
        },
      ],
      parameters: [{ name: "weight", tensor: "weight", dtype: "f32", shape: [128, 64] }],
    },
    safetensors({ weight: [128, 64] }),
  );
  const { device, buffers } = fakeDevice();
  const model = await loadModel(await runtime(device), "/models/emb");
  const plan = await model.compile({ ids: [2, 17] });

  expect(plan.plan.inputs[0]!.dtype).toBe("i32");
  expect(plan.plan.buffers[plan.plan.inputs[0]!.buffer]!.dtype).toBe("i32");
  const shader = plan.plan.shaders[plan.plan.dispatches[0]!.shader]!.wgsl;
  expect(shader).toContain("indices: array<u32>");

  const ids = new Int32Array(2 * 17).map((_, i) => i % 128);
  await plan.run({ ids });
  const written = buffers.find((buffer) => buffer.label === "input 0") as FakeBuffer;
  expect([...new Int32Array(written.bytes.buffer)]).toEqual([...ids]);

  const other = new Float32Array(2 * 17);
  await expect(plan.run({ ids: other })).rejects.toThrow("must be a Int32Array");
});

test("validates inputs", async () => {
  const { model, device } = await loadIdentity();
  const plan = await model.compile({ x: [1, 4] });
  const run = (x: unknown) => plan.run({ x } as Parameters<typeof plan.run>[0]);

  await expect(plan.run({})).rejects.toThrow("missing input `x`");
  await expect(run(42)).rejects.toThrow("requires a tensor");
  await expect(run(new Float32Array(3))).rejects.toThrow("has 12 bytes, expected 16");
  await expect(run(new Int32Array(4))).rejects.toThrow("must be a Float32Array");
  const small = device.createBuffer({ size: 12, usage: COPY_SRC });
  await expect(run(small)).rejects.toThrow("has 12 bytes, expected 16");
  const unreadable = device.createBuffer({ size: 16, usage: STORAGE });
  await expect(run(unreadable)).rejects.toThrow("needs COPY_SRC usage");
});

test("submits GPU consumers synchronously and preserves each run's outputs", async () => {
  const { model, device, calls } = await loadIdentity();
  const plan = await model.compile({ x: [2, 4] });
  const target = device.createBuffer({ size: 64, usage: COPY_DST | MAP_READ });
  const first = new Float32Array(8).fill(1);
  const second = new Float32Array(8).fill(2);
  const consume =
    (offset: number) =>
    (encoder: GPUCommandEncoder, outputs: Readonly<Record<string, GPUBuffer>>) => {
      encoder.copyBufferToBuffer(outputs.x!, 0, target, offset, 32);
    };
  plan.runGpu({ x: first }, consume(0));
  plan.runGpu({ x: second }, consume(32));
  expect(calls.filter((call) => call === "submit")).toHaveLength(2);
  expect(new Float32Array((target as unknown as FakeBuffer).bytes.buffer)).toEqual(
    new Float32Array([...first, ...second]),
  );
  const pending = plan.run({ x: first });
  expect(() => plan.runGpu({ x: second })).toThrow("waiting to submit");
  await pending;
  expect(() => plan.runGpu({ x: second })).not.toThrow();
});

test("graph media reaches the compiler and source settings separate cached plans", async () => {
  const graph = {
    formatVersion: 1,
    opset: "core-aten/1",
    parameters: [],
    nodes: [],
    inputs: [{ name: "x", dtype: "f32", shape: [1, 3, 2, 2] }],
    outputs: [{ name: "x", dtype: "f32", shape: [1, 3, 2, 2] }],
    media: { x: { source: { kind: "external" } } },
    metadata: { labels: ["ignored", { by: "the compiler" }] },
  };
  serve(graph);
  const fake = fakeDevice();
  const model = await loadModel(await runtime(fake.device), "/media");
  const plan = await model.compile({ x: [1, 3, 2, 2] });
  expect(plan.plan.shaders[0]!.wgsl).toContain("texture_external");
  expect(fake.buffers.some((buffer) => buffer.label.startsWith("input"))).toBe(false);
  expect(await model.compile({ x: [1, 3, 2, 2] })).toBe(plan);
  const normalized = await model.compile(
    { x: [1, 3, 2, 2] },
    { media: { x: { source: { kind: "external" }, scale: 2 } } },
  );
  expect(normalized).not.toBe(plan);
  await expect(plan.run({ x: mlpInput() })).rejects.toThrow("VideoFrame");
});

test("builder uses compiler shape inference and rejects foreign values", async () => {
  const rt = await runtime(fakeDevice().device);
  const g = rt.graph();
  const x = g.input("x", [2, 3]);
  const y = g.reshape(x, [3, 2]);
  expect(y.shape).toEqual([3, 2]);
  expect(() => g.reshape(x, [7])).toThrow();
  expect(() => g.add(x, rt.graph().input("other", [2, 3]))).toThrow("another graph");
  const model = ownModel(g.build([g.relu(y)]));
  const plan = await model.compile({ x: [2, 3] });
  expect(plan.plan.outputs[0]!.shape).toEqual([3, 2]);
  // An optional tensor left out keeps its slot.
  const image = g.input("image", [1, 2, 4, 4]);
  const weight = g.parameter("weight", [3, 2, 3, 3], new Float32Array(54));
  const conv = g.conv2d(image, weight, undefined, { padding: [1, 1] });
  expect(conv.shape).toEqual([1, 3, 4, 4]);
  expect(ownModel(g.build([conv])).graph.nodes.at(-1)!.inputs).toEqual(["image", "weight", null]);
  expect(() => g.input("x", [2, 3])).toThrow("duplicate");
});

test("custom registration stays outside the hot path and invalidates compiled plans", async () => {
  const fake = fakeDevice();
  const rt = await runtime(fake.device);
  let kernels = 0;
  const definition = {
    inputs: ["x"],
    shape: (inputs: readonly { shape: number[] }[]) => [...inputs[0]!.shape],
    kernel: () => {
      kernels++;
      return {
        wgsl: "@compute @workgroup_size(1) fn main() { output[0] = x[0]; }",
        workgroups: [1, 1, 1] as [number, number, number],
      };
    },
  };
  expect(() => rt.registerOp("aten.relu.default", definition)).toThrow();
  rt.registerOp("test.custom.default", definition);
  const g = rt.graph();
  const x = g.input("x", [4]);
  const y = g.op("test.custom.default", [x]);
  const model = ownModel(g.build([g.relu(y)]));
  const plan = await model.compile({ x: [4] });
  expect(kernels).toBe(1);
  expect(plan.plan.dispatches.find((d) => d.ops.includes(y.name))!.ops).toEqual([y.name]);
  await plan.run({ x: new Float32Array(4) });
  await model.compile({ x: [4] });
  expect(kernels).toBe(1);
  rt.registerOp("test.custom.default", definition);
  expect(await model.compile({ x: [4] })).not.toBe(plan);
  expect(kernels).toBe(2);
});

test("rejects malformed custom op registrations", async () => {
  const rt = await runtime(fakeDevice().device);
  const definition = {
    inputs: ["x"],
    shape: (inputs: readonly { shape: number[] }[]) => [...inputs[0]!.shape],
    kernel: () => ({
      wgsl: "@compute @workgroup_size(1) fn main() { output[0] = x[0]; }",
      workgroups: [1, 1, 1] as [number, number, number],
    }),
  };
  for (const name of ["relu", "test.custom", "aten.custom.default", "avel.custom.default"]) {
    expect(() => rt.registerOp(name, definition)).toThrow("custom op names");
  }
  expect(() => rt.registerOp("test.dup.default", { ...definition, inputs: ["x", "x"] })).toThrow(
    "unique",
  );
  expect(() =>
    rt.registerOp("test.reserved.default", { ...definition, inputs: ["output"] }),
  ).toThrow("unique");
  expect(() =>
    rt.registerOp("test.bad.default", { ...definition, inputs: ["not an identifier"] }),
  ).toThrow("unique");
});

test("rejects custom nodes with the wrong inputs, dtype or inferred shape", async () => {
  const rt = await runtime(fakeDevice().device);
  const kernel = () => ({
    wgsl: "@compute @workgroup_size(1) fn main() { output[0] = x[0]; }",
    workgroups: [1, 1, 1] as [number, number, number],
  });
  rt.registerOp("test.copy.default", {
    inputs: ["x"],
    shape: (inputs) => inputs[0]!.shape,
    kernel,
  });
  rt.registerOp("test.zero.default", { inputs: ["x"], shape: () => [0], kernel });
  const graph = (node: object, xDtype = "f32") => ({
    formatVersion: 1,
    opset: "core-aten/1",
    parameters: [],
    inputs: [valueInfo("x", xDtype, [4])],
    outputs: [valueInfo("y", "f32", [4])],
    nodes: [node],
  });
  const compile = async (node: object, xDtype?: string) => {
    serve(graph(node, xDtype));
    return loadModel(rt, "/custom").then((model) => model.compile({ x: [4] }));
  };
  const node = (op: string, inputs: unknown[]) => ({
    op,
    inputs,
    outputs: [valueInfo("y", "f32", [4])],
  });
  await expect(compile(node("test.copy.default", ["x", null]))).rejects.toThrow(
    "expects 1 f32 inputs",
  );
  await expect(compile(node("test.copy.default", ["x"]), "i32")).rejects.toThrow(
    "expects 1 f32 inputs",
  );
  await expect(compile(node("test.zero.default", ["x"]))).rejects.toThrow(
    "invalid shape for test.zero.default",
  );
});

test("builder rejects bad parameter sizes, empty outputs and invalid shapes", async () => {
  const rt = await runtime(fakeDevice().device);
  const g = rt.graph();
  expect(() => g.parameter("w", [4], new Float32Array(3))).toThrow("wrong parameter size");
  expect(() => g.build([])).toThrow("a graph needs an output");
  expect(() => g.input("zero", [0])).toThrow("invalid shape");
  expect(() => g.input("frac", [1.5])).toThrow("invalid shape");
  expect(() => g.input("", [1])).toThrow("duplicate or empty");
});

test("custom ops specialize requested and bucketed shapes separately", async () => {
  serve({
    formatVersion: 1,
    opset: "core-aten/1",
    parameters: [],
    inputs: [{ name: "x", dtype: "f32", shape: [{ name: "s", min: 1, max: 32 }, 2] }],
    outputs: [{ name: "y", dtype: "f32", shape: [{ name: "s" }, 2] }],
    nodes: [
      {
        op: "test.copy.default",
        inputs: ["x"],
        outputs: [{ name: "y", dtype: "f32", shape: [{ name: "s" }, 2] }],
        attributes: { rows: { $sym: "s" }, config: { nested: [1, { $sym: "2*s" }] } },
      },
    ],
  });
  const rt = await runtime(fakeDevice().device);
  const model = await loadModel(rt, "/custom");
  const seen: unknown[] = [];
  await expect(model.compile({ x: [3, 2] }, { bucket: 8 })).rejects.toThrow("not registered");
  rt.registerOp("test.copy.default", {
    inputs: ["x"],
    shape: (inputs, attributes) => {
      seen.push(attributes);
      return inputs[0]!.shape;
    },
    kernel: () => ({
      wgsl: "@compute @workgroup_size(1) fn main() { output[0] = x[0]; }",
      workgroups: [1, 1, 1],
    }),
  });
  const compiled = await model.compile({ x: [3, 2] }, { bucket: 8 });
  expect(compiled.plan.outputs[0]!.shape).toEqual([8, 2]);
  expect(compiled.plan.requested!.outputs).toEqual([[3, 2]]);
  // Symbolic attributes arrive resolved, for the bucketed and requested shapes.
  expect(seen).toContainEqual({ rows: 8, config: { nested: [1, 16] } });
  expect(seen).toContainEqual({ rows: 3, config: { nested: [1, 6] } });
});

test("autotune keeps heuristics without timestamp queries", async () => {
  const fake = fakeDevice();
  let reads = 0;
  const rt = await runtime(fake.device, {
    autotune: true,
    gpuClass: "mobile",
    tuningCache: {
      async get() {
        reads++;
        return undefined;
      },
      async set() {},
    },
  });
  serveMlp();
  const plan = await (await loadModel(rt, "/mlp")).compile({ input: [3, 4] });
  expect(reads).toBe(0);
  expect(plan.plan.shaders.some((s) => s.wgsl.includes("@workgroup_size(8, 8)"))).toBe(true);
});

test("rejects a model directory whose graph request fails", async () => {
  stubGlobal(
    "fetch",
    mock(async () => new Response("not found", { status: 404 })),
  );
  const rt = await runtime(fakeDevice().device);
  await expect(rt.load("/missing")).rejects.toThrow("HTTP 404");
});

test("requires WebGPU when explicitly selected", async () => {
  await expect(Runtime.create({ backend: "webgpu" })).rejects.toThrow("WebGPU is not available");
});

test("rejects unknown or missing compile-time input shapes", async () => {
  const { model } = await loadIdentity();
  await expect(model.compile({})).rejects.toThrow("missing shape for input `x`");
  await expect(model.compile({ y: [1, 4] })).rejects.toThrow("unknown input `y`");
});

test("rejects a plan whose parameters were not uploaded or outputs beyond the readback limit", async () => {
  const identity = await loadIdentity();
  const { model, plan, device } = await compileMlp();
  await expect(ExecutionPlan.create(device, model.graph, plan.plan, new Map())).rejects.toThrow(
    "tensor `0.weight` is not loaded",
  );

  const small = await identity.model.compile({ x: [2, 4] });
  // Shrink the limit after compiling: the check belongs to plan creation, not the compiler.
  (identity.device.limits as { maxBufferSize: number }).maxBufferSize = 16;
  await expect(
    ExecutionPlan.create(identity.device, identity.model.graph, small.plan, new Map()),
  ).rejects.toThrow("combined outputs exceed the device readback buffer limit");
});

test("a model validates its graph before downloading weights", async () => {
  const graph = await Bun.file(MLP).json();
  const requests = serve({ ...graph, formatVersion: 99 }, safetensors(MLP_TENSORS));
  const rt = await runtime(fakeDevice().device);
  await expect(loadModel(rt, "/models/mlp")).rejects.toThrow("format version 99");
  expect(requests).toHaveLength(0);
});

test("a model compiles from one loaded graph and frees it once on destroy", async () => {
  const free = spyOn(LoadedGraph.prototype, "free");
  const compile = spyOn(LoadedGraph.prototype, "compile");
  try {
    serveMlp();
    const model = await loadModel(await runtime(fakeDevice().device), "/models/mlp");
    await model.compile({ input: [3, 4] });
    await model.compile({ input: [5, 4] });
    expect(compile).toHaveBeenCalledTimes(2);
    // The compiler keeps its own copy, so the JS graph cannot drift from it.
    expect(() => {
      (model.graph as { f16?: boolean }).f16 = true;
    }).toThrow(TypeError);
    expect(() => model.graph.nodes[0]!.inputs.push("x")).toThrow(TypeError);
    await model.destroy();
    await model.destroy();
    expect(free).toHaveBeenCalledTimes(1);
    await expect(model.compile({ input: [3, 4] })).rejects.toThrow("destroyed");
  } finally {
    free.mockRestore();
    compile.mockRestore();
  }
});

test("a failed weight download frees the loaded graph", async () => {
  const free = spyOn(LoadedGraph.prototype, "free");
  try {
    serve(MLP, safetensors({}));
    const rt = await runtime(fakeDevice().device);
    await expect(loadModel(rt, "/models/mlp")).rejects.toThrow();
    expect(free).toHaveBeenCalledTimes(1);
  } finally {
    free.mockRestore();
  }
});
