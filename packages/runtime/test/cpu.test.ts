import { expect, test } from "bun:test";
import { Runtime } from "../src/index.js";
import { fakeDevice, serveMlp, setupTests, stubGlobal } from "./helpers.js";

setupTests();

test("auto selects CPU without an adapter and executes downloaded weights with shape buckets", async () => {
  stubGlobal("navigator", { gpu: { requestAdapter: async () => null } });
  serveMlp();
  const runtime = await Runtime.create();
  expect(runtime.backend).toBe("cpu");
  const model = await runtime.load("/models/mlp");
  try {
    const plan = await model.compile({ input: [2, 4] }, { bucket: 4 });
    expect(plan.backend).toBe("cpu");
    expect(await model.compile({ input: [1, 4] }, { bucket: 4 })).toBe(plan);
    const first = await plan.run({ input: new Float32Array(8) });
    expect(first.relu).toEqual(new Float32Array([12452, 14725, 12452, 14725]));
    const second = await plan.run({ input: new Float32Array(4).fill(1) });
    expect((second.relu as Float32Array).length).toBe(2);
    expect(second.relu).not.toEqual(new Float32Array([12452, 14725]));
    expect(first.relu).toEqual(new Float32Array([12452, 14725, 12452, 14725]));
    await plan.destroy();
    expect(await model.compile({ input: [1, 4] }, { bucket: 4 })).not.toBe(plan);
    await expect(plan.run({ input: new Float32Array(4) })).rejects.toThrow("destroyed");
  } finally {
    await model.destroy();
  }
});

test("auto falls back on device acquisition failure and respects an explicit GPU choice", async () => {
  stubGlobal("navigator", {
    gpu: {
      requestAdapter: async () => ({
        features: new Set(),
        limits: {},
        requestDevice: async () => {
          throw new Error("device unavailable");
        },
      }),
    },
  });
  expect((await Runtime.create()).backend).toBe("cpu");
  await expect(Runtime.create({ backend: "webgpu" })).rejects.toThrow("device unavailable");
  const { device } = fakeDevice();
  const gpu = await Runtime.create({ device });
  expect(gpu.backend).toBe("webgpu");
  expect(gpu.device).toBe(device);
  stubGlobal("navigator", {
    gpu: {
      requestAdapter: async () => ({
        features: new Set(),
        limits: {},
        requestDevice: async () => device,
      }),
    },
  });
  expect((await Runtime.create()).backend).toBe("webgpu");
});

test("CPU runs an explicit grouped convolution and pooling graph", async () => {
  const runtime = await Runtime.create({ backend: "cpu" });
  const graph = runtime.graph();
  const x = graph.input("x", [1, 2, 2, 2]);
  const weight = graph.parameter("weight", [2, 1, 1, 1], new Float32Array([2, -1]));
  const bias = graph.parameter("bias", [2], new Float32Array([1, 6]));
  const conv = graph.conv2d(x, weight, bias, { groups: 2 });
  const pooled = graph.op("aten._adaptive_avg_pool2d.default", [graph.relu(conv)], {
    output_size: [1, 1],
  });
  const model = graph.build([pooled]);
  try {
    const plan = await model.compile({ x: [1, 2, 2, 2] });
    const input = { x: new Float32Array([1, 2, 3, 4, 3, 4, 6, 8]) };
    expect((await plan.run(input))[pooled.name]).toEqual(new Float32Array([6, 1.25]));
    await expect(plan.run(input, { keepOnGpu: true })).rejects.toThrow("GPU outputs");
    await expect(model.compile({ x: [1, 2, 2, 2] }, { f16: true })).rejects.toThrow("f16");
  } finally {
    await model.destroy();
  }
});

test("CPU refreshes tensor views after another plan grows Wasm memory and narrows i64 inputs", async () => {
  const runtime = await Runtime.create({ backend: "cpu" });
  const graph = runtime.graph();
  const x = graph.input("x", [3], "i64");
  const rem = graph.op("aten.fmod.Scalar", [x], { other: 3 });
  const model = graph.build([rem]);
  const large = runtime.graph();
  const largeInput = large.input("y", [2_000_000]);
  const largeModel = large.build([large.relu(largeInput)]);
  try {
    const plan = await model.compile({ x: [3] });
    await largeModel.compile({ y: [2_000_000] });
    expect((await plan.run({ x: new BigInt64Array([-5n, 0n, 5n]) }))[rem.name]).toEqual(
      new BigInt64Array([-2n, 0n, 2n]),
    );
    await expect(plan.run({ x: new BigInt64Array([0n, 2147483648n, 0n]) })).rejects.toThrow(
      "outside the 32-bit range",
    );
  } finally {
    await largeModel.destroy();
    await model.destroy();
  }
});
