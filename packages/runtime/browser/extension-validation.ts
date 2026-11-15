import { assertClose } from "./numeric.js";
import {
  AvelError,
  ExecutionPlan,
  IndexedDbTuningCache,
  Runtime,
  type CustomOp,
  type TuningCache,
  type WebGpuRuntime,
} from "../src/index.js";
import type { RunOutput } from "../src/index.js";
import { loadGraph } from "../src/compiler.js";

const affine: CustomOp = {
  inputs: ["x"],
  shape: (inputs) => [...inputs[0]!.shape],
  kernel: ({ shape, attributes }) => {
    const count = shape.reduce((a, b) => a * b, 1);
    return {
      workgroups: [Math.ceil(count / 64), 1, 1],
      wgsl: `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < ${count}u) { output[id.x] = x[id.x] * ${Number(attributes.scale).toFixed(8)} + 0.25; }
}`,
    };
  },
};
export function registerValidationOp(rt: WebGpuRuntime): void {
  rt.registerOp("avel_validation.affine.default", affine);
}

function check(actual: RunOutput | undefined, expected: number[], tolerance = 1e-4): void {
  if (!(actual instanceof Float32Array) || actual.length !== expected.length)
    throw new Error("output shape differs");
  assertClose(actual, expected, { rtol: tolerance, atol: tolerance }, "extension output");
}

export async function validateExtensions(
  rt: WebGpuRuntime,
  test: (name: string, fn: () => Promise<void>) => Promise<void>,
): Promise<void> {
  await test("extensions/rejects-invalid-bind-group-at-compile-time", async () => {
    const op = "avel_validation.invalid_bindings.default";
    rt.registerOp(op, {
      inputs: ["x"],
      shape: (inputs) => [...inputs[0]!.shape],
      kernel: () => ({
        workgroups: [1, 1, 1],
        // The shader is valid but never reads x, so auto layout omits binding 0.
        wgsl: "@compute @workgroup_size(1) fn main() { output[0] = 1.0; }",
      }),
    });
    const graph = rt.graph();
    const x = graph.input("x", [1]);
    const model = graph.build([graph.op(op, [x])]);
    rt.device.pushErrorScope("validation");
    try {
      await model.compile({ x: [1] });
      throw new Error("compile accepted an invalid bind group");
    } catch (error) {
      if (!(error instanceof AvelError) || !error.message.includes("bind group")) throw error;
    } finally {
      await rt.device.popErrorScope();
      await model.destroy();
    }
  });

  await test("extensions/builder-custom-and-parameters", async () => {
    const g = rt.graph();
    const x = g.input("x", [2, 3]);
    const w = g.parameter("w", [3, 2], new Float32Array([1, 0, 0, 1, 1, 1]));
    const b = g.parameter("b", [2], new Float32Array([0.5, -0.5]));
    const mm = g.matmul(x, w, b);
    const custom = g.op("avel_validation.affine.default", [mm], { scale: 2 });
    const y = g.relu(custom);
    const plan = await g.build([y]).compile({ x: [2, 3] });
    check(
      (await plan.run({ x: new Float32Array([1, 2, 3, -1, -2, -3]) }))[y.name],
      [9.25, 9.25, 0, 0],
    );
    await plan.destroy();
  });
  await test("extensions/tile-variants", async () => {
    const g = rt.graph();
    const x = g.input("x", [3, 5]);
    const w = g.input("w", [5, 7]);
    const b = g.input("b", [7]);
    const mm = g.matmul(x, w, b);
    const image = g.input("image", [1, 32, 5, 7]);
    const filter = g.input("filter", [8, 32, 3, 3]);
    const conv = g.conv2d(image, filter, undefined, { padding: [1, 1] });
    const model = g.build([mm, conv]);
    const data = (n: number) => Float32Array.from({ length: n }, (_, i) => ((i % 13) - 6) / 17);
    const inputs = {
      x: data(15),
      w: data(35),
      b: data(7),
      image: data(32 * 5 * 7),
      filter: data(8 * 32 * 9),
    };
    const expectedMm: number[] = [];
    for (let row = 0; row < 3; row++)
      for (let col = 0; col < 7; col++) {
        let sum = inputs.b[col]!;
        for (let k = 0; k < 5; k++) sum += inputs.x[row * 5 + k]! * inputs.w[k * 7 + col]!;
        expectedMm.push(sum);
      }
    const expectedConv: number[] = [];
    for (let co = 0; co < 8; co++)
      for (let y = 0; y < 5; y++)
        for (let x = 0; x < 7; x++) {
          let sum = 0;
          for (let ci = 0; ci < 32; ci++)
            for (let ky = 0; ky < 3; ky++)
              for (let kx = 0; kx < 3; kx++) {
                const iy = y + ky - 1;
                const ix = x + kx - 1;
                if (iy >= 0 && iy < 5 && ix >= 0 && ix < 7)
                  sum +=
                    inputs.image[(ci * 5 + iy) * 7 + ix]! *
                    inputs.filter[((co * 32 + ci) * 3 + ky) * 3 + kx]!;
              }
          expectedConv.push(sum);
        }
    // The model graph is frozen: compile a copy that allows f16.
    const graph = { ...model.graph, f16: true };
    const loaded = loadGraph(graph);
    try {
      for (const half of rt.features.shaderF16 ? [false, true] : [false]) {
        for (const config of [
          { rows: 1, columns: 1, split: 1, matmul: 0 },
          { rows: 2, columns: 2, split: 2, matmul: 8 },
          { rows: 4, columns: 4, split: 4, matmul: 16 },
        ]) {
          const plan = loaded.compile({
            inputShapes: Object.fromEntries(
              model.graph.inputs.map((v) => [v.name, v.shape as number[]]),
            ),
            limits: rt.limits,
            features: rt.features,
            f16: half,
            gpuClass: rt.gpuClass,
            tiles: { [mm.name]: config, [conv.name]: config },
            media: {},
            customKernels: {},
          });
          const execution = await ExecutionPlan.create(rt.device, graph, plan, new Map());
          const result = await execution.run(inputs);
          check(result[mm.name], expectedMm);
          check(result[conv.name], expectedConv, half ? 1e-2 : 1e-4);
          await execution.destroy();
        }
      }
    } finally {
      loaded.free();
    }
  });
  await test("extensions/autotune-and-indexeddb-cache", async () => {
    const storage = new IndexedDbTuningCache();
    let reads = 0;
    let writes = 0;
    let key = "";
    const cache: TuningCache = {
      async get(k) {
        key = k;
        reads++;
        return storage.get(k);
      },
      async set(k, value) {
        writes++;
        await storage.set(k, value);
      },
    };
    const runtime = await Runtime.create({
      backend: "webgpu",
      device: rt.device,
      tuningCache: cache,
      autotune: true,
      wasm: undefined,
    });
    const make = () => {
      const g = runtime.graph();
      const x = g.input("x", [3, 5]);
      const w = g.parameter("w", [5, 7], new Float32Array(35).fill(0.5));
      const b = g.parameter("b", [7], new Float32Array(7).fill(0.25));
      const y = g.matmul(x, w, b);
      return { model: g.build([y]), y };
    };
    const { model, y } = make();
    const inputs = { x: new Float32Array(15).fill(1) };
    const plan = await model.compile({ x: [3, 5] });
    check((await plan.run(inputs))[y.name], new Array(21).fill(2.75));
    if (rt.device.features.has("timestamp-query")) {
      if (reads !== 1) throw new Error("tuning cache was not read");
      const persisted = await storage.get(key);
      if (!persisted) throw new Error("tuning result was not persisted");
      const before = writes;
      const second = await make().model.compile({ x: [3, 5] });
      check((await second.run(inputs))[y.name], new Array(21).fill(2.75));
      if (writes !== before) throw new Error("cache hit measured again");
      await second.destroy();
    }
    await plan.destroy();
    const marker = {};
    await storage.set("extension-smoke", marker);
    if (!(await storage.get("extension-smoke"))) throw new Error("IndexedDB round trip failed");
  });
}
