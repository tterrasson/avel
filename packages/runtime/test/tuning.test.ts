import { expect, mock, test } from "bun:test";
import type { TuningCache } from "../src/index.js";
import { gpuClass, IndexedDbTuningCache } from "../src/tuning.js";
import { fakeDevice, loadModel, runtime, serveMlp, setupTests } from "./helpers.js";

setupTests();

test("gpuClass classifies adapter info as mobile, discrete or integrated", () => {
  const info = (vendor: string, device = "") =>
    ({ vendor, architecture: "", device, description: "" }) as GPUAdapterInfo;
  expect(gpuClass(info("arm", "Mali-G78"))).toBe("mobile");
  expect(gpuClass(info("qualcomm", "Adreno 740"))).toBe("mobile");
  expect(gpuClass(info("apple", "Apple GPU"))).toBe("integrated");
  expect(gpuClass(info("nvidia", "GeForce RTX 4090"))).toBe("discrete");
  expect(gpuClass(info("amd", "Radeon RX 7900 XTX"))).toBe("discrete");
  expect(gpuClass(undefined)).toBe("integrated");
});

test("the IndexedDB tuning cache degrades to defaults when IndexedDB is unavailable", async () => {
  // Bun has no indexedDB: reads resolve to undefined and writes are dropped.
  const cache = new IndexedDbTuningCache();
  expect(await cache.get("key")).toBeUndefined();
  await cache.set("key", { addmm: { rows: 4, columns: 4, split: 1, matmul: 16 } });
  expect(await cache.get("key")).toBeUndefined();
});

test("autotuning replaces stale tiles, releases trial buffers and reuses the measured winner", async () => {
  // Make the 16x16 shader faster than the mobile default and other candidates.
  const fake = fakeDevice(["timestamp-query"], (shader) =>
    shader.includes("@workgroup_size(16, 16)") ? 3 : 10,
  );
  const winner = { rows: 4, columns: 4, split: 1, matmul: 16 };
  let saved: Awaited<ReturnType<TuningCache["get"]>>;
  const cache = {
    get: mock<TuningCache["get"]>(async () => saved ?? { addmm: { ...winner, rows: 3 } }),
    set: mock<TuningCache["set"]>(async (_, tiles) => {
      saved = tiles;
    }),
  };
  serveMlp();
  const rt = await runtime(fake.device, { gpuClass: "mobile", autotune: true, tuningCache: cache });
  const model = await loadModel(rt, "/mlp");
  const weights = [...fake.buffers];
  const tuned = await model.compile({ input: [3, 4] });
  expect(fake.calls).toContain("timed pass");
  const key = cache.get.mock.calls[0]![0];
  expect(cache.set).toHaveBeenCalledTimes(1);
  expect(cache.set).toHaveBeenCalledWith(key, { addmm: winner, addmm_1: winner });
  expect(tuned.plan.shaders.every(({ wgsl }) => wgsl.includes("@workgroup_size(16, 16)"))).toBe(
    true,
  );

  // After destroying the final plan, only the model's shared weights remain.
  await tuned.destroy();
  expect(fake.buffers.filter((buffer) => !buffer.destroyed)).toEqual(weights);

  fake.calls.length = 0;
  const cached = await model.compile({ input: [3, 4] });
  expect(cache.get.mock.calls).toEqual([[key], [key]]);
  expect(cache.set).toHaveBeenCalledTimes(1);
  expect(fake.calls).not.toContain("timed pass");
  expect(cached.plan.shaders).toEqual(tuned.plan.shaders);
});
