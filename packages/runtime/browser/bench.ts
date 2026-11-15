import { relativeError } from "./numeric.js";
/**
 * Benchmark executed inside headless Chrome, run by `run-bench.ts`.
 *
 * Every model exported by `avel_export.bench` runs on avel (f32, and f16 when
 * the device and the model allow it) and on ONNX Runtime Web's WebGPU backend,
 * on the same random input. A run is end to end: upload the input, execute,
 * read the output back. Outputs of both runtimes are compared, so a speedup
 * never comes from computing something else.
 */
import { Runtime } from "../src/index.js";
import type { Profile, WebGpuRuntime } from "../src/index.js";
import type { BenchmarkDevice } from "./bench-history.js";

const WASM_URL = "/packages/runtime/wasm/avel_wasm_bg.wasm";
const ARTIFACTS = "/packages/runtime/browser/artifacts/bench";
const ORT_DIST = "/packages/runtime/node_modules/onnxruntime-web/dist/";

const params = new URLSearchParams(location.search);
const WARMUP = Number(params.get("warmup") ?? 10);
const RUNS = Number(params.get("runs") ?? 50);
const F16_COMPUTE = params.get("f16Compute") === "1";

export interface Timing {
  runtime: string;
  /** Milliseconds per run, sorted. */
  ms: number[];
  /** Largest difference to ONNX Runtime's output, relative to its largest magnitude. */
  drift?: number;
}

export interface ModelResult {
  name: string;
  timings: Timing[];
  /** Per-dispatch GPU timings of the avel f32 plan, in Chrome trace format. */
  trace?: object;
}

type Ort = typeof import("onnxruntime-web");

const logs: string[] = [];

function log(...args: unknown[]): void {
  const line = args.map(String).join(" ");
  logs.push(line);
  console.log(line);
}

/** A deterministic input in [-1, 1). */
function randomInput(length: number): Float32Array {
  const data = new Float32Array(length);
  let state = 0x9e3779b9;
  for (let i = 0; i < length; i++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    data[i] = state / 2 ** 31 - 1;
  }
  return data;
}

async function time(runtime: string, run: () => Promise<unknown>): Promise<Timing> {
  for (let i = 0; i < WARMUP; i++) {
    await run();
  }
  const ms: number[] = [];
  for (let i = 0; i < RUNS; i++) {
    const start = performance.now();
    await run();
    ms.push(performance.now() - start);
  }
  return { runtime, ms: ms.sort((a, b) => a - b) };
}

async function benchmark(
  rt: WebGpuRuntime,
  ort: Ort,
  name: string,
  dir: string,
): Promise<ModelResult> {
  const url = `${ARTIFACTS}/${dir}`;
  const model = await rt.load(`${url}/`);
  let session: import("onnxruntime-web").InferenceSession | undefined;
  try {
    const [input] = model.graph.inputs;
    const [output] = model.graph.outputs;
    const shape = input!.shape.map(Number);
    const data = randomInput(shape.reduce((a, b) => a * b, 1));
    const inputs = { [input!.name]: data };

    session = await ort.InferenceSession.create(`${url}/model.onnx`, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "all",
    });
    const feeds = { [session.inputNames[0]!]: new ort.Tensor("float32", data, shape) };
    const reference = (await session.run(feeds))[session.outputNames[0]!]!.data;
    if (!(reference instanceof Float32Array)) throw new Error("expected f32 ONNX output");
    const timings = [await time("onnxruntime-web", () => session!.run(feeds))];

    let trace: object | undefined;
    const modes = [
      { runtime: "avel", f16: false, f16Compute: false },
      { runtime: "avel f16", f16: true, f16Compute: false },
      ...(F16_COMPUTE ? [{ runtime: "avel f16 compute", f16: true, f16Compute: true }] : []),
    ];
    for (const { runtime, f16, f16Compute } of modes) {
      if (f16 && !(rt.features.shaderF16 && model.graph.f16)) {
        continue;
      }
      const plan = await model.compile({ [input!.name]: shape }, { f16, f16Compute });
      const actual = (await plan.run(inputs))[output!.name] as Float32Array;
      const timing = await time(runtime, () => plan.run(inputs));
      timing.drift = relativeError(actual, reference);
      timings.push(timing);
      if (!f16 && plan.profiling) {
        const profile: Profile = await plan.profile(inputs);
        trace = profile.trace(name);
        log(
          `${name}: ${plan.plan.dispatches.length} dispatches, ${(profile.totalNs / 1e6).toFixed(2)} ms on the GPU`,
        );
      }
    }
    return { name, timings, trace };
  } finally {
    try {
      await session?.release();
    } finally {
      await model.destroy();
    }
  }
}

async function main(): Promise<{ results: ModelResult[]; device: BenchmarkDevice }> {
  if (!Number.isSafeInteger(WARMUP) || WARMUP < 0 || !Number.isSafeInteger(RUNS) || RUNS < 1) {
    throw new Error("benchmark warmup must be nonnegative and runs must be positive integers");
  }
  log(`warmup ${WARMUP}, runs ${RUNS}`);

  const ort = (await import(/* @vite-ignore */ `${ORT_DIST}ort.webgpu.bundle.min.mjs`)) as Ort;
  ort.env.wasm.wasmPaths = ORT_DIST;
  ort.env.wasm.numThreads = 1;
  const rt = await Runtime.create({
    backend: "webgpu",
    wasm: new URL(WASM_URL, location.href),
    preferF16: true,
    profiling: true,
  });

  const {
    vendor,
    architecture,
    device: id,
    description,
    isFallbackAdapter,
  } = rt.device.adapterInfo;
  const device = {
    vendor,
    architecture,
    device: id,
    description,
    isFallbackAdapter,
    browser: navigator.userAgent,
  };
  log("adapter:", JSON.stringify(device));

  const response = await fetch(`${ARTIFACTS}/manifest.json`);
  if (!response.ok) {
    throw new Error(
      `no manifest at ${ARTIFACTS}/manifest.json; run \`bun run generate:bench\` first`,
    );
  }
  const manifest = (await response.json()) as { entries: { name: string; dir: string }[] };
  const results = [];
  for (const { name, dir } of manifest.entries) {
    log(`benchmarking ${name}`);
    results.push(await benchmark(rt, ort, name, dir));
  }
  return { results, device };
}

async function report(payload: object): Promise<void> {
  await fetch("/report", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
}

await main().then(
  (result) => report({ ...result, logs }),
  (fatal) => report({ fatal: String(fatal), logs }),
);
