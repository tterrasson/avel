import { assertClose } from "./numeric.js";
/**
 * Validation suite executed inside headless Chrome, run by `run-validation.ts`.
 *
 * `bun test` runs against a fake `GPUDevice`; this suite exercises the real
 * WebGPU stack. Every case and model exported by `avel_export.validation` is
 * loaded, compiled, run, and compared against the exporter's test vectors,
 * with a tolerance per dtype.
 */
import { ExecutionPlan, Model, Runtime, type WebGpuRuntime } from "../src/index.js";

import { registerValidationOp, validateExtensions } from "./extension-validation.js";

import { validateMedia } from "./media-validation.js";

const WASM_URL = "/packages/runtime/wasm/avel_wasm_bg.wasm";
const ARTIFACTS = "/packages/runtime/browser/artifacts";

/**
 * Comparison tolerance per graph dtype. Integer and boolean dtypes are exact.
 * Plans with f16 activations compare at the `f16` tolerance, relative to the
 * largest expected magnitude, as the exporter validates them.
 */
const TOLERANCE: Record<string, { rtol: number; atol: number }> = {
  f32: { rtol: 1e-4, atol: 1e-5 },
  f16: { rtol: 1e-2, atol: 1e-3 },
  i32: { rtol: 0, atol: 0 },
  i64: { rtol: 0, atol: 0 },
  i8: { rtol: 0, atol: 0 },
  bool: { rtol: 0, atol: 0 },
};

interface Result {
  name: string;
  ok: boolean;
  ms: number;
  error?: string;
}

interface ManifestEntry {
  f16?: false;
  f16Compute?: boolean;
  kind: "case" | "model";
  name: string;
  dir: string;
  ops: string[];
  expectError?: string;
  /** Dynamic models: one run per test-vector length, with the shapes to compile for. */
  runs?: { shapes: Record<string, number[]>; vectors: string }[];
  bucket?: number;
}

const results: Result[] = [];
const logs: string[] = [];

function log(...args: unknown[]): void {
  logs.push(args.map(String).join(" "));
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) {
    throw new Error(message);
  }
}

async function test(name: string, fn: () => void | Promise<void>): Promise<void> {
  const start = performance.now();
  try {
    await fn();
    results.push({ name, ok: true, ms: performance.now() - start });
  } catch (error) {
    results.push({ name, ok: false, ms: performance.now() - start, error: String(error) });
  }
  const result = results[results.length - 1]!;
  console.log(
    `${result.ok ? "ok" : "FAIL"} ${result.name} (${result.ms.toFixed(0)}ms)${result.error ? `\n  ${result.error}` : ""}`,
  );
}

interface Tensor {
  dtype: string;
  shape: number[];
  data: Float32Array | Int32Array | BigInt64Array | Int8Array | Uint8Array;
}

async function readSafetensors(url: string): Promise<Map<string, Tensor>> {
  const response = await fetch(url);
  assert(response.ok, `failed to load ${url}: HTTP ${response.status}`);
  const buffer = await response.arrayBuffer();
  const headerLength = Number(new DataView(buffer).getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(buffer.slice(8, 8 + headerLength))) as Record<
    string,
    { dtype: string; shape: number[]; data_offsets: [number, number] }
  >;
  const dataStart = 8 + headerLength;
  const tensors = new Map<string, Tensor>();
  for (const [name, info] of Object.entries(header)) {
    if (name === "__metadata__") {
      continue;
    }
    const [start, end] = info.data_offsets;
    tensors.set(name, {
      dtype: info.dtype,
      shape: info.shape,
      data: decode(info.dtype, buffer.slice(dataStart + start, dataStart + end)),
    });
  }
  return tensors;
}

function decode(dtype: string, bytes: ArrayBuffer): Tensor["data"] {
  switch (dtype) {
    case "F32":
      return new Float32Array(bytes);
    case "I32":
      return new Int32Array(bytes);
    case "I64":
      return new BigInt64Array(bytes);
    case "I8":
      return new Int8Array(bytes);
    case "BOOL":
      return new Uint8Array(bytes);
    case "F16": {
      const view = new DataView(bytes);
      const out = new Float32Array(bytes.byteLength / 2);
      for (let i = 0; i < out.length; i++) {
        out[i] = f16ToF32(view.getUint16(i * 2, true));
      }
      return out;
    }
    default:
      throw new Error(`unsupported safetensors dtype ${dtype}`);
  }
}

function f16ToF32(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits & 0x7c00) >> 10;
  const fraction = bits & 0x03ff;
  if (exponent === 0) {
    return sign * 2 ** -14 * (fraction / 1024);
  }
  if (exponent === 0x1f) {
    return fraction === 0 ? sign * Infinity : NaN;
  }
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

/**
 * Every element must satisfy `|a - b| <= atol + rtol * |b|`, or with `scaled`,
 * `|a - b| <= atol + rtol * max |b|`.
 */
function compare(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  dtype: string,
  label: string,
  scaled = false,
): void {
  const tolerance = TOLERANCE[dtype];
  assert(tolerance !== undefined, `${label}: no tolerance for dtype ${dtype}`);
  assertClose(actual, expected, tolerance, label, scaled);
}

/** Concrete input shapes of a static entry: every dim must be fixed. */
function inputShapes(
  graph: {
    inputs: { name: string; shape: (number | { name: string; min?: number; max?: number })[] }[];
  },
  entry: string,
): Record<string, number[]> {
  const shapes: Record<string, number[]> = {};
  for (const input of graph.inputs) {
    shapes[input.name] = input.shape.map((dim) => {
      assert(typeof dim === "number", `${entry}: input ${input.name} has a symbolic dim`);
      return dim;
    });
  }
  return shapes;
}

/** The test vector of an input as the array the plan expects for its dtype. */
function inputArray(
  tensor: Tensor,
  label: string,
): Float32Array | Int32Array | Uint32Array | BigInt64Array {
  switch (tensor.dtype) {
    case "I64":
      // Narrowed to 32 bits by the runtime on upload.
      return tensor.data as BigInt64Array;
    case "F32":
      return tensor.data as Float32Array;
    case "I32":
      return tensor.data as Int32Array;
    case "BOOL":
      // Safetensors stores booleans one byte each; the plan reads them as u32.
      return Uint32Array.from(tensor.data as Uint8Array);
    default:
      throw new Error(`${label}: unsupported input dtype ${tensor.dtype}`);
  }
}

/**
 * Loads, compiles and runs an exported entry, comparing outputs to its test
 * vectors. With `f16`, activations are stored in f16. With `bucket`, the plan
 * is compiled with shape buckets and must still answer at the requested shape.
 * Returns the plan, so callers can check sharing between runs.
 */
async function runExported(
  rt: WebGpuRuntime,
  baseUrl: string,
  entry: ManifestEntry,
  f16 = false,
  shapes: Record<string, number[]> | undefined = undefined,
  vectorsFile = "model.testvec.safetensors",
  bucket?: number,
  model?: Model,
  f16Compute = false,
): Promise<ExecutionPlan | undefined> {
  model ??= await rt.load(`${baseUrl}/${entry.dir}/`);
  const vectors = await readSafetensors(`${baseUrl}/${entry.dir}/${vectorsFile}`);
  const inputShape = shapes ?? inputShapes(model.graph, entry.name);
  const inputs: Record<string, Float32Array | Int32Array | Uint32Array | BigInt64Array> = {};
  for (const input of model.graph.inputs) {
    const tensor = vectors.get(`input.${input.name}`);
    assert(tensor !== undefined, `${entry.name}: no test vector for input ${input.name}`);
    inputs[input.name] = inputArray(tensor, `${entry.name} input ${input.name}`);
  }

  const plan = await model.compile(inputShape, {
    f16,
    f16Compute,
    bucket,
    autotune: model.graph.parameters.some((p) => p.update !== undefined),
  });
  if (entry.name === "cached-prefill-decode") {
    const writes = plan.plan.dispatches.filter(
      ({ bindings }) => plan.plan.buffers[bindings.at(-1)!.buffer]!.role.kind === "state",
    );
    assert(writes.length === 2, "expected a single patch dispatch per state tensor");
    const sequence = Object.values(inputShape)[0]![2]!;
    assert(
      writes.every(({ workgroups }) => workgroups[0] === Math.ceil((2 * 3 * sequence * 4) / 64)),
      "cache writes must dispatch over the incoming block, independently of cache capacity",
    );
  }
  if (entry.name === "rope-attention" || entry.name === "rope-decode") {
    const { dispatches, shaders } = plan.plan;
    assert(dispatches.length === (entry.name === "rope-decode" ? 1 : 2), "expected rotary fusion");
    const attention = shaders[dispatches.at(-1)!.shader]!.wgsl;
    assert(attention.includes("q_row[d]"), "the rotated query must be cached per row");
  }
  if (entry.name.startsWith("int8-")) {
    assert(
      plan.plan.dispatches.some(({ shader }) =>
        plan.plan.shaders[shader]!.wgsl.includes("extractBits"),
      ),
      "quantized weights must be decoded in the consuming kernel",
    );
    assert(!plan.plan.prepare.length, "quantized weights must stay packed in the plan");
  }
  const outputs = await plan.run(inputs);
  for (const output of model.graph.outputs) {
    const expected = vectors.get(`output.${output.name}`);
    assert(expected !== undefined, `${entry.name}: no test vector for output ${output.name}`);
    const actual = outputs[output.name];
    assert(
      actual !== undefined && ArrayBuffer.isView(actual),
      `${entry.name}: output ${output.name} was not read back`,
    );
    // Integers and bools compare exactly, whatever their array type.
    const numbers = (data: ArrayLike<number | bigint>) => Array.from(data, Number);
    const dtype = f16 && output.dtype === "f32" ? "f16" : output.dtype;
    compare(
      numbers(actual),
      numbers(expected.data),
      dtype,
      `${entry.name} output ${output.name}`,
      dtype === "f16",
    );
  }
  return plan;
}

/** Runs every length of a dynamic entry, per shape and through buckets. */
async function runDynamic(
  rt: WebGpuRuntime,
  baseUrl: string,
  entry: ManifestEntry,
  f16: boolean,
): Promise<void> {
  const model = await rt.load(`${baseUrl}/${entry.dir}/`);
  const bucket = entry.bucket;
  if (bucket === undefined) {
    // Successive runs of a stateful model: each starts from the state the
    // previous one left, then a reset starts over.
    for (const run of entry.runs!) {
      await runExported(rt, baseUrl, entry, f16, run.shapes, run.vectors, undefined, model);
    }
    model.resetState();
    const first = entry.runs![0]!;
    await runExported(rt, baseUrl, entry, f16, first.shapes, first.vectors, undefined, model);
    return;
  }
  // One model, so plans are shared between the lengths of a bucket.
  const plans = new Map<number, ExecutionPlan>();
  for (const [index, run] of entry.runs!.entries()) {
    await runExported(rt, baseUrl, entry, f16, run.shapes, run.vectors, undefined, model);
    const bucketed = await runExported(
      rt,
      baseUrl,
      entry,
      f16,
      run.shapes,
      run.vectors,
      bucket,
      model,
    );
    assert(bucketed !== undefined, `${entry.name}: no plan`);
    plans.set(index, bucketed);
  }
  // Lengths in the same bucket share one plan: 17 and 20 both snap to 32.
  assert(
    plans.get(0) === plans.get(1),
    `${entry.name}: lengths 17 and 20 should share the bucket plan`,
  );
  assert(plans.get(0) !== plans.get(2), `${entry.name}: length 40 is in another bucket`);
}

/** Entries with a non-f32 dtype must be rejected by the compiler. */
async function expectCompileError(
  rt: WebGpuRuntime,
  baseUrl: string,
  entry: ManifestEntry,
): Promise<void> {
  const model = await rt.load(`${baseUrl}/${entry.dir}/`);
  let error: unknown;
  try {
    await model.compile(inputShapes(model.graph, entry.name));
  } catch (caught) {
    error = caught;
  }
  assert(error !== undefined, `${entry.name}: expected compilation to fail`);
  assert(
    String(error).includes(entry.expectError!),
    `${entry.name}: error "${String(error)}" does not contain "${entry.expectError}"`,
  );
}

async function main(): Promise<void> {
  window.addEventListener("error", (event) => log("window error:", event.message));
  window.addEventListener("unhandledrejection", (event) =>
    log("unhandled rejection:", String(event.reason)),
  );

  const adapter = await navigator.gpu?.requestAdapter();
  log("adapter:", JSON.stringify(adapter?.info ?? null));

  const rt = await Runtime.create({
    backend: "webgpu",
    wasm: new URL(WASM_URL, window.location.href),
    preferF16: true,
    profiling: true,
  });
  registerValidationOp(rt);
  log("shader-f16:", rt.features.shaderF16);
  log("subgroups:", rt.features.subgroups, "minimum lanes:", rt.features.subgroupMinSize);
  await test("acquires a WebGPU device", () => {
    assert(rt.device.limits.maxStorageBufferBindingSize > 0, "no device limits");
  });

  await test("forwards an input on a real GPU (identity fixture)", async () => {
    const model = await rt.load("/tests/fixtures/identity/");
    const plan = await model.compile({ x: [2, 4] });
    const input = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const outputs = await plan.run({ x: input });
    const actual = outputs.x;
    assert(actual instanceof Float32Array, "identity output was not read back");
    compare(actual, input, "f32", "identity output x");
  });

  const skipExternalVideo = new URLSearchParams(location.search).get("skipExternalVideo") === "1";
  await validateMedia(rt, test, async (name, fn) => {
    if (skipExternalVideo) {
      log(`SKIP ${name}: external video validation disabled for this run`);
      return;
    }
    await test(name, fn);
  });
  await validateExtensions(rt, test);

  const response = await fetch(`${ARTIFACTS}/manifest.json`);
  assert(
    response.ok,
    `no manifest at ${ARTIFACTS}/manifest.json; run the validation generator first`,
  );
  const manifest = (await response.json()) as { entries: ManifestEntry[] };
  for (const entry of manifest.entries) {
    const label = `${entry.kind} ${entry.name}`;
    if (entry.expectError) {
      await test(label, () => expectCompileError(rt, ARTIFACTS, entry));
    } else if (entry.runs) {
      await test(label, () => runDynamic(rt, ARTIFACTS, entry, false));
      // Only models are validated in f16 by the exporter.
      if (rt.features.shaderF16 && entry.kind === "model" && entry.f16 !== false) {
        await test(`${label} (f16)`, () => runDynamic(rt, ARTIFACTS, entry, true));
      }
    } else {
      await test(label, async () => {
        await runExported(rt, ARTIFACTS, entry);
      });
      if (entry.f16Compute && rt.features.shaderF16) {
        await test(`${label} (f16 compute)`, async () => {
          await runExported(
            rt,
            ARTIFACTS,
            entry,
            true,
            undefined,
            undefined,
            undefined,
            undefined,
            true,
          );
        });
      }
      if (entry.kind === "model" && rt.features.shaderF16 && entry.f16 !== false) {
        await test(`${label} (f16)`, async () => {
          await runExported(rt, ARTIFACTS, entry, true);
        });
      }
    }
  }
}

async function report(payload: object): Promise<void> {
  await fetch("/report", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
}

await main().then(
  () => report({ results, logs }),
  (fatal) => {
    log("fatal:", String(fatal));
    return report({ fatal: String(fatal), results, logs });
  },
);
