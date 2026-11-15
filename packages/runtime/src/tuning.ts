import type {
  CompileOptions,
  Graph,
  GpuClass,
  LoadedGraph,
  Plan,
  TileConfig,
} from "../wasm/avel_wasm.js";
import { ExecutionPlan, type RunInput } from "./plan.js";

export interface TuningCache {
  get(key: string): Promise<Record<string, TileConfig> | undefined>;
  set(key: string, tiles: Record<string, TileConfig>): Promise<void>;
}

/** Persistence is best effort: private browsing, quota and unavailable IDB use defaults. */
export class IndexedDbTuningCache implements TuningCache {
  private async access(
    mode: IDBTransactionMode,
    key: string,
    value?: Record<string, TileConfig>,
  ): Promise<Record<string, TileConfig> | undefined> {
    if (typeof indexedDB === "undefined") return undefined;
    return new Promise((resolve) => {
      const opening = indexedDB.open("avel-tile-tuning", 1);
      opening.onupgradeneeded = () => opening.result.createObjectStore("tiles");
      opening.onerror = () => resolve(undefined);
      opening.onblocked = () => resolve(undefined);
      opening.onsuccess = () => {
        const db = opening.result;
        try {
          const tx = db.transaction("tiles", mode);
          const store = tx.objectStore("tiles");
          const request = mode === "readonly" ? store.get(key) : store.put(value, key);
          let result: Record<string, TileConfig> | undefined;
          request.onsuccess = () => {
            if (mode === "readonly") result = request.result;
          };
          tx.oncomplete = () => {
            db.close();
            resolve(result);
          };
          tx.onabort = tx.onerror = () => {
            db.close();
            resolve(undefined);
          };
        } catch {
          db.close();
          resolve(undefined);
        }
      };
    });
  }
  async get(key: string): Promise<Record<string, TileConfig> | undefined> {
    try {
      return await this.access("readonly", key);
    } catch {
      return undefined;
    }
  }
  async set(key: string, value: Record<string, TileConfig>): Promise<void> {
    try {
      await this.access("readwrite", key, value);
    } catch {
      /* optional persistence */
    }
  }
}

/** Adapter descriptions are hints; applications may supply an explicit class. */
export function gpuClass(info?: GPUAdapterInfo): GpuClass {
  const text =
    `${info?.vendor} ${info?.architecture} ${info?.device} ${info?.description}`.toLowerCase();
  if (/adreno|mali|powervr|mobile/.test(text)) return "mobile";
  if (/nvidia|radeon rx|discrete/.test(text)) return "discrete";
  return "integrated";
}

export async function tune(
  device: GPUDevice,
  graph: Readonly<Graph>,
  loaded: LoadedGraph,
  options: CompileOptions,
  parameters: Map<string, GPUBuffer>,
  baseline: Plan,
  cache: TuningCache,
): Promise<Plan> {
  const names = graph.nodes
    .filter((n) =>
      [
        "aten.addmm.default",
        "aten.mm.default",
        "aten.bmm.default",
        "aten.convolution.default",
      ].includes(n.op),
    )
    .map((n) => n.outputs[0]!.name);
  if (
    !names.length ||
    Object.keys(baseline.media).length ||
    !device.features.has("timestamp-query")
  )
    return baseline;
  // Hash the generated baseline too: changes to codegen/heuristics invalidate old entries.
  const info = device.adapterInfo;
  const adapter = info && [info.vendor, info.architecture, info.device, info.description];
  const source = JSON.stringify([
    graph,
    options,
    adapter,
    baseline.shaders,
    device.limits.maxComputeWorkgroupStorageSize,
  ]);
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(source));
  const key = Array.from(new Uint8Array(digest), (n) => n.toString(16).padStart(2, "0")).join("");
  const cached = await cache.get(key).catch(() => undefined);
  if (cached) {
    try {
      return loaded.compile({ ...options, tiles: cached });
    } catch {
      /* stale or damaged cache */
    }
  }
  const inputs: Record<string, RunInput> = Object.create(null);
  for (const input of baseline.inputs) {
    const length = input.shape.reduce((a, b) => a * b, 1);
    inputs[input.name] =
      input.dtype === "i32" || input.dtype === "i64"
        ? new Int32Array(length)
        : input.dtype === "bool"
          ? new Uint32Array(length)
          : new Float32Array(length);
  }
  const best = new Map<string, number>();
  const selected: Record<string, TileConfig> = Object.create(null);
  // A small bounded sweep, including the default and scalar matmul fallback.
  const candidates: (TileConfig | undefined)[] = [
    undefined,
    { rows: 1, columns: 1, split: 1, matmul: 0 },
    { rows: 2, columns: 2, split: 1, matmul: 8 },
    { rows: 4, columns: 4, split: 1, matmul: 16 },
  ];
  const state = baseline.buffers.filter(({ role }) => role.kind === "state");
  for (const candidate of candidates) {
    const tiles = candidate ? Object.fromEntries(names.map((name) => [name, candidate])) : {};
    let execution: ExecutionPlan | undefined;
    const states: GPUBuffer[] = [];
    try {
      const plan = candidate ? loaded.compile({ ...options, tiles }) : baseline;
      // Trial runs must not change the model's live state. Each candidate
      // starts from the same snapshot and owns every state buffer it writes.
      const trialParameters = state.length ? new Map(parameters) : parameters;
      if (state.length) {
        const encoder = device.createCommandEncoder();
        for (const { role } of state) {
          if (role.kind !== "state") continue;
          const source = parameters.get(role.tensor)!;
          const copy = device.createBuffer({
            label: `tuning ${role.tensor}`,
            size: source.size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
          });
          states.push(copy);
          encoder.copyBufferToBuffer(source, 0, copy, 0, source.size);
          trialParameters.set(role.tensor, copy);
        }
        device.queue.submit([encoder.finish()]);
      }
      execution = await ExecutionPlan.create(device, graph, plan, trialParameters);
      execution.runGpu(inputs);
      const samples = [];
      for (let i = 0; i < 3; i++) samples.push(await execution.profile(inputs));
      for (const name of names) {
        const times = samples
          .map((sample) =>
            sample.dispatches
              .filter((d) => d.ops.some((op) => op === name || op.startsWith(`${name}.`)))
              .reduce((sum, d) => sum + d.durationNs, 0),
          )
          .sort((a, b) => a - b);
        const score = times[1]!;
        if (score > 0 && (!best.has(name) || score < best.get(name)! * 0.98)) {
          best.set(name, score);
          if (candidate) selected[name] = candidate;
          else delete selected[name];
        }
      }
    } catch (error) {
      if (!candidate) throw error;
    } finally {
      try {
        if (execution || states.length) await device.queue.onSubmittedWorkDone();
      } finally {
        try {
          await execution?.destroy();
        } finally {
          for (const state of states) state.destroy();
        }
      }
    }
  }
  await cache.set(key, selected).catch(() => {});
  return loaded.compile({ ...options, tiles: selected });
}
