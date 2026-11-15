/**
 * Runs the benchmark (`bench.ts`) in headless Chrome against ONNX Runtime Web.
 *
 * Writes `artifacts/bench/results.json` and one Chrome trace per model
 * (open them in `chrome://tracing` or https://ui.perfetto.dev), prints a
 * table, and appends it to `$GITHUB_STEP_SUMMARY` when set. Fails when a
 * runtime disagrees with ONNX Runtime beyond the tolerance of its precision.
 *
 * Environment, on top of `harness.ts`:
 * - `AVEL_BENCH_WARMUP`, `AVEL_BENCH_RUNS`: runs per model (default 10, 50).
 */
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { ModelResult } from "./bench.js";
import { browserDir, runPage, timeout } from "./harness.js";
import {
  cleanRevision,
  percentile,
  recordHistory,
  type BenchmarkDevice,
  type BenchmarkRow,
} from "./bench-history.js";

/** Largest accepted drift from ONNX Runtime, relative to its largest output. */
const MAX_DRIFT: Record<string, number> = {
  avel: 1e-4,
  "avel f16": 1e-2,
  "avel f16 compute": 1e-2,
};

const warmup = process.env.AVEL_BENCH_WARMUP ?? "10";
const runs = process.env.AVEL_BENCH_RUNS ?? "50";
const f16Compute = process.env.AVEL_BENCH_F16_COMPUTE === "1";
const revision = process.argv.includes("--history") ? cleanRevision() : undefined;
if (revision && process.env.AVEL_WEBGPU_ADAPTER === "swiftshader") {
  throw new Error("benchmark history requires hardware WebGPU");
}
const query = new URLSearchParams({ warmup, runs, f16Compute: f16Compute ? "1" : "0" });
const text = await runPage("bench.ts", timeout(1_200_000), query.toString());
const report = JSON.parse(text) as {
  fatal?: string;
  device?: BenchmarkDevice;
  results?: ModelResult[];
  logs: string[];
};
for (const line of report.logs) {
  console.log(`  | ${line}`);
}
if (report.fatal || !report.results?.length) {
  console.error(`fatal: ${report.fatal}`);
  process.exit(1);
}

const out = join(browserDir, "artifacts", "bench");
mkdirSync(out, { recursive: true });
const rows = [
  "| Model | Runtime | Median (ms) | p10 | p90 | vs ONNX Runtime | Drift |",
  "| --- | --- | --: | --: | --: | --: | --: |",
];
const summary: BenchmarkRow[] = [];
let failed = false;
for (const { name, timings, trace } of report.results) {
  if (trace) {
    writeFileSync(join(out, `${name}.trace.json`), JSON.stringify(trace));
  }
  const baseline = percentile(timings[0]!.ms, 0.5);
  for (const { runtime, ms, drift } of timings) {
    const median = percentile(ms, 0.5);
    const p10 = percentile(ms, 0.1);
    const p90 = percentile(ms, 0.9);
    const tooFar = drift !== undefined && !(drift <= (MAX_DRIFT[runtime] ?? 0));
    failed ||= tooFar;
    rows.push(
      `| ${name} | ${runtime} | ${median.toFixed(2)} | ${p10.toFixed(2)} | ${p90.toFixed(2)} | ` +
        `${(baseline / median).toFixed(2)}× | ${drift === undefined ? "" : `${drift.toExponential(1)}${tooFar ? " ✗" : ""}`} |`,
    );
    summary.push({
      model: name,
      runtime,
      medianMs: median,
      p10Ms: p10,
      p90Ms: p90,
      drift,
    });
  }
}
const table = rows.join("\n");
console.log(table);
writeFileSync(join(out, "results.json"), `${JSON.stringify(summary, null, 2)}\n`);
if (process.env.GITHUB_STEP_SUMMARY) {
  appendFileSync(process.env.GITHUB_STEP_SUMMARY, `## Benchmark\n\n${table}\n`);
}
if (!failed && revision && report.device) {
  console.log(
    `history: ${recordHistory(revision, report.device, Number(warmup), Number(runs), summary)}`,
  );
}
if (failed) {
  console.error("an avel output drifts from ONNX Runtime beyond tolerance");
}
process.exit(failed ? 1 : 0);
