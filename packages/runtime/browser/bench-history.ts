import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { arch, platform } from "node:os";
import { browserDir, repoRoot } from "./harness.js";

export interface BenchmarkDevice {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  isFallbackAdapter: boolean;
  browser: string;
}

export interface BenchmarkRow {
  model: string;
  runtime: string;
  medianMs: number;
  p10Ms: number;
  p90Ms: number;
  drift?: number;
}

/** Linearly interpolated percentile of ascending samples (`p` in [0, 1]). */
export function percentile(sorted: readonly number[], p: number): number {
  if (sorted.length === 0) throw new RangeError("percentile of an empty sample");
  const rank = p * (sorted.length - 1);
  const low = Math.floor(rank);
  const high = Math.min(sorted.length - 1, low + 1);
  return sorted[low]! + (sorted[high]! - sorted[low]!) * (rank - low);
}

function git(...args: string[]): string {
  const result = Bun.spawnSync(["git", ...args], { cwd: repoRoot });
  if (!result.success) throw new Error(result.stderr.toString());
  return result.stdout.toString().trim();
}

export function cleanRevision(): string {
  if (git("status", "--porcelain")) {
    throw new Error("benchmark history requires a clean worktree");
  }
  return git("rev-parse", "HEAD");
}

interface Snapshot {
  revision: string;
  time: string;
  rows: BenchmarkRow[];
}

const BAR = "▁▂▃▄▅▆▇█";

export function historyTable(snapshots: Snapshot[]): string {
  const series = new Map<string, Map<string, number>>();
  for (const { revision, rows } of snapshots) {
    for (const row of rows) {
      const label = `${row.model} / ${row.runtime}`;
      const values = series.get(label) ?? new Map<string, number>();
      values.set(revision, row.medianMs);
      series.set(label, values);
    }
  }
  const table = [
    "Medians in ms, ordered by recording time. Each row uses its own scale.",
    "",
    `Commits: ${snapshots.map(({ revision }) => `\`${revision.slice(0, 8)}\``).join(" → ")}`,
    "",
    "| Model / runtime | History | Latest | Min | Max |",
    "| --- | --- | --: | --: | --: |",
  ];
  for (const [label, seriesValues] of series) {
    const values = [...seriesValues.values()];
    const min = Math.min(...values);
    const max = Math.max(...values);
    const chart = snapshots
      .map(({ revision }) => {
        const value = seriesValues.get(revision);
        return value === undefined ? "·" : BAR[Math.round((7 * (value - min)) / (max - min || 1))];
      })
      .join("");
    const latest = seriesValues.get(snapshots.at(-1)!.revision);
    table.push(
      `| ${label} | ${chart} | ${latest?.toFixed(3) ?? "—"} | ${min.toFixed(3)} | ${max.toFixed(3)} |`,
    );
  }
  return table.join("\n");
}

export function recordHistory(
  revision: string,
  device: BenchmarkDevice,
  warmup: number,
  runs: number,
  rows: BenchmarkRow[],
): string {
  if (device.isFallbackAdapter || /swiftshader|llvmpipe|software/i.test(JSON.stringify(device))) {
    throw new Error("benchmark history requires a hardware WebGPU adapter");
  }
  if (cleanRevision() !== revision) throw new Error("revision changed during the benchmark");
  const modes = [...new Set(rows.map((row) => row.runtime))].sort();
  const configuration = { device, host: `${platform()}-${arch()}`, warmup, runs, modes };
  const key = new Bun.CryptoHasher("sha256").update(JSON.stringify(configuration)).digest("hex");
  const dir = join(browserDir, "artifacts", "bench-history", key.slice(0, 16));
  mkdirSync(dir, { recursive: true });
  const snapshot: Snapshot = { revision, time: new Date().toISOString(), rows };
  writeFileSync(
    join(dir, `${revision}.json`),
    `${JSON.stringify({ ...configuration, ...snapshot }, null, 2)}\n`,
  );
  const history = readdirSync(dir)
    .filter((name) => /^[a-f0-9]{40,64}\.json$/.test(name))
    .map((name) => JSON.parse(readFileSync(join(dir, name), "utf8")) as Snapshot)
    .sort((a, b) => a.time.localeCompare(b.time));
  const path = join(dir, "history.md");
  writeFileSync(
    path,
    `# WebGPU benchmark history\n\n\`\`\`json\n${JSON.stringify(configuration, null, 2)}\n\`\`\`\n\n${historyTable(history)}\n`,
  );
  return path;
}
