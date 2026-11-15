/**
 * Runs a page in headless Chrome and returns what it POSTs to `/report`.
 *
 * Bundles the page script, serves the repository over HTTP on 127.0.0.1, and
 * opens the page in a headless browser. Shared by the validation suite and
 * the benchmark, since `bun test` has no WebGPU.
 *
 * Environment:
 * - `CHROME_PATH`: browser binary (auto-detected otherwise).
 * - `AVEL_WEBGPU_ADAPTER`: WebGPU adapter, e.g. `swiftshader` on a GPU-less
 *   machine (CI). Omit to use the real GPU.
 * - `AVEL_NO_SANDBOX=1`: adds `--no-sandbox`.
 * - `AVEL_BROWSER_TIMEOUT_MS`: give up after this long (default 180000).
 */
import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, extname, join, normalize, resolve } from "node:path";

export const browserDir = import.meta.dir;
export const repoRoot = resolve(browserDir, "../../..");

const MIME: Record<string, string> = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".json": "application/json",
  ".wasm": "application/wasm",
  ".safetensors": "application/octet-stream",
};

/**
 * Bundles `script` (relative to this directory) into `.build/`, then opens
 * `page.html?script=<bundle>&<query>` and resolves with the body of its report.
 */
export async function runPage(script: string, timeoutMs: number, query = ""): Promise<string> {
  const build = await Bun.build({
    entrypoints: [
      join(browserDir, script),
      ...(script === "validation.ts" ? [join(browserDir, "media-worker.ts")] : []),
    ],
    target: "browser",
    outdir: join(browserDir, ".build"),
  });
  if (!build.success) {
    throw new AggregateError(build.logs, `failed to bundle ${script}`);
  }
  const bundle = basename(script).replace(/\.ts$/, ".js");
  // Resolve this before opening sockets or creating a temporary profile.
  const executable = findBrowser();

  let settleReport!: (text: string) => void;
  const reported = new Promise<string>((resolvePromise) => {
    settleReport = resolvePromise;
  });
  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    async fetch(request) {
      const url = new URL(request.url);
      if (url.pathname === "/report" && request.method === "POST") {
        settleReport(await request.text());
        return new Response("ok");
      }
      let path = normalize(decodeURIComponent(url.pathname));
      if (path.endsWith("/")) {
        path += "page.html";
      }
      // The fixtures are single files, not export directories.
      if (path === "/tests/fixtures/identity/model.graph.json") {
        path = "/tests/fixtures/identity.graph.json";
      }
      const file = join(repoRoot, path);
      if (!file.startsWith(`${repoRoot}/`) || !existsSync(file) || !statSync(file).isFile()) {
        console.log(`404 ${url.pathname}`);
        return new Response(`not found: ${url.pathname}`, { status: 404 });
      }
      return new Response(Bun.file(file), {
        headers: { "content-type": MIME[extname(file)] ?? "application/octet-stream" },
      });
    },
  });
  const base = `http://127.0.0.1:${server.port}`;
  console.log(`serving ${repoRoot} at ${base}`);

  const profile = mkdtempSync(join(tmpdir(), "avel-browser-"));
  const adapter = process.env.AVEL_WEBGPU_ADAPTER;
  const args = [
    "--headless=new",
    `--user-data-dir=${profile}`,
    "--disable-dev-shm-usage",
    "--enable-unsafe-webgpu",
    // Unquantized timestamp queries, for profiling.
    "--enable-webgpu-developer-features",
    ...(adapter
      ? [
          `--use-webgpu-adapter=${adapter}`,
          ...(adapter === "swiftshader" ? ["--enable-unsafe-swiftshader"] : []),
        ]
      : []),
    ...(process.env.AVEL_NO_SANDBOX ? ["--no-sandbox"] : []),
    `${base}/packages/runtime/browser/?script=${bundle}${query && `&${query}`}`,
  ];
  console.log(`launching ${executable} ${args.join(" ")}`);
  const browser = spawn(executable, args, { stdio: ["ignore", "ignore", "pipe"] });
  let browserLog = "";
  browser.stderr?.on("data", (chunk: Buffer) => {
    browserLog = (browserLog + chunk.toString()).slice(-4000);
  });
  const exited = new Promise<never>((_, reject) => {
    browser.on("error", reject);
    browser.on("exit", (code) =>
      reject(new Error(`browser exited with code ${code}\n${browserLog}`)),
    );
  });
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timedOut = new Promise<never>((_, reject) => {
    timer = setTimeout(
      () => reject(new Error(`timed out after ${timeoutMs} ms\n${browserLog}`)),
      timeoutMs,
    );
  });

  try {
    return await Promise.race([reported, exited, timedOut]);
  } finally {
    clearTimeout(timer);
    browser.kill();
    server.stop();
    rmSync(profile, { recursive: true, force: true });
  }
}

function findBrowser(): string {
  const candidates = [
    process.env.CHROME_PATH,
    ...(process.platform === "darwin"
      ? [
          "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
          "/Applications/Chromium.app/Contents/MacOS/Chromium",
          "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        ]
      : [
          "/usr/bin/google-chrome",
          "/usr/bin/google-chrome-stable",
          "/usr/bin/chromium",
          "/usr/bin/chromium-browser",
        ]),
  ].filter((candidate): candidate is string => candidate !== undefined);
  for (const candidate of candidates) {
    if (existsSync(candidate)) {
      return candidate;
    }
  }
  throw new Error("no Chrome or Chromium found; set CHROME_PATH");
}

export function timeout(fallbackMs: number): number {
  const value = Number(process.env.AVEL_BROWSER_TIMEOUT_MS ?? fallbackMs);
  if (!Number.isSafeInteger(value) || value < 1)
    throw new Error("AVEL_BROWSER_TIMEOUT_MS must be a positive integer");
  return value;
}
