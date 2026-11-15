import { expect, test } from "bun:test";
import { join } from "node:path";

test("loadCompiler() without arguments reads the shipped wasm module from disk", async () => {
  // Fresh process, so the module-level ready promise starts undefined and the
  // default source resolution is really exercised.
  const compiler = join(import.meta.dir, "..", "src", "compiler.ts");
  const proc = Bun.spawn({
    cmd: [
      process.execPath,
      "-e",
      `import(${JSON.stringify(compiler)}).then((m) => m.loadCompiler()).then(
        () => console.log("loaded"),
        (error) => { console.error(error); process.exit(1); },
      )`,
    ],
    stdout: "pipe",
    stderr: "pipe",
  });
  const exit = await proc.exited;
  expect(await new Response(proc.stdout).text()).toContain("loaded");
  expect(exit).toBe(0);
});

test("loadCompiler recovers after a failed source instead of caching the rejection", async () => {
  // Fresh process, so the module-level ready promise starts undefined.
  const compiler = join(import.meta.dir, "..", "src", "compiler.ts");
  const proc = Bun.spawn({
    cmd: [
      process.execPath,
      "-e",
      `import(${JSON.stringify(compiler)}).then(async (m) => {
        try {
          await m.loadCompiler(new Response("not wasm"));
          console.log("unexpected success");
          process.exit(1);
        } catch {
          console.log("failed as expected");
        }
        await m.loadCompiler();
        console.log("recovered");
      })`,
    ],
    stdout: "pipe",
    stderr: "pipe",
  });
  const exit = await proc.exited;
  const out = await new Response(proc.stdout).text();
  expect(out).toContain("failed as expected");
  expect(out).toContain("recovered");
  expect(exit).toBe(0);
});
