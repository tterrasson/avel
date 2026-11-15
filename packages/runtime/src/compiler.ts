import init, { LoadedGraph, type Graph } from "../wasm/avel_wasm.js";
import { AvelError } from "./errors.js";

export type WasmSource = BufferSource | string | URL | Response;

let ready: Promise<unknown> | undefined;

/** Loads the wasm compiler once. Defaults to the module shipped with the package. */
export async function loadCompiler(source?: WasmSource): Promise<void> {
  ready ??= resolveWasmSource(source, new URL("../wasm/avel_wasm_bg.wasm", import.meta.url))
    .then((module) => init(module === undefined ? undefined : { module_or_path: module }))
    .catch((error: unknown) => {
      ready = undefined;
      throw error;
    });
  await ready;
}

/**
 * Parses and validates `graph` once in the compiler, which then compiles it
 * for any input shapes. The caller owns the result and must `free()` it. Must
 * be called after {@link loadCompiler} has resolved.
 */
export function loadGraph(graph: Graph): LoadedGraph {
  return new LoadedGraph(graph);
}

export async function fetchGraph(base: string): Promise<Graph> {
  const response = await fetch(`${base}model.graph.json`);
  if (!response.ok) throw new AvelError(`failed to load ${response.url}: HTTP ${response.status}`);
  return (await response.json()) as Graph;
}

/**
 * Resolves the module to instantiate. Outside a browser (Node, Bun, Deno,
 * Electron) the module sits behind a `file:` URL that `fetch` cannot load —
 * Node rejects it and Bun workers never settle it — so its bytes are read
 * with `node:fs/promises`. In a browser the URL is served over HTTP(S) and
 * left to wasm-bindgen's streaming instantiation.
 */
export async function resolveWasmSource(
  source: WasmSource | undefined,
  defaultUrl: URL,
): Promise<WasmSource | undefined> {
  const url =
    source === undefined
      ? defaultUrl
      : typeof source === "string"
        ? new URL(source, import.meta.url)
        : source instanceof URL
          ? source
          : undefined;
  if (url === undefined || url.protocol !== "file:") return source;
  // A browser page opened from file:// has no node:fs; let fetch report that.
  // Read via globalThis so the published build needs no Node type definitions.
  const versions = (globalThis as { process?: { versions?: Record<string, string | undefined> } })
    .process?.versions;
  if (versions?.node === undefined) return source;
  const fs = "node:fs/promises";
  const { readFile } = (await import(/* @vite-ignore */ fs)) as {
    readFile(path: URL): Promise<Uint8Array<ArrayBuffer>>;
  };
  return readFile(url);
}
