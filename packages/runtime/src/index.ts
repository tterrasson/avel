/// <reference lib="esnext.disposable" preserve="true" />

export type {
  Attr,
  BufferDesc,
  BufferRole,
  CompileOptions,
  MediaInput,
  MediaSource,
  ImageLayout,
  ImageResize,
  Dtype,
  Graph,
  Plan,
  TensorDesc,
} from "../wasm/avel_wasm.js";
export { AvelError } from "./errors.js";
export { type CompileSettings, Model } from "./model.js";
export {
  ExecutionPlan,
  type RunInput,
  type RunOptions,
  type RunOutput,
  type TensorOutput,
} from "./plan.js";
export { type ChromeTrace, type DispatchTiming, Profile } from "./profile.js";
export {
  Runtime,
  type RuntimeBackend,
  type RuntimeOptions,
  type WebGpuRuntime,
} from "./runtime.js";
export type { CpuExecutionPlan, CpuModel, CpuRuntime } from "./cpu.js";
export type { MediaInputSource } from "./media.js";
export {
  hostWorker,
  WorkerSession,
  type WorkerInput,
  type WorkerRuntimeOptions,
} from "./worker.js";

export { GraphBuilder, type GraphValue } from "./builder.js";
export type { CustomAttributes, CustomOp } from "./custom.js";
export { IndexedDbTuningCache, type TuningCache } from "./tuning.js";
export type { GpuClass, TileConfig } from "../wasm/avel_wasm.js";
