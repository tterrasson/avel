import type { CompileSettings } from "./model.js";
import type { ExecutionPlan, TensorOutput } from "./plan.js";
import type { CpuExecutionPlan } from "./cpu.js";
import { Runtime, type RuntimeBackend } from "./runtime.js";
import { AvelError } from "./errors.js";

/** Transferable media and tensors; HTML elements stay on the main thread. */
export type WorkerInput = TensorOutput | VideoFrame | ImageBitmap | OffscreenCanvas;

export interface WorkerRuntimeOptions {
  backend?: RuntimeBackend | "auto";
  wasm?: URL;
  cpuWasm?: URL;
}

type Request =
  | {
      id: number;
      kind: "init";
      url: string;
      shapes: Record<string, number[]>;
      settings: CompileSettings;
      backend?: WorkerRuntimeOptions["backend"];
      wasm?: string;
      cpuWasm?: string;
    }
  | { id: number; kind: "run"; inputs: Record<string, WorkerInput>; keepOnGpu: boolean };
type Response = {
  id: number;
  backend?: RuntimeBackend;
  outputs?: Record<string, TensorOutput>;
  error?: string;
};

/** The part of a dedicated worker scope the host uses. */
interface WorkerScope {
  addEventListener(type: "message", listener: (event: MessageEvent<Request>) => void): void;
  postMessage(message: Response, transfer: Transferable[]): void;
}

/** Hosts one model in a dedicated Worker. Call once in the worker entry point. */
export function hostWorker(
  scope: WorkerScope,
  consume?: (
    encoder: GPUCommandEncoder,
    outputs: Readonly<Record<string, GPUBuffer>>,
    plan: ExecutionPlan,
  ) => void,
): void {
  let plan: ExecutionPlan | CpuExecutionPlan | undefined;
  let initializing = false;
  scope.addEventListener("message", async (event: MessageEvent<Request>) => {
    const request = event.data;
    const response: Response = { id: request.id };
    try {
      if (request.kind === "init") {
        if (initializing) throw new AvelError("worker is already initialized");
        initializing = true;
        const runtime = await Runtime.create({
          backend: request.backend,
          preferF16: request.settings.f16 ?? false,
          autotune: request.settings.autotune,
          wasm: request.wasm ? new URL(request.wasm) : undefined,
          cpuWasm: request.cpuWasm ? new URL(request.cpuWasm) : undefined,
        });
        const model = await runtime.load(request.url);
        try {
          plan = await model.compile(request.shapes, request.settings);
        } catch (error) {
          await model.destroy();
          throw error;
        }
        response.backend = runtime.backend;
      } else {
        try {
          const execution = plan;
          if (!execution) throw new AvelError("worker is not initialized");
          if (request.keepOnGpu) {
            if (execution.backend !== "webgpu")
              throw new AvelError("CPU backend does not support GPU outputs");
            if (!consume)
              throw new AvelError("GPU worker runs require a consume callback in hostWorker");
            execution.runGpu(request.inputs, (encoder, outputs) =>
              consume(encoder, outputs, execution),
            );
          } else {
            response.outputs = (await execution.run(request.inputs)) as Record<
              string,
              TensorOutput
            >;
          }
        } finally {
          for (const input of Object.values(request.inputs)) {
            if (
              (typeof VideoFrame !== "undefined" && input instanceof VideoFrame) ||
              (typeof ImageBitmap !== "undefined" && input instanceof ImageBitmap)
            )
              input.close();
          }
        }
      }
    } catch (error) {
      response.error = String(error);
    }
    const transfer = response.outputs
      ? Object.values(response.outputs).map((output) => output.buffer as ArrayBuffer)
      : [];
    scope.postMessage(response, transfer);
  });
}

/** Client for hostWorker. Owns the Worker; close() terminates it. */
export class WorkerSession {
  private id = 0;
  private closed = false;
  private selected?: RuntimeBackend;
  get backend(): RuntimeBackend {
    return this.selected!;
  }
  private readonly pending = new Map<
    number,
    { resolve: (response: Response) => void; reject: (error: Error) => void }
  >();

  private constructor(private readonly worker: Worker) {
    worker.addEventListener("message", (event: MessageEvent<Response>) => {
      const response = event.data;
      const pending = this.pending.get(response.id);
      if (!pending) return;
      this.pending.delete(response.id);
      if (response.error) pending.reject(new AvelError(response.error));
      else pending.resolve(response);
    });
    worker.addEventListener("error", (event) => this.close(new AvelError(event.message)));
    worker.addEventListener("messageerror", () =>
      this.close(new AvelError("worker message could not be decoded")),
    );
  }

  static async create(
    worker: Worker,
    url: string,
    shapes: Record<string, number[]>,
    settings: CompileSettings = {},
    options: WorkerRuntimeOptions = {},
  ): Promise<WorkerSession> {
    const session = new WorkerSession(worker);
    try {
      const response = await session.request({
        id: 0,
        kind: "init",
        url: new URL(url, globalThis.location.href).href,
        shapes,
        settings,
        backend: options.backend,
        wasm: options.wasm?.href,
        cpuWasm: options.cpuWasm?.href,
      });
      if (!response.backend) throw new AvelError("worker did not report its execution backend");
      session.selected = response.backend;
      return session;
    } catch (error) {
      session.close();
      throw error;
    }
  }

  /** Transfers ownership of every input. GPU-only runs acknowledge submission. */
  async run(
    inputs: Record<string, WorkerInput>,
    options: { keepOnGpu?: boolean } = {},
  ): Promise<Record<string, TensorOutput> | undefined> {
    const transfer = new Set<Transferable>();
    for (const input of Object.values(inputs)) {
      transfer.add(ArrayBuffer.isView(input) ? (input.buffer as ArrayBuffer) : input);
    }
    const response = await this.request(
      { id: 0, kind: "run", inputs, keepOnGpu: options.keepOnGpu ?? false },
      [...transfer],
    );
    return response.outputs;
  }

  close(error = new AvelError("worker session closed")): void {
    this.closed = true;
    this.worker.terminate();
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
  }

  private request(request: Request, transfer: Transferable[] = []): Promise<Response> {
    if (this.closed) return Promise.reject(new AvelError("worker session closed"));
    request.id = this.id++;
    return new Promise((resolve, reject) => {
      this.pending.set(request.id, { resolve, reject });
      try {
        this.worker.postMessage(request, transfer);
      } catch (error) {
        this.pending.delete(request.id);
        reject(error);
      }
    });
  }
}
