# @avel/runtime

Runs small PyTorch networks in a browser with WebGPU or a CPU fallback. Export the model with
[`avel-export`](https://github.com/tterrasson/avel/tree/main/python/avel-export)
and serve its files from the same origin as the application.

```ts
import { Runtime } from "@avel/runtime";

const runtime = await Runtime.create();
console.log(runtime.backend); // "webgpu" or "cpu"
const model = await runtime.load("/models/example/");
const plan = await model.compile({ input: [1, 4] });
const outputs = await plan.run({ input: new Float32Array([1, 2, 3, 4]) });
await model.destroy();
```

Use the input names in `model.graph.inputs`. WebGPU requires a secure context
(HTTPS or localhost). By default the runtime selects CPU when it cannot acquire
a WebGPU device. Pass `backend: "webgpu"` to require a GPU or `backend: "cpu"`
to force the fallback. CPU plans use typed tensor inputs and work in a dedicated
Worker. TypeScript declarations use the DOM and WebGPU types included with
TypeScript 7.

## Wasm loading

`Runtime.create()` loads the module for the selected backend automatically.

| Environment                                             | How the module is loaded                                                                                  |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Browser over HTTP(S), including dedicated workers       | Streams `avel_wasm_bg.wasm` for WebGPU or `avel_cpu_wasm_bg.wasm` for CPU, next to its JS glue. |
| Node, Bun, Deno, Electron (`file:` URLs)                | Read from disk with `node:fs/promises`; `fetch` cannot load `file:` there (and never settles in Bun workers). |
| Bundler (Vite, webpack, …) or edge runtime without it   | Pass `wasm` for WebGPU and `cpuWasm` for CPU as served URLs or module bytes. |

The package ships both modules in `wasm/`. With a bundler, include the modules
your application enables among its served assets.

See the [project README](https://github.com/tterrasson/avel) for supported
models, media inputs, workers and custom kernels.
