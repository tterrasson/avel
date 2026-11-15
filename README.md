# avel

avel is a web inference runtime for small PyTorch networks: MLPs, CNNs and
small transformers of 1 to 50 M parameters. Export a model from PyTorch, then
run it in the browser: avel specializes each graph for its input shapes,
compiles it to fused WGSL and runs it on WebGPU, with a Wasm CPU fallback.

- **Models:** MLPs, CNNs such as ResNet-18 and MobileNetV3, and small
  transformers such as a ViT or a text encoder.
- **Performance:** fused kernels, optional f16 storage, int8 weights and tile
  autotuning. See the [benchmarks](#benchmarks).
- **Web integration:** camera and video inputs without a copy, Worker
  hosting, GPU-side output consumers, custom WGSL operators and a graph
  builder.

## Packages

| Path | Package | Role |
| --- | --- | --- |
| `python/avel-export` | `avel-export` | Exports PyTorch models to `model.graph.json` and safetensors |
| `crates/avel-compiler` | `avel-compiler` | Compiles a graph into an execution plan |
| `crates/avel-wasm` | | Wasm bindings for the compiler and CPU executor, source of the TypeScript types |
| `crates/avel-cli` | `avel-cli` | Compiles plans ahead of time |
| `packages/runtime` | `@avel/runtime` | Runs plans with the native WebGPU API or on the CPU |
| `packages/tokenizer` | `@avel/tokenizer` | Optional BPE and WordPiece text encoding |

## Quick start

```ts
import { Runtime } from "@avel/runtime";

const rt = await Runtime.create({ preferF16: true });
console.log(rt.backend); // "webgpu" or "cpu"

const model = await rt.load("/models/resnet18/");
const plan = await model.compile({ input: [1, 3, 224, 224] });
const out = await plan.run({ input: pixels });

await model.destroy(); // Releases cached plans, shared weights and the compiler copy.
```

`load` validates the graph before downloading weights and hands it to the
compiler once. Every `compile` call reuses that copy, so `model.graph` is
frozen.

## Backends

| `backend` | Behavior |
| --- | --- |
| `"auto"` (default) | Uses WebGPU if a device can be acquired, otherwise loads the CPU Wasm module |
| `"webgpu"` | Requires WebGPU. Passing a `GPUDevice` also selects it |
| `"cpu"` | Forces CPU execution |

The selected backend is exposed on the runtime, models, plans and
`WorkerSession`, so applications can enable GPU-specific features accordingly.

**CPU.** Plans accept typed arrays and run synchronously inside the async `run`
call; use a [Worker](#worker-hosting) to keep inference off the main thread.
The CPU backend supports MLPs, CNNs and stateless transformers, including shape
buckets and int8 weight dequantization. Operators it does not implement
(sort/top-k, scatter, decoding ops) fail at compile time with a CPU backend
error.

**WebGPU only.** GPU buffers, media inputs, custom WGSL, f16 execution and
persistent model state.

## Compilation options

### f16

With `preferF16`, plans store activations in f16 when the device supports
`shader-f16` and the model was exported with `export(..., f16=True)`, which
checks that the model stays within tolerance in half precision.
`compile(..., { f16: true })` enables f16 for a single plan regardless of the
runtime preference.

By default, f16 plans compute in f32. `{ f16Compute: true }` additionally uses
f16 products with f32 accumulation. Products then have the range and rounding
of f16, so validate this mode on representative inputs.

### int8 weights

`export(..., int8=True)` quantizes eligible linear and convolution weights per
output channel. Compare the compiled model against the f32 test vectors to
measure the effect on your workload.

### Dynamic shapes and buckets

Models exported with dynamic shapes (for example a variable sequence length)
compile one plan per concrete shape. A `bucket` rounds symbolic dimensions up
to a multiple of its value, so all lengths in a bucket share one plan. Inputs
are padded at run time and outputs trimmed back:

```ts
const plan = await model.compile({ ids: [2, 37] }, { bucket: 16 }); // Compiled for [2, 48].
const out = await plan.run({ ids: tokenIds }); // tokenIds has 2×37 elements.
```

Buckets require that padding does not change the real output values. The
compiler rejects known incompatible patterns along variable axes:

- reductions, and batch or group statistics;
- unmasked non-causal attention over padded keys;
- returned reshapes that move row boundaries;
- flips, rolls, repeats, trailing pads, indexed reads and resizes.

These checks cannot prove padding invariance for every graph. Compare bucketed
and exact execution over the supported lengths before enabling buckets.

CPU inputs can infer one symbolic axis per tensor from the array length. GPU
inputs and outputs keep the compiled shape; trimming only applies to CPU
readback.

### Tile autotuning

```ts
const rt = await Runtime.create({ autotune: true });
const model = await rt.load("/models/resnet18/");
const plan = await model.compile({ input: [1, 3, 224, 224] });
```

At compile time, autotuning measures a bounded set of matmul and convolution
configurations with `timestamp-query` (warm-up, then three samples each):

- matmuls: a scalar fallback and shared-memory 8×8 and 16×16 tiles;
- convolutions: several output blocks, plus the heuristic reduction split.

Each node keeps the configuration with the lowest median GPU time, if it beats
the current choice by more than 2%. All resources are created before the first
run. No speedup is guaranteed.

Choices are cached in IndexedDB, keyed by graph, concrete shapes,
dtype/features, limits, adapter identity and baseline shaders. Provide your own
store with `RuntimeOptions.tuningCache`; if caching fails, tuning still runs.
`model.compile(shapes, { autotune: true })` enables it for a single plan. A
device you supply must already have `timestamp-query` enabled. Without
timestamps, and for media plans, compilation uses the heuristics directly.

Heuristics depend on the GPU class (integrated, discrete or mobile), inferred
from the adapter. Set `Runtime.create({ gpuClass: "mobile" })` (or
`"integrated"`, `"discrete"`) when the adapter hides or misreports it.

### Profiling

Create the runtime with `{ backend: "webgpu", profiling: true }`, which
requests `timestamp-query`, then:

```ts
const profile = await plan.profile({ input: pixels });
const trace = profile.trace("resnet18"); // Chrome tracing JSON for chrome://tracing or Perfetto.
```

## Running plans

| Method | Returns | Use it for |
| --- | --- | --- |
| `run(inputs)` | CPU arrays | General use |
| `run(inputs, { keepOnGpu: true })` | GPU buffers, asynchronously | Keeping outputs on the GPU |
| `runGpu(inputs, consumer?)` | GPU buffers, synchronously | Ordered GPU consumption in the same submit |

`run` uses a three-slot staging ring: submit the next frame before awaiting
the previous result to overlap inference and readback. Since submission may
wait for a free slot, keep inputs unchanged until `run` resolves. `runGpu`
consumes its inputs before returning. Await any pending `run` before switching
to `runGpu`.

`await plan.destroy()` releases one plan; the model recompiles it on the next
`compile` call. `await model.destroy()` also releases shared parameters and
prevents further compilation. Both wait for pending readbacks.

## Camera and media inputs

Compile an RGB image input with `media` settings, then pass a `VideoFrame` or a
video element directly:

```ts
const video = document.createElement("video");
video.muted = true;
video.srcObject = await navigator.mediaDevices.getUserMedia({ video: true });
await video.play();

const plan = await model.compile({ input: [1, 3, 224, 224] }, {
  media: {
    input: {
      source: { kind: "external" },
      resize: "centerCrop",
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
    },
  },
});

let callback = 0;
function infer() {
  plan.runGpu({ input: video }, (encoder, outputs) => {
    // Encode your renderer or downstream compute here.
    // Output buffers have STORAGE and COPY_SRC usage.
  });
  callback = video.requestVideoFrameCallback(infer);
}
callback = video.requestVideoFrameCallback(infer);

// Teardown:
// video.cancelVideoFrameCallback(callback);
// (video.srcObject as MediaStream).getTracks().forEach((track) => track.stop());
```

**Sources.**

- `{ kind: "external" }`: video elements and `VideoFrame`, imported with
  `importExternalTexture`. The video may change size without recompilation.
- `{ kind: "image", width, height }`: `ImageBitmap`, `HTMLCanvasElement` or
  `OffscreenCanvas`, copied with `copyExternalImageToTexture` into a texture
  allocated at compile time. `width` and `height` are the **source**
  dimensions; another size needs another plan.

**Preprocessing.** Inputs are f32 RGB, batch 1, with `layout: "nchw"` (default)
or `"nhwc"` matching the model's input. Pixels are read as sRGB values in
[0, 1], resized by nearest-center sampling with `"stretch"` (default) or
`"centerCrop"`, then normalized as `(pixel * scale - mean[c]) / std[c]`.
Defaults are `scale: 1`, `mean: [0, 0, 0]` and `std: [1, 1, 1]`.

The compiler folds these texture reads into the first consuming kernels,
including through views, so there is no pixel buffer and no separate
preprocessing dispatch. Only a graph that returns its image input adds a
materialization dispatch. Descriptors can also be stored in the graph with
`export(..., media={...})`; compile settings override them by input name.

**Lifetimes.** `runGpu` submits synchronously and never maps an output. Its
optional consumer records into the same command encoder and submit, which suits
a renderer or another model that needs this frame's results. Output buffers
stay valid until the next run. A `VideoFrame` can be closed as soon as `runGpu`
returns; with `run`, keep media sources open until it resolves. Pipelines,
image textures and static bind groups are created at compile time; external
textures are imported, and their bind groups refreshed, in the submitting task
as the [WebGPU lifetime rules](https://www.w3.org/TR/webgpu/#external-texture-creation)
require.

## Worker hosting

The runtime runs directly inside a dedicated Worker. The optional helpers
transfer frames and arrays, and close transferred media after submission or
readback:

```ts
// inference-worker.ts (bundle as a module worker)
import { hostWorker } from "@avel/runtime";
hostWorker(self); // An optional second argument encodes a GPU consumer.
```

```ts
// Main thread
import { WorkerSession } from "@avel/runtime";

const session = await WorkerSession.create(
  new Worker(new URL("./inference-worker.js", import.meta.url), { type: "module" }),
  "/models/resnet18/",
  { input: [1, 3, 224, 224] },
  { media: { input: { source: { kind: "external" } } } },
  { backend: "webgpu" },
);
const frame = new VideoFrame(video);
const outputs = await session.run({ input: frame }); // Ownership transfers.
session.close(); // Terminates the worker and rejects pending requests.
```

- Workers accept `VideoFrame`, `ImageBitmap`, `OffscreenCanvas` and typed
  arrays; HTML elements stay on the main thread.
- GPU objects stay in the worker that owns the device. `keepOnGpu: true`
  requires a consumer passed to `hostWorker` and only acknowledges submission.
- Pass `wasm` and `cpuWasm` URLs in the last argument of
  `WorkerSession.create` if your bundler needs explicit Wasm locations.
- Register custom ops separately in each worker.

## Autoregressive decoding

For fixed-capacity KV caches, export the model with `cache_write`,
`cached_attention` and `sample_topk` from `avel_export.generation`. Then:

1. Compile the prefill and decode shapes from the same loaded model, so they
   share the persistent K/V buffers.
2. Pass the insertion position and the sampling seed as scalar integer inputs
   on each run.
3. Call `model.resetState()` before each new sequence.

The decode output can stay on the GPU with `runGpu`. See the
[exporter example](python/avel-export/README.md#autoregressive-decode) and the
[tokenizer](packages/tokenizer/README.md) for model and input setup.

## Custom operators

Declare a functional
[PyTorch custom op](https://docs.pytorch.org/docs/stable/library.html#torch.library.custom_op)
with a reference implementation and a fake implementation, then allowlist it
at export:

```python
import torch
from avel_export import export

@torch.library.custom_op("example::affine", mutates_args=())
def affine(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x * scale + 0.25

@affine.register_fake
def affine_fake(x, scale):
    return torch.empty_like(x)

class Network(torch.nn.Module):
    def forward(self, x):
        return torch.relu(affine(x, 2.0))

export(Network(), (torch.randn(2, 3),), "model",
       custom_ops=("example.affine.default",))
```

Allowlisted nodes stay opaque through decomposition; other unknown operators
still fail export. Supported ops are functional, take f32 tensors and JSON
attributes (including nested lists and objects), and return one f32 tensor.
Aliasing and mutation are rejected. Symbolic integer attributes reach the
callbacks resolved to the compiled shapes.

Register the shape function and WGSL kernel before compiling:

```ts
rt.registerOp("example.affine.default", {
  inputs: ["x"], // Tensor bindings, in PyTorch schema order.
  shape: (inputs) => [...inputs[0].shape],
  kernel: ({ shape, attributes }) => {
    const count = shape.reduce((a, b) => a * b, 1);
    return {
      workgroups: [Math.ceil(count / 64), 1, 1],
      wgsl: `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < ${count}u) {
    output[id.x] = x[id.x] * ${Number(attributes.scale).toFixed(8)} + 0.25;
  }
}`,
    };
  },
});
const model = await rt.load("/models/custom/");
const plan = await model.compile({ x: [2, 3] });
```

**Kernel contract.**

- The runtime declares the group 0 storage bindings: one dense `array<f32>` per
  input, then `output`. Do not redeclare them.
- Bindings are `read_write` for arena compatibility, but the kernel must only
  write `output`, and must write every element of it.
- The entry point is `main`. Input names must be unique WGSL identifiers.
- WGSL compilation errors reject `compile`.

**Behavior.** Callbacks run only at compile time; with shape buckets, the
kernel is specialized at the padded shape. Views and media inputs are
materialized before a custom node, and these buffers stay f32 in f16 plans.
Custom kernels are not fused, but share the lifetime arena and the single
submit. Nodes that depend only on weights run once in `Plan.prepare`.
Re-registering an op invalidates the model's plan cache for later
compilations.

## Graph builder

The builder goes through the same Rust shape inference and compiler as
exported models:

```ts
const g = rt.graph();
const x = g.input("x", [2, 3]);
const w = g.parameter("w", [3, 2], new Float32Array([1, 0, 0, 1, 1, 1]));
const bias = g.parameter("bias", [2], new Float32Array([0.5, -0.5]));
const y = g.relu(g.matmul(x, w, bias));

const built = g.build([y]);
const plan = await built.compile({ x: [2, 3] });
const result = await plan.run({ x: new Float32Array([1, 2, 3, 4, 5, 6]) });
console.log(result[y.name]);
```

- Helpers: `input`, `parameter`, `relu`, `add`, `mul`, `reshape`, `matmul`
  (with optional bias) and `conv2d`.
- `g.op(name, inputs, attributes)` accepts any supported ATen op or registered
  custom op. `inputs` has one slot per tensor argument, `null` for an omitted
  optional one.
- Shapes are concrete, and values belong to a single builder.
- `build(outputs, metadata?)` snapshots the graph. Parameters are uploaded once
  and shared across its plans. Builder graphs use f32 storage.

## Graph format

`model.graph.json` is format version 1, and stays at 1 until the format is
declared stable. The Rust types in `crates/avel-compiler/src/graph.rs` are the
reference.

| Field | Content |
| --- | --- |
| `inputs`, `outputs` | Name, dtype and shape. A dimension is a size or a named symbol with optional bounds: `{"name": "s0", "min": 1, "max": 64}` |
| `nodes` | ATen op; `inputs`, one slot per tensor argument in schema order (`null` for an omitted optional tensor, one entry per element of a `Tensor[]`); `outputs` (`null` for an unused output); `attributes`, the non-tensor arguments by schema name |
| `parameters` | The safetensors tensor of each weight or buffer |
| `media` | Media input settings by input name |
| `metadata` | Free-form application data, ignored by the compiler |
| `f16` | The exporter validated the model in half precision |

Attributes are `null`, booleans, numbers, strings, lists or objects. A symbolic
integer is written `{"$sym": "2*s0"}` and resolved against the input shapes;
this single-key object form is reserved. Non-finite floats are rejected at
export.

## Benchmarks

Median end-to-end latency in milliseconds (input upload, run and output
readback), batch 1 at 224×224, compared with ONNX Runtime Web 1.31 on its
WebGPU backend, in headless Brave on an Apple GPU:

| Model | ONNX Runtime Web | avel | avel f16 |
| --- | --: | --: | --: |
| ResNet-18 | 6.3 | 4.4 (1.43×) | 3.6 (1.75×) |
| MobileNetV3 small | 2.3 | 1.3 (1.77×) | 1.4 (1.64×) |
| MobileNetV3 large | 3.4 | 2.8 (1.21×) | 1.9 (1.79×) |

The f16 column stores activations and constants in f16 and computes in f32. To
reproduce these numbers, see [Benchmarking](#benchmarking).

## Development

### Requirements

- Rust with the `wasm32-unknown-unknown` target
- Bun, at the version in `package.json`
- uv
- `wasm-bindgen-cli`, matching the version pinned in `Cargo.toml` (the Wasm
  build checks it):

```bash
cargo install wasm-bindgen-cli --version 0.2.130 --locked
```

### Tests

Rust:

```bash
cargo test --workspace
```

TypeScript:

```bash
bun install --frozen-lockfile && bun run build && bun run typecheck && bun run test
```

```bash
bun run format:check
```

Python, from `python/avel-export`:

```bash
cd python/avel-export && uv run ruff check . && uv run ruff format --check . && uv run pytest
```

The default Python suite uses small fixtures. `uv run pytest -m ""` also
exports the full validation corpus and exercises the ONNX exporter.

Format the runtime and browser sources with `bun run format`, and the Python
sources with `uv run ruff format .` (plus `uv run ruff check --fix .` for lint
fixes) from `python/avel-export`.

### Browser validation

WebGPU only exists in a browser, so the validation suite runs in headless
Chrome or Chromium. It regenerates per-op cases and models (MLP, ResNet-18,
MobileNetV3) from PyTorch, then loads, compiles and runs them on a real GPU and
compares the results with the exporter's test vectors. It needs Chrome or
Chromium (override with `CHROME_PATH`), uv and the Wasm build:

```bash
bun run build && bun run test:browser
```

- On machines without a GPU, set `AVEL_WEBGPU_ADAPTER=swiftshader`.
- `AVEL_SKIP_EXTERNAL_VIDEO=1` skips the external video texture tests. GitHub
  Actions sets it because those tests fail on its headless runner; local runs
  include them by default.

### Benchmarking

The benchmark exports ResNet-18 and MobileNetV3 for avel and to ONNX, times
both runtimes in headless Chrome and checks that their outputs agree. It writes
`results.json` and one Chrome trace per model to
`packages/runtime/browser/artifacts/bench/`.

```bash
bun run build && bun run bench
```

To track performance over time, run `bun run bench:history` on a real GPU from
a clean checkout. It records the current revision under
`packages/runtime/browser/artifacts/bench-history/` and updates the chart there.
Set `AVEL_BENCH_F16_COMPUTE=1` to include the f16 compute variant.

## License

[MIT](LICENSE)
