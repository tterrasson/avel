import { assertClose } from "./numeric.js";
import {
  Model,
  WorkerSession,
  type Graph,
  type MediaInput,
  type WebGpuRuntime,
  type RunInput,
} from "../src/index.js";
import { loadGraph } from "../src/compiler.js";

const config: MediaInput = {
  source: { kind: "external" },
  resize: "centerCrop",
  layout: "nchw",
  mean: [0.1, 0.2, 0.3],
  std: [0.5, 0.6, 0.7],
  scale: 2,
};

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}
function compare(actual: Float32Array, expected: Float32Array, tolerance = 1e-4): void {
  assertClose(actual, expected, { atol: tolerance, rtol: 0 }, "media output");
}

export async function validateMedia(
  rt: WebGpuRuntime,
  test: (name: string, fn: () => Promise<void>) => Promise<void>,
  testVideo = test,
): Promise<void> {
  const url = "/packages/runtime/browser/artifacts/models/camera-cnn/";
  const model = await rt.load(url);
  const name = model.graph.inputs[0]!.name;
  const output = model.graph.outputs[0]!.name;
  const shapes = { [name]: [1, 3, 4, 4] };
  const canvas = document.createElement("canvas");
  canvas.width = 8;
  canvas.height = 6;
  const context = canvas.getContext("2d")!;
  const pixels = new ImageData(8, 6);
  for (let p = 0; p < 48; p++) {
    for (let c = 0; c < 3; c++) pixels.data[p * 4 + c] = (p * 3 + c) % 251;
    pixels.data[p * 4 + 3] = 255;
  }
  context.putImageData(pixels, 0, 0);
  const normalized = new Float32Array(3 * 4 * 4);
  for (let c = 0; c < 3; c++) {
    for (let y = 0; y < 4; y++) {
      for (let x = 0; x < 4; x++) {
        const sx = 1 + Math.floor(((x + 0.5) * 6) / 4);
        const sy = Math.floor(((y + 0.5) * 6) / 4);
        normalized[c * 16 + y * 4 + x] =
          ((pixels.data[(sy * 8 + sx) * 4 + c]! / 255) * 2 - config.mean![c]!) / config.std![c]!;
      }
    }
  }
  const baseline = await (await model.compile(shapes, { f16: false })).run({ [name]: normalized });
  const expected = baseline[output] as Float32Array;
  const external = await model.compile(shapes, { media: { [name]: config }, f16: false });
  const image = await model.compile(shapes, {
    media: { [name]: { ...config, source: { kind: "image", width: 8, height: 6 } } },
    f16: false,
  });
  assert(external !== image, "media source must be part of the plan cache key");
  assert(
    external.plan.dispatches.length ===
      (await model.compile(shapes, { f16: false })).plan.dispatches.length,
    "preprocessing added a dispatch",
  );
  const check = async (plan: typeof image, input: RunInput, tolerance?: number) => {
    const values = await plan.run({ [name]: input });
    compare(values[output] as Float32Array, expected, tolerance);
  };

  await test("media CNN: canvas, ImageBitmap and OffscreenCanvas match PyTorch inputs", async () => {
    await check(image, canvas);
    const bitmap = await createImageBitmap(canvas);
    try {
      await check(image, bitmap);
    } finally {
      bitmap.close();
    }
    const offscreen = new OffscreenCanvas(8, 6);
    offscreen.getContext("2d")!.drawImage(canvas, 0, 0);
    await check(image, offscreen);
  });
  await testVideo("media CNN: VideoFrame import and overlapping readbacks", async () => {
    const frames = Array.from(
      { length: 7 },
      (_, timestamp) => new VideoFrame(canvas, { timestamp }),
    );
    try {
      const outputs = await Promise.all(frames.map((frame) => external.run({ [name]: frame })));
      for (const result of outputs) compare(result[output] as Float32Array, expected);
    } finally {
      frames.forEach((frame) => frame.close());
    }
  });
  if (rt.features.shaderF16) {
    await testVideo("media CNN: fused texture prologue with f16 activations", async () => {
      const half = await model.compile(shapes, { media: { [name]: config }, f16: true });
      const frame = new VideoFrame(canvas, { timestamp: 0 });
      try {
        await check(half, frame, 1e-3);
      } finally {
        frame.close();
      }
    });
  }
  await testVideo("media CNN: live video element through importExternalTexture", async () => {
    const video = document.createElement("video");
    const stream = canvas.captureStream(30);
    video.srcObject = stream;
    video.muted = true;
    const pump = setInterval(() => {
      context.putImageData(pixels, 0, 0);
      (stream.getVideoTracks()[0] as CanvasCaptureMediaStreamTrack).requestFrame();
    }, 33);
    try {
      await Promise.race([
        video.play(),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error("video did not start in 10 seconds")), 10_000),
        ),
      ]);
      await check(external, video, 0.015);
    } finally {
      clearInterval(pump);
      video.pause();
      video.srcObject = null;
      stream.getTracks().forEach((track) => track.stop());
    }
  });
  await testVideo(
    "media CNN: GPU consumer and repeated submits without pixel uploads or readbacks",
    async () => {
      const size = expected.byteLength;
      const staging = rt.device.createBuffer({
        size: size * 5,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const writeBuffer = rt.device.queue.writeBuffer;
      let uploads = 0;
      rt.device.queue.writeBuffer = function (...args: Parameters<GPUQueue["writeBuffer"]>) {
        uploads++;
        return writeBuffer.apply(this, args);
      };
      try {
        for (let i = 0; i < 5; i++) {
          const frame = new VideoFrame(canvas, { timestamp: i });
          try {
            external.runGpu({ [name]: frame }, (encoder, outputs) =>
              encoder.copyBufferToBuffer(outputs[output]!, 0, staging, i * size, size),
            );
          } finally {
            frame.close();
          }
        }
        assert(uploads === 0, "media inference uploaded a CPU tensor");
        await staging.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(staging.getMappedRange());
        for (let i = 0; i < 5; i++)
          compare(data.subarray(i * expected.length, (i + 1) * expected.length), expected);
        staging.unmap();
      } finally {
        rt.device.queue.writeBuffer = writeBuffer;
        staging.destroy();
      }
    },
  );
  await test("media: identity, NHWC, strided views and stretch resize", async () => {
    const shape = [1, 3, 2, 3];
    const graph: Graph = {
      formatVersion: model.graph.formatVersion,
      opset: model.graph.opset,
      parameters: [],
      inputs: [{ name: "x", dtype: "f32", shape }],
      outputs: [
        { name: "x", dtype: "f32", shape },
        { name: "permuted", dtype: "f32", shape: [1, 3, 3, 2] },
      ],
      nodes: [
        {
          op: "aten.permute.default",
          inputs: ["x"],
          outputs: [{ name: "permuted", dtype: "f32", shape: [1, 3, 3, 2] }],
          attributes: { dims: [0, 3, 1, 2] },
        },
      ],
    };
    const input = new Float32Array(18);
    for (let y = 0; y < 3; y++)
      for (let x = 0; x < 2; x++)
        for (let c = 0; c < 3; c++) {
          const sx = Math.floor(((x + 0.5) * 8) / 2),
            sy = Math.floor(((y + 0.5) * 6) / 3);
          input[(y * 2 + x) * 3 + c] = pixels.data[(sy * 8 + sx) * 4 + c]! / 255;
        }
    const identity = new Model(rt, graph, loadGraph(graph), new Map());
    const reference = await (await identity.compile({ x: shape })).run({ x: input });
    const plan = await identity.compile(
      { x: shape },
      { media: { x: { source: { kind: "image", width: 8, height: 6 }, layout: "nhwc" } } },
    );
    const actual = await plan.run({ x: canvas });
    for (const key of Object.keys(reference))
      compare(actual[key] as Float32Array, reference[key] as Float32Array);
    await identity.destroy();
  });
  await testVideo("media CNN: transferable VideoFrame hosted in a Worker", async () => {
    const session = await WorkerSession.create(
      new Worker("/packages/runtime/browser/.build/media-worker.js", { type: "module" }),
      url,
      shapes,
      { media: { [name]: config }, f16: false },
      {
        backend: "webgpu",
        wasm: new URL("/packages/runtime/wasm/avel_wasm_bg.wasm", location.href),
      },
    );
    try {
      const frame = new VideoFrame(canvas, { timestamp: 0 });
      const result = await session.run({ [name]: frame });
      compare(result![output]! as Float32Array, expected);
      assert(frame.displayWidth === 0, "frame was not transferred to the worker");
      const next = new VideoFrame(canvas, { timestamp: 1 });
      await session.run({ [name]: next }, { keepOnGpu: true });
    } finally {
      session.close();
    }
  });
}
