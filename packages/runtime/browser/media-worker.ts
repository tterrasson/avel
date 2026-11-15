import { hostWorker } from "../src/index.js";

// A GPU consumer can encode rendering or another network here. This test
// consumer copies the output to another GPU buffer in the inference submit.
let target: GPUBuffer | undefined;
hostWorker(self as unknown as DedicatedWorkerGlobalScope, (encoder, outputs, plan) => {
  const output = Object.values(outputs)[0]!;
  target ??= plan.device.createBuffer({
    size: output.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
  });
  encoder.copyBufferToBuffer(output, 0, target, 0, output.size);
});
