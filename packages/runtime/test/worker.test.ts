import { expect, test } from "bun:test";
import { hostWorker, WorkerSession } from "../src/index.js";
import { setupTests, stubGlobal } from "./helpers.js";

type Reply = Parameters<Parameters<typeof hostWorker>[0]["postMessage"]>[0];

class FakeWorker extends EventTarget {
  readonly messages: unknown[] = [];
  terminated = false;
  sendError?: Error;

  postMessage(message: unknown, transfer: Transferable[]) {
    if (this.sendError) throw this.sendError;
    this.messages.push(structuredClone(message, { transfer }));
  }

  reply(data: Reply) {
    this.dispatchEvent(new MessageEvent("message", { data }));
  }

  terminate() {
    this.terminated = true;
  }
}

setupTests();

function startSession(worker: FakeWorker) {
  stubGlobal("location", new URL("https://example.test/app/"));
  return WorkerSession.create(worker as unknown as Worker, "model", { x: [2] });
}

async function openSession(worker: FakeWorker) {
  const session = startSession(worker);
  worker.reply({ id: 0, backend: "webgpu" });
  return session;
}

test("transfers aliased inputs once and matches replies to concurrent requests", async () => {
  const worker = new FakeWorker();
  const session = await openSession(worker);
  try {
    expect(session.backend).toBe("webgpu");
    expect(worker.messages[0]).toMatchObject({
      kind: "init",
      url: "https://example.test/app/model",
      shapes: { x: [2] },
    });
    const x = new Float32Array([1, 2]);
    const first = session.run({ x, alias: x.subarray(1) });
    const second = session.run({ x: new Float32Array([3, 4]) }, { keepOnGpu: true });
    const results = Promise.all([first, second]);
    // The worker can acknowledge submission before an earlier CPU readback.
    worker.reply({ id: 2 });
    worker.reply({ id: 1, outputs: { y: new Float32Array([5, 6]) } });
    expect(await results).toEqual([{ y: new Float32Array([5, 6]) }, undefined]);
    expect(x.byteLength).toBe(0);
    expect(worker.messages[1]).toMatchObject({
      inputs: { x: new Float32Array([1, 2]), alias: new Float32Array([2]) },
      keepOnGpu: false,
    });
    expect(worker.messages[2]).toMatchObject({ keepOnGpu: true });
  } finally {
    session.close();
  }
});

test("terminates the worker when initialization fails", async () => {
  const worker = new FakeWorker();
  const session = startSession(worker);
  worker.reply({ id: 0, error: "model could not be loaded" });
  await expect(session).rejects.toThrow("model could not be loaded");
  expect(worker.terminated).toBe(true);
});

test("recovers from a send failure and rejects pending runs if the worker crashes", async () => {
  const worker = new FakeWorker();
  const session = await openSession(worker);
  try {
    worker.sendError = new Error("could not transfer input");
    await expect(session.run({})).rejects.toThrow("could not transfer input");
    worker.sendError = undefined;

    const settled = Promise.allSettled([session.run({}), session.run({})]);
    worker.dispatchEvent(new ErrorEvent("error", { message: "worker crashed" }));
    expect(await settled).toEqual([
      { status: "rejected", reason: expect.objectContaining({ message: "worker crashed" }) },
      { status: "rejected", reason: expect.objectContaining({ message: "worker crashed" }) },
    ]);
    expect(worker.terminated).toBe(true);
    await expect(session.run({})).rejects.toThrow("worker session closed");
    expect(worker.messages).toHaveLength(3);
  } finally {
    if (!worker.terminated) session.close();
  }
});

test("the host rejects a second init while one is in flight", async () => {
  const replies: Reply[] = [];
  let receive!: Parameters<Parameters<typeof hostWorker>[0]["addEventListener"]>[1];
  hostWorker({
    addEventListener(_, listener) {
      receive = listener;
    },
    postMessage: (message) => replies.push(message),
  });
  const init = {
    id: 1,
    kind: "init" as const,
    url: "https://example.test/model",
    shapes: { x: [2] },
    settings: {},
    backend: "webgpu" as const,
  };
  receive(new MessageEvent("message", { data: init }));
  receive(new MessageEvent("message", { data: { ...init, id: 2 } }));
  await Bun.sleep(0);
  expect(replies).toHaveLength(2);
  // The first init fails here (no WebGPU in this process); the second must not start over.
  const errors = replies.map((reply) => reply.error);
  expect(errors.some((error) => error?.includes("WebGPU is not available"))).toBe(true);
  expect(errors.some((error) => error?.includes("worker is already initialized"))).toBe(true);
});

test("the worker host closes transferred images even when a run fails", async () => {
  let closed = 0;
  class FakeImageBitmap {
    close() {
      closed++;
    }
  }
  stubGlobal("ImageBitmap", FakeImageBitmap);
  const reply = Promise.withResolvers<Reply>();
  let receive!: Parameters<Parameters<typeof hostWorker>[0]["addEventListener"]>[1];
  hostWorker({
    addEventListener(_, listener) {
      receive = listener;
    },
    postMessage: reply.resolve,
  });
  receive(
    new MessageEvent("message", {
      data: {
        id: 1,
        kind: "run",
        inputs: { image: new FakeImageBitmap() as unknown as ImageBitmap },
        keepOnGpu: false,
      },
    }),
  );
  expect(await reply.promise).toMatchObject({
    id: 1,
    error: expect.stringContaining("not initialized"),
  });
  expect(closed).toBe(1);
});
