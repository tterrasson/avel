import type { Dtype, Parameter } from "../wasm/avel_wasm.js";
import { AvelError } from "./errors.js";

interface TensorInfo {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

// Safetensors packs bools as bytes; WebGPU stores them as u32.
const DTYPES: Record<Dtype, { name: string; bytes: number }> = {
  f32: { name: "F32", bytes: 4 },
  f16: { name: "F16", bytes: 2 },
  i32: { name: "I32", bytes: 4 },
  // The exporter narrows int64 tensors: WGSL has no 64-bit integers.
  i64: { name: "I32", bytes: 4 },
  i8: { name: "I8", bytes: 1 },
  bool: { name: "BOOL", bytes: 1 },
};
const MAX_HEADER_SIZE = 100_000_000;
const DOWNLOAD_CONCURRENCY = 4;

/** A safetensors file read by byte ranges, or at once if the server ignores Range. */
class SafetensorsFile {
  private constructor(
    private readonly url: string,
    private readonly header: Record<string, TensorInfo>,
    private readonly dataStart: number,
    private readonly contents: Uint8Array<ArrayBuffer> | undefined,
  ) {}

  static async open(url: string): Promise<SafetensorsFile> {
    const response = await fetch(url, { headers: { Range: "bytes=0-7" } });
    if (!response.ok) throw new AvelError(`failed to load ${url}: HTTP ${response.status}`);
    const body = await response.arrayBuffer();
    if (body.byteLength < 8) throw new AvelError(`${url} is too short to be a safetensors file`);
    const length = Number(new DataView(body).getBigUint64(0, true));
    if (!Number.isSafeInteger(length) || length < 2 || length > MAX_HEADER_SIZE) {
      throw new AvelError(`${url} has an invalid header length`);
    }
    const dataStart = 8 + length;
    const contents = response.status === 206 ? undefined : new Uint8Array(body);
    const header = contents?.subarray(8, dataStart) ?? (await fetchRange(url, 8, dataStart));
    if (header.byteLength !== length) throw new AvelError(`${url} has a truncated header`);
    const decoded: unknown = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(header));
    if (!decoded || typeof decoded !== "object" || Array.isArray(decoded)) {
      throw new AvelError(`${url} has an invalid header`);
    }
    return new SafetensorsFile(url, decoded as Record<string, TensorInfo>, dataStart, contents);
  }

  /** Returns a byte view without copying a whole-file download. */
  async read(parameter: Parameter): Promise<Uint8Array<ArrayBuffer>> {
    const { tensor } = parameter;
    const info = Object.hasOwn(this.header, tensor) ? this.header[tensor] : undefined;
    if (!info || tensor === "__metadata__")
      throw new AvelError(`${this.url} has no tensor \`${tensor}\``);
    const dtype = DTYPES[parameter.dtype];
    if (!dtype || info.dtype !== dtype.name) {
      throw new AvelError(
        `tensor \`${tensor}\` has dtype ${info.dtype}, expected ${parameter.dtype}`,
      );
    }
    if (
      !Array.isArray(info.shape) ||
      info.shape.length !== parameter.shape.length ||
      info.shape.some(
        (size, i) => !Number.isSafeInteger(size) || size < 1 || size !== parameter.shape[i],
      )
    ) {
      throw new AvelError(
        `tensor \`${tensor}\` has shape [${info.shape}], expected [${parameter.shape}]`,
      );
    }
    const size = info.shape.reduce((n, dim) => n * dim, dtype.bytes);
    const offsets = info.data_offsets;
    if (
      !Number.isSafeInteger(size) ||
      !Array.isArray(offsets) ||
      offsets.length !== 2 ||
      offsets.some((n) => !Number.isSafeInteger(n) || n < 0) ||
      offsets[1] - offsets[0] !== size
    ) {
      throw new AvelError(`tensor \`${tensor}\` has invalid data offsets`);
    }
    const start = this.dataStart + offsets[0];
    const end = this.dataStart + offsets[1];
    if (!Number.isSafeInteger(end))
      throw new AvelError(`tensor \`${tensor}\` has invalid data offsets`);
    const bytes = this.contents?.subarray(start, end) ?? (await fetchRange(this.url, start, end));
    if (bytes.byteLength !== size)
      throw new AvelError(`tensor \`${tensor}\` is truncated in ${this.url}`);
    return bytes;
  }
}

async function fetchRange(
  url: string,
  start: number,
  end: number,
): Promise<Uint8Array<ArrayBuffer>> {
  const response = await fetch(url, { headers: { Range: `bytes=${start}-${end - 1}` } });
  if (response.status !== 206) {
    throw new AvelError(
      `failed to read bytes ${start}-${end - 1} of ${url}: HTTP ${response.status}`,
    );
  }
  return new Uint8Array(await response.arrayBuffer());
}

/** Uploads each unique tensor with bounded download memory, cleaning up on failure. */
export async function loadParameters(
  device: GPUDevice,
  url: string,
  parameters: Parameter[],
): Promise<Map<string, GPUBuffer>> {
  const buffers = new Map<string, GPUBuffer>();
  try {
    await readParameters(url, parameters, (parameter, bytes) => {
      const size = Math.ceil((bytes.byteLength * (parameter.dtype === "bool" ? 4 : 1)) / 4) * 4;
      if (size > Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize))
        throw new AvelError(`tensor \`${parameter.tensor}\` exceeds the device buffer limit`);
      const buffer = device.createBuffer({
        label: parameter.tensor,
        size,
        // State buffers are rewritten by plans and reset by copies.
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          (parameter.update ? GPUBufferUsage.COPY_DST : 0),
        mappedAtCreation: true,
      });
      buffers.set(parameter.tensor, buffer);
      try {
        if (parameter.dtype === "bool") new Uint32Array(buffer.getMappedRange()).set(bytes);
        else new Uint8Array(buffer.getMappedRange()).set(bytes);
      } finally {
        buffer.unmap();
      }
    });
  } catch (error) {
    for (const buffer of buffers.values()) buffer.destroy();
    throw error;
  }
  return buffers;
}

/** Reads each unique tensor, with bounded download memory. */
export async function readParameters(
  url: string,
  parameters: Parameter[],
  consume: (parameter: Parameter, bytes: Uint8Array<ArrayBuffer>) => void,
): Promise<void> {
  if (!parameters.length) return;
  const unique = new Map<string, Parameter>();
  for (const parameter of parameters) {
    const previous = unique.get(parameter.tensor);
    if (
      previous &&
      (previous.dtype !== parameter.dtype ||
        previous.shape.join() !== parameter.shape.join() ||
        previous.update !== undefined ||
        parameter.update !== undefined)
    ) {
      throw new AvelError(`conflicting descriptions for tensor \`${parameter.tensor}\``);
    }
    unique.set(parameter.tensor, parameter);
  }
  const file = await SafetensorsFile.open(url);
  const pending = unique.values();
  let failed = false;
  const read = async () => {
    try {
      while (!failed) {
        const next = pending.next();
        if (next.done) return;
        const parameter = next.value;
        const bytes = await file.read(parameter);
        if (failed) return;
        consume(parameter, bytes);
      }
    } catch (error) {
      failed = true;
      throw error;
    }
  };
  const results = await Promise.allSettled(
    Array.from({ length: Math.min(DOWNLOAD_CONCURRENCY, unique.size) }, read),
  );
  const failure = results.find((result) => result.status === "rejected");
  if (failure?.status === "rejected") throw failure.reason;
}
