import type { MediaInput } from "../wasm/avel_wasm.js";
import { AvelError } from "./errors.js";

export type MediaInputSource =
  VideoFrame | HTMLVideoElement | ImageBitmap | HTMLCanvasElement | OffscreenCanvas;

export function isExternal(source: unknown): source is VideoFrame | HTMLVideoElement {
  return (
    (typeof VideoFrame !== "undefined" && source instanceof VideoFrame) ||
    (typeof HTMLVideoElement !== "undefined" && source instanceof HTMLVideoElement)
  );
}

export class MediaBinding {
  readonly texture?: GPUTexture;
  readonly view?: GPUTextureView;
  external?: GPUExternalTexture;
  private readonly externalDescriptor: GPUExternalTextureDescriptor = {
    source: undefined as unknown as VideoFrame,
    colorSpace: "srgb",
  };
  private readonly copySource: GPUCopyExternalImageSourceInfo = {
    source: undefined as unknown as ImageBitmap,
  };
  private readonly copyDestination?: GPUCopyExternalImageDestInfo;
  private readonly copySize?: [number, number];

  constructor(
    readonly device: GPUDevice,
    readonly config: MediaInput,
  ) {
    if (config.source.kind === "image") {
      const { width, height } = config.source;
      if (
        width > device.limits.maxTextureDimension2D ||
        height > device.limits.maxTextureDimension2D
      ) {
        throw new AvelError("media source exceeds maxTextureDimension2D");
      }
      this.texture = device.createTexture({
        size: [width, height],
        format: "rgba8unorm",
        usage:
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.view = this.texture.createView();
      this.copyDestination = {
        texture: this.texture,
        colorSpace: "srgb",
        premultipliedAlpha: false,
      };
      this.copySize = [width, height];
    }
  }

  destroy(): void {
    this.texture?.destroy();
    this.external = undefined;
    this.externalDescriptor.source = undefined as unknown as VideoFrame;
    this.copySource.source = undefined as unknown as ImageBitmap;
  }

  write(source: unknown): void {
    if (this.config.source.kind === "external") {
      if (!isExternal(source)) {
        throw new AvelError("external media input requires a VideoFrame or HTMLVideoElement");
      }
      this.externalDescriptor.source = source;
      this.external = this.device.importExternalTexture(this.externalDescriptor);
      return;
    }
    const image =
      (typeof ImageBitmap !== "undefined" && source instanceof ImageBitmap) ||
      (typeof HTMLCanvasElement !== "undefined" && source instanceof HTMLCanvasElement) ||
      (typeof OffscreenCanvas !== "undefined" && source instanceof OffscreenCanvas);
    if (!image) {
      throw new AvelError("image media input requires an ImageBitmap or canvas");
    }
    const { width, height } = this.config.source;
    if (source.width !== width || source.height !== height) {
      throw new AvelError(
        `image source is ${source.width}×${source.height}, expected ${width}×${height}`,
      );
    }
    this.copySource.source = source;
    this.device.queue.copyExternalImageToTexture(
      this.copySource,
      this.copyDestination!,
      this.copySize!,
    );
  }
}
