import { Tokenizer } from "@huggingface/tokenizers";

export interface EncodeSettings {
  addSpecialTokens?: boolean;
  textPair?: string;
  /** Pad to an exact model input length. An oversized encoding is an error. */
  pad?: { length: number; id: number; side?: "left" | "right" };
}

export interface EncodedText {
  inputIds: Int32Array;
  attentionMask: Int32Array;
  tokenTypeIds: Int32Array;
}

function tokenId(value: number | bigint): number {
  const id = Number(value);
  if (!Number.isInteger(id) || id < 0 || id > 0x7fffffff) {
    throw new RangeError("token ids must fit in a non-negative i32");
  }
  return id;
}

/** Loads a Hugging Face BPE or WordPiece pipeline from parsed JSON. */
export class TextTokenizer {
  private readonly tokenizer: Tokenizer;

  constructor(json: object, config: object = {}) {
    this.tokenizer = new Tokenizer(json, config);
  }

  encode(text: string, settings: EncodeSettings = {}): EncodedText {
    const { ids, attention_mask, token_type_ids } = this.tokenizer.encode(text, {
      add_special_tokens: settings.addSpecialTokens ?? true,
      text_pair: settings.textPair ?? null,
      return_token_type_ids: true,
    });
    const { pad } = settings;
    const length = pad?.length ?? ids.length;
    if (length === 0)
      throw new RangeError("the encoding is empty and models need at least one token");
    if (!Number.isSafeInteger(length) || length < ids.length || length > 0x7fffffff) {
      throw new RangeError("padding length must fit the encoded sequence and a signed 32-bit size");
    }
    const inputIds = new Int32Array(length);
    const attentionMask = new Int32Array(length);
    const tokenTypeIds = new Int32Array(length);
    const offset = pad?.side === "left" ? length - ids.length : 0;
    if (pad) inputIds.fill(tokenId(pad.id));
    for (let i = 0; i < ids.length; i++) {
      inputIds[offset + i] = tokenId(ids[i]!);
      attentionMask[offset + i] = attention_mask[i]!;
      tokenTypeIds[offset + i] = token_type_ids?.[i] ?? 0;
    }
    return { inputIds, attentionMask, tokenTypeIds };
  }

  /** Skips special tokens unless `skipSpecialTokens` is false. */
  decode(ids: ArrayLike<number | bigint>, skipSpecialTokens = true): string {
    return this.tokenizer.decode(Array.from(ids, tokenId), {
      skip_special_tokens: skipSpecialTokens,
    });
  }
}
