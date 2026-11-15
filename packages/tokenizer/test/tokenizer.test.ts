import { expect, test } from "bun:test";
import { TextTokenizer } from "../src/index.js";

const base = {
  added_tokens: [],
  normalizer: null,
  pre_tokenizer: { type: "Whitespace" },
  post_processor: null,
  decoder: null,
};

test("runs WordPiece normalization, special tokens and left padding", () => {
  const tokenizer = new TextTokenizer({
    ...base,
    normalizer: { type: "Lowercase" },
    model: {
      type: "WordPiece",
      vocab: { "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, hello: 4, world: 5, "##s": 6 },
      unk_token: "[UNK]",
      continuing_subword_prefix: "##",
      max_input_chars_per_word: 100,
    },
    post_processor: { type: "BertProcessing", sep: ["[SEP]", 3], cls: ["[CLS]", 2] },
    decoder: { type: "WordPiece", prefix: "##", cleanup: true },
  });
  const encoded = tokenizer.encode("Hello Worlds", { pad: { length: 7, id: 0, side: "left" } });
  expect(encoded.inputIds).toEqual(Int32Array.of(0, 0, 2, 4, 5, 6, 3));
  expect(encoded.attentionMask).toEqual(Int32Array.of(0, 0, 1, 1, 1, 1, 1));
  expect(encoded.tokenTypeIds).toEqual(new Int32Array(7));
  expect(tokenizer.decode(BigInt64Array.of(4n, 5n, 6n))).toBe("hello worlds");
  expect(() => tokenizer.encode("hello", { pad: { length: 2, id: 0 } })).toThrow(RangeError);
});

test("uses BPE merges and returns runtime-ready arrays", () => {
  const tokenizer = new TextTokenizer({
    ...base,
    model: { type: "BPE", vocab: { a: 0, b: 1, ab: 2 }, merges: [["a", "b"]] },
  });
  expect(tokenizer.encode("ab a").inputIds).toEqual(Int32Array.of(2, 0));
  expect(() => tokenizer.decode([2147483648])).toThrow(RangeError);
  expect(() => tokenizer.encode("", { addSpecialTokens: false })).toThrow(RangeError);
});

test("encodes text pairs with token type ids and pads on the right by default", () => {
  const tokenizer = new TextTokenizer({
    ...base,
    model: {
      type: "WordPiece",
      vocab: { "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, hello: 4, world: 5 },
      unk_token: "[UNK]",
      continuing_subword_prefix: "##",
      max_input_chars_per_word: 100,
    },
    post_processor: { type: "BertProcessing", sep: ["[SEP]", 3], cls: ["[CLS]", 2] },
  });
  const encoded = tokenizer.encode("hello", { textPair: "world", pad: { length: 7, id: 0 } });
  expect(encoded.inputIds).toEqual(Int32Array.of(2, 4, 3, 5, 3, 0, 0));
  expect(encoded.attentionMask).toEqual(Int32Array.of(1, 1, 1, 1, 1, 0, 0));
  expect(encoded.tokenTypeIds).toEqual(Int32Array.of(0, 0, 0, 1, 1, 0, 0));
  expect(() => tokenizer.encode("hello", { pad: { length: 0x80000000, id: 0 } })).toThrow(
    RangeError,
  );
});

test("decode skips registered special tokens unless asked to keep them", () => {
  const tokenizer = new TextTokenizer({
    added_tokens: [
      {
        id: 7,
        content: "[MASK]",
        single_word: false,
        lstrip: false,
        rstrip: false,
        normalized: false,
        special: true,
      },
    ],
    normalizer: null,
    pre_tokenizer: { type: "Whitespace" },
    post_processor: null,
    decoder: null,
    model: {
      type: "WordPiece",
      vocab: { "[UNK]": 1, hello: 4, "[MASK]": 7 },
      unk_token: "[UNK]",
      continuing_subword_prefix: "##",
      max_input_chars_per_word: 100,
    },
  });
  expect(tokenizer.decode([7, 4])).toBe("hello");
  expect(tokenizer.decode([7, 4], false)).toContain("[MASK]");
});
