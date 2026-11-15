# @avel/tokenizer

Optional browser and Node.js text encoding for avel models. Use the
`tokenizer.json` exported with the original Hugging Face model. Pass its
`tokenizer_config.json` when the model uses extra tokenizer metadata.

```ts
import { TextTokenizer } from "@avel/tokenizer";

const json = await fetch("/model/tokenizer.json").then((response) => response.json());
const tokenizer = new TextTokenizer(json);
const { inputIds, attentionMask, tokenTypeIds } = tokenizer.encode("Hello world");
const text = tokenizer.decode(inputIds);
```

Pass the arrays to the corresponding inputs of the model and use
`[1, inputIds.length]` for a batched sequence. All three arrays are
`Int32Array` values, which i32 and i64 inputs accept. For a bool mask input,
pass `new Uint32Array(attentionMask.buffer)`. An empty encoding raises an
error. `decode` skips special tokens unless its second argument is `false`.
Use `pad: { length, id, side }` with the model's padding ID when compiling
for an exact length. An encoding larger than that length raises an error.
