# avel-export

Exports PyTorch models to the avel format for WebGPU inference. Requires Python
3.15 or later and a supported PyTorch installation.

```python
import torch
from avel_export import export

model = torch.nn.Sequential(torch.nn.Linear(4, 2), torch.nn.ReLU())
export(model, (torch.ones(1, 4),), "public/models/example")
```

Serve the exported directory over HTTP and load its URL with the
[avel runtime](https://github.com/tterrasson/avel#usage).

## Custom operators

Pass `custom_ops=("namespace.op.default",)` to `export` to preserve explicitly
allowed `torch.library.custom_op` calls through decomposition. Ops must be
functional, without aliases, and use f32 tensor inputs and one f32 tensor output.
Register a fake implementation with PyTorch for shape propagation. The runtime
needs a matching `registerOp` definition before compilation; see the complete
[Python and WGSL example](https://github.com/tterrasson/avel#custom-operators).

## Autoregressive decode

Keep fixed-capacity key and value buffers on the module. Write the new
block and run attention over the valid causal prefix.

```python
from avel_export.generation import cache_write, cached_attention

# In forward, q, k and v have shape [batch, heads, tokens, depth].
self.keys.copy_(cache_write(self.keys, k, position))
self.values.copy_(cache_write(self.values, v, position))
features = cached_attention(q, self.keys, self.values, position)
```

Use an integer scalar tensor for `position` and give the token dimension a
symbolic range including 1. Prefill starts at position 0. Decode supplies the
next free position and the one-token inputs. Use `sample_topk(logits, seed, k)`
to return sampled IDs from the model. Advance the scalar seed per run for a
different sample.

## Model census

Run `uv sync --extra census` and then
`uv run --extra census python -m avel_export.census --external --check census.json`
to include the optional Hugging Face and timm models. All models use random weights.
