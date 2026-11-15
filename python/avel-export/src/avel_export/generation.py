"""Exportable primitives for fixed-capacity autoregressive decoding."""

import torch
from torch.nn.functional import scaled_dot_product_attention


@torch.library.custom_op("avel::cache_write", mutates_args=())
def cache_write(cache: torch.Tensor, patch: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    """Returns the cache with the new block written along the sequence axis."""
    axis = cache.ndim - 2
    start = position.reshape(()).clamp(0, cache.shape[axis] - patch.shape[axis])
    indices = torch.arange(patch.shape[axis], device=cache.device) + start
    return cache.index_copy(axis, indices, patch)


@cache_write.register_fake
def _cache_write_fake(cache, patch, position):
    return torch.empty_like(cache)


@torch.library.custom_op("avel::cached_attention", mutates_args=())
def cached_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    position: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Causal attention starting at position over the updated key/value caches."""
    q_len, capacity = query.shape[-2], key.shape[-2]
    start = position.reshape(()).clamp(0, capacity - q_len)
    queries = torch.arange(q_len, device=query.device) + start
    keys = torch.arange(capacity, device=query.device)
    return scaled_dot_product_attention(query, key, value, keys <= queries[:, None], scale=scale)


@cached_attention.register_fake
def _cached_attention_fake(query, key, value, position, scale=None):
    return query.new_empty((*query.shape[:-1], value.shape[-1]))


@torch.library.custom_op("avel::sample_candidates", mutates_args=())
def sample_candidates(
    logits: torch.Tensor, indices: torch.Tensor, seed: torch.Tensor
) -> torch.Tensor:
    """Samples a candidate per row using a scalar seed and a PCG hash."""
    rows = torch.arange(logits.numel() // logits.shape[-1], device=logits.device)
    state = ((seed.reshape(()).long() + rows * 0x9E3779B9) * 747796405 + 2891336453) & 0xFFFFFFFF
    word = (((state >> ((state >> 28) + 4)) ^ state) * 277803737) & 0xFFFFFFFF
    fraction = (((word >> 22) ^ word) >> 8).float() * (1.0 / 16777216.0)
    maximum = logits.amax(-1, keepdim=True)
    weights = (logits - maximum).exp()
    cumulative = weights.cumsum(-1)
    threshold = fraction.reshape(logits.shape[:-1]) * cumulative[..., -1]
    selected = (cumulative <= threshold.unsqueeze(-1)).sum(-1)
    last = logits.shape[-1] - 1 - (weights.flip((-1,)) > 0).int().argmax(-1)
    selected = torch.where(maximum.squeeze(-1).isfinite(), selected.minimum(last), 0)
    return indices.gather(-1, selected.unsqueeze(-1)).squeeze(-1).long()


@sample_candidates.register_fake
def _sample_candidates_fake(logits, indices, seed):
    return indices.new_empty(logits.shape[:-1], dtype=torch.int64)


def sample_topk(logits: torch.Tensor, seed: torch.Tensor, k: int) -> torch.Tensor:
    """Runs top-k and categorical sampling, returning only the sampled ids."""
    scores, indices = torch.topk(logits, k, dim=-1)
    return sample_candidates(scores, indices, seed)


@torch.library.custom_op("avel::rotary", mutates_args=())
def rotary(x: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor) -> torch.Tensor:
    """Split-half rotary embeddings with broadcastable sine and cosine tables."""
    lower, upper = x.chunk(2, dim=-1)
    return x * cosine + torch.cat((-upper, lower), dim=-1) * sine


@rotary.register_fake
def _rotary_fake(x, cosine, sine):
    return torch.empty_like(x)
