import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from aleph.model.ffn import FeedForward


class MoEStats(NamedTuple):
    load_balance_loss: jax.Array
    router_z_loss: jax.Array
    load_fraction: jax.Array
    overflow_fraction: jax.Array


@nnx.vmap(in_axes=(0, 0), out_axes=0)
def _apply_experts(expert: FeedForward, x: jax.Array) -> jax.Array:
    return expert(x)


class MoE(nnx.Module):
    def __init__(
        self,
        dim: int,
        n_experts: int,
        top_k: int,
        ffn_hidden: int,
        *,
        capacity_factor: float = 1.25,
        rngs: nnx.Rngs,
    ):
        assert top_k <= n_experts, "top_k cannot exceed n_experts"

        self.dim = dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        self.router = nnx.Linear(dim, n_experts, use_bias=False, rngs=rngs)

        @nnx.split_rngs(splits=n_experts)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def make_experts(rngs: nnx.Rngs) -> FeedForward:
            return FeedForward(dim, ffn_hidden, rngs=rngs)

        self.experts = make_experts(rngs)

    def __call__(self, x: jax.Array) -> tuple[jax.Array, MoEStats]:
        B, T, dim = x.shape
        E, k = self.n_experts, self.top_k
        N = B * T
        x_flat = x.reshape(N, dim)

        C = max(1, math.ceil(self.capacity_factor * N / E))

        logits = self.router(x_flat).astype(jnp.float32)
        probs = jax.nn.softmax(logits, axis=-1)

        gate_vals, expert_idx = jax.lax.top_k(probs, k)

        dispatch = jnp.zeros((N, E, C), dtype=jnp.float32)
        combine = jnp.zeros((N, E, C), dtype=jnp.float32)
        expert_counts = jnp.zeros((E,), dtype=jnp.int32)

        for i in range(k):
            idx_i = expert_idx[:, i]
            gate_i = gate_vals[:, i]
            mask_i = jax.nn.one_hot(idx_i, E, dtype=jnp.int32)

            prefix = jnp.cumsum(mask_i, axis=0) - mask_i
            prefix = prefix + expert_counts[None, :]
            slot_i = (prefix * mask_i).sum(axis=-1)
            expert_counts = expert_counts + mask_i.sum(axis=0)

            slot_oh = jax.nn.one_hot(slot_i, C, dtype=jnp.float32)
            contrib = mask_i.astype(jnp.float32)[:, :, None] * slot_oh[:, None, :]
            dispatch = dispatch + contrib
            combine = combine + contrib * gate_i[:, None, None]

        expert_in = jnp.einsum("nec,nd->ecd", dispatch, x_flat.astype(jnp.float32))
        expert_in = expert_in.astype(x.dtype)

        expert_out = _apply_experts(self.experts, expert_in)

        y = jnp.einsum("nec,ecd->nd", combine, expert_out.astype(jnp.float32))
        y = y.reshape(B, T, dim).astype(x.dtype)

        f = expert_counts.astype(jnp.float32) / (N * k)
        P = probs.mean(axis=0)
        load_balance_loss = E * jnp.sum(f * P)

        router_z_loss = jnp.mean(jax.nn.logsumexp(logits, axis=-1) ** 2)

        overflow_fraction = 1.0 - dispatch.sum() / (N * k)

        stats = MoEStats(load_balance_loss, router_z_loss, f, overflow_fraction)
        return y, stats
