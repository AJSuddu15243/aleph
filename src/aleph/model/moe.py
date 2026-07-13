import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from aleph.model.ffn import FeedForward


class MoEStats(NamedTuple):
    """Side outputs of one MoE layer, threaded up to the training loop.

    The two losses come back *unweighted* — TrainConfig's coefficients scale them
    where the total loss is assembled. load_fraction is for logging only: falling
    task loss alone can't detect router collapse (all tokens stampeding to one
    expert), so we watch the per-expert load from step one.
    """

    load_balance_loss: jax.Array   # scalar, unweighted (× aux_loss_coef in the loss)
    router_z_loss: jax.Array       # scalar, unweighted (× router_z_loss_coef)
    load_fraction: jax.Array       # (E,)   fraction of token-slots routed to each expert
    overflow_fraction: jax.Array   # scalar fraction of token-slots dropped by capacity


# nnx.vmap maps expert e over its own slice of the dispatched batch. `experts`
# carries a leading E axis on every param (built by the ensemble ctor below), and
# `x` is (E, C, dim); mapping axis 0 of both runs all experts as one batched call.
@nnx.vmap(in_axes=(0, 0), out_axes=0)
def _apply_experts(expert: FeedForward, x: jax.Array) -> jax.Array:
    return expert(x)


class MoE(nnx.Module):
    """Top-k Mixture-of-Experts feed-forward layer with capacity + token dropping.

    A drop-in replacement for a block's single FeedForward sublayer: instead of
    one FFN every token pays for, we keep E experts and let a tiny learned router
    send each token to only its top-k. The residual structure around it is
    unchanged — only what happens inside the FFN box changes.

    Exactly two sets of learned weights live here: the router (dim → E) and the E
    experts (SwiGLU FFNs from ffn.py). Everything between them — dispatch and
    combine — is weightless: a one-hot routing tensor and two einsums that move
    tokens to experts and blend the results back. The router decides; dispatch and
    combine are the conveyor belts.

    Args:
        dim:             model / residual width.
        n_experts:       number of experts, E. Only ``top_k`` of them fire per token.
        top_k:           experts fired per token. 1 → Switch-style; 2 → Mixtral-style.
        ffn_hidden:      SwiGLU inner width of each expert (see ffn.py).
        capacity_factor: slots per expert = ceil(capacity_factor · N / E), N = tokens.
                         >1 leaves slack; tokens past an expert's slots are dropped.
    """

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

        # The router: the layer's only routing weights. dim → E logits, bias-free
        # (like every other projection in the model). Softmax of one logit is 1.0,
        # so n_experts == 1 makes the whole layer identical to a plain FeedForward
        # — the debugging knob the build plan keeps in reserve.
        self.router = nnx.Linear(dim, n_experts, use_bias=False, rngs=rngs)

        # The experts: E independent SwiGLU FFNs, created as ONE ensemble whose
        # params all gained a leading E axis. split_rngs gives each expert its own
        # init; nnx.vmap stacks them. This is ffn.py reused, restructured to batch.
        @nnx.split_rngs(splits=n_experts)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def make_experts(rngs: nnx.Rngs) -> FeedForward:
            return FeedForward(dim, ffn_hidden, rngs=rngs)

        self.experts = make_experts(rngs)

    def __call__(self, x: jax.Array) -> tuple[jax.Array, MoEStats]:
        # x: (B, T, dim). Flatten the batch so every token is one row — routing
        # is per-token and doesn't care which sequence a token came from.
        B, T, dim = x.shape
        E, k = self.n_experts, self.top_k
        N = B * T
        x_flat = x.reshape(N, dim)                       # (N, dim)

        # Capacity: how many tokens each expert has room for. Static given fixed
        # batch/seq, so the (E, C, dim) buffers below have compile-time shapes.
        C = max(1, math.ceil(self.capacity_factor * N / E))

        # Router: score the experts for every token, then softmax to a
        # distribution. fp32 for the softmax and every downstream loss —
        # exponentials in bf16 lose accuracy and can destabilize the aux losses.
        logits = self.router(x_flat).astype(jnp.float32)  # (N, E)
        probs = jax.nn.softmax(logits, axis=-1)           # (N, E)

        # Top-k: collapse the soft distribution to the k winning experts + their
        # gate weights (the winning probabilities). For top-1 this is the argmax.
        gate_vals, expert_idx = jax.lax.top_k(probs, k)   # (N, k), (N, k)

        # Build dispatch / combine (weightless plumbing).
        #   dispatch[n,e,c] = 1  if token n is the c-th token assigned to expert e.
        #   combine[n,e,c]  = its gate weight at that same slot (0 elsewhere).
        # We accumulate over the k choices, tracking a running per-expert count so
        # each expert's slots fill in token order across all choices.
        dispatch = jnp.zeros((N, E, C), dtype=jnp.float32)
        combine = jnp.zeros((N, E, C), dtype=jnp.float32)
        expert_counts = jnp.zeros((E,), dtype=jnp.int32)  # tokens placed per expert so far

        for i in range(k):                                # k is a static int → loop unrolls
            idx_i = expert_idx[:, i]                       # (N,) chosen expert for this slot
            gate_i = gate_vals[:, i]                       # (N,) its probability
            mask_i = jax.nn.one_hot(idx_i, E, dtype=jnp.int32)   # (N, E)

            # 0-indexed slot within the chosen expert = tokens already placed there.
            # cumsum − mask is the exclusive prefix count; add the offset carried
            # from earlier choices so slots don't collide across the k passes.
            prefix = jnp.cumsum(mask_i, axis=0) - mask_i          # (N, E)
            prefix = prefix + expert_counts[None, :]
            slot_i = (prefix * mask_i).sum(axis=-1)               # (N,) slot for chosen expert
            expert_counts = expert_counts + mask_i.sum(axis=0)    # (E,) update running totals

            # one_hot returns all-zeros for slot ≥ C, so overflow tokens are
            # dropped automatically — no explicit capacity mask needed.
            slot_oh = jax.nn.one_hot(slot_i, C, dtype=jnp.float32)      # (N, C)
            contrib = mask_i.astype(jnp.float32)[:, :, None] * slot_oh[:, None, :]  # (N, E, C)
            dispatch = dispatch + contrib
            combine = combine + contrib * gate_i[:, None, None]

        # Dispatch: scatter tokens into per-expert buckets. (N,E,C)·(N,dim) →
        # (E,C,dim). Cast to x's dtype so the heavy expert matmuls run in the
        # compute dtype (bf16 in training).
        expert_in = jnp.einsum("nec,nd->ecd", dispatch, x_flat.astype(jnp.float32))
        expert_in = expert_in.astype(x.dtype)             # (E, C, dim)

        # Experts: all E run as one batched SwiGLU.
        expert_out = _apply_experts(self.experts, expert_in)   # (E, C, dim)

        # Combine: gather results back, weighted by the gate. Because `combine`
        # carries the gate weight, this single einsum both pulls each token's
        # output from its expert AND scales it by how much the router wanted that
        # expert. Dropped tokens have all-zero combine rows → 0 here, so the
        # block's residual carries them through unchanged. fp32 keeps the router
        # gradient (which flows through the gate weights) clean.
        y = jnp.einsum("nec,ecd->nd", combine, expert_out.astype(jnp.float32))
        y = y.reshape(B, T, dim).astype(x.dtype)          # (B, T, dim)

        # Load-balance loss (Switch eq. 4):  L = E · Σ_e f_e · P_e
        #   f_e = fraction of token-slots the router SENT to expert e (hard, no grad)
        #   P_e = mean router probability for expert e               (soft, carries grad)
        # Minimized when load is uniform; punishes an expert that is both
        # over-chosen and over-confident. Gradient flows through P, nudging the
        # router toward an even spread. f_e falls out of expert_counts for free.
        f = expert_counts.astype(jnp.float32) / (N * k)   # (E,), Σ = 1
        P = probs.mean(axis=0)                            # (E,), Σ = 1
        load_balance_loss = E * jnp.sum(f * P)

        # Router z-loss (ST-MoE): keep logits from growing large in magnitude.
        router_z_loss = jnp.mean(jax.nn.logsumexp(logits, axis=-1) ** 2)

        # Overflow: share of the N·k token-slots that capacity dropped. A health
        # gauge — high overflow means capacity_factor is too tight or load is skewed.
        overflow_fraction = 1.0 - dispatch.sum() / (N * k)

        stats = MoEStats(load_balance_loss, router_z_loss, f, overflow_fraction)
        return y, stats
