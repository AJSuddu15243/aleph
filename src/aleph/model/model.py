import jax
import jax.numpy as jnp
from flax import nnx

from aleph.config import ModelConfig
from aleph.model.block import Block
from aleph.model.embed import Embedding
from aleph.model.moe import MoEStats
from aleph.model.norm import RMSNorm


class Aleph(nnx.Module):
    """The full MoE decoder: embed → n_layers of Block → final norm → LM head.

    Pure forward. It returns raw logits and the per-layer routing stats and stops
    there — no cross-entropy, no aux-loss weighting. The MoE coefficients live in
    TrainConfig, so folding load-balance / z-loss into the objective is the
    training loop's job, not the model's.

        ids (B,T) int
          │  embed.encode      gather rows              (B,T,dim)
          ▼
         Block × n_layers      each → (x, MoEStats)     (B,T,dim)
          │  keep x, collect one MoEStats per layer
          ▼
         final_norm            the pre-decode RMSNorm   (B,T,dim)
          │
          ▼
         embed.decode          h @ Eᵀ, cast to fp32     (B,T,vocab)

    The final norm is load-bearing, not decoration: pre-norm blocks add each
    sublayer's output straight back into the residual stream and never normalize
    the stream itself, so the activations arriving at the top are un-normed. Feed
    those raw into the tied head — whose matrix embed.py deliberately inits small
    (std 0.02) — and the first-step logits would be hot and high-variance. One
    RMSNorm before decode fixes the scale the head is counting on.

    The block stack is a plain Python list walked by a Python for-loop, so the
    graph unrolls to n_layers copies. Fine at n_layers=8; the later speedup is to
    fold identical layers into one `nnx.scan`ned block (one compiled body, layer
    index as the scan axis) once compile time actually bites.
    """

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.embed = Embedding(
            cfg.vocab_size,
            cfg.dim,
            tie_embeddings=cfg.tie_embeddings,
            rngs=rngs,
        )
        # nnx.data marks the list as a pytree (data) attribute so each Block's
        # params are tracked state; a bare list would be treated as static config.
        self.layers = nnx.data([Block(cfg, rngs=rngs) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.dim, eps=cfg.rms_eps)

    def __call__(self, ids: jax.Array) -> tuple[jax.Array, MoEStats]:
        # Token ids → vectors. Position is injected later by RoPE inside attention,
        # so nothing positional happens here.
        x = self.embed.encode(ids)                       # (B, T, dim)

        # Walk the stack. Each block returns the updated stream plus its own MoE
        # routing stats; keep the stream, stash one MoEStats per layer.
        per_layer: list[MoEStats] = []
        for layer in self.layers:
            x, stats = layer(x)                          # (B, T, dim), MoEStats
            per_layer.append(stats)

        # Normalize the un-normed stream once before projecting to vocab logits.
        x = self.final_norm(x)                           # (B, T, dim)
        logits = self.embed.decode(x)                    # (B, T, vocab), fp32

        # Stack the per-layer stats along a new leading n_layers axis. MoEStats is
        # a NamedTuple (a pytree), so one tree.map lifts every field at once:
        #   load_balance_loss/router_z_loss/overflow_fraction → (n_layers,)
        #   load_fraction                                      → (n_layers, E)
        # The loop then sums/means across layer 0 in a single reduction instead of
        # bookkeeping per layer.
        stats = jax.tree.map(lambda *s: jnp.stack(s), *per_layer)
        return logits, stats
