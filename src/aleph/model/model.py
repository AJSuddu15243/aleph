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
        self.layers = nnx.data([Block(cfg, rngs=rngs) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.dim, eps=cfg.rms_eps)

    def __call__(self, ids: jax.Array) -> tuple[jax.Array, MoEStats]:
        x = self.embed.encode(ids)

        per_layer: list[MoEStats] = []
        for layer in self.layers:
            x, stats = layer(x)
            per_layer.append(stats)

        x = self.final_norm(x)
        logits = self.embed.decode(x)

        stats = jax.tree.map(lambda *s: jnp.stack(s), *per_layer)
        return logits, stats
