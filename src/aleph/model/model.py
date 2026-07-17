import jax
import jax.numpy as jnp
from flax import nnx

from aleph.config import ModelConfig
from aleph.model.block import Block
from aleph.model.embed import Embedding
from aleph.model.moe import MoEStats
from aleph.model.norm import RMSNorm


class Aleph(nnx.Module):
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
