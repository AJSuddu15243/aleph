import jax
from flax import nnx

from aleph.config import ModelConfig
from aleph.model.attention import Attention
from aleph.model.moe import MoE, MoEStats
from aleph.model.norm import RMSNorm


class Block(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.attn_norm = RMSNorm(cfg.dim, eps=cfg.rms_eps)
        self.attn = Attention(
            cfg.dim,
            cfg.n_q_heads,
            cfg.n_kv_heads,
            head_dim=cfg.head_dim,
            rope_base=cfg.rope_base,
            rngs=rngs,
        )
        self.moe_norm = RMSNorm(cfg.dim, eps=cfg.rms_eps)
        self.moe = MoE(
            cfg.dim,
            cfg.n_experts,
            cfg.top_k,
            cfg.ffn_hidden,
            capacity_factor=cfg.capacity_factor,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> tuple[jax.Array, MoEStats]:
        x = x + self.attn(self.attn_norm(x))
        moe_out, stats = self.moe(self.moe_norm(x))
        x = x + moe_out
        return x, stats
