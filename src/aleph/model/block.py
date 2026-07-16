import jax
from flax import nnx

from aleph.config import ModelConfig
from aleph.model.attention import Attention
from aleph.model.moe import MoE, MoEStats
from aleph.model.norm import RMSNorm


class Block(nnx.Module):
    """One pre-norm decoder layer: an attention sublayer then an MoE sublayer.

    Pre-norm means each sublayer normalizes a *copy* of the residual stream,
    computes, and adds its result back — the stream itself is never normalized
    in place. That leaves a clean identity path from input to output (the
    gradient highway that lets deep stacks train), while every sublayer still
    sees a normalized input.

        x ─┬─ RMSNorm ─ Attention ─┬─(+)─┬─ RMSNorm ─ MoE ─┬─(+)─→ out
           └───────────────────────┘     └────────────────┘

    Shape is preserved end to end: (B, T, dim) in, (B, T, dim) out — which is
    exactly why model.py can stack n_layers of these.

    The MoE sublayer returns per-layer routing stats (load-balance loss, z-loss,
    expert load, overflow) alongside its output. The block threads them out so
    model.py can collect one MoEStats per layer and fold the aux losses into the
    training objective. Attention has no such side output, so only the MoE call
    is unpacked.
    """

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
        # Attention sublayer: norm a copy, attend, add the delta back to the stream.
        x = x + self.attn(self.attn_norm(x))
        # MoE sublayer: same shape in/out; catch the routing stats to thread up.
        moe_out, stats = self.moe(self.moe_norm(x))
        x = x + moe_out
        return x, stats
