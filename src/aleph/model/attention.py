import jax
import jax.numpy as jnp
from flax import nnx


def rope_tables(seq_len: int, head_dim: int, base: float = 10000.0):
    inv_freq = base ** (-jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)

    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, inv_freq)

    cos = jnp.concatenate([jnp.cos(angles), jnp.cos(angles)], axis=-1)
    sin = jnp.concatenate([jnp.sin(angles), jnp.sin(angles)], axis=-1)
    return cos, sin


def rotate_half(x: jax.Array) -> jax.Array:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    dtype = x.dtype
    x = x.astype(jnp.float32)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    out = x * cos + rotate_half(x) * sin
    return out.astype(dtype)


class Attention(nnx.Module):

    def __init__(
        self,
        dim: int,
        n_q_heads: int,
        n_kv_heads: int,
        *,
        head_dim: int | None = None,
        rope_base: float = 10000.0,
        rngs: nnx.Rngs,
    ):
        assert n_q_heads % n_kv_heads == 0, "n_q_heads must be a multiple of n_kv_heads"

        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim or dim // n_q_heads
        self.group_size = n_q_heads // n_kv_heads
        self.rope_base = rope_base

        q_out = n_q_heads * self.head_dim
        kv_out = n_kv_heads * self.head_dim
        self.w_q = nnx.Linear(dim, q_out, use_bias=False, rngs=rngs)
        self.w_k = nnx.Linear(dim, kv_out, use_bias=False, rngs=rngs)
        self.w_v = nnx.Linear(dim, kv_out, use_bias=False, rngs=rngs)
        self.w_o = nnx.Linear(q_out, dim, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        B, T, _ = x.shape
        H, G, d_h = self.n_q_heads, self.n_kv_heads, self.head_dim

        q = self.w_q(x).reshape(B, T, H, d_h)
        k = self.w_k(x).reshape(B, T, G, d_h)
        v = self.w_v(x).reshape(B, T, G, d_h)

        cos, sin = rope_tables(T, d_h, self.rope_base)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k = jnp.repeat(k, self.group_size, axis=2)
        v = jnp.repeat(v, self.group_size, axis=2)

        scores = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(d_h)

        causal = jnp.tril(jnp.ones((T, T), dtype=bool))
        scores = jnp.where(causal, scores, -jnp.inf)

        attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(v.dtype)

        out = jnp.einsum("bhts,bshd->bthd", attn, v)
        out = out.reshape(B, T, H * d_h)
        return self.w_o(out)
