import jax
import jax.numpy as jnp
from flax import nnx


# ─────────────────────────────────────────────────────────────────────────────
# RoPE — Rotary Position Embedding
#
# The problem: attention's q·k dot product is order-blind. Shuffle the tokens
# and the scores are identical, so the model has no idea which token came first.
# We have to inject position somehow.
#
# RoPE's trick: *rotate* each query and key vector by an angle that grows with
# its position in the sequence. Token 0 gets rotated a little, token 5 more,
# token 100 a lot. The beautiful part is what happens in the dot product:
#
#       (rotate q by angle m) · (rotate k by angle n)  depends only on (m − n)
#
# So the score between two tokens ends up encoding their *relative* distance,
# which falls out for free — we never add a position vector, we just spin q & k.
# ─────────────────────────────────────────────────────────────────────────────


def rope_tables(seq_len: int, head_dim: int, base: float = 10000.0):
    """Precompute the cos/sin rotation tables for positions 0..seq_len-1.

    We don't rotate all `head_dim` numbers by one angle. Instead we pair the
    dimensions up and give each pair its own rotation *speed*: early pairs spin
    fast (they track fine, local position), later pairs spin slowly (they track
    coarse, long-range position). `base` sets how fast those speeds fall off —
    raise it to stretch RoPE over longer contexts.
    """
    # One frequency per dimension-pair. θ_i = base^(-2i/head_dim), i = 0,1,2,...
    # Small i → θ near 1 → fast rotation.  Large i → θ tiny → slow rotation.
    inv_freq = base ** (-jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)  # (head_dim/2,)

    pos = jnp.arange(seq_len, dtype=jnp.float32)              # (seq_len,)
    angles = jnp.outer(pos, inv_freq)                         # (seq_len, head_dim/2): angle[p,i] = p·θ_i

    # We duplicate the table so it lines up with `rotate_half` below, which
    # treats the vector as [first_half, second_half]. Each half needs the same
    # set of angles, hence the concat.
    cos = jnp.concatenate([jnp.cos(angles), jnp.cos(angles)], axis=-1)  # (seq_len, head_dim)
    sin = jnp.concatenate([jnp.sin(angles), jnp.sin(angles)], axis=-1)  # (seq_len, head_dim)
    return cos, sin


def rotate_half(x: jax.Array) -> jax.Array:
    """Rotate the vector 90° in each dimension-pair: [a, b] → [-b, a].

    This is the partner of the cos/sin tables. A 2-D rotation by θ is
        x·cos(θ) + rotate_half(x)·sin(θ)
    and doing it this "split in half" way (instead of interleaving pairs) is
    just a layout choice — it's the Llama/HF convention, and it vectorizes
    cleanly because the whole first half maps to the whole second half.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Rotate q or k by position.  x: (B, T, n_heads, head_dim)."""
    dtype = x.dtype
    x = x.astype(jnp.float32)                    # rotate in fp32 — sin/cos in bf16 lose precision
    # cos/sin are (T, head_dim); add batch and head axes so they broadcast over
    # every sequence and every head (all heads share the same rotation schedule).
    cos = cos[None, :, None, :]                  # (1, T, 1, head_dim)
    sin = sin[None, :, None, :]
    out = x * cos + rotate_half(x) * sin
    return out.astype(dtype)                      # back to the compute dtype (e.g. bf16)


# ─────────────────────────────────────────────────────────────────────────────
# Grouped-Query Attention
# ─────────────────────────────────────────────────────────────────────────────


class Attention(nnx.Module):
    """Causal grouped-query self-attention with RoPE.

    One knob, `n_kv_heads`, spans the whole family:
        n_kv_heads == n_q_heads   → plain multi-head attention (MHA)
        n_kv_heads == 1           → multi-query attention (MQA)
        in between                → grouped-query attention (GQA)

    Every query head asks its own question (its own W_q), but several query
    heads *share* one key/value table. Sharing K/V is what shrinks the KV cache
    at decode time — the thing we stream from memory for every generated token —
    without touching the queries, where the model's expressive power actually
    lives.

    Args:
        dim:         model / residual width.
        n_q_heads:   number of query heads (H).
        n_kv_heads:  number of shared key/value heads (G). Must divide n_q_heads.
        head_dim:    width of one head. Defaults to dim // n_q_heads.
        rope_base:   RoPE frequency base (raise for longer contexts).
    """

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
        self.group_size = n_q_heads // n_kv_heads   # how many query heads reuse one K/V head
        self.rope_base = rope_base

        # The four projections. Note the K/V matrices are *narrower* than Q:
        # they output only n_kv_heads worth of head_dim, not n_q_heads. That
        # narrowness is where GQA's parameter + KV-cache savings physically sit.
        q_out = n_q_heads * self.head_dim
        kv_out = n_kv_heads * self.head_dim
        self.w_q = nnx.Linear(dim, q_out, use_bias=False, rngs=rngs)   # dim → H·d_h
        self.w_k = nnx.Linear(dim, kv_out, use_bias=False, rngs=rngs)  # dim → G·d_h
        self.w_v = nnx.Linear(dim, kv_out, use_bias=False, rngs=rngs)  # dim → G·d_h
        self.w_o = nnx.Linear(q_out, dim, use_bias=False, rngs=rngs)   # H·d_h → dim  (mix heads back)

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (B, T, dim).  B = batch, T = sequence length.
        B, T, _ = x.shape
        H, G, d_h = self.n_q_heads, self.n_kv_heads, self.head_dim

        # 1) Project the residual stream into queries, keys, values, then split
        #    the flat feature axis into (heads, head_dim). After this, each head
        #    has its own d_h-wide q/k/v slice to work with.
        q = self.w_q(x).reshape(B, T, H, d_h)   # (B, T, H, d_h)  — H distinct questions
        k = self.w_k(x).reshape(B, T, G, d_h)   # (B, T, G, d_h)  — G shared keys
        v = self.w_v(x).reshape(B, T, G, d_h)   # (B, T, G, d_h)  — G shared values

        # 2) Inject position by rotating q and k.  v is left alone — values carry
        #    *content* to hand over, not *where* the token is.
        cos, sin = rope_tables(T, d_h, self.rope_base)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # 3) GQA expansion: give each of the G key/value heads to its whole group
        #    of query heads. This repeat is conceptually just "broadcast a shared
        #    K/V head across its group"; a fused kernel would skip the copy, but
        #    an explicit repeat keeps the einsum below dead simple.
        k = jnp.repeat(k, self.group_size, axis=2)   # (B, T, H, d_h)
        v = jnp.repeat(v, self.group_size, axis=2)   # (B, T, H, d_h)

        # 4) Scores: for every head, dot each query against every key.
        #    einsum axes: b=batch, t=query pos, s=key pos, h=head, d=head_dim.
        #    Dividing by √d_h keeps the dot products from blowing up as d_h grows,
        #    which would otherwise push softmax into a near-one-hot, low-gradient
        #    corner.
        scores = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(d_h)   # (B, H, T, T)

        # 5) Causal mask: token t may attend to key s only if s <= t (its own
        #    past and itself). Future keys get −inf so softmax sends them to 0.
        #    This is what makes it a *decoder* — no peeking ahead at the answer.
        causal = jnp.tril(jnp.ones((T, T), dtype=bool))       # lower-triangular True
        scores = jnp.where(causal, scores, -jnp.inf)          # broadcasts over (B, H)

        # 6) Softmax each query's row into a probability distribution over keys.
        #    Done in fp32: softmax exponentiates, and bf16 there loses accuracy.
        attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(v.dtype)

        # 7) Blend: each token's output is the attention-weighted sum of the
        #    values it attended to. Then flatten heads back into one vector and
        #    let W_o mix information across heads.
        out = jnp.einsum("bhts,bshd->bthd", attn, v)          # (B, T, H, d_h)
        out = out.reshape(B, T, H * d_h)                      # (B, T, H·d_h)
        return self.w_o(out)                                  # (B, T, dim)
