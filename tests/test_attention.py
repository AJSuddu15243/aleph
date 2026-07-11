import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from aleph.model.attention import Attention, apply_rope, rope_tables


def _make(dim=64, n_q_heads=8, n_kv_heads=2, seed=0):
    return Attention(dim, n_q_heads, n_kv_heads, rngs=nnx.Rngs(seed))


def test_shape_preserved():
    m = _make()
    x = jnp.ones((2, 5, 64))          # [batch, tokens, dim]
    assert m(x).shape == x.shape


def test_projection_shapes_and_bias_free():
    dim, H, G, d_h = 64, 8, 2, 8
    m = _make(dim, H, G)
    # K/V matrices are narrower than Q: only G heads' worth of head_dim.
    assert m.w_q.kernel[...].shape == (dim, H * d_h)
    assert m.w_k.kernel[...].shape == (dim, G * d_h)
    assert m.w_v.kernel[...].shape == (dim, G * d_h)
    assert m.w_o.kernel[...].shape == (H * d_h, dim)
    # Bias-free (Llama-style).
    for w in (m.w_q, m.w_k, m.w_v, m.w_o):
        assert w.bias is None


def test_mha_is_special_case():
    # n_kv_heads == n_q_heads is plain multi-head attention: group size 1.
    m = _make(dim=64, n_q_heads=8, n_kv_heads=8)
    assert m.group_size == 1
    x = jnp.ones((2, 5, 64))
    assert m(x).shape == x.shape


def test_causal_no_future_leak():
    # Changing a *future* token must not change the current token's output.
    # This is the whole point of the causal mask.
    m = _make()
    key = jax.random.key(0)
    x = jax.random.normal(key, (1, 6, 64))

    y_full = m(x)

    # Corrupt only the last token, recompute, and check earlier outputs are
    # untouched. If the mask leaked, token 0..4 would shift.
    x2 = x.at[:, -1, :].set(jax.random.normal(jax.random.key(1), (1, 64)))
    y_corrupt = m(x2)

    np.testing.assert_allclose(y_full[:, :-1], y_corrupt[:, :-1], rtol=1e-5, atol=1e-5)


def test_rope_preserves_norm():
    # A rotation changes direction, never length. Each head vector's L2 norm
    # must survive apply_rope unchanged.
    cos, sin = rope_tables(seq_len=7, head_dim=8)
    x = jax.random.normal(jax.random.key(2), (2, 7, 4, 8))  # (B, T, heads, d_h)
    y = apply_rope(x, cos, sin)
    np.testing.assert_allclose(
        jnp.linalg.norm(x, axis=-1), jnp.linalg.norm(y, axis=-1), rtol=1e-5, atol=1e-5
    )


def test_rope_position_zero_is_identity():
    # Position 0 rotates by angle 0 → no change. Good sanity anchor.
    cos, sin = rope_tables(seq_len=4, head_dim=8)
    x = jax.random.normal(jax.random.key(3), (1, 4, 2, 8))
    y = apply_rope(x, cos, sin)
    np.testing.assert_allclose(y[:, 0], x[:, 0], rtol=1e-5, atol=1e-6)
