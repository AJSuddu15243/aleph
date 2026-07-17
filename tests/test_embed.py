import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from aleph.model.embed import Embedding


def _make(vocab=32, dim=8, tie=True, seed=0):
    return Embedding(vocab, dim, tie_embeddings=tie, rngs=nnx.Rngs(seed))


def test_encode_shape_and_is_lookup():
    m = _make(vocab=32, dim=8)
    ids = jnp.array([[1, 2, 3], [4, 5, 6]])
    out = m.encode(ids)
    assert out.shape == (2, 3, 8)
    np.testing.assert_array_equal(out[1, 1], m.tok.embedding[...][5])


def test_decode_shape_and_is_float32():
    m = _make(vocab=32, dim=8)
    h = jax.random.normal(jax.random.key(1), (2, 3, 8), dtype=jnp.bfloat16)
    logits = m.decode(h)
    assert logits.shape == (2, 3, 32)
    assert logits.dtype == jnp.float32


def test_tied_head_reuses_embedding_matrix():
    m = _make(vocab=32, dim=8, tie=True)
    assert m.head is None
    h = jax.random.normal(jax.random.key(2), (4, 8))
    expected = h @ m.tok.embedding[...].T
    np.testing.assert_allclose(m.decode(h), expected, rtol=1e-5, atol=1e-6)


def test_untied_head_is_independent():
    m = _make(vocab=32, dim=8, tie=False)
    assert m.head is not None
    assert m.head.kernel[...].shape == (8, 32)
    assert m.head.bias is None
    h = jax.random.normal(jax.random.key(3), (4, 8))
    expected = h @ m.head.kernel[...]
    np.testing.assert_allclose(m.decode(h), expected, rtol=1e-5, atol=1e-6)


def test_embedding_init_scale_is_small():
    m = _make(vocab=4096, dim=64)
    std = float(jnp.std(m.tok.embedding[...]))
    assert 0.01 < std < 0.03
