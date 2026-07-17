import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from aleph.model.ffn import FeedForward


def _make(dim=8, hidden=16, seed=0):
    return FeedForward(dim, hidden, rngs=nnx.Rngs(seed))


def test_shape_preserved():
    m = _make(dim=8, hidden=16)
    x = jnp.ones((2, 5, 8))
    assert m(x).shape == x.shape


def test_projection_shapes_and_bias_free():
    dim, hidden = 8, 16
    m = _make(dim, hidden)
    assert m.w_gate.kernel[...].shape == (dim, hidden)
    assert m.w_up.kernel[...].shape == (dim, hidden)
    assert m.w_down.kernel[...].shape == (hidden, dim)
    assert m.w_gate.bias is None
    assert m.w_up.bias is None
    assert m.w_down.bias is None


def test_zero_input_gives_zero_output():
    m = _make(dim=8, hidden=16)
    x = jnp.zeros((3, 8))
    np.testing.assert_allclose(m(x), jnp.zeros((3, 8)), atol=1e-6)


def test_matches_manual_swiglu():
    m = _make(dim=8, hidden=16)
    x = jax.random.normal(jax.random.key(1), (4, 8))

    w_gate = m.w_gate.kernel[...]
    w_up = m.w_up.kernel[...]
    w_down = m.w_down.kernel[...]

    gate = jax.nn.silu(x @ w_gate)
    expected = (gate * (x @ w_up)) @ w_down

    np.testing.assert_allclose(m(x), expected, rtol=1e-5, atol=1e-6)
