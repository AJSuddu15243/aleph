import jax
import jax.numpy as jnp
import numpy as np

from aleph.model.norm import RMSNorm


def test_shape_preserved():
    m = RMSNorm(dim=8)
    x = jnp.ones((2, 3, 8))
    assert m(x).shape == x.shape


def test_output_has_unit_rms():
    m = RMSNorm(dim=16, eps=0.0)
    x = jax.random.normal(jax.random.key(0), (4, 16)) * 5.0
    y = m(x)
    rms = jnp.sqrt(jnp.mean(jnp.square(y), axis=-1))
    np.testing.assert_allclose(rms, jnp.ones(4), rtol=1e-4)


def test_matches_manual_computation():
    m = RMSNorm(dim=3, eps=0.0)
    x = jnp.array([[3.0, 0.0, 4.0]])
    expected = x / jnp.sqrt(25.0 / 3.0)
    np.testing.assert_allclose(m(x), expected, rtol=1e-5)


def test_scale_is_learnable():
    m = RMSNorm(dim=4)
    assert m.scale[...].shape == (4,)
    np.testing.assert_array_equal(m.scale[...], jnp.ones(4))
