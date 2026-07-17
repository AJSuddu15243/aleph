import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from aleph.model.moe import MoE, MoEStats


def _make(dim=16, n_experts=4, top_k=1, hidden=32, capacity_factor=1.25, seed=0):
    return MoE(dim, n_experts, top_k, hidden, capacity_factor=capacity_factor, rngs=nnx.Rngs(seed))


def test_shape_preserved():
    m = _make()
    x = jax.random.normal(jax.random.key(0), (2, 8, 16))
    y, stats = m(x)
    assert y.shape == x.shape
    assert isinstance(stats, MoEStats)


def test_router_shape_and_bias_free():
    dim, E = 16, 4
    m = _make(dim=dim, n_experts=E)
    assert m.router.kernel[...].shape == (dim, E)
    assert m.router.bias is None


def test_experts_are_stacked_ensemble():
    dim, E, hidden = 16, 4, 32
    m = _make(dim=dim, n_experts=E, hidden=hidden)
    assert m.experts.w_gate.kernel[...].shape == (E, dim, hidden)
    assert m.experts.w_down.kernel[...].shape == (E, hidden, dim)


def test_load_fraction_is_a_distribution():
    m = _make(n_experts=4, top_k=1)
    x = jax.random.normal(jax.random.key(3), (4, 16, 16))
    _, stats = m(x)
    assert stats.load_fraction.shape == (4,)
    np.testing.assert_allclose(stats.load_fraction.sum(), 1.0, atol=1e-5)


def test_single_expert_equals_plain_swiglu():
    dim, hidden = 16, 32
    m = _make(dim=dim, n_experts=1, top_k=1, hidden=hidden, capacity_factor=2.0)
    x = jax.random.normal(jax.random.key(4), (2, 8, dim))
    y, stats = m(x)

    wg = m.experts.w_gate.kernel[...][0]
    wu = m.experts.w_up.kernel[...][0]
    wd = m.experts.w_down.kernel[...][0]
    xf = x.reshape(-1, dim)
    expected = (jax.nn.silu(xf @ wg) * (xf @ wu)) @ wd
    expected = expected.reshape(x.shape)

    np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(stats.load_fraction, jnp.ones(1), atol=1e-6)
    np.testing.assert_allclose(stats.overflow_fraction, 0.0, atol=1e-6)


def test_capacity_drops_tokens():
    m = _make(n_experts=2, top_k=1, capacity_factor=0.1)
    x = jax.random.normal(jax.random.key(5), (2, 16, 16))
    y, stats = m(x)
    assert float(stats.overflow_fraction) > 0.0
    assert y.shape == x.shape


def test_top2_routing_runs():
    m = _make(n_experts=4, top_k=2)
    x = jax.random.normal(jax.random.key(6), (2, 8, 16))
    y, stats = m(x)
    assert y.shape == x.shape
    np.testing.assert_allclose(stats.load_fraction.sum(), 1.0, atol=1e-5)


def test_jits_and_is_differentiable():
    m = _make(n_experts=4, top_k=1)
    x = jax.random.normal(jax.random.key(7), (2, 8, 16))

    def loss_fn(model, x):
        y, stats = model(x)
        return y.sum() + stats.load_balance_loss + stats.router_z_loss

    val = nnx.jit(loss_fn)(m, x)
    assert jnp.isfinite(val)

    grads = nnx.grad(loss_fn)(m, x)
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves, "expected gradient leaves"
    assert all(jnp.all(jnp.isfinite(g)) for g in leaves)
    assert sum(float(jnp.sum(jnp.abs(g))) for g in leaves) > 0.0
