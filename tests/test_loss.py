import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from aleph.config import ModelConfig, TrainConfig
from aleph.model import Aleph
from aleph.model.moe import MoEStats
from aleph.train.loss import LossMetrics, compute_loss


def _stats(n_layers=4, n_experts=8, seed=0):
    k = jax.random.key(seed)
    k1, k2, k3 = jax.random.split(k, 3)
    load = jax.random.uniform(k1, (n_layers, n_experts)) + 0.1
    load = load / load.sum(axis=-1, keepdims=True)
    return MoEStats(
        load_balance_loss=jax.random.uniform(k2, (n_layers,)),
        router_z_loss=jax.random.uniform(k3, (n_layers,)),
        load_fraction=load,
        overflow_fraction=jnp.full((n_layers,), 0.03),
    )


def _logits_targets(B=2, T=8, V=32, seed=1):
    k = jax.random.key(seed)
    k1, k2 = jax.random.split(k)
    logits = jax.random.normal(k1, (B, T, V)).astype(jnp.float32)
    targets = jax.random.randint(k2, (B, T), 0, V).astype(jnp.int32)
    return logits, targets


def test_returns_scalar_total_and_metrics():
    logits, targets = _logits_targets()
    total, m = compute_loss(logits, targets, _stats(), TrainConfig())
    assert total.shape == ()
    assert isinstance(m, LossMetrics)
    assert m.total is total
    for field in (m.ce, m.load_balance, m.router_z, m.overflow, m.load_min, m.load_max):
        assert field.shape == ()


def test_total_matches_explicit_formula():
    cfg = TrainConfig()
    logits, targets = _logits_targets()
    stats = _stats()
    total, m = compute_loss(logits, targets, stats, cfg)

    ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    lb = stats.load_balance_loss.mean()
    z = stats.router_z_loss.mean()
    expected = ce + cfg.aux_loss_coef * lb + cfg.router_z_loss_coef * z

    np.testing.assert_allclose(total, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(m.ce, ce, rtol=1e-6, atol=1e-6)


def test_aux_is_averaged_over_layers_not_summed():
    stats = _stats(n_layers=4)
    _, m = compute_loss(*_logits_targets(), stats, TrainConfig())
    np.testing.assert_allclose(m.load_balance, stats.load_balance_loss.mean(), rtol=1e-6)
    assert float(m.load_balance) < float(stats.load_balance_loss.sum()) - 1e-6


def test_perfect_prediction_drives_ce_to_zero():
    B, T, V = 2, 8, 32
    _, targets = _logits_targets(B, T, V)
    logits = jax.nn.one_hot(targets, V, dtype=jnp.float32) * 30.0
    zero = MoEStats(jnp.zeros(2), jnp.zeros(2), jnp.ones((2, V)) / V, jnp.zeros(2))
    total, m = compute_loss(logits, targets, zero, TrainConfig())
    assert float(m.ce) < 1e-5
    np.testing.assert_allclose(total, m.ce, atol=1e-6)


def test_load_min_max_span_all_layers():
    stats = _stats(n_layers=4, n_experts=8)
    _, m = compute_loss(*_logits_targets(), stats, TrainConfig())
    np.testing.assert_allclose(m.load_min, stats.load_fraction.min(), rtol=1e-6)
    np.testing.assert_allclose(m.load_max, stats.load_fraction.max(), rtol=1e-6)


def test_grad_flows_to_model_params():
    cfg = ModelConfig(
        dim=32, n_layers=2, vocab_size=64, n_q_heads=4, n_kv_heads=2,
        head_dim=8, ffn_hidden=64, n_experts=4, top_k=1,
    )
    model = Aleph(cfg, rngs=nnx.Rngs(0))
    tcfg = TrainConfig()
    ids = jax.random.randint(jax.random.key(2), (2, 16), 0, cfg.vocab_size).astype(jnp.int32)
    targets = jax.random.randint(jax.random.key(3), (2, 16), 0, cfg.vocab_size).astype(jnp.int32)

    def loss_fn(model):
        logits, stats = model(ids)
        return compute_loss(logits, targets, stats, tcfg)

    (total, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    assert jnp.isfinite(total)
    assert isinstance(metrics, LossMetrics)
    leaves = jax.tree_util.tree_leaves(nnx.state(grads))
    assert leaves, "expected gradient leaves"
    assert all(jnp.all(jnp.isfinite(g)) for g in leaves)
    assert sum(float(jnp.sum(jnp.abs(g))) for g in leaves) > 0.0
