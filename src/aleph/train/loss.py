from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from aleph.config import TrainConfig
from aleph.model.moe import MoEStats


class LossMetrics(NamedTuple):
    total: jax.Array
    ce: jax.Array
    load_balance: jax.Array
    router_z: jax.Array
    overflow: jax.Array
    load_min: jax.Array
    load_max: jax.Array


def compute_loss(
    logits: jax.Array,
    targets: jax.Array,
    stats: MoEStats,
    cfg: TrainConfig,
) -> tuple[jax.Array, LossMetrics]:
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

    load_balance = stats.load_balance_loss.mean()
    router_z = stats.router_z_loss.mean()

    total = ce + cfg.aux_loss_coef * load_balance + cfg.router_z_loss_coef * router_z

    metrics = LossMetrics(
        total=total,
        ce=ce,
        load_balance=load_balance,
        router_z=router_z,
        overflow=stats.overflow_fraction.mean(),
        load_min=stats.load_fraction.min(),
        load_max=stats.load_fraction.max(),
    )
    return total, metrics
