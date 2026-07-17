from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from aleph.config import TrainConfig
from aleph.model.moe import MoEStats


class LossMetrics(NamedTuple):
    """Everything one step wants to log, plus the one scalar it differentiates.

    Only ``total`` carries a gradient into the params; the rest are detached
    readouts. The two aux pieces are stored *unweighted* (pre-coefficient) so the
    logs show the raw router health, not a number that moves when we retune a
    coefficient. ``overflow`` and ``load_*`` never enter ``total`` at all — they're
    the router-collapse alarm the build plan insists we watch from step one.
    """

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
    """Assemble the full training objective from a forward pass.

        total = CE  +  aux_loss_coef · mean_layers(load_balance)
                    +  router_z_loss_coef · mean_layers(router_z)

    Args:
        logits:  (B, T, V) fp32 — next-token scores, one row per position.
        targets: (B, T) int — the token each position should predict. Already
                 shifted by the data loader (targets[b,t] = input[b,t+1]); this
                 function does NO shifting, so a bug there would silently teach the
                 model to copy its input.
        stats:   MoEStats with every field carrying a leading (n_layers,) axis —
                 the per-layer routing signals model.py stacked.
        cfg:     TrainConfig; supplies the two aux-loss coefficients.

    Returns (total, metrics). Shaped for ``nnx.value_and_grad(..., has_aux=True)``:
    grad differentiates ``total``; ``metrics`` rides along untouched.
    """
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
