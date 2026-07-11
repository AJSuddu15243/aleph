import jax
import jax.numpy as jnp
from flax import nnx


class RMSNorm(nnx.Module):
    """RMS normalization over the last (feature) dimension.

        y = x / sqrt(mean(x**2) + eps) * scale

    The normalize step runs in float32 for numerical stability (squaring and
    dividing in bf16 loses precision), then casts back to the input dtype.
    """

    def __init__(self, dim: int, *, eps: float = 1e-6):
        self.scale = nnx.Param(jnp.ones((dim,)))
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        dtype = x.dtype
        x = x.astype(jnp.float32)
        mean_sq = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(mean_sq + self.eps)
        x = x.astype(dtype)
        return x * self.scale.value
