import jax
import jax.numpy as jnp
from flax import nnx


class RMSNorm(nnx.Module):
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
