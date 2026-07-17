import jax
from flax import nnx


class FeedForward(nnx.Module):
    def __init__(self, dim: int, hidden: int, *, rngs: nnx.Rngs):
        self.w_gate = nnx.Linear(dim, hidden, use_bias=False, rngs=rngs)
        self.w_up = nnx.Linear(dim, hidden, use_bias=False, rngs=rngs)
        self.w_down = nnx.Linear(hidden, dim, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = jax.nn.silu(self.w_gate(x))
        return self.w_down(gate * self.w_up(x))
