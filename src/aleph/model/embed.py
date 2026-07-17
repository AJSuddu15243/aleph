import jax
import jax.numpy as jnp
from flax import nnx


class Embedding(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        *,
        tie_embeddings: bool = True,
        rngs: nnx.Rngs,
    ):
        self.tie_embeddings = tie_embeddings
        init = nnx.initializers.normal(stddev=0.02)

        self.tok = nnx.Embed(vocab_size, dim, embedding_init=init, rngs=rngs)
        self.head = (
            None
            if tie_embeddings
            else nnx.Linear(dim, vocab_size, use_bias=False, kernel_init=init, rngs=rngs)
        )

    def encode(self, ids: jax.Array) -> jax.Array:
        return self.tok(ids)

    def decode(self, h: jax.Array) -> jax.Array:
        logits = self.tok.attend(h) if self.head is None else self.head(h)
        return logits.astype(jnp.float32)
