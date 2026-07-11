import jax
import jax.numpy as jnp
from flax import nnx


class Embedding(nnx.Module):
    """Token embedding + LM head, sharing one matrix when weights are tied.

    One matrix ``E`` of shape ``(vocab_size, dim)`` is the whole dictionary that
    maps tokens ↔ vectors, read in two directions:

      • ``encode(ids)`` — a *gather*. Token 42 grabs row 42. No matmul.
      • ``decode(h)``   — a *matmul* ``h @ Eᵀ`` turning each dim-vector back into
                          one logit per vocab word (the LM head / unembed).

    When ``tie_embeddings`` is True (the default) both directions use the *same*
    ``E`` — 25.7M params here on the gpt2 vocab, so tying saves a full second
    copy (~15% of the whole tiny model) for free. ``nnx.Embed.attend`` does the
    tied unembed for us. Untied, a separate bias-free ``Linear`` is the head.

    No positional information lives here — position is injected by RoPE inside
    attention. This module is pure token identity.

    Init note: Flax's default embedding init is unit-variance, which is too hot
    for a *tied* head — with RMSNorm feeding decode, initial logits would have
    std ≈ √dim and blow the softmax into a peaky, high-loss corner. We override
    to normal(0.02), the GPT-2 / nanoGPT recipe, keeping init logits small.
    """

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
        # Untied: an independent output projection. Tied: none — decode reuses E.
        self.head = (
            None
            if tie_embeddings
            else nnx.Linear(dim, vocab_size, use_bias=False, kernel_init=init, rngs=rngs)
        )

    def encode(self, ids: jax.Array) -> jax.Array:
        # ids: (B, T) int token ids → (B, T, dim) vectors. Pure lookup.
        return self.tok(ids)

    def decode(self, h: jax.Array) -> jax.Array:
        # h: (B, T, dim) → (B, T, vocab_size) logits, in float32 so the
        # downstream softmax / cross-entropy stays stable under bf16 compute.
        logits = self.tok.attend(h) if self.head is None else self.head(h)
        return logits.astype(jnp.float32)
