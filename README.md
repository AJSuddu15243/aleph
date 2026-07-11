# aleph

A Mixture-of-Experts decoder-only LLM, trained from scratch in **JAX / Flax NNX**
on **Modal** (NVIDIA GPUs), with custom **Pallas** kernels for the hot spots.

## Architecture

- **Decoder-only** transformer (causal masking, next-token prediction).
- **Mixture-of-Experts** FFNs — a router dispatches each token to its top-k experts.
- **GQA** attention (grouped-query) to keep the KV cache small.
- **RMSNorm** (pre-norm) and **SwiGLU** feed-forward blocks.

## Layout

```
src/aleph/
  model/        # the network, built one module at a time
    norm.py     # RMSNorm
    ffn.py      # SwiGLU expert FFN
    attention.py# GQA attention
    embed.py    # token embedding + LM head
    moe.py      # router + experts (top-k, load balancing)
    block.py    # one decoder layer
    model.py    # the full stack
    kernels/    # custom Pallas kernels (added last)
  data/         # tokenization + data pipeline
  train/        # training loop
  utils/
configs/        # model / data / train configs
modal/          # Modal app for remote training
tests/          # per-module tests
```

## Development

```bash
uv sync --extra dev      # set up the local CPU env (Python 3.12)
uv run pytest            # run the tests
```

GPU/TPU JAX is installed inside the Modal image, not locally; local deps are
CPU-safe for smoke tests.
