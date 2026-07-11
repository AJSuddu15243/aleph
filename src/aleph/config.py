from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """The single source of truth for aleph's shape.

    Locked "tiny MoE" starter: ~170M total params, ~48M active per token.
    Threaded into every module so the whole model is described by one object.

    Two mental buckets to keep straight (see why they differ in the field notes):
      • TOTAL params  → your training MEMORY wall (weights + grads + Adam moments).
      • ACTIVE params → your training COMPUTE / step time / GPU bill (top-k routing
                        only fires k experts per token, in fwd AND bwd).
    """

    # ── core dimensions ──────────────────────────────────────────────────────
    dim: int = 512               # residual / model width
    n_layers: int = 8            # decoder blocks stacked
    vocab_size: int = 50257      # tiktoken gpt2 vocab

    # ── attention (GQA) ──────────────────────────────────────────────────────
    n_q_heads: int = 8           # query heads (H).  dim = H · head_dim
    n_kv_heads: int = 2          # shared K/V heads (G).  8→ MHA, 1→ MQA, 2→ GQA
    head_dim: int = 64           # width of one head
    rope_base: float = 10000.0   # RoPE frequency base (raise for longer context)

    # ── feed-forward / MoE ───────────────────────────────────────────────────
    ffn_hidden: int = 1408       # SwiGLU inner width ≈ (8/3)·dim, rounded to 128·11
    n_experts: int = 8           # experts per MoE layer (drives TOTAL params)
    top_k: int = 1               # experts fired per token (drives ACTIVE params); 1→2 later

    # ── misc ─────────────────────────────────────────────────────────────────
    tie_embeddings: bool = True  # share embed & LM-head matrix — halves the 25.7M vocab tax
    rms_eps: float = 1e-6

    def __post_init__(self):
        assert self.dim == self.n_q_heads * self.head_dim, "dim must equal n_q_heads · head_dim"
        assert self.n_q_heads % self.n_kv_heads == 0, "n_q_heads must be a multiple of n_kv_heads"
        assert self.top_k <= self.n_experts

    # ── derived sizes (so the config documents its own param budget) ──────────
    @property
    def _attn_params(self) -> int:
        # W_q, W_o are dim×(H·d_h); W_k, W_v are dim×(G·d_h). All bias-free.
        q = self.dim * self.n_q_heads * self.head_dim
        kv = self.dim * self.n_kv_heads * self.head_dim
        return 2 * q + 2 * kv

    @property
    def _expert_params(self) -> int:
        return 3 * self.dim * self.ffn_hidden          # SwiGLU: w_gate, w_up, w_down

    @property
    def _embed_params(self) -> int:
        n = self.vocab_size * self.dim
        return n if self.tie_embeddings else 2 * n

    @property
    def total_params(self) -> int:
        """Every param on the GPU — sets the training memory footprint."""
        per_block = self._attn_params + self.n_experts * self._expert_params
        per_block += self.dim * self.n_experts          # router logits
        per_block += 2 * self.dim                       # two RMSNorm scales
        return self.n_layers * per_block + self._embed_params + self.dim

    @property
    def active_params(self) -> int:
        """Params touched per token — sets the compute / FLOPs / cost."""
        per_block = self._attn_params + self.top_k * self._expert_params
        return self.n_layers * per_block + self._embed_params + self.dim


# The locked starter. Import this everywhere until we deliberately scale.
ALEPH_TINY = ModelConfig()


@dataclass(frozen=True)
class TrainConfig:
    """The optimization budget — the second half of the locked spec.

    Locked plan: over-train the tiny MoE on ~20B tokens of code. That's ~120×
    total params / ~410× active — well past Chinchilla, deliberately. For a code
    *completion* model you'll serve, over-training a small net buys a strong
    model whose inference stays cheap (small active count). TinyLlama/StarCoder
    playbook.
    """

    # ── data / token budget ──────────────────────────────────────────────────
    total_tokens: int = 20_000_000_000     # 20B — the whole training budget
    seq_len: int = 2048                     # context window (code likes long-ish)
    dataset: str = "bigcode/starcoderdata"  # HF code corpus (~250B tokens on tap)

    # ── batching (global = per optimizer step; the loop micro-batches to fit) ─
    batch_size: int = 256                   # sequences per optimizer step

    # ── optimizer: AdamW ─────────────────────────────────────────────────────
    peak_lr: float = 6e-4                   # nanoGPT-tested for this scale
    min_lr_ratio: float = 0.1               # cosine floor = 0.1·peak
    warmup_steps: int = 1000                # linear 0→peak, then cosine decay
    weight_decay: float = 0.1
    adam_b1: float = 0.9
    adam_b2: float = 0.95                    # 0.95 not 0.999 — the LLM convention
    adam_eps: float = 1e-8
    grad_clip: float = 1.0                   # clip global grad norm

    # ── MoE auxiliary losses (weights live here; they're an optimization term) ─
    aux_loss_coef: float = 1e-2             # load-balance loss (Switch-style)
    router_z_loss_coef: float = 1e-3        # keep router logits from exploding
    # (capacity_factor joins here when we build moe.py — it's a dispatch knob.)

    # ── misc ─────────────────────────────────────────────────────────────────
    compute_dtype: str = "bfloat16"         # bf16 compute + fp32 master weights
    seed: int = 0

    def __post_init__(self):
        assert self.warmup_steps < self.total_steps, "warmup longer than the whole run"

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.seq_len

    @property
    def total_steps(self) -> int:
        return self.total_tokens // self.tokens_per_step

    @property
    def min_lr(self) -> float:
        return self.peak_lr * self.min_lr_ratio


# The locked training budget. Pairs with ALEPH_TINY.
ALEPH_TINY_TRAIN = TrainConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Cost estimation — the Modal pricing analysis, captured as code so we can
# re-run it against measured MFU after the first smoke test instead of guessing.
#
# Rates confirmed from modal.com/pricing (per-second × 3600). Peak = bf16 dense
# TFLOP/s. MFU (how much of that peak a tiny model actually realizes) is the big
# unknown — pass the measured value once we have it.
# ─────────────────────────────────────────────────────────────────────────────

MODAL_GPUS = {
    #        peak bf16 TFLOP/s,  $/hr
    "h100":     (990.0, 3.95),
    "l40s":     (362.0, 1.95),
    "a100-80":  (312.0, 2.50),
    "a100-40":  (312.0, 2.10),
    "a10g":     (125.0, 1.10),
    "l4":       (121.0, 0.80),
}


def estimate_cost(model: ModelConfig, train: TrainConfig, gpu: str, mfu: float = 0.35):
    """Rough wall-clock + dollar cost of a training run on a Modal GPU.

    Training FLOPs ≈ 6 · active_params · tokens  (the 6 = fwd + bwd + weight
    update, and active_params because MoE only fires top_k experts per token).
    """
    peak_tflops, usd_per_hour = MODAL_GPUS[gpu]
    flops = 6 * model.active_params * train.total_tokens
    seconds = flops / (peak_tflops * 1e12 * mfu)
    hours = seconds / 3600
    return {"pflops": flops / 1e15, "hours": hours, "usd": hours * usd_per_hour}
