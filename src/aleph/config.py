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

    dim: int = 512
    n_layers: int = 8
    vocab_size: int = 50257

    n_q_heads: int = 8
    n_kv_heads: int = 2
    head_dim: int = 64
    rope_base: float = 10000.0

    ffn_hidden: int = 1408
    n_experts: int = 8
    top_k: int = 1
    capacity_factor: float = 1.25

    tie_embeddings: bool = True
    rms_eps: float = 1e-6

    def __post_init__(self):
        assert self.dim == self.n_q_heads * self.head_dim, "dim must equal n_q_heads · head_dim"
        assert self.n_q_heads % self.n_kv_heads == 0, "n_q_heads must be a multiple of n_kv_heads"
        assert self.top_k <= self.n_experts

    @property
    def _attn_params(self) -> int:
        q = self.dim * self.n_q_heads * self.head_dim
        kv = self.dim * self.n_kv_heads * self.head_dim
        return 2 * q + 2 * kv

    @property
    def _expert_params(self) -> int:
        return 3 * self.dim * self.ffn_hidden

    @property
    def _embed_params(self) -> int:
        n = self.vocab_size * self.dim
        return n if self.tie_embeddings else 2 * n

    @property
    def total_params(self) -> int:
        """Every param on the GPU — sets the training memory footprint."""
        per_block = self._attn_params + self.n_experts * self._expert_params
        per_block += self.dim * self.n_experts
        per_block += 2 * self.dim
        return self.n_layers * per_block + self._embed_params + self.dim

    @property
    def active_params(self) -> int:
        """Params touched per token — sets the compute / FLOPs / cost."""
        per_block = self._attn_params + self.top_k * self._expert_params
        return self.n_layers * per_block + self._embed_params + self.dim


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

    total_tokens: int = 20_000_000_000
    seq_len: int = 2048
    dataset: str = "bigcode/starcoderdata"

    batch_size: int = 256

    peak_lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip: float = 1.0

    aux_loss_coef: float = 1e-2
    router_z_loss_coef: float = 1e-3

    compute_dtype: str = "bfloat16"
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


ALEPH_TINY_TRAIN = TrainConfig()


MODAL_GPUS = {
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
