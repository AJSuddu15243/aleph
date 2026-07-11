import pytest

from aleph.config import ALEPH_TINY, ALEPH_TINY_TRAIN, ModelConfig, TrainConfig, estimate_cost


def test_locked_param_budget():
    # The whole point of the locked spec: ~170M total, ~48M active.
    c = ALEPH_TINY
    assert round(c.total_params / 1e6) == 169
    assert round(c.active_params / 1e6) == 48
    # Active < total is the MoE signature: top_k(1) of n_experts(8) fires.
    assert c.active_params < c.total_params


def test_dim_head_consistency_enforced():
    # dim must equal n_q_heads · head_dim, or the reshape in attention breaks.
    with pytest.raises(AssertionError):
        ModelConfig(dim=512, n_q_heads=8, head_dim=32)  # 8·32 = 256 ≠ 512


def test_gqa_divisibility_enforced():
    with pytest.raises(AssertionError):
        ModelConfig(n_q_heads=8, n_kv_heads=3)  # 3 does not divide 8


def test_train_derived_steps():
    t = ALEPH_TINY_TRAIN
    assert t.tokens_per_step == 256 * 2048          # 524,288
    assert t.total_steps == 20_000_000_000 // t.tokens_per_step
    assert t.min_lr == pytest.approx(6e-5)          # 0.1 · peak


def test_warmup_shorter_than_run():
    with pytest.raises(AssertionError):
        TrainConfig(total_tokens=1_000_000, seq_len=2048, batch_size=256, warmup_steps=1000)


def test_cost_estimate_matches_plan():
    # The 20B run should land near ~$25 on an L40S — the number we committed to.
    est = estimate_cost(ALEPH_TINY, ALEPH_TINY_TRAIN, "l40s", mfu=0.35)
    assert 20 < est["usd"] < 30
    assert 10 < est["hours"] < 15
