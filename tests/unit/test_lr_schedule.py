"""
Unit tests for the SAM fine-tuner's warmup -> cosine LR multiplier (issue #85).

`warmup_cosine_lambda` returns a pure ``step -> multiplier`` callable: a linear
ramp 0 -> 1 over the first ``warmup_frac`` of steps, then cosine decay to a
``floor`` fraction over the rest, clamped past ``total_steps``.
"""

import math

from src.digitalsreeni_image_annotator.training.lr_schedule import warmup_cosine_lambda


def test_warmup_ramps_from_low_to_peak():
    f = warmup_cosine_lambda(100)  # warmup = 10 steps
    assert f(0) == 0.1            # (0 + 1) / 10
    assert f(9) == 1.0           # peak reached at the end of warmup
    assert f(4) < f(9)           # monotone ramp up


def test_cosine_decays_to_floor():
    f = warmup_cosine_lambda(100, warmup_frac=0.1, floor=0.1)
    assert math.isclose(f(99), 0.1, abs_tol=1e-3)  # ~floor at the end
    assert 0.1 < f(55) < 1.0                        # mid-decay between peak and floor


def test_clamps_past_total_steps():
    f = warmup_cosine_lambda(100)
    assert math.isclose(f(500), 0.1, abs_tol=1e-9)  # never dips below the floor


def test_monotonic_decay_after_warmup():
    f = warmup_cosine_lambda(100)
    vals = [f(s) for s in range(10, 100)]
    assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:]))


def test_tiny_total_steps_is_safe():
    f = warmup_cosine_lambda(1)  # warmup clamps to 1 step; no div-by-zero
    assert f(0) == 1.0
    assert f(5) == 0.1
