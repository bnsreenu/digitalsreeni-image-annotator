"""Learning-rate schedules for the custom training loops.

Only the SAM fine-tuner needs this — YOLO gets warmup→cosine from Ultralytics'
own ``cos_lr`` / ``warmup_epochs`` / ``lrf`` knobs (see ``YOLOTrainer``). The
function here is a pure ``step -> multiplier`` callable so it drops straight
into ``torch.optim.lr_scheduler.LambdaLR`` and is unit-testable without torch.

Recipe (issue bnsreenu#85): linear warmup over the first ``warmup_frac`` of the
total steps (ramp 0 → peak), then cosine decay to a ``floor`` fraction of the
peak over the remainder — the modern fine-tuning default (smooth, zero-slope
landing). The multiplier is relative to the optimizer's base (peak) LR.
"""

from __future__ import annotations

import math


def warmup_cosine_lambda(total_steps, warmup_frac: float = 0.1, floor: float = 0.1):
    """Return ``fn(step) -> lr_multiplier`` for ``LambdaLR``.

    - ``step`` in ``[0, warmup)``: linear ramp ``(step + 1) / warmup`` (so the
      very first step already trains a little rather than at LR 0).
    - ``step >= warmup``: ``floor + (1 - floor) * 0.5 * (1 + cos(pi * progress))``
      where ``progress`` is the fraction through the post-warmup span, clamped
      to ``[0, 1]`` so stepping past ``total_steps`` (gradient accumulation makes
      the real step count approximate) just pins the multiplier at ``floor``.
    """
    total = max(1, int(total_steps))
    warmup = max(1, int(round(total * warmup_frac)))

    def _fn(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(1.0, max(0.0, progress))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return _fn
