"""Patience-based early stopping for the SAM fine-tuning loop.

Kept as a tiny, dependency-free state machine so the stop/best-epoch logic is
unit-testable without torch or a real training run. YOLO uses Ultralytics'
native ``patience`` instead, so this is SAM-only.

Tracks the best (lowest) validation metric seen and how many epochs have passed
without improvement. ``patience == 0`` disables stopping (the best checkpoint is
still tracked, so the run always saves its best epoch rather than its last).
"""

from __future__ import annotations

import math


class EarlyStopper:
    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.best_epoch = 0
        self.num_bad = 0

    def update(self, value: float, epoch: int) -> bool:
        """Record ``value`` for ``epoch``; return True if it improved on the best.

        "Improved" means strictly lower than the previous best by more than
        ``min_delta`` — the caller snapshots its best checkpoint on a True.
        """
        improved = value < self.best - self.min_delta
        if improved:
            self.best = value
            self.best_epoch = epoch
            self.num_bad = 0
        else:
            self.num_bad += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.num_bad >= self.patience
