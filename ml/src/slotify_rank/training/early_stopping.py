"""Early stopping on validation NDCG@3.

Two details that decide whether this helps or silently never fires:

``min_delta`` -- without it, a fourth-decimal wobble counts as an improvement
and the patience counter resets forever. Training then always runs the full
epoch budget and "early stopping enabled" is decorative.

``None`` metrics -- a validation epoch whose NDCG is undefined (no candidate
above the gain offset) is *not* an improvement and is *not* a failure to
improve. It is skipped, and counted, so a run whose validation set went
degenerate is visible rather than being read as a long plateau.

The whole state serializes to primitives so a resumed run continues the same
patience count rather than restarting it -- otherwise resuming would quietly
extend training past where it should have stopped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = ["EarlyStopping"]


@dataclass
class EarlyStopping:
    """Tracks the best validation metric and how long since it improved."""

    patience: int = 8
    min_delta: float = 1e-4
    #: Higher is better for NDCG. Kept explicit so switching to a loss-based
    #: criterion is a configuration change rather than a sign bug.
    mode: str = "max"

    best_metric: float | None = None
    best_epoch: int = -1
    epochs_without_improvement: int = 0
    undefined_epochs: int = 0
    stopped: bool = False
    stop_reason: str = ""

    def __post_init__(self) -> None:
        if self.patience < 0:
            raise ValueError("patience must be non-negative")
        if self.min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        if self.mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {self.mode!r}")

    @property
    def enabled(self) -> bool:
        return self.patience > 0

    def is_improvement(self, metric: float) -> bool:
        if self.best_metric is None:
            return True
        if self.mode == "max":
            return metric > self.best_metric + self.min_delta
        return metric < self.best_metric - self.min_delta

    def update(self, metric: float | None, epoch: int) -> bool:
        """Record an epoch. Returns True when this epoch is the new best."""
        if metric is None:
            self.undefined_epochs += 1
            return False
        if self.is_improvement(metric):
            self.best_metric = float(metric)
            self.best_epoch = int(epoch)
            self.epochs_without_improvement = 0
            return True
        self.epochs_without_improvement += 1
        if self.enabled and self.epochs_without_improvement >= self.patience:
            self.stopped = True
            self.stop_reason = (
                f"validation metric did not improve by more than {self.min_delta} "
                f"for {self.patience} epoch(s); best was {self.best_metric} at "
                f"epoch {self.best_epoch}"
            )
        return False

    def state_dict(self) -> dict[str, Any]:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
            "undefined_epochs": self.undefined_epochs,
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.patience = int(state.get("patience", self.patience))
        self.min_delta = float(state.get("min_delta", self.min_delta))
        self.mode = str(state.get("mode", self.mode))
        best = state.get("best_metric")
        self.best_metric = None if best is None else float(best)
        self.best_epoch = int(state.get("best_epoch", -1))
        self.epochs_without_improvement = int(
            state.get("epochs_without_improvement", 0)
        )
        self.undefined_epochs = int(state.get("undefined_epochs", 0))
        self.stopped = bool(state.get("stopped", False))
        self.stop_reason = str(state.get("stop_reason", ""))
