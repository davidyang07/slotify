"""Phase 5 experiment layer: the readiness gate and the frozen label snapshot.

Nothing here trains a model. This package answers one question -- *is there
enough genuine human-labelled data to run the real comparison?* -- and, when the
answer is yes, freezes exactly what that comparison must be reproduced against.
The gate is deliberately hard to pass: a small or leaky label set produces a
number that looks like evidence and is not.
"""

from __future__ import annotations

__all__ = ["readiness", "freeze"]
