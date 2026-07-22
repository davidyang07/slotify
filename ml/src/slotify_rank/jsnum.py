"""ECMAScript numeric semantics, for byte-exact parity with the TypeScript baseline.

The canonical baseline is defined by the running TypeScript product code
(`backend/src/lib/candidates.ts` and the slot finalisation in
`backend/src/routes/insert-sections.ts`). Python and JavaScript both use IEEE-754
doubles, so arithmetic agrees exactly as long as the *order of operations* is
preserved. Rounding does **not** agree, and those differences are what this
module exists to eliminate:

* ``Math.round`` rounds halves toward ``+Infinity`` (``Math.round(-0.5) === -0``,
  ``Math.round(-1.5) === -1``), whereas Python's :func:`round` uses banker's
  rounding (``round(0.5) == 0``, ``round(2.5) == 2``). Note that this is *not*
  the same as round-half-away-from-zero, so a single ``ROUND_HALF_UP`` is wrong
  for negative ties.
* ``Number.prototype.toFixed`` takes the absolute value first and prepends the
  sign, so its ties round **away from zero** -- a different rule from
  ``Math.round``. Within the magnitude it picks the integer ``n`` minimising
  ``|n / 10**digits - x|``, breaking ties toward the larger ``n``.

Both helpers operate on a :class:`decimal.Decimal` constructed directly from the
float, which is exact. Rounding the decimal *literal* instead would be wrong:
``(1.0005).toFixed(3) === "1.000"`` because the nearest double to 1.0005 is
slightly below the midpoint.
"""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_DOWN, ROUND_HALF_UP

__all__ = ["js_round", "js_to_fixed"]


def js_round(value: float) -> int:
    """Return ``Math.round(value)`` as an ``int``.

    Ties round toward positive infinity, matching ECMA-262: for a positive value
    that is away from zero (``ROUND_HALF_UP``), for a negative value it is
    toward zero (``ROUND_HALF_DOWN``). Raises on NaN or infinity rather than
    returning a silently wrong integer.
    """
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"js_round is undefined for non-finite input: {value!r}")
    rounding = ROUND_HALF_UP if value >= 0 else ROUND_HALF_DOWN
    return int(Decimal(value).quantize(Decimal(1), rounding=rounding))


def js_to_fixed(value: float, digits: int) -> float:
    """Return ``Number(value.toFixed(digits))``.

    Used for the ``insertion_time_seconds`` field, which the product emits as
    ``Number(clampedTimeSeconds.toFixed(3))``
    (``backend/src/routes/insert-sections.ts:216``). Ties round away from zero,
    because ECMA-262 applies the rounding to the magnitude and restores the sign
    afterwards.
    """
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"js_to_fixed is undefined for non-finite input: {value!r}")
    if digits < 0:
        raise ValueError(f"digits must be non-negative, got {digits}")
    exponent = Decimal(1).scaleb(-digits)
    return float(Decimal(value).quantize(exponent, rounding=ROUND_HALF_UP))
