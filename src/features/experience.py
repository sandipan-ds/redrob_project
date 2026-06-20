"""
experience.py — s_exp_band.

Soft experience band peaking in [ideal_min_yrs, ideal_max_yrs], tapering
to the acceptable range, and near-zero below hard_min_yrs. The JD
explicitly says "5–9 years (range, not a requirement)" and ideal 6–8 —
so this is a TAPER, not a cliff. An over-qualified 14-year candidate gets
a mild penalty, not a disqualification.
"""

from __future__ import annotations

from typing import Any


def s_exp_band(yoe: float, cfg: dict[str, Any]) -> float:
    """
    Soft band score in [0, 1].

    Args:
        yoe: Years of experience (from candidate["profile"]["years_of_experience"]).
        cfg: Full scoring config dict (must contain `experience` block).

    Returns:
        A float in [0, 1].
    """
    if not isinstance(yoe, (int, float)):
        return 0.0
    yoe = float(yoe)
    if yoe < 0:
        return 0.0

    e = cfg.get("experience", {}) or {}
    ideal_min = float(e.get("ideal_min_yrs", 6.0))
    ideal_max = float(e.get("ideal_max_yrs", 8.0))
    acceptable_min = float(e.get("acceptable_min_yrs", 4.0))
    acceptable_max = float(e.get("acceptable_max_yrs", 12.0))
    hard_min = float(e.get("hard_min_yrs", 2.0))

    # Ideal band: full credit.
    if ideal_min <= yoe <= ideal_max:
        return 1.0

    # Acceptable taper between acceptable and ideal (and ideal and max).
    if acceptable_min <= yoe < ideal_min:
        return _taper(yoe, acceptable_min, ideal_min, lo=0.5, hi=1.0)
    if ideal_max < yoe <= acceptable_max:
        return _taper(yoe, ideal_max, acceptable_max, lo=1.0, hi=0.7)

    # Hard min: steep penalty, not zero.
    if hard_min <= yoe < acceptable_min:
        return _taper(yoe, hard_min, acceptable_min, lo=0.05, hi=0.5)

    # Below hard min: near zero (not exactly zero — the JD says
    # "sub-30-day notice loved" etc., so a tiny non-zero avoids
    # zero-scoring the feature entirely).
    if yoe < hard_min:
        # A 0-year candidate gets a very small but non-zero score.
        return max(0.0, 0.05 * (yoe / hard_min))

    # Over acceptable_max: mild penalty (over-qualified).
    # Decay linearly to 0.5 at 2× acceptable_max, floor at 0.4.
    over = yoe - acceptable_max
    span = acceptable_max  # decay over `span` extra years
    return max(0.4, 0.7 - 0.3 * (over / span))


def _taper(x: float, x0: float, x1: float, *, lo: float, hi: float) -> float:
    """Linear interpolation from `lo` at x0 to `hi` at x1 (or hi→lo)."""
    if x1 == x0:
        return hi
    t = (x - x0) / (x1 - x0)
    t = max(0.0, min(1.0, t))
    return lo + (hi - lo) * t
