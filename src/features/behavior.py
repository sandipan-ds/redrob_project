"""
behavior.py — m_behavior (EXCEUTION_PLAN §2.5.g, criteria_map §E).

The behavior multiplier modulates fit_score in a narrow band [0.5, 1.1].
It nudges — it does not decide. A perfectly-available weak fit must not
outrank a strong fit (that's what corrupts NDCG@10).

Signals used (per criteria_map §E — the rest are dropped):
  - last_active_date  → recency tier
  - recruiter_response_rate (0–1)
  - interview_completion_rate (0–1)
  - open_to_work_flag
  - notice_period_days
  - avg_response_time_hours  → minor modifier (fast response = slight bonus)
  - saved_by_recruiters_30d  → very minor market-interest signal
  - profile_completeness_score → floor only (low completeness penalised)

**github_activity_score is DROPPED** (EXCEUTION_PLAN §2.5.g) — sentinel
`-1` for a large fraction of the pool, so it only discriminates a
self-selected subset. Not read here.

**Sentinels** (`-1` for numerics, `{}` for dicts) are treated as
*"unknown"* — the signal contributes nothing and the multiplier falls
back toward `neutral_base` for that channel. Missing keys also count
as unknown.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)


# A numeric signal is a "sentinel" when it equals this exact value
# (the dataset's "unknown" placeholder for `>= 0` numerics).
_SENTINEL_NUMERIC = -1.0


def m_behavior(
    signals: dict[str, Any] | None, cfg: dict[str, Any]
) -> float:
    """
    Compute the behavior multiplier in [cfg.behavior.min_multiplier,
    cfg.behavior.max_multiplier].

    Args:
        signals: The candidate's `redrob_signals` dict (may be empty/None).
        cfg: Full scoring config dict (must contain `behavior` block).

    Returns:
        A float in [min_multiplier, max_multiplier].
    """
    bcfg = cfg.get("behavior", {}) or {}
    min_m = float(bcfg.get("min_multiplier", 0.50))
    max_m = float(bcfg.get("max_multiplier", 1.10))
    neutral_base = float(bcfg.get("neutral_base", 0.85))

    if not signals:
        return _clamp(neutral_base, min_m, max_m)

    multiplier = neutral_base

    # Recency from last_active_date (days since last activity).
    multiplier += _recency_contribution(signals.get("last_active_date"), bcfg)

    # recruiter_response_rate (0–1) — weighted contribution.
    multiplier += _weighted_signal(
        signals.get("recruiter_response_rate"),
        float(bcfg.get("response_rate_weight", 0.3)),
    )

    # interview_completion_rate (0–1) — weighted contribution.
    multiplier += _weighted_signal(
        signals.get("interview_completion_rate"),
        float(bcfg.get("interview_completion_weight", 0.2)),
    )

    # open_to_work_flag — binary bonus.
    if signals.get("open_to_work_flag") is True:
        multiplier += float(bcfg.get("open_to_work_bonus", 0.1))

    # notice_period_days — categorical bonus/penalty.
    multiplier += _notice_contribution(signals.get("notice_period_days"), bcfg)

    # avg_response_time_hours — minor modifier (fast response → slight
    # bonus; slow → small penalty). Treat -1 as unknown.
    multiplier += _response_time_modifier(signals.get("avg_response_time_hours"))

    # saved_by_recruiters_30d — very minor market-interest signal.
    # Small positive contribution, scaled (capped). Sentinel-safe.
    multiplier += _saved_by_recruiters_modifier(
        signals.get("saved_by_recruiters_30d")
    )

    # profile_completeness_score — floor only: penalise < 40 as
    # low-engagement signal. Sentinel-safe.
    multiplier += _profile_completeness_modifier(
        signals.get("profile_completeness_score")
    )

    return _clamp(multiplier, min_m, max_m)


# -----------------------------------------------------------------------------
# Per-signal contributors (all sentinel-safe)
# -----------------------------------------------------------------------------

def _recency_contribution(last_active: Any, bcfg: dict) -> float:
    """
    Map days-since-active to a contribution. Below the very_active
    threshold → +0.1; between active and moderate → +0.05; stale →
    0; inactive → -0.1. These deltas are heuristic — the config
    thresholds (very_active_days, active_days, etc.) anchor the tiers.
    """
    days = _days_since(last_active)
    if days is None:
        return 0.0
    th = bcfg.get("recency_thresholds", {}) or {}
    very_active = int(th.get("very_active_days", 30))
    active = int(th.get("active_days", 90))
    moderate = int(th.get("moderate_days", 180))
    stale = int(th.get("stale_days", 365))
    if days <= very_active:
        return 0.10
    if days <= active:
        return 0.05
    if days <= moderate:
        return 0.0
    if days <= stale:
        return -0.05
    return -0.10


def _weighted_signal(value: Any, weight: float) -> float:
    """A 0–1 signal contributes `weight × value`; sentinel/missing → 0."""
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 0.0  # bool counts as sentinel
    if not isinstance(value, (int, float)):
        return 0.0
    if float(value) == _SENTINEL_NUMERIC:
        return 0.0
    v = max(0.0, min(1.0, float(value)))
    return weight * v


def _notice_contribution(notice_days: Any, bcfg: dict) -> float:
    """
    notice_period_days → sub_30_bonus / sub_60_neutral / above_90_penalty.
    30 ≤ days ≤ 90 → 0. 0 days or negative → treat as sub-30.
    """
    if notice_days is None or not isinstance(notice_days, (int, float)):
        return 0.0
    if int(notice_days) <= 0:
        return float((bcfg.get("notice_period", {}) or {}).get("sub_30_bonus", 0.05))
    d = int(notice_days)
    sub_30 = float((bcfg.get("notice_period", {}) or {}).get("sub_30_bonus", 0.05))
    sub_60 = float((bcfg.get("notice_period", {}) or {}).get("sub_60_neutral", 0.0))
    above_90 = float((bcfg.get("notice_period", {}) or {}).get("above_90_penalty", -0.05))
    if d < 30:
        return sub_30
    if d < 60:
        return sub_60
    if d <= 90:
        return 0.0
    return above_90


def _response_time_modifier(hours: Any) -> float:
    """
    Fast response (< 6h) → +0.02. 6–24h → 0. 24–72h → -0.02. > 72h → -0.05.
    Sentinel / missing → 0.
    """
    if hours is None or not isinstance(hours, (int, float)):
        return 0.0
    if float(hours) == _SENTINEL_NUMERIC:
        return 0.0
    h = float(hours)
    if h < 0:
        return 0.0
    if h < 6:
        return 0.02
    if h < 24:
        return 0.0
    if h < 72:
        return -0.02
    return -0.05


def _saved_by_recruiters_modifier(value: Any) -> float:
    """
    Very minor market-interest signal. Saturating: 0 saved → 0,
    1 saved → +0.01, 5+ saved → +0.03 (capped). Sentinel / missing → 0.
    """
    if value is None or not isinstance(value, (int, float)):
        return 0.0
    if float(value) == _SENTINEL_NUMERIC:
        return 0.0
    n = max(0, int(value))
    return min(0.03, 0.01 * n)


def _profile_completeness_modifier(value: Any) -> float:
    """
    Floor-only check: < 40 → -0.05 (low engagement); 40+ → 0.
    Sentinel / missing → 0.
    """
    if value is None or not isinstance(value, (int, float)):
        return 0.0
    if float(value) == _SENTINEL_NUMERIC:
        return 0.0
    if float(value) < 40:
        return -0.05
    return 0.0


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _days_since(date_str: Any) -> int | None:
    if not date_str or not isinstance(date_str, str):
        return None
    try:
        d = date.fromisoformat(date_str)
    except (TypeError, ValueError):
        return None
    today = date.today()
    return (today - d).days


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return float(x)
