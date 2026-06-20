"""
education.py — s_education.

Tier score (from `cfg.education.tier_scores`, with `unknown → 0.30`
neutral — missing tier is NOT penalized) combined with a non-linear
CGPA ramp and a field-relevance multiplier.

The CGPA ramp is:
  below min_threshold (default 7.0): score = cgpa × below_threshold_scale
  above min_threshold:                score = 1.0 (flat, since
                                      recruiter_max is the 10-point CGPA
                                      ceiling, not a score cap > 1)
  The combined function is clipped to [0, 1].

Field relevance:
  - field_of_study in `relevant_fields`  → multiplier = bonus (1.0)
  - otherwise                              → multiplier = penalty (0.7)
  Missing field → treat as irrelevant (apply penalty).

For multiple education entries, the **best** education dominates (max),
since the JD is about the candidate's strongest credential.
"""

from __future__ import annotations

import re
from typing import Any

_CGPA_RE = re.compile(r"-?\d+(?:\.\d+)?")


def s_education(
    education: list[dict[str, Any]] | None, cfg: dict[str, Any]
) -> float:
    """
    Compute the education score in [0, 1] for one candidate.

    Args:
        education: The candidate's `education` list (may be empty/None).
        cfg: Full scoring config dict (must contain `education` block).

    Returns:
        A float in [0, 1].
    """
    ecfg = cfg.get("education", {}) or {}
    tier_scores = ecfg.get("tier_scores", {}) or {}
    cgpa_cfg = ecfg.get("cgpa", {}) or {}
    min_threshold = float(cgpa_cfg.get("min_threshold", 7.0))
    below_scale = float(cgpa_cfg.get("below_threshold_scale", 1.0 / 7.0))
    relevant = {f.strip().lower() for f in (ecfg.get("relevant_fields", []) or [])}
    bonus = float(ecfg.get("relevant_field_bonus", 1.0))
    penalty = float(ecfg.get("irrelevant_field_penalty", 0.7))

    if not education:
        return 0.0

    best = 0.0
    for edu in education:
        tier = (edu.get("tier") or "unknown").strip().lower()
        tier_score = float(tier_scores.get(tier, tier_scores.get("unknown", 0.30)))

        cgpa = _parse_cgpa(edu.get("grade"))
        cgpa_score = _cgpa_score(cgpa, min_threshold, below_scale)

        # Field relevance multiplier.
        field = (edu.get("field_of_study") or "").strip().lower()
        if field and field in relevant:
            field_mult = bonus
        else:
            field_mult = penalty

        entry_score = _clamp01(tier_score * cgpa_score * field_mult)
        if entry_score > best:
            best = entry_score

    return best


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _parse_cgpa(grade: Any) -> float | None:
    """
    Parse a numeric CGPA from the `grade` field. The field is free-form:
    "8.24 CGPA", "8.5/10", "First Class", "74%", "3.8 GPA". We extract
    the first numeric token. Returns None when no number is present.
    """
    if grade is None:
        return None
    if isinstance(grade, (int, float)):
        return float(grade)
    s = str(grade).strip()
    m = _CGPA_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except (TypeError, ValueError):
        return None


def _cgpa_score(
    cgpa: float | None, min_threshold: float, below_scale: float
) -> float:
    """
    Non-linear CGPA ramp in [0, 1].
      cgpa < min_threshold: linear `cgpa × below_scale` (saturates at 1.0)
      cgpa ≥ min_threshold: 1.0
    Missing CGPA → 0.5 (neutral — don't punish, don't reward).
    """
    if cgpa is None:
        return 0.5
    if cgpa < 0:
        return 0.0
    if cgpa < min_threshold:
        return _clamp01(cgpa * below_scale)
    return 1.0


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)
