"""
scoring.py — P4 scoring assembly (EXCEUTION_PLAN §2).

` final_score ` assembles the 5 fit components into `fit_score`, then
multiplies by `m_behavior` and `p_penalty`. Pure function: reads only
the candidate dict, a pre-computed feature dict, and the config. No I/O.

The `breakdown` dict returned alongside the final score feeds the P6
reasoning generator (it can cite which sub-scores dominated, which
gates fired, etc.). It is intentionally simple for P4 — P6 will
enrich it.
"""

from __future__ import annotations

import logging
from typing import Any

from src.disqualifiers import compute_penalty
from src.features.behavior import m_behavior

logger = logging.getLogger(__name__)


def final_score(
    candidate: dict[str, Any],
    feats: dict[str, float],
    cfg: dict[str, Any],
    role_fit_text: str = "",
) -> tuple[float, dict[str, Any]]:
    """
    Compute `final = fit × m_behavior × p_penalty` for one candidate.

    Args:
        candidate: One candidate dict.
        feats: Pre-computed feature values (from P3 extractors):
            {"s_role_fit", "s_skill", "s_exp_band", "s_education", "s_location"}.
        cfg: Full scoring config dict.
        role_fit_text: Concatenated career-description text (for the
            research_only gate's production-lexicon scan). Defaults to "".

    Returns:
        (final, breakdown) where `final` is a float (P_penalty may have
        zeroed it out — a honeypot) and `breakdown` is a dict of the
        sub-scores and gate information, intended for the P6 reasoning
        generator.
    """
    weights = cfg.get("weights", {}) or {}

    s_role = float(feats.get("s_role_fit", 0.0))
    s_skill = float(feats.get("s_skill", 0.0))
    s_exp = float(feats.get("s_exp_band", 0.0))
    s_edu = float(feats.get("s_education", 0.0))
    s_loc = float(feats.get("s_location", 0.0))

    fit = (
        float(weights.get("role_fit", 0.0)) * s_role
        + float(weights.get("skill", 0.0)) * s_skill
        + float(weights.get("experience", 0.0)) * s_exp
        + float(weights.get("education", 0.0)) * s_edu
        + float(weights.get("location", 0.0)) * s_loc
    )

    behavior = m_behavior(candidate.get("redrob_signals"), cfg)
    p_penalty, gate_reasons = compute_penalty(candidate, cfg, role_fit_text)

    final = fit * behavior * p_penalty

    breakdown = {
        "s_role_fit": s_role,
        "s_skill": s_skill,
        "s_exp_band": s_exp,
        "s_education": s_edu,
        "s_location": s_loc,
        "fit_score": fit,
        "m_behavior": behavior,
        "p_penalty": p_penalty,
        "gate_reasons": gate_reasons,
    }
    return float(final), breakdown
