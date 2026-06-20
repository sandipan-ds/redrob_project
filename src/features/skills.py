"""
skills.py — s_skill (EXCEUTION_PLAN §2.5, criteria_map §B).

JD-core skill match with synonym collapse, endorsement curve, duration
trust, and platform-verified assessment override.

The synonym-collapse step is the most important: a candidate listing
both "RAG" and "Retrieval-Augmented Generation" scores the canonical
RAG skill **once**, not twice. Without collapse, keyword-stuffers could
double-dip on synonymous skills (GLM-v2 #A4).

Scoring per matched skill d:
  raw_d = jd_weight(d) × endorsement_score(endorsements_d) × duration_trust(d)
where:
  endorsement_score = min(endorsements_d, endorse_floor) / endorse_floor  ∈ [0, 1]
  duration_trust    = a function of duration_months (see _duration_trust)

Final s_skill = sum(raw_d) / max_possible, clamped to [0, 1].

Platform-verified `skill_assessment_scores` are preferred when present
(§5.1 trust platform-verified over self-reported): if a skill has a
Redrob assessment score (0–100), it overrides the self-reported
proficiency/endorsements for that skill.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def s_skill(candidate: dict[str, Any], cfg: dict[str, Any]) -> float:
    """
    Compute the JD-skill match score in [0, 1] for one candidate.

    Args:
        candidate: One candidate dict.
        cfg: Full scoring config dict (must contain `skills` block).

    Returns:
        A float in [0, 1].
    """
    scfg = cfg.get("skills", {})
    endorse_floor = int(scfg.get("endorse_floor", 30))
    synonym_map = _normalize_synonyms(scfg.get("skill_synonyms", {}) or {})
    jd_core = _normalize_core(scfg.get("jd_core_skills", []) or [])
    noise = {n.strip().lower() for n in (scfg.get("noise_skills", []) or [])}

    if not jd_core:
        return 0.0

    # Apply synonym collapse to the candidate's skills.
    raw_skills = candidate.get("skills", []) or []
    canonical_skills: dict[str, dict] = {}  # canonical_name → skill dict
    for s in raw_skills:
        name = (s.get("name") or "").strip()
        if not name:
            continue
        canonical = synonym_map.get(name.lower(), name)
        canonical_key = canonical.lower()
        if canonical_key in noise:
            continue
        # Only keep skills that exist in jd_core (others are ignored per spec).
        if canonical_key not in {c["key"] for c in jd_core}:
            continue
        # If duplicate, keep the one with more evidence (longer duration).
        existing = canonical_skills.get(canonical_key)
        if existing is None or int(s.get("duration_months", 0) or 0) > int(
            existing.get("duration_months", 0) or 0
        ):
            canonical_skills[canonical_key] = s

    # Platform-verified assessment scores (preferred over self-reported).
    assessments = (
        candidate.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}
    )
    normalized_assessments = {
        k.strip().lower(): float(v) for k, v in assessments.items() if k
    }

    total = 0.0
    max_possible = 0.0
    for core in jd_core:
        canonical_key = core["key"]
        max_possible += float(core["weight"])  # max if endorsed + trusted + verified
        matched = canonical_skills.get(canonical_key)
        if matched is None:
            continue
        # Prefer platform-verified assessment score when present.
        assessment = normalized_assessments.get(canonical_key)
        if assessment is not None:
            # Assessment is 0–100 → [0, 1]. Weight by jd weight.
            confidence = _clamp01(assessment / 100.0)
        else:
            endorsements = int(matched.get("endorsements", 0) or 0)
            endorsement_score = min(endorsements, endorse_floor) / endorse_floor
            duration_trust = _duration_trust(matched.get("duration_months", 0))
            confidence = endorsement_score * duration_trust
        total += float(core["weight"]) * confidence

    if max_possible <= 0:
        return 0.0
    return _clamp01(total / max_possible)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _normalize_synonyms(raw: dict) -> dict[str, str]:
    """Lowercase keys/values; map variant → canonical."""
    out: dict[str, str] = {}
    for k, v in raw.items():
        kl = (k or "").strip().lower()
        vl = (v or "").strip()
        if kl and vl:
            out[kl] = vl
    return out


def _normalize_core(raw: list) -> list[dict]:
    """Return a list of {key (lowercase), name, weight} dicts."""
    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or "").strip()
        if not name:
            continue
        out.append(
            {
                "key": name.lower(),
                "name": name,
                "weight": float(item.get("weight", 0) or 0),
            }
        )
    return out


def _duration_trust(duration_months: Any) -> float:
    """
    Trust multiplier on self-reported proficiency. Saturates at 24 months
    of usage: 0mo → 0.0, 12mo → 0.5, 24mo+ → 1.0. Combined with
    endorsement_score, a skill with no months-of-use cannot earn a
    high self-reported score (this is the structural-impossibility
    check the honeypot detector also enforces at a candidate level).
    """
    months = max(0, int(duration_months or 0))
    return min(months / 24.0, 1.0)


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)
