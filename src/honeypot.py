"""
honeypot.py — Structural-integrity honeypot detector (P2).

A honeypot is, by construction, a profile that *looks* great on keywords but
falls apart when you read the career (EXCEUTION_PLAN §4). The dominant
` s_role_fit ` feature naturally scores them low, so this detector is a
**safety net**, not the primary defense — its job is to catch the
*structural* impossibilities a semantic reader might miss
(EXCEUTION_PLAN §4.1, "belt and suspenders").

The detector is a pure function: no I/O, no model, no network. It reads only
the candidate dict and the `honeypot_detection` block of the config.

Do **not** special-case the sample honeypots — keep rules general so the
hidden pool's variants are caught too.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Tiny hard-coded dict of known company founding years (bonus check #4).
# Keep small on purpose — a missed company is a false-negative, not a
# false-positive; over-broad lists create false-kills on legitimate
# candidates (the failure mode the plan explicitly warns against, §4.1).
# Only companies with a well-known founding year that is *recent enough*
# that the check is meaningful go here.
KNOWN_FOUNDING_YEARS: dict[str, int] = {
    "OpenAI": 2015,
    "Anthropic": 2021,
    "Mistral AI": 2023,
    "Cohere": 2019,
    "Hugging Face": 2016,
    "Pinecone": 2019,
    "Weaviate": 2019,
    "Qdrant": 2021,
    "LangChain": 2022,
    "LlamaIndex": 2023,
    "Weights & Biases": 2017,
}


def detect_honeypot(
    candidate: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[bool, list[str]]:
    """
    Run all structural-impossibility checks. Return (is_honeypot, reasons).

    Args:
        candidate: One candidate dict (schema per data/samples/candidate_schema.json).
        cfg: The full scoring config dict (must contain `honeypot_detection` block).

    Returns:
        (is_honeypot, reasons) where `is_honeypot` is True if **any** check fires
        and `reasons` is the list of check names that fired (human-readable,
        used in the `P_penalty` reasons list and for debugging).
    """
    hcfg = cfg.get("honeypot_detection", {})
    reasons: list[str] = []

    if _check_experience_mismatch(candidate, hcfg):
        reasons.append("experience_mismatch")
    if _check_expert_zero_duration(candidate, hcfg):
        reasons.append("expert_with_zero_duration")
    if _check_education_dates(candidate, hcfg):
        reasons.append("education_end_before_start")
    if _check_tenure_before_existence(candidate, hcfg):
        reasons.append("tenure_before_company_existence")

    is_honeypot = bool(reasons)
    if is_honeypot:
        cid = candidate.get("candidate_id", "<unknown>")
        logger.debug("Honeypot detected: %s — %s", cid, reasons)
    return is_honeypot, reasons


def _sum_career_duration(candidate: dict) -> int:
    """Sum of `duration_months` across all career_history entries. Defensive on type."""
    total = 0
    for entry in candidate.get("career_history", []) or []:
        d = entry.get("duration_months", 0)
        if isinstance(d, (int, float)):
            total += int(d)
    return total


def _check_experience_mismatch(candidate: dict, hcfg: dict) -> bool:
    """
    Check 1: |yoe*12 − Σ career duration| > tolerance.

    A real candidate's declared years_of_experience should roughly match the
    sum of their career-entry durations. A 14.5-yr yoe with only 3 months
    of career history is a fabrication.
    """
    yoe = candidate.get("profile", {}).get("years_of_experience")
    if not isinstance(yoe, (int, float)):
        return False
    tolerance = hcfg.get("experience_mismatch_tolerance_months", 12)
    diff = abs(yoe * 12 - _sum_career_duration(candidate))
    return diff > tolerance


def _check_expert_zero_duration(candidate: dict, hcfg: dict) -> bool:
    """
    Check 2: count of skills with `proficiency ∈ {advanced, expert}` AND
    `duration_months == 0` exceeds the configured maximum.

    The classic "expert in 10 skills with 0 years used" honeypot. A real
    expert has months-of-use evidence.
    """
    threshold = hcfg.get("max_zero_duration_expert_skills", 2)
    count = 0
    for skill in candidate.get("skills", []) or []:
        prof = skill.get("proficiency")
        dur = skill.get("duration_months", 0) or 0
        if prof in ("advanced", "expert") and dur == 0:
            count += 1
    return count > threshold


def _check_education_dates(candidate: dict, hcfg: dict) -> bool:
    """
    Check 3: any education entry with end_year < start_year (when enabled).

    Disabled by setting `education_year_sanity: false` in config.
    """
    if not hcfg.get("education_year_sanity", True):
        return False
    for edu in candidate.get("education", []) or []:
        sy = edu.get("start_year")
        ey = edu.get("end_year")
        if isinstance(sy, int) and isinstance(ey, int) and ey < sy:
            return True
    return False


def _check_tenure_before_existence(candidate: dict, hcfg: dict) -> bool:
    """
    Check 4 (bonus, keep simple): a career_history entry's start_date is
    earlier than a known company founding year.

    Only fires for the small set of companies in `KNOWN_FOUNDING_YEARS` so
    the false-positive rate stays low. A missed company is a false-negative,
    not a false-positive — better to under-detect than to false-kill a
    legitimate candidate (EXCEUTION_PLAN §4.1).
    """
    if not KNOWN_FOUNDING_YEARS:
        return False
    for entry in candidate.get("career_history", []) or []:
        company = entry.get("company")
        start_date = entry.get("start_date")
        if not company or not start_date or not isinstance(start_date, str):
            continue
        if company in KNOWN_FOUNDING_YEARS:
            try:
                start_year = int(start_date[:4])
            except (ValueError, IndexError):
                continue
            if start_year < KNOWN_FOUNDING_YEARS[company]:
                return True
    return False
