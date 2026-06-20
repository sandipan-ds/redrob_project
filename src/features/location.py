"""
location.py — s_location (soft tie-breaker, w_loc = 0.05).

Substring match on `cfg.location.preferred_cities` /
`also_welcome_cities` against the candidate's `profile.location`
(which is `"City, Region"` per the data — match by **substring**,
not equality).

Fallback ladder when the city doesn't match:
  - willing_to_relocate is true        → willing_to_relocate_score
  - location is in India (heuristic)   → other_india_score
  - otherwise                           → outside_india_score

This is the softest fit component (5% of fit) and only differentiates
candidates who are otherwise ~equal.
"""

from __future__ import annotations

from typing import Any


def s_location(
    profile: dict[str, Any] | None,
    signals: dict[str, Any] | None,
    cfg: dict[str, Any],
) -> float:
    """
    Compute the location score in [0, 1].

    Args:
        profile: The candidate's `profile` dict (may be empty/None).
        signals: The candidate's `redrob_signals` dict (may be empty/None).
        cfg: Full scoring config dict (must contain `location` block).

    Returns:
        A float in [0, 1].
    """
    lcfg = cfg.get("location", {}) or {}
    preferred = [c.strip() for c in (lcfg.get("preferred_cities", []) or []) if c]
    also_welcome = [c.strip() for c in (lcfg.get("also_welcome_cities", []) or []) if c]
    preferred_score = float(lcfg.get("preferred_score", 1.0))
    also_welcome_score = float(lcfg.get("also_welcome_score", 0.85))
    willing_score = float(lcfg.get("willing_to_relocate_score", 0.7))
    other_india_score = float(lcfg.get("other_india_score", 0.5))
    outside_india_score = float(lcfg.get("outside_india_score", 0.2))

    location = ((profile or {}).get("location") or "").strip()
    willing = bool((signals or {}).get("willing_to_relocate"))

    # Substring match (case-insensitive). The data is "City, Region"
    # (e.g. "Noida, Uttar Pradesh") — match by substring so "Noida" hits
    # the full field, and "Delhi" hits "Delhi NCR" / "New Delhi".
    low_loc = location.lower()
    for city in preferred:
        if city and city.lower() in low_loc:
            return _clamp01(preferred_score)
    for city in also_welcome:
        if city and city.lower() in low_loc:
            return _clamp01(also_welcome_score)

    # Fallback: willing_to_relocate, then India vs non-India.
    if willing:
        return _clamp01(willing_score)
    if _looks_indian(location):
        return _clamp01(other_india_score)
    return _clamp01(outside_india_score)


def _looks_indian(location: str) -> bool:
    """
    Heuristic: Indian city / region substrings. Not exhaustive — the
    data's `country` field would be more reliable, but the JD is
    India-focused and this is only a 5% tie-breaker.
    """
    if not location:
        return False
    low = location.lower()
    indian_markers = (
        "india", "bengal", "karnataka", "tamil nadu", "maharashtra",
        "uttar pradesh", "delhi", "noida", "gurgaon", "gurugram",
        "bangalore", "bengaluru", "chennai", "hyderabad", "pune",
        "mumbai", "kolkata", "ahmedabad", "jaipur", "kerala",
        "andhra pradesh", "telangana", "gujarat", "madhya pradesh",
        "haryana", "punjab", "rajasthan", "odisha", "assam",
    )
    return any(m in low for m in indian_markers)


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)
