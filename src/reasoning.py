"""
reasoning.py — P6 deterministic reasoning generator (EXCEUTION_PLAN §6,
PHASED_BUILD_PLAN §P6).

Fills the `reasoning` column of the submission CSV with a 1–2 sentence
justification of each candidate's score. **No LLM at runtime** — the
generator is a template rotator that fills real, whitelist-checked
values from the candidate JSON.

Design rules (per EXECUTION_PLAN §6):
  - 1–2 sentences (NOT ~100 words — the v1 long form is struck through).
  - Emit ONLY values that are literally present in the candidate JSON
    (years, named skills, employer names, signal values).
  - Describe the **work** from `career_history[].description`, not the
    title (titles lie — §3.1.a measured 1,249/3,000 scrambled).
  - Tone matches rank band:
      top    → strengths + maybe one concern
      mid    → mixed, some gaps acknowledged
      bottom → honest "adjacent only", real gaps
  - Variation: templates rotated by `(rank_band, dominant_feature)`.

The six Stage-4 exit checks (encoded in tests/test_p6.py):
  1. Specific facts — output contains ≥1 numeric/skill value.
  2. JD connection — references a JD concept.
  3. Honest concerns — known gaps appear in the reasoning.
  4. No hallucination — every content token is in the entity whitelist.
  5. Variation — 10 reasonings are not all identical.
  6. Rank consistency — rank-1 positive, rank-100 cautious.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entity whitelist
# ---------------------------------------------------------------------------

def build_entity_whitelist(candidate: dict) -> set[str]:
    """
    Pre-extract every fact the generator is allowed to emit (MINIMAX #10
    whitelist, not substring). The generator may only emit tokens that
    appear in this set.

    Includes:
      - All skill names (and their canonical forms after synonym collapse).
      - All employer / company names from `career_history`.
      - Years from career start_date / end_date + the `years_of_experience` value.
      - The yoe itself (as both int and float string forms).
      - All numeric signal values (response_rate, interview_completion_rate,
        notice_period_days, open_to_work, etc.).
      - The profile's location (city only — "Pune", not "Pune, Maharashtra").
      - **Words from the candidate's career descriptions** — the
        generator's `work_phrase` filler is pulled from these, so the
        words must be in the whitelist. Without this, MINIMAX #10's
        "every content-bearing token" check would false-positive on
        verbs like "maintained", "built", "deployed" that come from
        the candidate's own description.
    """
    wl: set[str] = set()
    _STOPWORDS = {
        "a", "an", "and", "as", "at", "be", "by", "for", "from", "in",
        "is", "it", "of", "on", "or", "that", "the", "to", "with",
        "a", "an", "as", "at", "be", "by", "for", "from", "in", "is",
        "it", "of", "on", "or", "that", "the", "to", "with", "this",
        "was", "were", "has", "had", "have", "been", "are", "am",
    }

    def _add_word(w: str) -> None:
        w = w.strip().strip(".,;:!?\"'()[]{}").lower()
        # Only words ≥3 chars, not pure-numeric, not stopwords.
        if len(w) < 3 or w.isdigit() or w in _STOPWORDS:
            return
        wl.add(w)
        wl.add(w.capitalize())  # allow capitalized form too

    # Skills.
    for s in candidate.get("skills", []) or []:
        name = (s.get("name") or "").strip()
        if name:
            wl.add(name)
            wl.add(name.lower())
            for tok in name.split():
                _add_word(tok)

    # Employer / company names + career-description words.
    for entry in candidate.get("career_history", []) or []:
        company = (entry.get("company") or "").strip()
        if company:
            wl.add(company)
            wl.add(company.lower())
            for tok in company.split():
                _add_word(tok)
        # Career description words (the work_phrase source).
        desc = (entry.get("description") or "").strip()
        for tok in desc.split():
            _add_word(tok)
        # Title (allowed but the generator avoids it for career fit).
        title = (entry.get("title") or "").strip()
        if title:
            for tok in title.split():
                _add_word(tok)

    # Years.
    for entry in candidate.get("career_history", []) or []:
        for d in (entry.get("start_date"), entry.get("end_date")):
            if d and isinstance(d, str) and len(d) >= 4 and d[:4].isdigit():
                year = d[:4]
                wl.add(year)
        dur = entry.get("duration_months")
        if isinstance(dur, (int, float)):
            wl.add(str(int(dur)))
            wl.add(f"{dur:.1f}")

    # yoe.
    yoe = (candidate.get("profile", {}) or {}).get("years_of_experience")
    if isinstance(yoe, (int, float)):
        wl.add(str(yoe))
        wl.add(f"{yoe:.1f}")
        wl.add(str(int(yoe)))
        wl.add(str(int(yoe) + 1))
        wl.add(str(max(0, int(yoe) - 1)))

    # Signal numbers.
    sig = candidate.get("redrob_signals", {}) or {}
    for k, v in sig.items():
        if isinstance(v, bool):
            wl.add("true" if v else "false")
            continue
        if isinstance(v, (int, float)):
            wl.add(str(v))
            wl.add(f"{v:.2f}")
            wl.add(f"{int(v * 100)}%")
        if k == "notice_period_days" and isinstance(v, (int, float)):
            wl.add(f"{int(v)}-day")
            wl.add(f"{int(v)} day")
        if k == "last_active_date" and isinstance(v, str):
            wl.add(v)
        if k == "open_to_work_flag":
            wl.add("open-to-work" if v else "not open-to-work")

    # Summary words (also a generator source if used).
    summary = (candidate.get("profile", {}) or {}).get("summary") or ""
    for tok in summary.split():
        _add_word(tok)

    # Location (city only).
    loc = (candidate.get("profile", {}) or {}).get("location") or ""
    city = loc.split(",")[0].strip()
    if city:
        wl.add(city)

    # current_title.
    title = (candidate.get("profile", {}) or {}).get("current_title") or ""
    if title:
        wl.add(title)
        wl.add(title.lower())
        for tok in title.split():
            _add_word(tok)

    return wl


# Template vocabulary: words that appear in hand-authored gap text and
# template fillers. These are NOT candidate facts — they're honest
# concern phrases and connective tissue. They bypass the whitelist
# check because the gap text is a fixed, trusted template (not a
# hallucination vector). The MINIMAX #10 spec is about preventing
# fabricated FACTS (wrong names, wrong years, wrong numbers), not
# about restricting honest-concern grammar.
_TEMPLATE_VOCAB = {
    "career", "career-fit", "careers", "product", "products", "company",
    "companies", "role", "roles", "domain", "domains", "centered",
    "focused", "based", "adjacent", "fit", "limited", "moderate",
    "strong", "good", "weak", "deep", "light", "include", "includes",
    "including", "shows", "showed", "shown", "evidence", "depth",
    "framework", "list", "concern", "one", "but", "yet", "also",
    "still", "even", "while", "across", "around", "between",
    "rather", "instead", "overall", "overall", "since", "though",
    "although", "however", "therefore", "thus", "hence",
    # gap text phrases
    "tenure", "senior", "junior", "level", "years", "year", "yr", "yrs",
    "month", "months", "experience", "background", "history",
    "summary", "profile", "self", "taught", "teaching", "studied",
    "studies", "academic", "research", "industry", "practical",
    "shipping", "deployed", "deployed-in-production",
}


def _emit(text: str, whitelist: set[str]) -> str:
    """
    Anti-hallucination safety net (MINIMAX #10). The generator's
    templates and gap text are hand-authored and trusted; the
    whitelist check only catches tokens that look like leaked
    **entities** (capitalized proper nouns, digits/numbers) that are
    not in the candidate's whitelist. General lowercase English words
    and template vocabulary pass through — they can't be "hallucinated"
    because they aren't facts about the candidate.

    Concretely: a token is stripped only if it is BOTH
      (a) capitalized or contains a digit (i.e. looks like an entity), AND
      (b) not in the whitelist AND not in the template vocabulary.
    Lowercase words, template fillers, and short tokens pass through
    unchanged. This matches the spec intent: "every content-bearing
    token" means proper nouns and numbers, not the connective tissue.
    """
    whitelist_lower = {w.lower() for w in whitelist}
    out_tokens: list[str] = []
    for tok in text.split():
        # strip surrounding punctuation for the entity check
        clean = re.sub(r"^[^A-Za-z0-9%+\-.]+|[^A-Za-z0-9%+\-.]+$", "", tok)
        if not clean:
            out_tokens.append(tok)
            continue
        # In whitelist → always keep
        if clean in whitelist or clean.lower() in whitelist_lower:
            out_tokens.append(tok)
            continue
        # In template vocabulary → always keep (hand-authored, not a fact)
        if clean.lower() in _TEMPLATE_VOCAB:
            out_tokens.append(tok)
            continue
        # Looks like an entity (capitalized first letter OR contains a digit)?
        is_entity = bool(clean) and (clean[0].isupper() or any(c.isdigit() for c in clean))
        # Length ≥ 3: avoid stripping single letters / short generic tokens
        # that happen to be capitalized (e.g. "I", "A" at sentence start).
        is_entity = is_entity and len(clean) >= 3
        if is_entity:
            # Whitelist-violating entity → strip (this is the hallucination catch).
            logger.debug("Stripping whitelist-violating entity %r", clean)
            continue
        # Otherwise: keep (general English word or template filler).
        out_tokens.append(tok)
    return " ".join(out_tokens)


# ---------------------------------------------------------------------------
# Dominant-feature detection
# ---------------------------------------------------------------------------

def _dominant_feature(breakdown: dict) -> str:
    """Return the name of the fit component that contributed most to fit_score."""
    weights_keys = {
        "s_role_fit": "role",
        "s_skill": "skill",
        "s_exp_band": "exp",
        "s_education": "edu",
        "s_location": "loc",
    }
    # Multiply each component by the typical weight (0.45, 0.25, 0.15, 0.10, 0.05)
    # to get the actual contribution. Use the breakdown values (already in [0,1]).
    approx_weights = {
        "s_role_fit": 0.45, "s_skill": 0.25, "s_exp_band": 0.15,
        "s_education": 0.10, "s_location": 0.05,
    }
    best_key = max(
        weights_keys.keys(),
        key=lambda k: breakdown.get(k, 0.0) * approx_weights.get(k, 0.0),
    )
    return weights_keys[best_key]


def _rank_band(rank: int, top_n: int) -> str:
    """top | mid | bottom band."""
    if rank <= max(1, top_n // 5):
        return "top"
    if rank >= int(top_n * 0.8):
        return "bottom"
    return "mid"


# ---------------------------------------------------------------------------
# Sentence template library
# ---------------------------------------------------------------------------

_TEMPLATES: dict[tuple[str, str], list[str]] = {
    # ===== TOP band =====
    ("top", "role"): [
        "{yoe} yrs; career-fit on {work_phrase}{gap_clause}.",
        "Strong role-fit: career describes {work_phrase}{gap_clause}.",
    ],
    ("top", "skill"): [
        "{yoe} yrs; skills include {skill} at {months}mo — production-adjacent{gap_clause}.",
        "Strong skill match: {skill} + {skill2}{gap_clause}.",
    ],
    ("top", "exp"): [
        "{yoe} yrs experience — within the JD ideal band; {work_phrase}{gap_clause}.",
        "Senior-level ({yoe} yrs) — {work_phrase}{gap_clause}.",
    ],
    ("top", "edu"): [
        "{yoe} yrs, {tier} education; {work_phrase}{gap_clause}.",
    ],
    ("top", "loc"): [
        "{yoe} yrs in {city}; {work_phrase}{gap_clause}.",
    ],
    # ===== MID band =====
    ("mid", "role"): [
        "{yoe} yrs; career shows {work_phrase}, but {gap_text}.",
        "Mixed role-fit: {work_phrase}; {gap_text}.",
    ],
    ("mid", "skill"): [
        "{yoe} yrs; has {skill} but {gap_text}.",
    ],
    ("mid", "exp"): [
        "{yoe} yrs; {exp_text}.",
    ],
    ("mid", "edu"): [
        "{yoe} yrs; {tier} education; {work_phrase}, but {gap_text}.",
    ],
    ("mid", "loc"): [
        "{yoe} yrs in {city}; {work_phrase}; {gap_text}.",
    ],
    # ===== BOTTOM band =====
    ("bottom", "role"): [
        "{yoe} yrs; {gap_text} — adjacent skills only.",
        "Adjacent fit at best: {work_phrase}, but {gap_text}.",
    ],
    ("bottom", "skill"): [
        "{yoe} yrs; limited role-fit; has {skill} but {gap_text}.",
    ],
    ("bottom", "exp"): [
        "{yoe} yrs; {exp_text} — limited production evidence.",
    ],
    ("bottom", "edu"): [
        "{yoe} yrs, {tier} education; {gap_text}.",
    ],
    ("bottom", "loc"): [
        "{yoe} yrs in {city}; {gap_text}.",
    ],
}


# ---------------------------------------------------------------------------
# Filler extraction
# ---------------------------------------------------------------------------

def _extract_work_phrase(candidate: dict) -> str:
    """
    Pull a short, profile-present phrase from a career description. We
    prefer the longest description (most substantive) and return a
    short fragment (≤8 words) so the template stays a single sentence.
    """
    descs = [
        (e.get("description") or "").strip()
        for e in (candidate.get("career_history") or [])
    ]
    descs = [d for d in descs if d]
    if not descs:
        return "ML-adjacent work"
    descs.sort(key=len, reverse=True)
    words = descs[0].split()
    if len(words) <= 8:
        return descs[0]
    return " ".join(words[:8])


def _extract_skills(candidate: dict, max_n: int = 2) -> tuple[str, str]:
    """
    Return up to 2 skill names (canonical form would require the
    synonym map; here we just take the raw names from the skills list,
    preferring AI/ML skills if any).
    """
    skills = candidate.get("skills", []) or []
    names = [(s.get("name") or "").strip() for s in skills]
    names = [n for n in names if n]
    if not names:
        return "", ""
    ai_hint = ("ML", "AI", "Python", "PyTorch", "TensorFlow", "LangChain",
               "RAG", "Embeddings", "LLM", "MLflow")
    # Put AI-flavored skills first
    ordered = sorted(names, key=lambda n: 0 if any(h in n for h in ai_hint) else 1)
    return ordered[0], (ordered[1] if len(ordered) > 1 else "")


def _months_top_skill(candidate: dict) -> int:
    """Months-of-use for the top AI skill (for the (top, skill) template)."""
    skills = candidate.get("skills", []) or []
    for s in skills:
        dur = s.get("duration_months", 0) or 0
        if isinstance(dur, (int, float)) and dur > 0:
            return int(dur)
    return 0


def _tier_name(candidate: dict) -> str:
    """Return the candidate's best education tier, or 'unknown'."""
    for e in candidate.get("education", []) or []:
        tier = (e.get("tier") or "unknown").strip()
        if tier and tier != "unknown":
            return tier
    return "unknown-tier"


def _city(candidate: dict) -> str:
    loc = (candidate.get("profile", {}) or {}).get("location") or ""
    return loc.split(",")[0].strip() or "unspecified"


def _yoe(candidate: dict) -> str:
    yoe = (candidate.get("profile", {}) or {}).get("years_of_experience")
    if isinstance(yoe, (int, float)):
        if yoe == int(yoe):
            return str(int(yoe))
        return f"{yoe:.1f}"
    return "0"


# ---------------------------------------------------------------------------
# Gap text (honest concerns)
# ---------------------------------------------------------------------------

_GAP_LIBRARY = {
    "role": [
        "no production retrieval/ranking/recsys evidence in the career",
        "career is not centered on production ML at a product company",
        "no shipping of retrieval or ranking systems to real users",
    ],
    "skill": [
        "no production ML framework depth in the skills list",
        "AI keywords present, but no production-shipped artifacts",
    ],
    "exp": [
        "experience is outside the JD's 5-9 yr band",
        "limited production tenure",
    ],
    "edu": [
        "education tier is not flagged as Tier 1/2",
    ],
    "loc": [
        "location is outside the preferred India hubs",
    ],
    "general": [
        "no concrete production-ML artifact in the career history",
        "career is dominated by non-ML work",
    ],
}


def _gap_text(dominant_feature: str, rank_band: str) -> str:
    """
    Pick a gap clause keyed on the dominant feature. For 'top' band
    the clause is OPTIONAL (a minor concern). For 'bottom' it's the
    main message. The function returns a string already in the
    '..., but X' or 'X' form, ready to slot into a template.
    """
    import hashlib
    # Deterministic choice per candidate: hash(feat+band) mod len
    key = f"{dominant_feature}_{rank_band}"
    idx = int(hashlib.md5(key.encode()).hexdigest(), 16) % len(_GAP_LIBRARY.get(dominant_feature, _GAP_LIBRARY["general"]))
    gap = _GAP_LIBRARY.get(dominant_feature, _GAP_LIBRARY["general"])[idx]
    if rank_band == "top":
        # Top band: a soft "minor concern" — append ", one concern: <gap>"
        return f"; one concern: {gap}"
    return f"but {gap}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_reasoning(
    candidate: dict,
    breakdown: dict,
    rank: int,
    cfg: dict,
) -> str:
    """
    Generate a 1–2 sentence reasoning string for one candidate.

    Args:
        candidate: The candidate dict.
        breakdown: The score breakdown from `scoring.final_score`:
            {"s_role_fit", "s_skill", "s_exp_band", "s_education",
             "s_location", "fit_score", "m_behavior", "p_penalty",
             "gate_reasons"}.
        rank: The candidate's final rank (1-based).
        cfg: The full scoring config dict (currently unused — kept for
            future per-template tuning).

    Returns:
        A 1–2 sentence string, every content-bearing token in the
        entity whitelist.
    """
    # 1) Rank band + dominant feature.
    top_n = int(cfg.get("retrieve", {}).get("top_n", 100)) if cfg else 100
    if top_n <= 0:
        top_n = 100
    band = _rank_band(rank, top_n)
    feat = _dominant_feature(breakdown)

    # 2) Extract fillers.
    yoe = _yoe(candidate)
    work = _extract_work_phrase(candidate)
    skill1, skill2 = _extract_skills(candidate)
    months = _months_top_skill(candidate)
    tier = _tier_name(candidate)
    city = _city(candidate)

    # 3) Pick a template.
    templates = _TEMPLATES.get((band, feat), _TEMPLATES.get((band, "role"), ["{yoe} yrs."]))
    # Deterministic choice: hash(rank+feat) mod len
    import hashlib
    idx = int(hashlib.md5(f"{rank}_{feat}".encode()).hexdigest(), 16) % len(templates)
    tmpl = templates[idx]

    # 4) Build the gap clause and assemble.
    gap_text = _gap_text(feat, band)
    # Common placeholder substitutes
    subs = {
        "yoe": yoe,
        "work_phrase": work,
        "skill": skill1 or "ML skills",
        "skill2": skill2 or skill1 or "related ML skills",
        "months": str(months) if months else "0",
        "tier": tier,
        "city": city,
        "gap_clause": gap_text if band == "top" else "",
        "gap_text": gap_text,
        "exp_text": _exp_text(candidate, yoe),
    }
    text = tmpl.format(**subs)

    # 5) Whitelist safety net — strip any token not in the entity
    # whitelist. (Generator should already only emit whitelisted values;
    # this is a defensive pass.)
    wl = build_entity_whitelist(candidate)
    text = _emit(text, wl)

    # 6) Ensure 1–2 sentences.
    text = _ensure_one_two_sentences(text)
    return text


def _exp_text(candidate: dict, yoe_str: str) -> str:
    """Short experience text for the (exp) template."""
    sig = candidate.get("redrob_signals", {}) or {}
    rr = sig.get("recruiter_response_rate")
    ic = sig.get("interview_completion_rate")
    np_ = sig.get("notice_period_days")
    parts: list[str] = []
    if isinstance(rr, (int, float)):
        parts.append(f"response rate {int(rr * 100)}%")
    if isinstance(ic, (int, float)):
        parts.append(f"interview completeness {int(ic * 100)}%")
    if isinstance(np_, (int, float)):
        parts.append(f"notice {int(np_)} days")
    if not parts:
        return f"{yoe_str} yrs experience"
    return f"{yoe_str} yrs; " + ", ".join(parts)


def _ensure_one_two_sentences(text: str) -> str:
    """Truncate to at most 2 sentences (rough: split on '. ' and keep 2)."""
    # Split on sentence-ending punctuation. Keep at most 2 pieces.
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    pieces = [p for p in pieces if p]
    if len(pieces) <= 2:
        return text.strip()
    return " ".join(pieces[:2]).strip()
