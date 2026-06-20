"""
disqualifiers.py — Multiplicative penalty gates (P2).

Computes `P_penalty ∈ (0, 1]` per candidate as the product of independent
hard-reject gates from the JD (EXCEUTION_PLAN §1.1, §2.0.c, §4).

Gates (EXCEUTION_PLAN §2.5.j + §2.5.d + §2.5.e):
  - honeypot               → always at full strength (0.01)
  - consulting_only        → generalized detector (name list OR industry+size),
                             with prior-product-co exemption
  - research_only          → conjunctive (3 conditions must ALL hold)
  - no_recent_code         → no ML-relevant career entry in last 18 months
  - domain_mismatch        → CV/speech/robotics primary, no NLP/IR evidence
  - langchain_only_junior  → DEMOTED (§2.5.j): only fires when ALL 3
                             conjunctive conditions hold

Combination (§2.5.d): worst-gate-full + geometric softening of secondaries.
  P = (honeypot_score if honeypot else 1.0)
      * min(non_hp) * prod(others)^0.5
where each non-honeypot gate score is pre-scaled by `cfg.penalties.p_scale`
(§2.5.e — a single calibratable knob, replaces the 6 previously-hidden
per-gate constants).
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any, Iterable

from src.honeypot import detect_honeypot

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Lexicons (broad, to defeat the "keyword trap inverted" failure mode §2.5.c)
# -----------------------------------------------------------------------------

# Production-shipping verbs + retrieval/ranking/recsys/search terms.
# Used by: research_only (condition 1), no_recent_code (ML-relevance),
#           langchain_only_junior (condition 2 pre-2022 ML evidence).
PRODUCTION_LEXICON: tuple[str, ...] = (
    # production-shipping verbs (broad synonym set — §2.5.c)
    "shipped", "deploy", "deployed", "deploying", "deployment",
    "launch", "launched", "launching",
    "roll out", "rolled out", "rollout",
    "served", "serving", "in production", "productionized", "productionised",
    "a/b test", "a-b test", "ab test",
    "inference service", "inference api", "online",
    # retrieval / ranking / recsys / search / embeddings (§2.5.a core)
    "retrieval", "ranking", "recommendation system", "recommender system",
    "recommender", "recsys", "search", "search system",
    "embedding", "embeddings", "vector search", "vector database",
    "dense retrieval", "hybrid search", "bm25",
    "ndcg", "mrr", "map",  # eval metrics = strong production signal
)

# Explicit research framing (condition 3 of research_only).
# "applied scientist" is borderline — included because real research framings
# often use it; the conjunctive design (needs ALL 3 conditions) keeps the
# false-positive rate low even if it's a soft signal.
RESEARCH_FRAMING: tuple[str, ...] = (
    "research scientist", "researcher", "research engineer",
    "academic", "academia", "professor", "faculty",
    "lab ", "laboratory", "research lab",
    "thesis", "phd", "ph.d", "postdoc", "post-doctoral",
    "university", "institute of technology",
    "applied scientist", "research intern",
)

# Broad "AI/ML" keyword set for:
#   - langchain_only_junior condition (1): "total AI experience" = sum of
#     duration_months for skills whose name matches this set.
#   - langchain_only_junior condition (3): "LangChain is the ONLY AI signal" =
#     candidate has AI skills/description evidence OUTSIDE LangChain.
AI_SKILL_KEYWORDS: tuple[str, ...] = (
    "machine learning", "deep learning", "neural", "transformer", "transformers",
    "embedding", "embeddings", "llm", "llms", "large language model",
    "fine-tun", "lora", "qlora", "peft",
    "rag", "retrieval-augmented",
    "pytorch", "tensorflow", "scikit-learn", "sklearn",
    "hugging face", "huggingface", "bert", "gpt",
    "vector", "faiss", "pinecone", "weaviate", "milvus", "qdrant",
    "recommendation", "recsys", "search", "ranking", "retrieval",
    "nlp", "natural language", "computer vision", "speech", "robotics",
    "langchain", "langgraph", "llamaindex",
)


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def compute_penalty(
    candidate: dict[str, Any],
    cfg: dict[str, Any],
    role_fit_text: str = "",
) -> tuple[float, list[str]]:
    """
    Compute P_penalty ∈ (0, 1] for a single candidate.

    Args:
        candidate: One candidate dict (schema per data/samples/candidate_schema.json).
        cfg: The full scoring config dict (must contain `penalties` block).
        role_fit_text: The concatenated career description text used by the
            role-fit feature. Used by research_only to scan for production
            evidence. If empty, falls back to concatenating career_history
            descriptions in-place.

    Returns:
        (P_penalty, reasons) where P_penalty ∈ (0, 1] and `reasons` is a
        human-readable list of gate names that fired. `P_penalty == 1.0` and
        `reasons == []` when no gate fires.
    """
    pcfg = cfg.get("penalties", {})
    p_scale = float(pcfg.get("p_scale", 1.0))
    reasons: list[str] = []

    # ---- honeypot: always full strength, never scaled ----
    is_honeypot, hp_reasons = detect_honeypot(candidate, cfg)
    if is_honeypot:
        reasons.append("honeypot")
        # Honeypot: 1.0 from this arm, 0.01 from the honeypot gate.
        return float(pcfg.get("honeypot", {}).get("score", 0.01)), reasons

    # ---- fall back role_fit_text if not provided ----
    if not role_fit_text:
        role_fit_text = " ".join(
            (e.get("description") or "")
            for e in candidate.get("career_history", []) or []
        )

    # ---- evaluate each non-honeypot gate ----
    non_hp: list[tuple[str, float]] = []  # (gate_name, scaled_score)

    score = _eval_consulting_only(candidate, pcfg)
    if score is not None:
        non_hp.append(("consulting_only", score * p_scale))
        reasons.append("consulting_only")

    score = _eval_research_only(candidate, role_fit_text, pcfg)
    if score is not None:
        non_hp.append(("research_only", score * p_scale))
        reasons.append("research_only")

    score = _eval_no_recent_code(candidate, pcfg)
    if score is not None:
        non_hp.append(("no_recent_code", score * p_scale))
        reasons.append("no_recent_code")

    score = _eval_domain_mismatch(candidate, pcfg)
    if score is not None:
        non_hp.append(("domain_mismatch", score * p_scale))
        reasons.append("domain_mismatch")

    score = _eval_langchain_only_junior(candidate, role_fit_text, pcfg)
    if score is not None:
        non_hp.append(("langchain_only_junior", score * p_scale))
        reasons.append("langchain_only_junior")

    if not non_hp:
        return 1.0, []

    # ---- §2.5.d combination: worst gate full, secondaries geometric ----
    non_hp_sorted = sorted(non_hp, key=lambda kv: kv[1])  # ascending
    worst_name, worst_score = non_hp_sorted[0]
    other_scores = [s for _, s in non_hp_sorted[1:]]
    if other_scores:
        from math import prod
        secondary = prod(other_scores) ** 0.5
    else:
        secondary = 1.0
    p_penalty = worst_score * secondary

    # Numerical safety: clamp into (0, 1].
    if p_penalty > 1.0:
        p_penalty = 1.0
    if p_penalty <= 0.0:
        p_penalty = 1e-6  # never zero so sorts are deterministic

    return float(p_penalty), reasons


# -----------------------------------------------------------------------------
# Gate evaluators — each returns the configured score if the gate FIRES, else None.
# -----------------------------------------------------------------------------

def _eval_consulting_only(candidate: dict, pcfg: dict) -> float | None:
    """
    §2.5.j generalized consulting-only detector.

    A stint is "consulting" if (a) its company is in `consulting_companies`,
    OR (b) its industry ∈ consulting_industries AND company_size ∈
    consulting_company_sizes.

    A candidate is consulting-only if EVERY stint is consulting. The gate
    fires on that condition. (Equivalently: any stint with industry ∉
    consulting_industries is a product-co stint → exemption → no fire.)
    """
    cfg_block = pcfg.get("consulting_only", {})
    score = float(cfg_block.get("score", 0.15))
    name_list = set(cfg_block.get("consulting_companies", []) or [])
    consulting_industries = set(cfg_block.get("consulting_industries", []) or [])
    consulting_sizes = set(cfg_block.get("consulting_company_sizes", []) or [])

    career = candidate.get("career_history", []) or []
    if not career:
        return None  # No career data → can't claim consulting-only

    for entry in career:
        company = entry.get("company") or ""
        industry = entry.get("industry") or ""
        size = entry.get("company_size") or ""

        is_consulting = (
            company in name_list
            or (industry in consulting_industries and size in consulting_sizes)
        )
        if not is_consulting:
            return None  # found a non-consulting stint → exempt

    return score  # every stint is consulting → fire


def _eval_research_only(
    candidate: dict, role_fit_text: str, pcfg: dict
) -> float | None:
    """
    §2.5.c conjunctive research-only detector. All 3 conditions must hold:

      (1) no production-lexicon match in role_fit_text
      (2) no product-company tenure ever (every stint has industry ∈
          consulting_industries — i.e. is consulting)
      (3) explicit research framing in descriptions or current_title

    Otherwise → return None (gate does NOT fire; the mild soft demotion is
    a separate signal, not a gate).
    """
    cfg_block = pcfg.get("research_only", {})
    score = float(cfg_block.get("score", 0.20))
    consulting_industries = set(
        pcfg.get("consulting_only", {}).get("consulting_industries", []) or []
    )
    research_industries = set(cfg_block.get("research_industries", []) or [])

    # A stint is "not a product company" if its industry is consulting OR
    # academic. A stint in any OTHER industry is a product-co stint.
    non_product_industries = consulting_industries | research_industries

    # (1) no production evidence in descriptions
    if _has_any_term(role_fit_text, PRODUCTION_LEXICON):
        return None

    # (2) no product-company tenure ever
    career = candidate.get("career_history", []) or []
    if not career:
        return None  # no data → don't claim research-only
    for entry in career:
        industry = entry.get("industry") or ""
        if industry and industry not in non_product_industries:
            return None  # found a product-co stint → condition (2) fails

    # (3) explicit research framing
    title = (candidate.get("profile", {}).get("current_title") or "").lower()
    if _has_any_term(title, RESEARCH_FRAMING):
        return score
    if _has_any_term(role_fit_text, RESEARCH_FRAMING):
        return score

    return None


def _eval_no_recent_code(candidate: dict, pcfg: dict) -> float | None:
    """
    No ML-relevant career entry within `lookback_months` (default 18).

    An entry is "ML-relevant" if its description contains any production/
    ML term. The lookback is from today (or from `end_date` of the
    most-recent entry — approximate).

    The github_activity_score sentinel issue (§2.5.g) means we MUST lean
    on description text, not the github signal.
    """
    cfg_block = pcfg.get("no_recent_code", {})
    score = float(cfg_block.get("score", 0.25))
    lookback = int(cfg_block.get("lookback_months", 18))

    career = candidate.get("career_history", []) or []
    if not career:
        return None  # no data → don't claim no-recent-code

    # Reference date: today. (In practice rank.py is run at submission
    # time; using `date.today()` keeps the test deterministic for a given
    # run-day. For an even more stable approach, the lookback could be
    # pegged to the max end_date in the data — but that requires loading
    # all candidates, which rank.py does NOT do at this step.)
    today = date.today()

    for entry in career:
        desc = entry.get("description") or ""
        if not _has_any_term(desc, PRODUCTION_LEXICON):
            continue  # not ML-relevant
        # ML-relevant: check the entry falls within the lookback window.
        # Use end_date if present (and not is_current → use today),
        # else fall back to start_date as a lower bound.
        end = entry.get("end_date")
        start = entry.get("start_date")
        is_current = entry.get("is_current", False)
        if is_current or not end:
            ref = today
        else:
            try:
                ref = date.fromisoformat(end)
            except (TypeError, ValueError):
                continue
        # How recent? months between ref and today
        months_since = (today.year - ref.year) * 12 + (today.month - ref.month)
        if months_since <= lookback:
            return None  # found a recent ML stint → gate does NOT fire

    return score  # no recent ML stint → fire


def _eval_domain_mismatch(candidate: dict, pcfg: dict) -> float | None:
    """
    Primary expertise is CV / speech / robotics AND no NLP/IR evidence.

    Fires if skills include a `mismatch_domains` entry (or descriptions
    mention them) AND the profile has no NLP/IR signal.
    """
    cfg_block = pcfg.get("domain_mismatch", {})
    score = float(cfg_block.get("score", 0.30))
    mismatch = set(cfg_block.get("mismatch_domains", []) or [])

    has_mismatch_skill = False
    has_nlp_ir_skill = False
    for s in candidate.get("skills", []) or []:
        name = (s.get("name") or "").strip()
        if name in mismatch:
            has_mismatch_skill = True
        if any(kw in name.lower() for kw in ("nlp", "natural language", "information retrieval", "ir ")):
            has_nlp_ir_skill = True

    desc_text = " ".join(
        (e.get("description") or "")
        for e in candidate.get("career_history", []) or []
    ).lower()
    has_mismatch_desc = any(d.lower() in desc_text for d in mismatch)
    has_nlp_ir_desc = any(
        kw in desc_text
        for kw in ("nlp", "natural language processing", "information retrieval")
    )

    if (has_mismatch_skill or has_mismatch_desc) and not (has_nlp_ir_skill or has_nlp_ir_desc):
        return score
    return None


def _eval_langchain_only_junior(
    candidate: dict, role_fit_text: str, pcfg: dict
) -> float | None:
    """
    §2.5.j DEMOTED langchain-only-junior detector. ALL 3 conjunctive
    conditions must hold for the gate to fire; otherwise return None.

      (1) total AI experience < junior_exp_months (sum of duration_months
          for skills in the AI set, excluding unknown/empty)
      (2) no pre-2022 ML production evidence in any career entry
      (3) LangChain is the ONLY AI signal — no other AI skills and no
          AI evidence in descriptions
    """
    cfg_block = pcfg.get("langchain_only_junior", {})
    score = float(cfg_block.get("score", 0.40))
    junior_months = int(cfg_block.get("junior_exp_months", 12))
    pre_llm_year = int(cfg_block.get("pre_llm_cutoff_year", 2022))

    skills = candidate.get("skills", []) or []

    # (1) total AI experience
    ai_months = 0
    has_langchain = False
    has_other_ai_skill = False
    for s in skills:
        name = (s.get("name") or "").strip()
        if not name:
            continue
        if "langchain" in name.lower():
            has_langchain = True
            continue
        if any(kw in name.lower() for kw in AI_SKILL_KEYWORDS):
            has_other_ai_skill = True
            dur = s.get("duration_months", 0) or 0
            if isinstance(dur, (int, float)):
                ai_months += int(dur)
    if ai_months >= junior_months:
        return None  # not a junior

    # (2) no pre-cutoff-year ML production
    career = candidate.get("career_history", []) or []
    for entry in career:
        end = entry.get("end_date")
        start = entry.get("start_date")
        # Use the entry's start year to check pre-cutoff
        ref_date = start or end
        if not ref_date or not isinstance(ref_date, str):
            continue
        try:
            yr = int(ref_date[:4])
        except (ValueError, IndexError):
            continue
        if yr >= pre_llm_year:
            continue  # not pre-cutoff
        if _has_any_term(entry.get("description") or "", PRODUCTION_LEXICON):
            return None  # pre-cutoff ML production found → condition (2) fails

    # (3) LangChain is the ONLY AI signal
    if not has_langchain:
        return None  # no LangChain at all → this gate doesn't apply
    if has_other_ai_skill:
        return None  # other AI skills present → condition (3) fails
    if _has_any_term(role_fit_text, AI_SKILL_KEYWORDS):
        return None  # AI evidence in descriptions → condition (3) fails

    return score


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _has_any_term(text: str, terms: Iterable[str]) -> bool:
    """Case-insensitive word-boundary check: does `text` contain any of `terms`?

    Uses word-boundary anchors so that, e.g., "search" in PRODUCTION_LEXICON
    does NOT match inside "research" (the keyword-absence trap in mirror
    form — §2.5.c). This is critical for research_only to ever fire on
    academic descriptions that contain the word "research".
    """
    if not text:
        return False
    low = text.lower()
    for t in terms:
        # (?<!\w) ... (?!\w) enforces that the term is bounded by non-word
        # characters (or string edges) on both sides. Equivalent to \b...\b
        # but explicit for clarity.
        pattern = r"(?<!\w)" + re.escape(t.lower()) + r"(?!\w)"
        if re.search(pattern, low):
            return True
    return False
