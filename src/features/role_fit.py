"""
role_fit.py — The DOMINANT feature (EXCEUTION_PLAN §2.5.a/b/f/h).

` s_role_fit ` is the single largest contributor to fit_score (w_role = 0.45)
and the feature that decides most of NDCG@10. It is the ranker's defense
against the keyword trap: it reads what the career *shows* (descriptions)
not what the title *says*. Titles are distrusted per §3.1.a (measured
1,249/3,000 title↔description mismatches).

The signal is **blended** (EXCEUTION_PLAN §2.5.a):

  s_role_fit = w_dense · s_dense  +  w_lex · s_lex

where:
  - s_dense is max cosine over a small set of frozen JD-intent vectors,
    then top-K-mean pooled across descriptions with a single combined
    per-description weight (duration_norm × recency_decay, §2.5.f).
  - s_lex is a production-evidence lexical match against a broad synonym
    set, so an engineer who wrote "rolled out / served / A-B tested"
    isn't missed by the dense retrieval's guessed vocabulary.

A thin-description fallback to the role-affinity title prior is the
**single** justified use of the otherwise-demoted title (§2.5.h).

The function is pure: no I/O, no model loading. The caller (P4 rank.py)
passes pre-computed embeddings (`embs`) and the frozen intent set
(`jd_intents`). At runtime the model is NOT loaded — vectors are read
from `config/*.npy`.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# Broad production-evidence lexicon (mirrors src/disqualifiers.py but kept
# independent so role_fit can be tuned separately). Word-boundary matching
# via the same approach to prevent the keyword-absence trap (§2.5.c).
PRODUCTION_LEXICON: tuple[str, ...] = (
    # production-shipping verbs
    "shipped", "deploy", "deployed", "deploying", "deployment",
    "launch", "launched", "launching",
    "roll out", "rolled out", "rollout",
    "served", "serving", "in production", "productionized", "productionised",
    "a/b test", "a-b test", "ab test",
    "inference service", "inference api", "online",
    # retrieval / ranking / recsys / search / embeddings
    "retrieval", "ranking", "recommendation system", "recommender system",
    "recommender", "recsys", "search", "search system",
    "embedding", "embeddings", "vector search", "vector database",
    "dense retrieval", "hybrid search", "bm25",
    "ndcg", "mrr", "map",
)


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def s_role_fit(
    candidate: dict[str, Any],
    embs: np.ndarray | None,
    jd_intents: np.ndarray,
    cfg: dict[str, Any],
) -> float:
    """
    Compute the dominant role-fit score in [0, 1] for one candidate.

    Args:
        candidate: One candidate dict.
        embs: Pre-computed per-description embeddings, shape (D, dim),
            L2-normalized. D is the number of career entries. May be None
            if the candidate has no career descriptions (thin-desc path).
        jd_intents: Frozen multi-query intent set, shape (Q, dim),
            L2-normalized rows. Loaded at runtime from
            `config/jd_intent_embeddings.npy`.
        cfg: Full scoring config dict (must contain `role_fit` block and
            `role_affinity` block).

    Returns:
        A float in [0, 1].
    """
    rcfg = cfg.get("role_fit", {})
    min_desc_chars = int(rcfg.get("min_desc_chars", 40))

    # Concatenate all career descriptions for the lexical signal and the
    # thin-description check. Titles are NOT included.
    career = candidate.get("career_history", []) or []
    desc_texts: list[str] = [
        (e.get("description") or "") for e in career
    ]
    total_desc_chars = sum(len(t) for t in desc_texts)

    # §2.5.h thin-description fallback: if there is no usable text, fall
    # back to the role-affinity title prior. This is the single justified
    # use of the otherwise-demoted title lookup (fallback, never a bonus).
    if total_desc_chars < min_desc_chars or not career:
        return _title_prior(candidate, cfg)

    # Pre-check that we have matching embeddings for the descriptions.
    if embs is None or len(embs) != len(career):
        # Mismatch: either no embeddings or shape mismatch. Fall back to
        # the lexical-only signal (still meaningful) blended with the
        # title prior as a soft prior.
        s_dense = 0.0
    else:
        s_dense = _s_dense_pooled(career, embs, jd_intents, rcfg)

    s_lex = _s_lex(desc_texts)

    w_dense = float(rcfg.get("w_dense", 0.7))
    w_lex = float(rcfg.get("w_lex", 0.3))
    blended = w_dense * s_dense + w_lex * s_lex
    return _clamp01(blended)


# -----------------------------------------------------------------------------
# Dense component: multi-query max cosine, top-K-mean pooled, single weight
# -----------------------------------------------------------------------------

def _s_dense_pooled(
    career: list[dict],
    embs: np.ndarray,
    jd_intents: np.ndarray,
    rcfg: dict,
) -> float:
    """
    Multi-query dense role signal.

    Per description d:
      cosine_d = max over Q intent queries of cosine(emb_d, intent_q)
    Then pool across descriptions:
      pool = "max"      → top-1 cosine
      pool = "topk_mean"→ weighted mean of the K highest cosines
                          (single combined weight w_d = duration_norm × recency_decay)
    """
    pool = rcfg.get("pool", "topk_mean")
    k = int(rcfg.get("pool_k", 2))
    use_duration_weight = bool(rcfg.get("duration_weight", True))
    recency_half_life = float(rcfg.get("recency_half_life_months", 36))

    # Per-description cosines: (D, Q) @ (Q,)? → (D, Q) → max over Q → (D,)
    sims = embs @ jd_intents.T
    per_desc_cos = sims.max(axis=1)

    if pool == "max" or len(per_desc_cos) <= 1:
        return float(per_desc_cos.max()) if len(per_desc_cos) else 0.0

    # topk_mean with single combined weight (§2.5.f / SYSTEM_DESIGN §4.1.1).
    # w_d = duration_norm(d) × recency_decay(d) — ONE weight, not two stacked.
    weights = np.array(
        [
            _per_description_weight(d, use_duration_weight, recency_half_life)
            for d in career
        ],
        dtype=np.float64,
    )
    # If all weights are zero (degenerate), fall back to unweighted top-K.
    if float(weights.sum()) <= 0:
        return _topk_mean_unweighted(per_desc_cos, k)

    k = min(k, len(per_desc_cos))
    # argsort descending by cosine, take top K
    top_idx = np.argsort(-per_desc_cos)[:k]
    top_cos = per_desc_cos[top_idx]
    top_w = weights[top_idx]
    return float((top_w * top_cos).sum() / top_w.sum())


def _topk_mean_unweighted(per_desc_cos: np.ndarray, k: int) -> float:
    k = min(k, len(per_desc_cos))
    top_idx = np.argsort(-per_desc_cos)[:k]
    return float(per_desc_cos[top_idx].mean())


def _per_description_weight(
    entry: dict, use_duration_weight: bool, recency_half_life_months: float
) -> float:
    """
    §2.5.f / SYSTEM_DESIGN §4.1.1: ONE combined per-description weight.
    w_d = duration_norm(d) × recency_decay(d), where
      duration_norm = min(duration_months / 24, 1.0)
      recency_decay  = 0.5 ** (months_since_end / recency_half_life_months)
    """
    recency = _recency_decay(entry, recency_half_life_months)
    if not use_duration_weight:
        return recency
    duration_months = max(0, int(entry.get("duration_months", 0) or 0))
    duration_norm = min(duration_months / 24.0, 1.0)
    return duration_norm * recency


def _recency_decay(entry: dict, half_life_months: float) -> float:
    """
    0.5 ** (months_since_end / half_life_months). 1.0 for the most-recent
    stint (is_current or end within the current month); decays to 0 over
    multiple half-lives.
    """
    if half_life_months <= 0:
        return 1.0
    end = entry.get("end_date")
    is_current = entry.get("is_current", False)
    if is_current or not end:
        ref = date.today()
    else:
        try:
            ref = date.fromisoformat(end)
        except (TypeError, ValueError):
            return 0.5  # unparseable date → middling weight, not zero
    today = date.today()
    months_since = (today.year - ref.year) * 12 + (today.month - ref.month)
    if months_since < 0:
        months_since = 0  # future-dated end → treat as current
    return 0.5 ** (months_since / half_life_months)


# -----------------------------------------------------------------------------
# Lexical component: production-evidence term match
# -----------------------------------------------------------------------------

def _s_lex(desc_texts: list[str]) -> float:
    """
    Production-evidence lexical match.

    A simple, corpus-independent proxy for BM25/TF-IDF (P3 scope; a real
    BM25 needs a document-frequency table computed in P4 over the whole
    pool). We use a **saturating hit count**: each production term that
    appears at least once contributes a fixed amount, and the score
    saturates as more distinct terms are found. This keeps s_lex in [0, 1]
    without requiring corpus statistics.

    Word-boundary matching to avoid the keyword-absence trap (§2.5.c):
    e.g. "search" does not match inside "research".
    """
    combined = " ".join(desc_texts)
    if not combined.strip():
        return 0.0
    low = combined.lower()
    # Count distinct production terms present.
    hits = 0
    for term in PRODUCTION_LEXICON:
        if re.search(r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)", low):
            hits += 1
    # Saturating map: 0 hits → 0; 3 distinct terms → ~0.5; 6+ → ~1.0.
    # Score = 1 - 0.5 ** (hits / 3)  →  hits=0→0, 3→0.5, 6→0.75, 9→0.875, 12→0.9375
    if hits <= 0:
        return 0.0
    return 1.0 - 0.5 ** (hits / 3.0)


# -----------------------------------------------------------------------------
# Thin-description fallback: role-affinity title prior
# -----------------------------------------------------------------------------

def _title_prior(candidate: dict, cfg: dict) -> float:
    """
    §2.5.h fallback. Look up `current_title` in `role_affinity`; unknown
    titles get the `default` value. This is a FALLBACK (not a bonus):
    only invoked when the candidate has no usable description text.
    """
    title = (candidate.get("profile", {}).get("current_title") or "").strip()
    affinity = cfg.get("role_affinity", {}) or {}
    if title in affinity:
        return _clamp01(float(affinity[title]))
    return _clamp01(float(affinity.get("default", 0.2)))


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)
