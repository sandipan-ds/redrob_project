"""
metrics.py — P5 ranking metrics (submission_spec §4).

Exact implementations of:
  NDCG@10, NDCG@50, MAP, P@10
and the composite:
  composite = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10

Relevance is on a 0–5 tier scale (from `data/labels/proxy_tiers.json`).
"Relevant" for P@10 is tier ≥ 3 (a reasonable threshold; tier 0–2 is
"not a match", tier 3+ is "matches to some degree"). The threshold is
exposed as a parameter for tests.

All metrics are pure functions of (ordered candidate_ids, tier_map).
No I/O, no model, no numpy required (works on plain lists).
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

# Composite weights from submission_spec §4 (exact).
COMPOSITE_WEIGHTS = {
    "ndcg_at_10": 0.50,
    "ndcg_at_50": 0.30,
    "map": 0.15,
    "p_at_10": 0.05,
}

# Default relevance threshold for P@k: tier >= 3 = "relevant".
DEFAULT_RELEVANCE_THRESHOLD = 3


def dcg_at_k(
    ranked_ids: Sequence[str],
    tiers: Mapping[str, int],
    k: int,
) -> float:
    """
    Discounted Cumulative Gain @ k (with the standard log2(i+1) discount
    and 2^rel - 1 gain). Pure function.
    """
    total = 0.0
    for i, cid in enumerate(ranked_ids[:k], start=1):
        rel = max(0, int(tiers.get(cid, 0)))
        # 2^rel - 1 (standard NDCG gain, clipped at rel=0)
        gain = (2 ** rel) - 1.0
        total += gain / math.log2(i + 1)
    return total


def ndcg_at_k(
    ranked_ids: Sequence[str],
    tiers: Mapping[str, int],
    k: int,
) -> float:
    """
    Normalized DCG @ k. IDCG@k uses the same `k` positions as DCG@k.
    If no relevant item exists in the top k (or in the entire list, when
    k exceeds the list length), returns 0.0.
    """
    dcg = dcg_at_k(ranked_ids, tiers, k)
    # Ideal ordering: sort all relevant items (tier >= 1, since gain
    # is 0 for tier 0) by tier descending.
    relevant = [t for t in tiers.values() if t > 0]
    relevant.sort(reverse=True)
    ideal_ranked = [f"__ideal_{i}__" for i in range(len(relevant))]
    # IDCG uses tier values directly.
    idcg = dcg_at_k(ideal_ranked, {f"__ideal_{i}__": relevant[i] for i in range(len(relevant))}, k)
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def average_precision(
    ranked_ids: Sequence[str],
    tiers: Mapping[str, int],
    relevance_threshold: int = DEFAULT_RELEVANCE_THRESHOLD,
) -> float:
    """
    Average Precision for a single ranked list. AP = sum_{i} [P(i) × rel(i)]
    / R, where P(i) is precision at rank i, rel(i) is 1 if item at rank i
    is relevant (tier >= threshold), and R is the total number of
    relevant items. Returns 0.0 if R == 0.
    """
    relevant_total = sum(1 for t in tiers.values() if t >= relevance_threshold)
    if relevant_total == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, cid in enumerate(ranked_ids, start=1):
        if int(tiers.get(cid, 0)) >= relevance_threshold:
            hits += 1
            precision_sum += hits / i
    return precision_sum / relevant_total


def precision_at_k(
    ranked_ids: Sequence[str],
    tiers: Mapping[str, int],
    k: int,
    relevance_threshold: int = DEFAULT_RELEVANCE_THRESHOLD,
) -> float:
    """
    Precision @ k: fraction of the top-k that are relevant (tier >= threshold).
    """
    if k <= 0:
        return 0.0
    top = ranked_ids[:k]
    hits = sum(1 for cid in top if int(tiers.get(cid, 0)) >= relevance_threshold)
    return hits / k


def composite_score(
    ranked_ids: Sequence[str],
    tiers: Mapping[str, int],
    relevance_threshold: int = DEFAULT_RELEVANCE_THRESHOLD,
) -> dict:
    """
    All four metrics + the exact composite from submission_spec §4.
    Returns a dict with keys: ndcg_at_10, ndcg_at_50, map, p_at_10, composite.
    """
    n10 = ndcg_at_k(ranked_ids, tiers, 10)
    n50 = ndcg_at_k(ranked_ids, tiers, 50)
    m = average_precision(ranked_ids, tiers, relevance_threshold)
    p10 = precision_at_k(ranked_ids, tiers, 10, relevance_threshold)
    comp = (
        COMPOSITE_WEIGHTS["ndcg_at_10"] * n10
        + COMPOSITE_WEIGHTS["ndcg_at_50"] * n50
        + COMPOSITE_WEIGHTS["map"] * m
        + COMPOSITE_WEIGHTS["p_at_10"] * p10
    )
    return {
        "ndcg_at_10": n10,
        "ndcg_at_50": n50,
        "map": m,
        "p_at_10": p10,
        "composite": comp,
    }
