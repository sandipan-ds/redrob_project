"""
retrieve.py — Two-band retrieval (P4, EXCEUTION_PLAN §2.5 / MINIMAX #7).

The ranker does NOT score all 100K candidates at runtime — that's too
slow. Instead, we shortlist a small set (1–2K) and rerank.

Two-band retrieval:
  1. First band: top-`k` by `s_dense` (max cosine over the multi-query
     intent set). This catches the obvious fits.
  2. Second-chance band: candidates NOT in the first band with
     (s_skill + s_exp_band) / 2 > `second_chance_threshold` — real
     fits with jargon-heavy phrasing that the dense centroid may
     underestimate. This is the "jargon-heavy real fit" guard.
  3. Union → shortlist. Ranker reranks this set with the full feature
     suite.

The function is pure: numpy in, numpy out. No I/O. Called by rank.py.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def shortlist(
    s_dense: np.ndarray,
    cfg: dict[str, Any],
    s_skill: np.ndarray | None = None,
    s_exp: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute the shortlisted candidate indices via two-band retrieval.

    Args:
        s_dense: (M,) per-candidate max cosine over the JD-intent set.
        cfg: Full scoring config dict (must contain `retrieve` block
            with `k` and `second_chance_threshold`).
        s_skill: (M,) optional per-candidate s_skill, for the second-
            chance band. If None, the second-chance band is skipped.
        s_exp: (M,) optional per-candidate s_exp_band, for the second-
            chance band. If None, the second-chance band is skipped.

    Returns:
        np.ndarray of integer indices into the candidate list, sorted
        descending by s_dense. Size ≤ k + |second-chance band|.
    """
    rcfg = cfg.get("retrieve", {}) or {}
    k = int(rcfg.get("k", 1500))
    threshold = float(rcfg.get("second_chance_threshold", 0.7))
    max_second = int(rcfg.get("second_chance_max", 500))

    M = len(s_dense)
    if M == 0:
        return np.array([], dtype=np.int64)

    # First band: top-k by s_dense.
    k_eff = min(k, M)
    first_band = np.argsort(-s_dense)[:k_eff]
    first_set = set(first_band.tolist())

    # Second-chance band: NOT in first, (s_skill + s_exp)/2 > threshold.
    second_band_list: list[int] = []
    if s_skill is not None and s_exp is not None and threshold > 0:
        avg = 0.5 * (np.asarray(s_skill, dtype=np.float64)
                     + np.asarray(s_exp, dtype=np.float64))
        # candidates NOT in first band with avg > threshold
        mask = (avg > threshold)
        # exclude first band
        for idx in np.where(mask)[0].tolist():
            if idx not in first_set:
                second_band_list.append(idx)
        # cap second band
        if len(second_band_list) > max_second:
            # rank by avg desc, take top max_second
            ranked = sorted(
                second_band_list,
                key=lambda i: float(0.5 * (s_skill[i] + s_exp[i])),
                reverse=True,
            )
            second_band_list = ranked[:max_second]

    # Union, preserving first-band order (descending s_dense), then
    # second-chance candidates appended in their ranked order.
    shortlist_idx = list(first_band.tolist()) + second_band_list
    return np.array(shortlist_idx, dtype=np.int64)
