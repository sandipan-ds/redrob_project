#!/usr/bin/env python3
"""
calibrate.py — P5 calibration driver (EXECUTION_PLAN §5.1).

Coordinate search over the documented macro knobs, maximizing the
proxy composite (0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10)
on the 50-sample proxy set + 3 adversarial decoys.

**Honest constraint (§5.1):** ~50 hand-labels is far too little to fit
the ~40 skill weights + 6 top-level weights + 6 penalty constants +
behavior model. Fitting all of them on 50 points = guaranteed overfit.
So this script sweeps ONLY the documented macro knobs:
  - Top-level weight split (role/skill/exp/edu/loc), including w_edu
  - Behavior band (min_multiplier / max_multiplier / neutral_base)
  - p_scale (global penalty severity)
  - retrieve.k (shortlist size)
  - w_dense / w_lex (role-fit blend)

Per-skill weights and role-affinity decimals are FROZEN (do not fit).

The script writes the best config to a CANDIDATE file
(`scoring_config.candidate.yaml`) for human review — NEVER silently
overwrites `scoring_config.yaml`.

Usage:
    python scripts/calibrate.py [--grid coarse|medium|fine] [--out PATH]
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from src.config_loader import load_config
from src.data_loader import load_candidates_json
from src.eval.metrics import composite_score
from src.eval.proxy_labels import adversarial_decoys, load_proxy_tiers
from src.features.behavior import m_behavior
from src.features.education import s_education
from src.features.experience import s_exp_band
from src.features.location import s_location
from src.features.role_fit import s_role_fit
from src.features.skills import s_skill
from src.jd_embedding import load_jd_intent_set
from src.precompute import run_precompute

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


SAMPLE_PATH = REPO / "data" / "samples" / "sample_candidates.json"
SAMPLE_CACHE = REPO / "artifacts" / "sample"
CANDIDATE_OUT = REPO / "config" / "scoring_config.candidate.yaml"


def _eval_config(
    cfg: dict,
    candidates: list[dict],
    tiers: dict[str, int],
    jd_intents,
    embeddings,
    offsets_arr,
    cached_ids,
    id_to_index,
) -> tuple[float, list[str]]:
    """
    Score all candidates with the given config, return (composite, ranked_ids).
    """
    rows: list[tuple[float, str]] = []
    for c in candidates:
        cid = c["candidate_id"]
        feats = _compute_features(c, cfg, jd_intents, embeddings, offsets_arr, id_to_index)
        profile = c.get("profile", {})
        signals = c.get("redrob_signals")
        # Quick m_behavior + p_penalty for final score
        behavior = m_behavior(signals, cfg)
        from src.disqualifiers import compute_penalty
        role_text = " ".join(
            (e.get("description") or "") for e in (c.get("career_history") or [])
        )
        p_pen, _ = compute_penalty(c, cfg, role_text)
        weights = cfg.get("weights", {})
        fit = (
            float(weights.get("role_fit", 0)) * feats["s_role_fit"]
            + float(weights.get("skill", 0)) * feats["s_skill"]
            + float(weights.get("experience", 0)) * feats["s_exp_band"]
            + float(weights.get("education", 0)) * feats["s_education"]
            + float(weights.get("location", 0)) * feats["s_location"]
        )
        score = fit * behavior * p_pen
        rows.append((score, cid))
    # Sort desc; tie-break by candidate_id asc.
    rows.sort(key=lambda r: (-r[0], r[1]))
    ranked_ids = [r[1] for r in rows]
    comp_dict = composite_score(ranked_ids, tiers)
    return comp_dict["composite"], ranked_ids


def _compute_features(
    c, cfg, jd_intents, embeddings, offsets_arr, id_to_index,
) -> dict:
    """Compute the 5 fit features for one candidate."""
    cid = c["candidate_id"]
    profile = c.get("profile", {})
    descs = [(e.get("description") or "") for e in (c.get("career_history") or [])]
    total_chars = sum(len(t) for t in descs)
    min_desc_chars = int(cfg.get("role_fit", {}).get("min_desc_chars", 40))
    if cid in id_to_index and total_chars >= min_desc_chars:
        idx = id_to_index[cid]
        start = int(offsets_arr[idx])
        end = int(offsets_arr[idx + 1])
        cand_embs = embeddings[start:end]
    else:
        cand_embs = None
    return {
        "s_role_fit": s_role_fit(c, cand_embs, jd_intents, cfg),
        "s_skill": s_skill(c, cfg),
        "s_exp_band": s_exp_band(profile.get("years_of_experience", 0), cfg),
        "s_education": s_education(c.get("education", []), cfg),
        "s_location": s_location(profile, c.get("redrob_signals"), cfg),
    }


def _normalize_weights(w: dict) -> dict:
    """Renormalize a weight split so the values sum to 1.0."""
    total = sum(w.values())
    if total <= 0:
        return w
    return {k: v / total for k, v in w.items()}


def _grid(grid_size: str) -> dict:
    """Return the macro-knob grid for the given size."""
    if grid_size == "coarse":
        return {
            "w_edu": [0.05, 0.10, 0.15],
            "p_scale": [0.5, 1.0, 1.5],
            "w_dense": [0.6, 0.7, 0.8],
            "retrieve_k": [1500],
            "behavior_neutral": [0.85],  # frozen by default
        }
    # medium
    return {
        "w_edu": [0.05, 0.10, 0.15],
        "p_scale": [0.5, 0.75, 1.0, 1.25, 1.5],
        "w_dense": [0.6, 0.7, 0.8],
        "retrieve_k": [1500],
        "behavior_neutral": [0.85],
    }


def _set_in(d: dict, path: tuple, val) -> None:
    """Set d[path[0]][path[1]]... = val, creating intermediate dicts as needed."""
    cur = d
    for k in path[:-1]:
        cur = cur.setdefault(k, {})
    cur[path[-1]] = val


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate macro knobs on the proxy set.")
    parser.add_argument("--grid", choices=["coarse", "medium"], default="coarse")
    parser.add_argument("--out", type=Path, default=CANDIDATE_OUT)
    parser.add_argument("--decoy-tier", type=int, default=1,
                        help="Tier to assign to adversarial decoys (default 1)")
    args = parser.parse_args()

    # Load base config + proxy tiers + candidates.
    base_cfg = load_config()
    tiers = load_proxy_tiers()
    decoys = adversarial_decoys()
    for d in decoys:
        tiers[d["candidate_id"]] = args.decoy_tier

    candidates = load_candidates_json(SAMPLE_PATH) + decoys
    logger.info("Loaded %d candidates (%d sample + %d decoys)",
                len(candidates), len(candidates) - len(decoys), len(decoys))

    # Ensure precompute is run.
    if not (SAMPLE_CACHE / "career_embeddings.npy").exists():
        run_precompute(SAMPLE_PATH, SAMPLE_CACHE, force=True)
    from src.precompute import load_precomputed
    embeddings, off, _meta = load_precomputed(SAMPLE_CACHE)
    jd_intents = load_jd_intent_set()
    cached_ids = list(off["candidate_ids"])
    offsets_arr = off["offsets"]
    id_to_index = {cid: i for i, cid in enumerate(cached_ids)}

    grid = _grid(args.grid)
    # Cartesian product over the macro knobs.
    keys = sorted(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    logger.info("Searching %d configurations (grid=%s)…", len(combos), args.grid)

    best_score = -1.0
    best_cfg: dict | None = None
    best_ranked: list[str] = []
    t0 = time.perf_counter()
    for combo in combos:
        cfg = copy.deepcopy(base_cfg)
        for k, v in zip(keys, combo):
            if k == "w_edu":
                # Re-balance the top-level weight split: keep role/skill/exp ratios,
                # shift the w_edu share, redistribute the freed mass proportionally.
                cur_w = cfg["weights"]
                freed = 0.10 - v
                # Redistribute freed mass to role/skill/exp in their current proportion.
                if v != 0.10:
                    donors = {kk: cur_w[kk] for kk in ("role_fit", "skill", "experience")}
                    donor_sum = sum(donors.values()) or 1.0
                    for kk, dv in donors.items():
                        cur_w[kk] = round(dv + freed * (dv / donor_sum), 6)
                cur_w["education"] = v
                # Normalize to sum exactly 1.0.
                cfg["weights"] = _normalize_weights(cur_w)
            elif k == "p_scale":
                cfg["penalties"]["p_scale"] = v
            elif k == "w_dense":
                cfg["role_fit"]["w_dense"] = v
                cfg["role_fit"]["w_lex"] = round(1.0 - v, 6)
            elif k == "behavior_neutral":
                cfg["behavior"]["neutral_base"] = v
            elif k == "retrieve_k":
                cfg["retrieve"]["k"] = v
        try:
            score, ranked = _eval_config(
                cfg, candidates, tiers, jd_intents,
                embeddings, offsets_arr, cached_ids, id_to_index,
            )
        except Exception as exc:
            logger.debug("Config %s failed: %s", combo, exc)
            continue
        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_ranked = ranked
    elapsed = time.perf_counter() - t0
    logger.info("Best composite: %.4f in %.1fs (%d configs)", best_score, elapsed, len(combos))

    # Report top-10 with tier.
    logger.info("Top-10 (rank | id | tier):")
    for i, cid in enumerate(best_ranked[:10], start=1):
        logger.info("  %2d | %s | tier %d", i, cid, tiers.get(cid, 0))

    # Write candidate config (NEVER overwrite scoring_config.yaml).
    import yaml
    if best_cfg is not None:
        with args.out.open("w", encoding="utf-8") as fh:
            yaml.dump(best_cfg, fh, default_flow_style=False, sort_keys=False)
        logger.info("Wrote candidate config to %s (review before applying).", args.out)


if __name__ == "__main__":
    main()
