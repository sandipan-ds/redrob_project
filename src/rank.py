"""
rank.py — Runtime entrypoint (P4, EXCEUTION_PLAN §3, §7.1).

Ranks the top-N candidates from a candidates file and writes a
submission CSV. **No network, no model loading at runtime** — all
embedding work is offline (precompute.py). The ranking step must fit
in ≤5 min on the full 100K pool, CPU-only, ≤16 GB RAM (spec §3 / §7.1).

Usage:
    python rank.py --candidates data/originals/candidates.jsonl \
                   --cache artifacts/ \
                   --out outputs/submission.csv \
                   [--top-n 100] [--shortlist-k 1500]

For the dev 50-sample:
    python rank.py --candidates data/samples/sample_candidates.json \
                   --cache artifacts/sample/ \
                   --out outputs/sample_submission.csv \
                   --top-n 50
"""

from __future__ import annotations

import argparse
import csv
import logging
import socket
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# No-network guard (spec §3: "ranking code must not make external API calls")
# Overrides socket.socket at import time. An accidental network call raises.
# ---------------------------------------------------------------------------
_orig_socket_factory = socket.socket


def _blocked_socket(*args, **kwargs):
    raise RuntimeError(
        "Network access blocked at ranking runtime "
        "(submission_spec §3). socket.socket was called."
    )


socket.socket = _blocked_socket  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Redrob ranking step (runtime).")
    parser.add_argument("--candidates", required=True,
                        help="Path to candidates .jsonl or .json")
    parser.add_argument("--cache", required=True,
                        help="Path to precompute artifacts dir "
                             "(career_embeddings.npy + candidate_offsets.npz)")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--top-n", type=int, default=100,
                        help="Number of candidates in the output CSV (default 100)")
    parser.add_argument("--shortlist-k", type=int, default=None,
                        help="Override retrieve.k (default: from config)")
    args = parser.parse_args()

    t0 = time.perf_counter()
    rank(
        candidates_path=Path(args.candidates),
        cache_dir=Path(args.cache),
        out_path=Path(args.out),
        top_n=args.top_n,
        shortlist_k_override=args.shortlist_k,
    )
    elapsed = time.perf_counter() - t0
    logger.info("Ranking step completed in %.2fs", elapsed)


def rank(
    candidates_path: Path,
    cache_dir: Path,
    out_path: Path,
    top_n: int = 100,
    shortlist_k_override: int | None = None,
) -> None:
    """
    The ranking step. Loads cached embeddings, shortlists, reranks, writes CSV.
    """
    from src.config_loader import load_config
    from src.data_loader import iter_candidates_jsonl, load_candidates_json
    from src.features.education import s_education
    from src.features.experience import s_exp_band
    from src.features.location import s_location
    from src.features.role_fit import s_role_fit
    from src.features.skills import s_skill
    from src.jd_embedding import load_jd_intent_set
    from src.precompute import load_precomputed
    from src.retrieve import shortlist
    from src.scoring import final_score

    cfg = load_config()

    # ---- Load cached artifacts (embeddings + offsets + jd_intents) ----
    embeddings, offsets_data, _meta = load_precomputed(cache_dir)
    jd_intents = load_jd_intent_set()
    cached_ids = list(offsets_data["candidate_ids"])
    offsets_arr = np.asarray(offsets_data["offsets"], dtype=np.int64)
    # Index candidate_id → position in cached_ids (for offsets_arr[i]/[i+1]).
    id_to_index = {cid: i for i, cid in enumerate(cached_ids)}

    # ---- Stream candidates from the input ----
    if candidates_path.suffix.lower() == ".jsonl":
        candidates = list(iter_candidates_jsonl(candidates_path))
    else:
        candidates = load_candidates_json(candidates_path)
    logger.info("Loaded %d candidates from %s", len(candidates), candidates_path)

    # ---- First pass: compute s_dense + s_skill + s_exp for ALL candidates
    # (s_skill and s_exp are scalar — fast even on 100K).
    s_dense = np.zeros(len(candidates), dtype=np.float64)
    s_skill_arr = np.zeros(len(candidates), dtype=np.float64)
    s_exp_arr = np.zeros(len(candidates), dtype=np.float64)
    desc_texts_list: list[str] = []  # for the research_only gate
    for i, c in enumerate(candidates):
        cid = c.get("candidate_id", "<missing>")
        # s_dense: max cosine over jd_intents for the candidate's descriptions.
        if cid in id_to_index:
            idx_in_cache = id_to_index[cid]
            start = int(offsets_arr[idx_in_cache])
            end = int(offsets_arr[idx_in_cache + 1])
            if end > start:
                cand_embs = embeddings[start:end]
                sims = cand_embs @ jd_intents.T
                s_dense[i] = float(sims.max())
        # Scalar features (fast).
        s_skill_arr[i] = s_skill(c, cfg)
        s_exp_arr[i] = s_exp_band(c.get("profile", {}).get("years_of_experience", 0), cfg)
        # Concatenated descriptions for the research_only gate.
        desc_texts_list.append(
            " ".join(
                (e.get("description") or "")
                for e in (c.get("career_history") or [])
            )
        )

    # ---- Two-band retrieval ----
    rcfg = dict(cfg.get("retrieve", {}) or {})
    if shortlist_k_override is not None:
        rcfg["k"] = shortlist_k_override
    indices = shortlist(s_dense, cfg, s_skill=s_skill_arr, s_exp=s_exp_arr)
    # If the shortlist is smaller than the pool (e.g. small k on big pool),
    # ensure we still have a workable set; on the 50-sample with default k=1500
    # the shortlist is all 50.
    if len(indices) == 0:
        indices = np.arange(len(candidates), dtype=np.int64)
    logger.info("Shortlist: %d candidates (out of %d)", len(indices), len(candidates))

    # ---- Rerank: compute full feature set + final score for shortlisted ----
    rows: list[dict] = []
    for idx in indices:
        c = candidates[int(idx)]
        cid = c.get("candidate_id", "<missing>")
        profile = c.get("profile", {}) or {}
        descs = [(e.get("description") or "") for e in (c.get("career_history") or [])]
        total_chars = sum(len(t) for t in descs)
        # s_role_fit needs the pre-computed per-description embeddings.
        if cid in id_to_index and total_chars >= int(cfg.get("role_fit", {}).get("min_desc_chars", 40)):
            idx_in_cache = id_to_index[cid]
            start = int(offsets_arr[idx_in_cache])
            end = int(offsets_arr[idx_in_cache + 1])
            cand_embs = embeddings[start:end]
        else:
            cand_embs = None

        feats = {
            "s_role_fit": s_role_fit(c, cand_embs, jd_intents, cfg),
            "s_skill": s_skill(c, cfg),
            "s_exp_band": s_exp_band(profile.get("years_of_experience", 0), cfg),
            "s_education": s_education(c.get("education", []), cfg),
            "s_location": s_location(profile, c.get("redrob_signals"), cfg),
        }
        score, breakdown = final_score(c, feats, cfg, role_fit_text=desc_texts_list[int(idx)])

        rows.append(
            {
                "candidate_id": cid,
                "score": float(score),
                "reasoning": _minimal_reasoning(profile, breakdown),
            }
        )

    # ---- Sort: score desc, tie-break by candidate_id asc ----
    rows.sort(key=lambda r: (-r["score"], r["candidate_id"]))

    # ---- Take top N and write CSV ----
    top = rows[:top_n]
    _write_csv(out_path, top)
    logger.info("Wrote %d rows to %s", len(top), out_path)


def _minimal_reasoning(profile: dict, breakdown: dict) -> str:
    """
    A 1-sentence placeholder for the reasoning column. P6 replaces this
    with the deterministic template-driven generator. The output must
    be a non-empty string and must contain only values present in the
    profile (anti-hallucination — spec §3 reasoning checks).
    """
    yoe = profile.get("years_of_experience", 0)
    title = (profile.get("current_title") or "").strip()
    return f"{title or 'Candidate'}, {yoe} yrs."


def _write_csv(out_path: Path, rows: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, row in enumerate(rows, start=1):
            w.writerow(
                [
                    row["candidate_id"],
                    rank,
                    f"{row['score']:.6f}",
                    row["reasoning"],
                ]
            )


if __name__ == "__main__":
    main()
