"""
precompute.py — Offline embedding cache (P4, dev-time only).

Streams candidates, embeds each `career_history[].description` with
`all-MiniLM-L6-v2` (the same model as P1's JD-intent embedding), and
writes three artifacts:

  artifacts/career_embeddings.npy   (N, 384) float32, L2-normalized.
                                     N = total descriptions across all candidates.
  artifacts/candidate_offsets.npz    candidate_ids (M,), offsets (M+1,)
                                     so candidate i's descriptions are
                                     embeddings[offsets[i]:offsets[i+1]].
                                     Also: desc_char_counts (M,) — per-candidate
                                     total description char count, for the
                                     thin-desc fallback in role_fit.
  artifacts/precompute_meta.yaml    model, dim, count, date, intent_queries.

The runtime (rank.py) reads these. It does NOT load the sentence-transformers
model. All embedding work is dev-time per submission_spec §3 / §10.3.

Usage (offline, dev-time only):
    python -m src.precompute --candidates data/samples/sample_candidates.json \
                             --output artifacts/

The 100K production run: `python -m src.precompute --candidates
data/originals/candidates.jsonl --output artifacts/`. This takes minutes
to tens of minutes (uncapped per §10.3); the ≤5 min constraint applies
only to the RANKING step.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from src.data_loader import iter_candidates_jsonl, load_candidates_json
from src.jd_embedding import DEFAULT_MODEL, embed_texts

logger = logging.getLogger(__name__)


def run_precompute(
    candidates_path: str | Path,
    output_dir: str | Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 256,
    force: bool = False,
) -> dict:
    """
    Run the full precompute pass. Returns a meta dict (also written to YAML).

    Args:
        candidates_path: Path to candidates.jsonl or .json.
        output_dir: Where to write the artifacts.
        model_name: Sentence-transformers model (must match the JD-intent model).
        batch_size: Description-encoding batch size.
        force: Overwrite existing artifacts if True.

    Returns:
        Meta dict with model, dim, count, date, intent_queries.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_path = output_dir / "career_embeddings.npy"
    off_path = output_dir / "candidate_offsets.npz"
    meta_path = output_dir / "precompute_meta.yaml"

    if emb_path.exists() and off_path.exists() and meta_path.exists() and not force:
        logger.info("Precompute artifacts already exist at %s. Use force=True to redo.", output_dir)
        return _load_meta(meta_path)

    # Load intent_queries from config so the meta records what was used.
    from src.config_loader import DEFAULT_CONFIG_PATH, load_config

    cfg = load_config()
    intent_queries = list(cfg.get("role_fit", {}).get("intent_queries", []) or [])

    # Stream candidates depending on file format.
    cand_path = Path(candidates_path)
    if cand_path.suffix.lower() == ".jsonl":
        candidates = list(iter_candidates_jsonl(cand_path))
    else:
        candidates = load_candidates_json(cand_path)

    logger.info("Loaded %d candidates from %s", len(candidates), cand_path)

    # Collect all descriptions and per-candidate offsets.
    all_texts: list[str] = []
    offsets: list[int] = [0]
    candidate_ids: list[str] = []
    desc_char_counts: list[int] = []
    for c in candidates:
        cid = c.get("candidate_id", "<missing>")
        descs = [(e.get("description") or "") for e in (c.get("career_history") or [])]
        all_texts.extend(descs)
        candidate_ids.append(cid)
        offsets.append(len(all_texts))
        desc_char_counts.append(sum(len(t) for t in descs))

    logger.info("Encoding %d descriptions (batch_size=%d)…", len(all_texts), batch_size)

    # Encode (batched) and L2-normalize.
    embeddings = embed_texts(all_texts, model_name=model_name) if all_texts else np.zeros((0, 384), dtype=np.float32)
    if embeddings.size > 0:
        # embed_texts already normalizes; ensure float32.
        embeddings = embeddings.astype(np.float32, copy=False)

    # Persist.
    np.save(emb_path, embeddings)
    np.savez(
        off_path,
        candidate_ids=np.array(candidate_ids, dtype=object),
        offsets=np.array(offsets, dtype=np.int64),
        desc_char_counts=np.array(desc_char_counts, dtype=np.int64),
    )

    meta = {
        "model": model_name,
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "num_candidates": int(len(candidates)),
        "num_descriptions": int(len(all_texts)),
        "generated_date": str(date.today()),
        "normalized": True,
        "intent_queries": intent_queries,
        "input_path": str(cand_path),
        "note": "Dev-time only. rank.py loads this at runtime (no model).",
    }
    with meta_path.open("w", encoding="utf-8") as fh:
        yaml.dump(meta, fh, default_flow_style=False)
    logger.info(
        "Wrote precompute artifacts: %d candidates, %d descriptions → %s",
        len(candidates), len(all_texts), output_dir,
    )
    return meta


def load_precomputed(
    output_dir: str | Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """
    Load precompute artifacts. Used by rank.py at runtime.

    Returns:
        (embeddings, offsets_data, meta) where:
          - embeddings: (N, D) float32, L2-normalized
          - offsets_data: dict with arrays 'candidate_ids' (object), 'offsets'
            (int64), 'desc_char_counts' (int64)
          - meta: dict from the YAML
    """
    output_dir = Path(output_dir)
    emb = np.load(output_dir / "career_embeddings.npy", mmap_mode="r")
    off = np.load(output_dir / "candidate_offsets.npz", allow_pickle=True)
    meta = _load_meta(output_dir / "precompute_meta.yaml")
    return emb, dict(off), meta


def _load_meta(meta_path: Path) -> dict:
    with meta_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Precompute career-description embeddings.")
    parser.add_argument("--candidates", required=True, help="Path to candidates .jsonl or .json")
    parser.add_argument("--output", required=True, help="Output directory for artifacts")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_precompute(
        args.candidates,
        args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        force=args.force,
    )


if __name__ == "__main__":
    main()
