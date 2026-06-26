"""
app.py — Streamlit web UI for the Hugging Face Space.

The spec §10.5 requires the sandbox to:
  - Accept a small candidate sample (≤100 candidates) as input
  - Run the ranking system end-to-end and produce a ranked CSV
  - Complete within the compute budget (≤5 min on CPU)

This UI does exactly that. Users upload a candidate JSON/JSONL file
(≤100 candidates), the app runs the ranker on it, and the user can
download the ranked CSV. The precomputed embeddings (from the
vendored 50-sample) are used as-is; the ranker reuses them when the
candidate is in the sample, and falls back to a title-prior for
unseen candidates.

HuggingFace Spaces runs Streamlit on port 7860 by convention.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO))

# ---------------------------------------------------------------------------
# Streamlit page config (must be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached model artifacts (loaded once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading JD-intent embeddings + ranker...")
def load_ranker():
    """Load the JD-intent set, precomputed cache, and config."""
    from src.config_loader import load_config
    from src.jd_embedding import load_jd_intent_set
    from src.precompute import load_precomputed

    cfg = load_config()
    jd_intents = load_jd_intent_set()
    emb, off, _meta = load_precomputed(REPO / "artifacts" / "sample")
    return cfg, jd_intents, emb, off


def parse_uploaded_candidates(uploaded_file) -> list[dict]:
    """
    Parse the uploaded file into a list of candidate dicts.

    Accepts:
      - JSON: a list of dicts, or a single dict
      - JSONL: one dict per line
    """
    raw = uploaded_file.read()
    text = raw.decode("utf-8")
    # Try JSON first (list or single object)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass
    # Fall back to JSONL
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def score_candidates(candidates: list[dict], cfg, jd_intents, emb, off) -> tuple[list[dict], float]:
    """
    Run the ranker on the uploaded candidates. Reuses precomputed
    embeddings for any candidate in the sample; falls back to a
    title-prior for unseen candidates.

    Returns (rows_for_csv, elapsed_seconds).
    """
    from src.features.role_fit import s_role_fit
    from src.features.skills import s_skill
    from src.features.experience import s_exp_band
    from src.features.education import s_education
    from src.features.location import s_location
    from src.features.behavior import m_behavior
    from src.disqualifiers import compute_penalty
    from src.reasoning import generate_reasoning
    import numpy as np

    cached_ids = list(off["candidate_ids"])
    offsets_arr = off["offsets"]
    id_to_index = {cid: i for i, cid in enumerate(cached_ids)}

    weights = cfg.get("weights", {}) or {}
    rows: list[dict] = []
    t0 = time.perf_counter()
    for cand in candidates:
        cid = cand.get("candidate_id", "?")
        # Get precomputed embeddings if available
        cand_embs = None
        if cid in id_to_index:
            idx = id_to_index[cid]
            start = int(offsets_arr[idx])
            end = int(offsets_arr[idx + 1])
            cand_embs = emb[start:end]

        feats = {
            "s_role_fit": s_role_fit(cand, cand_embs, jd_intents, cfg),
            "s_skill": s_skill(cand, cfg),
            "s_exp_band": s_exp_band(cand.get("profile", {}).get("years_of_experience", 0), cfg),
            "s_education": s_education(cand.get("education", []), cfg),
            "s_location": s_location(cand.get("profile", {}), cand.get("redrob_signals"), cfg),
        }
        fit = (
            float(weights.get("role_fit", 0)) * feats["s_role_fit"]
            + float(weights.get("skill", 0)) * feats["s_skill"]
            + float(weights.get("experience", 0)) * feats["s_exp_band"]
            + float(weights.get("education", 0)) * feats["s_education"]
            + float(weights.get("location", 0)) * feats["s_location"]
        )
        beh = m_behavior(cand.get("redrob_signals"), cfg)
        pen, reasons = compute_penalty(cand, cfg)
        score = fit * beh * pen

        rows.append({
            "candidate_id": cid,
            "score": score,
            "fit_score": fit,
            "m_behavior": beh,
            "p_penalty": pen,
            "gate_reasons": reasons,
            "breakdown": feats,
            "candidate": cand,
        })

    # Sort by score desc, tie-break by candidate_id asc
    rows.sort(key=lambda r: (-r["score"], r["candidate_id"]))

    # Assign ranks and generate reasoning
    for rank, r in enumerate(rows, start=1):
        r["rank"] = rank
        r["reasoning"] = generate_reasoning(
            r["candidate"], r["breakdown"], rank=rank, cfg=cfg
        )

    elapsed = time.perf_counter() - t0
    return rows, elapsed


def to_csv(rows: list[dict], top_n: int = 100) -> str:
    """Build the submission CSV string from the top-N rows."""
    out_lines = ["candidate_id,rank,score,reasoning"]
    for r in rows[:top_n]:
        # CSV-escape the reasoning (it can contain commas/quotes)
        reasoning = str(r.get("reasoning", ""))
        if '"' in reasoning or ',' in reasoning or '\n' in reasoning:
            reasoning = '"' + reasoning.replace('"', '""') + '"'
        out_lines.append(f"{r['candidate_id']},{r['rank']},{r['score']:.6f},{reasoning}")
    return "\n".join(out_lines) + "\n"


# ---------------------------------------------------------------------------
# Page body
# ---------------------------------------------------------------------------
st.title("🧠 Redrob Candidate Ranker")
st.markdown(
    """
Rank the top 100 of a candidate pool against a **Senior AI Engineer JD**.
Upload a candidate JSON/JSONL file (≤100 candidates), and the ranker
will score, sort, and produce a downloadable ranked CSV.

**Hard constraints (submission_spec §3):** CPU-only, ≤16 GB RAM,
≤5 min ranking step, no network, no LLM at runtime.
"""
)

with st.expander("ℹ️ About the ranker", expanded=False):
    st.markdown(
        """
The system reads career descriptions (not titles — 1,249/3,000 title-
description mismatches in the data), embeds them with
`all-MiniLM-L6-v2`, and compares to 4 frozen JD-intent queries.
The blend is `0.4765·dense + 0.2647·lex` with top-K-mean pooling and
recency weighting. Hard penalty gates (honeypot, consulting-only,
research-only, etc.) are applied multiplicatively.

For the full architecture see `docs/project_explanation/`. For the
weights derivation see `docs/project_explanation/WEIGHT_REVISIONS.md`.
"""
)

# File uploader
uploaded = st.file_uploader(
    "Upload candidate file (JSON or JSONL, ≤100 candidates)",
    type=["json", "jsonl", "txt"],
    help="JSON: a list of candidate dicts, or a single dict. JSONL: one candidate per line.",
)

# Sample data option
use_sample = st.checkbox(
    "Or use the vendored 50-sample (data/samples/sample_candidates.json)",
    value=False,
)

if uploaded is not None or use_sample:
    try:
        if use_sample:
            sample_path = REPO / "data" / "samples" / "sample_candidates.json"
            candidates = json.loads(sample_path.read_text(encoding="utf-8"))
            st.info(f"Using vendored 50-sample ({len(candidates)} candidates).")
        else:
            candidates = parse_uploaded_candidates(uploaded)
            st.info(f"Loaded {len(candidates)} candidates from upload.")
    except Exception as e:
        st.error(f"Failed to parse file: {e}")
        st.stop()

    if len(candidates) > 100:
        st.warning(
            f"⚠️ You uploaded {len(candidates)} candidates. The spec §10.5 sandbox "
            f"expects ≤100. The ranker will still work, but the top-100 will be truncated."
        )

    top_n = st.slider("Top-N to include in the output CSV", min_value=1, max_value=100, value=100)

    if st.button("🚀 Run ranker", type="primary"):
        with st.spinner("Scoring candidates..."):
            try:
                cfg, jd_intents, emb, off = load_ranker()
                rows, elapsed = score_candidates(candidates, cfg, jd_intents, emb, off)
            except Exception as e:
                st.error(f"Ranking failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()

        st.success(f"✅ Ranked {len(rows)} candidates in {elapsed:.2f}s.")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Candidates", len(rows))
        col2.metric("Top score", f"{rows[0]['score']:.4f}" if rows else "—")
        col3.metric("Median score", f"{rows[len(rows)//2]['score']:.4f}" if rows else "—")
        col4.metric("Time", f"{elapsed:.2f}s")

        # Top-N table
        st.subheader(f"Top {min(top_n, len(rows))} ranked candidates")
        table_data = []
        for r in rows[:top_n]:
            table_data.append({
                "Rank": r["rank"],
                "Candidate ID": r["candidate_id"],
                "Score": round(r["score"], 4),
                "Fit": round(r["fit_score"], 4),
                "M_behavior": round(r["m_behavior"], 4),
                "P_penalty": round(r["p_penalty"], 4),
                "Gates": ", ".join(r["gate_reasons"]) or "—",
                "Reasoning": r["reasoning"],
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        # CSV download
        csv_str = to_csv(rows, top_n=top_n)
        st.download_button(
            label="⬇️ Download ranked CSV",
            data=csv_str,
            file_name="submission.csv",
            mime="text/csv",
            help="Save the ranked top-N as a submission.csv file.",
        )

        # Store in session for re-download without re-running
        st.session_state["last_csv"] = csv_str
        st.session_state["last_rows"] = rows

elif "last_csv" in st.session_state:
    st.info("Showing results from the last run. Upload a new file above to re-run.")
    st.download_button(
        label="⬇️ Re-download last CSV",
        data=st.session_state["last_csv"],
        file_name="submission.csv",
        mime="text/csv",
    )
else:
    st.info("👆 Upload a candidate file or check the sample option above to get started.")
