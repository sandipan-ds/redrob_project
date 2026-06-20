"""
jd_embedding.py — Offline JD-intent embedding generator.

This module is run ONCE at dev-time (P1) to produce a frozen embedding of the
JD's intent — what a good candidate *does*, not what keywords they list.

The embedding is saved to config/jd_intent_embedding.npy and loaded at runtime
by the ranker. No LLM or network call happens during ranking.

Usage (offline, dev-time only):
    python -m src.jd_embedding

Output:
    config/jd_intent_embedding.npy  — shape (embedding_dim,), float32
    config/jd_embedding_meta.yaml   — model name, dim, date generated
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JD Intent Text
# ---------------------------------------------------------------------------
# This is NOT the full JD. It is a distilled description of what a GOOD
# candidate's career history *looks like* — written to match the semantic
# space of career_history[].description text in the candidate profiles.
#
# Design rationale (from criteria_map.md Section B):
#   - The JD says the right answer is NOT keyword matching.
#   - The dominant signal is: did this person BUILD retrieval/ranking/recsys
#     systems at a product company and ship them to real users?
#   - We embed THAT intent, then compute cosine similarity against each
#     candidate's concatenated career_history descriptions.
#   - This defeats the keyword trap because a "Marketing Manager" with all
#     the right skill keywords will have career descriptions about campaigns
#     and KPIs, not about embedding drift and NDCG regression.
#
# The text below is written in the same register as career_history descriptions
# so the embedding space is well-aligned.

JD_INTENT_TEXT = """
Built and deployed production retrieval and ranking systems serving real users at scale.
Designed and shipped embedding-based candidate or product search using dense vector retrieval,
hybrid BM25 plus semantic search, and vector databases such as Milvus, Pinecone, Weaviate,
FAISS, or Elasticsearch. Owned the full ML lifecycle: data pipeline, model training,
offline evaluation using NDCG, MRR, and MAP, A/B testing, and production monitoring.
Worked on recommendation systems, information retrieval, or search ranking at a product company,
not a consulting firm. Applied machine learning in production with Python, PyTorch, or TensorFlow.
Fine-tuned or adapted large language models using LoRA or PEFT for domain-specific retrieval tasks.
Built evaluation frameworks to measure retrieval quality and catch regression before deployment.
Shipped features to real users and iterated based on recruiter or user engagement metrics.
Experience with sentence-transformers, Hugging Face transformers, BGE, E5, or OpenAI embeddings
in a production setting, handling embedding drift and index refresh at scale.
Strong software engineering practices: code review, testing, CI/CD, containerization with Docker.
"""

JD_INTENT_TEXT = JD_INTENT_TEXT.strip()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_DIR = Path(__file__).parent.parent / "config"
EMBEDDING_PATH = CONFIG_DIR / "jd_intent_embedding.npy"           # legacy single vector
INTENT_SET_PATH = CONFIG_DIR / "jd_intent_embeddings.npy"          # multi-query set (Q, dim)
INTENT_SET_META_PATH = CONFIG_DIR / "jd_intent_embeddings_meta.yaml"
SCORING_CONFIG_PATH = CONFIG_DIR / "scoring_config.yaml"
META_PATH = CONFIG_DIR / "jd_embedding_meta.yaml"

# Model: lightweight, CPU-friendly, strong on technical text
# all-MiniLM-L6-v2: 384-dim, ~80MB, fast on CPU, good semantic alignment
#
# We vendor the model into models/all-MiniLM-L6-v2/ so a clean machine
# can run precompute with no network (PHASED_BUILD_PLAN §P7 task 1).
# DEFAULT_MODEL points at the vendored path; SentenceTransformer accepts
# both short names and local directories.
REPO_ROOT = Path(__file__).parent.parent
VENDORED_MODEL_DIR = REPO_ROOT / "models" / "all-MiniLM-L6-v2"
DEFAULT_MODEL = str(VENDORED_MODEL_DIR)


def generate_jd_embedding(
    model_name: str = DEFAULT_MODEL,
    output_path: Path = EMBEDDING_PATH,
    meta_path: Path = META_PATH,
    force: bool = False,
) -> np.ndarray:
    """
    Generate and save the JD-intent embedding.

    Args:
        model_name: Sentence-transformers model to use.
        output_path: Where to save the .npy embedding.
        meta_path: Where to save the metadata YAML.
        force: Regenerate even if embedding already exists.

    Returns:
        The embedding as a float32 numpy array of shape (dim,).
    """
    if output_path.exists() and not force:
        logger.info(
            "JD embedding already exists at %s. Use force=True to regenerate.", output_path
        )
        return load_jd_embedding(output_path)

    logger.info("Loading sentence-transformers model: %s", model_name)
    from sentence_transformers import SentenceTransformer  # lazy import — not needed at runtime

    model = SentenceTransformer(model_name)

    logger.info("Encoding JD intent text (%d chars)...", len(JD_INTENT_TEXT))
    embedding = model.encode(JD_INTENT_TEXT, normalize_embeddings=True)
    embedding = embedding.astype(np.float32)

    # Save embedding
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embedding)
    logger.info("Saved JD embedding to %s (dim=%d)", output_path, embedding.shape[0])

    # Save metadata
    meta = {
        "model": model_name,
        "embedding_dim": int(embedding.shape[0]),
        "generated_date": str(date.today()),
        "jd_intent_text_chars": len(JD_INTENT_TEXT),
        "normalized": True,
        "note": (
            "Frozen at dev-time. Do not regenerate between submissions unless "
            "JD_INTENT_TEXT is intentionally updated."
        ),
    }
    with meta_path.open("w", encoding="utf-8") as fh:
        yaml.dump(meta, fh, default_flow_style=False)
    logger.info("Saved embedding metadata to %s", meta_path)

    return embedding


def load_jd_embedding(path: Path = EMBEDDING_PATH) -> np.ndarray:
    """
    Load the frozen JD-intent embedding from disk.

    Called at ranking runtime — no model loading, no network.

    Returns:
        float32 numpy array of shape (embedding_dim,).

    Raises:
        FileNotFoundError: If the embedding has not been generated yet.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"JD embedding not found at {path}. "
            "Run: python -m src.jd_embedding  (offline, dev-time only)"
        )
    embedding = np.load(path).astype(np.float32)
    logger.debug("Loaded JD embedding from %s (dim=%d)", path, embedding.shape[0])
    return embedding


def embed_texts(texts: list[str], model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """
    Embed a list of texts using the same model as the JD embedding.

    Used offline to precompute candidate career-description embeddings.
    NOT called at ranking runtime.

    Args:
        texts: List of strings to embed.
        model_name: Must match the model used for the JD embedding.

    Returns:
        float32 numpy array of shape (len(texts), embedding_dim), L2-normalized.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return embeddings.astype(np.float32)


def cosine_similarity_batch(
    candidate_embeddings: np.ndarray,
    jd_embedding: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between each candidate embedding and the JD embedding.

    Since both are L2-normalized, cosine similarity = dot product.
    Vectorized over the full candidate pool — fast on CPU.

    Args:
        candidate_embeddings: shape (N, dim), float32, L2-normalized.
        jd_embedding: shape (dim,), float32, L2-normalized.

    Returns:
        shape (N,), float32 — similarity scores in [-1, 1], typically [0, 1].
    """
    return candidate_embeddings @ jd_embedding


def _load_intent_queries() -> list[str]:
    """Read the multi-query JD-intent strings from scoring_config.yaml (role_fit.intent_queries)."""
    with SCORING_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    queries = (cfg.get("role_fit") or {}).get("intent_queries") or []
    if not queries:
        raise ValueError(
            "scoring_config.yaml role_fit.intent_queries is empty — "
            "the multi-query role signal (EXECUTION_PLAN §2.5.a) needs at least one query."
        )
    return list(queries)


def generate_jd_intent_set(
    model_name: str = DEFAULT_MODEL,
    output_path: Path = INTENT_SET_PATH,
    meta_path: Path = INTENT_SET_META_PATH,
    force: bool = False,
) -> np.ndarray:
    """
    Generate and save the MULTI-QUERY JD-intent embedding set (EXECUTION_PLAN §2.5.a).

    Each string in config.role_fit.intent_queries is embedded into one L2-normalized
    row. Runtime `s_dense` = max cosine over these rows (per candidate description).

    Returns:
        float32 array of shape (Q, dim), L2-normalized rows.
    """
    if output_path.exists() and not force:
        logger.info("JD-intent set already exists at %s. Use force=True to regenerate.", output_path)
        return load_jd_intent_set(output_path)

    queries = _load_intent_queries()
    logger.info("Loading sentence-transformers model: %s", model_name)
    from sentence_transformers import SentenceTransformer  # lazy import — not needed at runtime

    model = SentenceTransformer(model_name)
    logger.info("Encoding %d JD-intent queries...", len(queries))
    embeddings = model.encode(queries, normalize_embeddings=True).astype(np.float32)
    if embeddings.ndim == 1:  # single query edge case
        embeddings = embeddings[np.newaxis, :]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    logger.info("Saved JD-intent set to %s (shape=%s)", output_path, embeddings.shape)

    meta = {
        "model": model_name,
        "embedding_dim": int(embeddings.shape[1]),
        "num_queries": int(embeddings.shape[0]),
        "generated_date": str(date.today()),
        "normalized": True,
        "queries": queries,
        "note": (
            "Frozen multi-query JD-intent set. s_dense = max cosine over rows. "
            "Regenerate only when role_fit.intent_queries changes."
        ),
    }
    with meta_path.open("w", encoding="utf-8") as fh:
        yaml.dump(meta, fh, default_flow_style=False, allow_unicode=True)
    logger.info("Saved JD-intent set metadata to %s", meta_path)
    return embeddings


def load_jd_intent_set(path: Path = INTENT_SET_PATH) -> np.ndarray:
    """Load the frozen multi-query JD-intent set (Q, dim) at runtime — no model, no network."""
    if not path.exists():
        raise FileNotFoundError(
            f"JD-intent set not found at {path}. "
            "Run: python -m src.jd_embedding  (offline, dev-time only)"
        )
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    logger.debug("Loaded JD-intent set from %s (shape=%s)", path, arr.shape)
    return arr


def max_query_similarity(
    candidate_embeddings: np.ndarray,
    intent_set: np.ndarray,
) -> np.ndarray:
    """
    s_dense per row: for each candidate embedding, the MAX cosine over all intent queries.

    Args:
        candidate_embeddings: (N, dim), L2-normalized.
        intent_set:           (Q, dim), L2-normalized.

    Returns:
        (N,) float32 — max-over-queries cosine, the multi-query dense role signal.
    """
    sims = candidate_embeddings @ intent_set.T   # (N, Q)
    return sims.max(axis=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # Keep the legacy single vector (back-compat) AND generate the multi-query set.
    generate_jd_embedding(force=False)
    generate_jd_intent_set(force=False)
