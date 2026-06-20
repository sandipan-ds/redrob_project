"""
test_p1.py — P1 exit criterion tests.

P1 exit criterion: criteria_map.md exists and is human-reviewed;
JD-intent embedding is frozen, loadable, and semantically correct.

Run with: pytest tests/test_p1.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.jd_embedding import (
    JD_INTENT_TEXT,
    cosine_similarity_batch,
    load_jd_embedding,
    load_jd_intent_set,
    max_query_similarity,
)

CRITERIA_MAP = Path("docs/project_docs/criteria_map.md")
EMBEDDING_PATH = Path("config/jd_intent_embedding.npy")
META_PATH = Path("config/jd_embedding_meta.yaml")
INTENT_SET_PATH = Path("config/jd_intent_embeddings.npy")
INTENT_SET_META_PATH = Path("config/jd_intent_embeddings_meta.yaml")


# ---------------------------------------------------------------------------
# Criteria map tests
# ---------------------------------------------------------------------------

class TestCriteriaMap:
    def test_criteria_map_exists(self):
        assert CRITERIA_MAP.exists(), "criteria_map.md not found"

    def test_criteria_map_has_all_sections(self):
        content = CRITERIA_MAP.read_text(encoding="utf-8")
        required_sections = [
            "Section A",   # Common criteria
            "Section B",   # JD-only criteria
            "Section C",   # Schema fields not in JD
            "Section D",   # Signals doc architecture decision
            "Section E",   # Signals selected for M_behavior
            "Section F",   # Hard disqualifier gates
            "Section G",   # What is NOT scored
        ]
        for section in required_sections:
            assert section in content, f"criteria_map.md missing: {section}"

    def test_criteria_map_mentions_key_disqualifiers(self):
        content = CRITERIA_MAP.read_text(encoding="utf-8").lower()
        disqualifiers = {
            "research only / pure research": ["research only", "pure research"],
            "consulting": ["consulting"],
            "LangChain": ["langchain"],
            "honeypot": ["honeypot"],
            "domain mismatch / CV-speech-robotics": ["cv / speech", "domain_mismatch", "robotics"],
        }
        for label, variants in disqualifiers.items():
            assert any(v in content for v in variants), (
                f"criteria_map.md missing disqualifier mention: {label}"
            )

    def test_criteria_map_mentions_sentinel_handling(self):
        content = CRITERIA_MAP.read_text(encoding="utf-8")
        assert "sentinel" in content.lower() or "-1" in content, (
            "criteria_map.md should document sentinel value handling"
        )

    def test_criteria_map_not_empty(self):
        content = CRITERIA_MAP.read_text(encoding="utf-8")
        assert len(content) > 2000, "criteria_map.md seems too short — may be incomplete"


# ---------------------------------------------------------------------------
# JD embedding — file & metadata tests
# ---------------------------------------------------------------------------

class TestJdEmbeddingFiles:
    def test_embedding_file_exists(self):
        assert EMBEDDING_PATH.exists(), (
            "JD embedding not found. Run: python -m src.jd_embedding"
        )

    def test_meta_file_exists(self):
        assert META_PATH.exists(), "jd_embedding_meta.yaml not found"

    def test_meta_has_required_fields(self):
        import yaml
        meta = yaml.safe_load(META_PATH.read_text(encoding="utf-8"))
        for field in ["model", "embedding_dim", "generated_date", "normalized"]:
            assert field in meta, f"jd_embedding_meta.yaml missing field: {field}"

    def test_meta_normalized_is_true(self):
        import yaml
        meta = yaml.safe_load(META_PATH.read_text(encoding="utf-8"))
        assert meta["normalized"] is True, "Embedding must be L2-normalized for cosine sim = dot product"


# ---------------------------------------------------------------------------
# JD embedding — shape & numerical tests
# ---------------------------------------------------------------------------

class TestJdEmbeddingNumerical:
    def test_loads_as_float32_array(self):
        emb = load_jd_embedding()
        assert isinstance(emb, np.ndarray)
        assert emb.dtype == np.float32

    def test_embedding_is_1d(self):
        emb = load_jd_embedding()
        assert emb.ndim == 1, f"Expected 1D embedding, got shape {emb.shape}"

    def test_embedding_dim_matches_meta(self):
        import yaml
        meta = yaml.safe_load(META_PATH.read_text(encoding="utf-8"))
        emb = load_jd_embedding()
        assert emb.shape[0] == meta["embedding_dim"], (
            f"Embedding dim {emb.shape[0]} does not match meta {meta['embedding_dim']}"
        )

    def test_embedding_is_unit_norm(self):
        emb = load_jd_embedding()
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < 1e-5, f"Embedding is not unit-normalized (norm={norm:.6f})"

    def test_embedding_has_no_nan_or_inf(self):
        emb = load_jd_embedding()
        assert not np.any(np.isnan(emb)), "Embedding contains NaN values"
        assert not np.any(np.isinf(emb)), "Embedding contains Inf values"


# ---------------------------------------------------------------------------
# JD embedding — semantic sanity tests
# ---------------------------------------------------------------------------

class TestJdEmbeddingSemantics:
    """
    Verify the embedding captures the right semantic space.
    A good ML/retrieval career description should score higher than
    a clearly irrelevant one (marketing manager, civil engineer).
    These are not unit tests of the model — they are sanity checks
    that the JD intent text is pointing in the right direction.
    """

    def _score(self, text: str) -> float:
        """Embed a single text and return cosine similarity to JD embedding."""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode(text, normalize_embeddings=True).astype(np.float32)
        jd_emb = load_jd_embedding()
        return float(np.dot(emb, jd_emb))

    def test_ml_engineer_scores_higher_than_marketing_manager(self):
        ml_text = (
            "Built production embedding-based retrieval system using FAISS and sentence-transformers. "
            "Designed offline evaluation with NDCG and MRR. Shipped ranking model to 2M users."
        )
        marketing_text = (
            "Led demand-generation campaigns, owned content marketing and SEO strategy. "
            "Managed a team of 5 across performance marketing and marketing operations."
        )
        ml_score = self._score(ml_text)
        mkt_score = self._score(marketing_text)
        assert ml_score > mkt_score, (
            f"ML engineer ({ml_score:.3f}) should score higher than marketing manager ({mkt_score:.3f})"
        )

    def test_recsys_engineer_scores_higher_than_civil_engineer(self):
        recsys_text = (
            "Designed and deployed recommendation system using collaborative filtering and "
            "vector similarity search. Evaluated with MAP and NDCG@10 on offline test sets."
        )
        civil_text = (
            "Led structural design of bridge subsystems. Managed construction teams and "
            "coordinated with municipal authorities on compliance and safety standards."
        )
        recsys_score = self._score(recsys_text)
        civil_score = self._score(civil_text)
        assert recsys_score > civil_score, (
            f"RecSys engineer ({recsys_score:.3f}) should score higher than civil engineer ({civil_score:.3f})"
        )

    def test_keyword_stuffer_scores_lower_than_genuine_ml_career(self):
        """
        A profile that just lists AI keywords in a summary should score lower
        than one that describes actually building ML systems in career history.
        This is the core anti-keyword-stuffing check.
        """
        genuine_ml = (
            "Owned the ranking pipeline for a job-search product. Built hybrid retrieval "
            "combining BM25 and dense embeddings. Set up A/B testing infrastructure and "
            "improved NDCG@10 by 18% over the baseline BM25 system."
        )
        keyword_stuffer = (
            "Skills: Python, LangChain, RAG, Pinecone, Milvus, FAISS, NDCG, MRR, MAP, "
            "embeddings, vector database, retrieval, ranking, LLM, fine-tuning, PyTorch, "
            "TensorFlow, Hugging Face, sentence-transformers, OpenAI, GPT-4."
        )
        genuine_score = self._score(genuine_ml)
        stuffer_score = self._score(keyword_stuffer)
        assert genuine_score > stuffer_score, (
            f"Genuine ML career ({genuine_score:.3f}) should score higher than "
            f"keyword stuffer ({stuffer_score:.3f})"
        )


# ---------------------------------------------------------------------------
# cosine_similarity_batch tests
# ---------------------------------------------------------------------------

class TestCosineSimilarityBatch:
    def test_batch_similarity_shape(self):
        jd_emb = load_jd_embedding()
        dim = jd_emb.shape[0]
        fake_candidates = np.random.randn(10, dim).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(fake_candidates, axis=1, keepdims=True)
        fake_candidates = fake_candidates / norms
        scores = cosine_similarity_batch(fake_candidates, jd_emb)
        assert scores.shape == (10,), f"Expected shape (10,), got {scores.shape}"

    def test_identical_vector_scores_one(self):
        jd_emb = load_jd_embedding()
        scores = cosine_similarity_batch(jd_emb[np.newaxis, :], jd_emb)
        assert abs(scores[0] - 1.0) < 1e-5, f"Self-similarity should be 1.0, got {scores[0]:.6f}"

    def test_orthogonal_vector_scores_near_zero(self):
        jd_emb = load_jd_embedding()
        dim = jd_emb.shape[0]
        # Build a vector orthogonal to jd_emb using Gram-Schmidt
        random_vec = np.random.randn(dim).astype(np.float32)
        random_vec -= random_vec.dot(jd_emb) * jd_emb
        random_vec /= np.linalg.norm(random_vec)
        scores = cosine_similarity_batch(random_vec[np.newaxis, :], jd_emb)
        assert abs(scores[0]) < 1e-4, f"Orthogonal vector should score ~0, got {scores[0]:.6f}"


# ---------------------------------------------------------------------------
# Multi-query JD-intent set (EXECUTION_PLAN §2.5.a) — the blended role signal
# ---------------------------------------------------------------------------

class TestJdIntentSet:
    def test_intent_set_files_exist(self):
        assert INTENT_SET_PATH.exists(), (
            "Multi-query JD-intent set not found. Run: python -m src.jd_embedding"
        )
        assert INTENT_SET_META_PATH.exists(), "jd_intent_embeddings_meta.yaml not found"

    def test_intent_set_is_2d_normalized_matching_config(self):
        import yaml
        arr = load_jd_intent_set()
        assert arr.ndim == 2, f"Intent set must be (Q, dim), got shape {arr.shape}"
        assert arr.dtype == np.float32
        # rows L2-normalized
        norms = np.linalg.norm(arr, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), f"Intent rows not unit-norm: {norms}"
        # Q matches the number of intent_queries in scoring_config.yaml
        cfg = yaml.safe_load(Path("config/scoring_config.yaml").read_text(encoding="utf-8"))
        n_queries = len(cfg["role_fit"]["intent_queries"])
        assert arr.shape[0] == n_queries, (
            f"Intent set has {arr.shape[0]} rows but config lists {n_queries} queries — "
            "regenerate with python -m src.jd_embedding"
        )
        assert arr.shape[1] == load_jd_embedding().shape[0], "dim mismatch vs legacy JD vector"

    def test_max_query_similarity_shape_and_self(self):
        intents = load_jd_intent_set()
        scores = max_query_similarity(intents, intents)  # each row matches itself → ~1.0
        assert scores.shape == (intents.shape[0],)
        assert np.all(scores > 1.0 - 1e-4), f"Each intent should self-match ~1.0, got {scores}"

    def test_multi_query_ranks_ml_over_marketing(self):
        """The blended s_dense (max over queries) must still beat the keyword/marketing trap."""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        intents = load_jd_intent_set()
        ml_text = (
            "Built production embedding-based retrieval and ranking; evaluated with NDCG/MRR; "
            "shipped a recommendation system to millions of users at a product company."
        )
        marketing_text = (
            "Led demand-generation campaigns, content marketing and SEO strategy; "
            "managed a performance-marketing team."
        )
        embs = model.encode([ml_text, marketing_text], normalize_embeddings=True).astype(np.float32)
        s = max_query_similarity(embs, intents)
        assert s[0] > s[1], f"ML ({s[0]:.3f}) should beat marketing ({s[1]:.3f}) on multi-query dense"
