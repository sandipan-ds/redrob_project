"""
proxy_labels.py — P5 proxy labels (EXCEUTION_PLAN §5, GLM #3).

Loads `data/labels/proxy_tiers.json` and returns:
  - `load_proxy_tiers()` — {candidate_id: tier} for the 50 sample
    candidates (tier 0–5; 0 = no-fit, 5 = ideal).
  - `adversarial_decoys()` — synthesized near-miss profiles that LOOK
    like fits by raw formula but should rank LOW (the §5 independence
    guard against the weight-setter / labeler being the same person).

Tier rubric (documented in proxy_tiers.json too):
  0 = honeypot or no-fit whatsoever
  1 = clearly not a fit (wrong domain, scattered AI keywords, no ML production)
  2 = weak fit (some ML exposure but no production retrieval/ranking/recsys)
  3 = possible fit (some production ML, unclear seniority or domain)
  4 = likely fit (clear production ML at a product company)
  5 = ideal match (6–8 yrs, production retrieval/ranking/recsys, JD-aligned)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LABELS_PATH = Path(__file__).parent.parent.parent / "data" / "labels" / "proxy_tiers.json"


def load_proxy_tiers(path: Path | None = None) -> dict[str, int]:
    """
    Load the hand-labeled proxy tiers. Returns {candidate_id: tier}.
    The tiers live in `data/labels/proxy_tiers.json`; the file is
    committed and human-reviewed.
    """
    p = Path(path) if path else LABELS_PATH
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    tiers = data.get("tiers", {})
    return {str(k): int(v) for k, v in tiers.items()}


def load_proxy_metadata(path: Path | None = None) -> dict[str, Any]:
    """Load the full proxy labels file (tiers + rubric + decoy notes)."""
    p = Path(path) if path else LABELS_PATH
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def adversarial_decoys() -> list[dict[str, Any]]:
    """
    Synthesized near-miss profiles that LOOK like fits by raw formula
    (high AI-skill count, mentions "production ML" in summary) but
    should rank LOW because the career is consulting-only / research-only /
    marketing with scattered AI keywords — exactly the keyword-stuffing
    trap the JD warns about.

    The §5 independence guard (GLM #3 / MINIMAX): these are the
    "second reviewer" stand-in. They carry candidate_ids in the
    `CAND_DECOY_*` range so they're easy to identify in the eval set.

    Returns a list of candidate dicts in the standard schema.
    """
    return [
        _consulting_decoy(),
        _marketing_keyword_stuffer(),
        _research_only_decoy(),
    ]


def _consulting_decoy() -> dict[str, Any]:
    """Consulting-only career with 10+ AI skills — the §2.5.j trap."""
    return {
        "candidate_id": "CAND_DECOY_0001",
        "profile": {
            "anonymized_name": "Decoy One",
            "headline": "ML Engineer | 10+ AI skills",
            "summary": "Built production ML systems for enterprise clients. "
                       "Python, PyTorch, TensorFlow, RAG, Pinecone, Milvus, FAISS, "
                       "LangChain, NDCG, MRR, MAP, embeddings, vector search, "
                       "Hugging Face, sentence-transformers, OpenAI, GPT-4, "
                       "recommendation systems, search, ranking, retrieval, LLM, "
                       "fine-tuning, LoRA, PEFT, MLflow, Docker, Kubernetes, AWS, GCP.",
            "location": "Toronto, Canada",
            "country": "Canada",
            "years_of_experience": 7.0,
            "current_title": "ML Engineer",
            "current_company": "TCS",
            "current_company_size": "10001+",
            "current_industry": "IT Services",
        },
        "career_history": [
            {
                "company": "TCS", "title": "Consultant",
                "start_date": "2018-01-01", "end_date": None,
                "duration_months": 84, "is_current": True,
                "industry": "IT Services", "company_size": "10001+",
                "description": "Client consulting, project staffing, status reports. "
                               "Deployed dashboards in production for client reporting.",
            },
            {
                "company": "Infosys", "title": "Senior Consultant",
                "start_date": "2014-01-01", "end_date": "2018-01-01",
                "duration_months": 48, "is_current": False,
                "industry": "IT Services", "company_size": "10001+",
                "description": "Enterprise consulting, knowledge transfer, billing.",
            },
        ],
        "education": [
            {"institution": "State University", "degree": "B.Tech",
             "field_of_study": "Computer Science", "start_year": 2010,
             "end_year": 2014, "grade": "7.0 CGPA", "tier": "tier_3"},
        ],
        "skills": [
            {"name": s, "proficiency": "advanced", "endorsements": 5, "duration_months": 6}
            for s in ("Python", "PyTorch", "TensorFlow", "RAG", "Pinecone",
                      "Milvus", "FAISS", "LangChain", "Embeddings", "Hugging Face",
                      "sentence-transformers", "LLM", "Fine-tuning LLMs")
        ],
        "redrob_signals": {
            "profile_completeness_score": 90, "signup_date": "2023-01-01",
            "last_active_date": "2026-06-01", "open_to_work_flag": True,
            "profile_views_received_30d": 20, "applications_submitted_30d": 0,
            "recruiter_response_rate": 0.5, "avg_response_time_hours": 8.0,
            "skill_assessment_scores": {"Python": 40.0},
            "connection_count": 200, "endorsements_received": 50,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 15, "max": 25},
            "preferred_work_mode": "hybrid", "willing_to_relocate": True,
            "github_activity_score": -1, "search_appearance_30d": 5,
            "saved_by_recruiters_30d": 1, "interview_completion_rate": 0.6,
            "offer_acceptance_rate": 0.3, "verified_email": True,
            "verified_phone": True, "linkedin_connected": True,
        },
    }


def _marketing_keyword_stuffer() -> dict[str, Any]:
    """Marketing Manager with scattered AI keywords — the canonical bad output."""
    return {
        "candidate_id": "CAND_DECOY_0002",
        "profile": {
            "anonymized_name": "Decoy Two",
            "headline": "Marketing Manager | AI enthusiast",
            "summary": "Led demand-generation campaigns. Skills: Python, LangChain, "
                       "RAG, Pinecone, embeddings, vector search, LLM, GPT-4, "
                       "recommendation systems, recsys, ML, deep learning, NLP, "
                       "Hugging Face, sentence-transformers, AI, machine learning, "
                       "data science, analytics.",
            "location": "Mumbai, Maharashtra",
            "country": "India",
            "years_of_experience": 6.0,
            "current_title": "Marketing Manager",
            "current_company": "AdCo",
            "current_company_size": "51-200",
            "current_industry": "Marketing",
        },
        "career_history": [
            {
                "company": "AdCo", "title": "Marketing Manager",
                "start_date": "2020-01-01", "end_date": None,
                "duration_months": 72, "is_current": True,
                "industry": "Marketing", "company_size": "51-200",
                "description": "Led demand-generation campaigns, content marketing, "
                               "SEO strategy. Managed a team of 5. No production ML work.",
            },
            {
                "company": "MediaCo", "title": "Marketing Manager",
                "start_date": "2017-01-01", "end_date": "2020-01-01",
                "duration_months": 36, "is_current": False,
                "industry": "Marketing", "company_size": "201-500",
                "description": "Content strategy, campaign management, KPIs.",
            },
        ],
        "education": [
            {"institution": "Business School", "degree": "MBA",
             "field_of_study": "Marketing", "start_year": 2014,
             "end_year": 2016, "grade": "8.0 CGPA", "tier": "tier_2"},
        ],
        "skills": [
            {"name": s, "proficiency": "intermediate", "endorsements": 2, "duration_months": 3}
            for s in ("Python", "LangChain", "RAG", "Pinecone", "Embeddings",
                      "LLM", "GPT-4", "Hugging Face", "NLP", "ML")
        ],
        "redrob_signals": {
            "profile_completeness_score": 85, "signup_date": "2024-01-01",
            "last_active_date": "2026-06-01", "open_to_work_flag": True,
            "profile_views_received_30d": 5, "applications_submitted_30d": 0,
            "recruiter_response_rate": 0.3, "avg_response_time_hours": 12.0,
            "skill_assessment_scores": {},
            "connection_count": 100, "endorsements_received": 10,
            "notice_period_days": 60,
            "expected_salary_range_inr_lpa": {"min": 20, "max": 30},
            "preferred_work_mode": "flexible", "willing_to_relocate": False,
            "github_activity_score": -1, "search_appearance_30d": 1,
            "saved_by_recruiters_30d": 0, "interview_completion_rate": 0.4,
            "offer_acceptance_rate": -1, "verified_email": True,
            "verified_phone": True, "linkedin_connected": False,
        },
    }


def _research_only_decoy() -> dict[str, Any]:
    """Research-only academic with no production deployment — the §2.5.c trap."""
    return {
        "candidate_id": "CAND_DECOY_0003",
        "profile": {
            "anonymized_name": "Decoy Three",
            "headline": "Research Scientist | ML, NLP",
            "summary": "PhD researcher. Studies embedding spaces, retrieval models, "
                       "recsys, ranking, NDCG, MRR, MAP, evaluation frameworks. "
                       "Python, PyTorch, Hugging Face, sentence-transformers.",
            "location": "Bangalore, Karnataka",
            "country": "India",
            "years_of_experience": 5.0,
            "current_title": "Research Scientist",
            "current_company": "University Lab",
            "current_company_size": "1001-5000",
            "current_industry": "Research",
        },
        "career_history": [
            {
                "company": "University Lab", "title": "Research Scientist",
                "start_date": "2021-01-01", "end_date": None,
                "duration_months": 60, "is_current": True,
                "industry": "Research", "company_size": "1001-5000",
                "description": "PhD researcher. Studies embedding spaces, retrieval "
                               "models, recsys, ranking, NDCG, MRR, MAP. Publishes "
                               "papers, academic collaborations, university teaching. "
                               "No production deployment.",
            },
        ],
        "education": [
            {"institution": "IIT", "degree": "PhD",
             "field_of_study": "Computer Science", "start_year": 2017,
             "end_year": 2021, "grade": "9.2 CGPA", "tier": "tier_1"},
        ],
        "skills": [
            {"name": s, "proficiency": "advanced", "endorsements": 10, "duration_months": 48}
            for s in ("Python", "PyTorch", "Hugging Face", "sentence-transformers",
                      "Embeddings", "NLP")
        ],
        "redrob_signals": {
            "profile_completeness_score": 95, "signup_date": "2022-01-01",
            "last_active_date": "2026-05-01", "open_to_work_flag": False,
            "profile_views_received_30d": 30, "applications_submitted_30d": 2,
            "recruiter_response_rate": 0.2, "avg_response_time_hours": 24.0,
            "skill_assessment_scores": {"Python": 80.0},
            "connection_count": 300, "endorsements_received": 80,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 18, "max": 28},
            "preferred_work_mode": "onsite", "willing_to_relocate": False,
            "github_activity_score": 25.0, "search_appearance_30d": 10,
            "saved_by_recruiters_30d": 0, "interview_completion_rate": 0.5,
            "offer_acceptance_rate": -1, "verified_email": True,
            "verified_phone": True, "linkedin_connected": True,
        },
    }
