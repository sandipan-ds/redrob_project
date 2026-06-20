"""
test_p3.py — P3 exit criterion tests.

P3 exit criterion (PHASED_BUILD_PLAN §P3):
  - Each function returns a float in its declared range for the 50 sample
    candidates (no exceptions).
  - Monotonicity sanity: higher CGPA → higher edu score; 7 yrs → higher
    exp score than 1 yr; a top-K-mean ML description → higher role_fit
    than a marketing one (reuse P1 semantic pairs).
  - Sentinel inputs → behavior returns neutral_base, never < min_multiplier.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.config_loader import load_config
from src.features.behavior import m_behavior
from src.features.education import s_education
from src.features.experience import s_exp_band
from src.features.location import s_location
from src.features.role_fit import s_role_fit
from src.features.skills import s_skill
from src.jd_embedding import load_jd_intent_set

CFG = load_config()
JD_INTENTS = load_jd_intent_set()  # (Q, 384) L2-normalized
EMBED_MODEL = None  # lazy-loaded for role_fit tests

SAMPLE_PATH = Path("data/samples/sample_candidates.json")
SAMPLE = json.loads(SAMPLE_PATH.read_text(encoding="utf-8")) if SAMPLE_PATH.exists() else []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_embed_model():
    """Lazy-load the sentence-transformers model (dev-time only, used
    in tests to produce deterministic embeddings for synthetic text)."""
    global EMBED_MODEL
    if EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return EMBED_MODEL


def _embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts → (N, 384) float32, L2-normalized."""
    model = _get_embed_model()
    return model.encode(texts, normalize_embeddings=True).astype(np.float32)


def _make_candidate(
    *,
    candidate_id: str = "CAND_TEST0001",
    yoe: float = 6.0,
    career: list[dict] | None = None,
    education: list[dict] | None = None,
    skills: list[dict] | None = None,
    signals: dict | None = None,
    current_title: str = "ML Engineer",
    current_industry: str = "Software",
    current_company_size: str = "201-500",
    current_company: str = "Acme",
    location: str = "Pune, Maharashtra",
) -> dict:
    return {
        "candidate_id": candidate_id,
        "profile": {
            "anonymized_name": "Test",
            "headline": "Test",
            "summary": "Test",
            "location": location,
            "country": "India",
            "years_of_experience": yoe,
            "current_title": current_title,
            "current_company": current_company,
            "current_company_size": current_company_size,
            "current_industry": current_industry,
        },
        "career_history": career if career is not None else [],
        "education": education if education is not None else [
            {"institution": "X", "degree": "B.Tech", "field_of_study": "Computer Science",
             "start_year": 2016, "end_year": 2020, "grade": "8.0 CGPA", "tier": "tier_2"},
        ],
        "skills": skills if skills is not None else [],
        "redrob_signals": signals if signals is not None else {
            "profile_completeness_score": 80,
            "signup_date": "2023-01-01",
            "last_active_date": "2026-06-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 10,
            "applications_submitted_30d": 0,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 4.0,
            "skill_assessment_scores": {},
            "connection_count": 100,
            "endorsements_received": 20,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 20, "max": 35},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 5.0,
            "search_appearance_30d": 5,
            "saved_by_recruiters_30d": 2,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.5,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }


# ---------------------------------------------------------------------------
# s_role_fit (DOMINANT, §2.5.a/b/f/h)
# ---------------------------------------------------------------------------

class TestRoleFit:
    def test_returns_float_in_range(self):
        cand = _make_candidate(
            career=[{
                "company": "Acme", "title": "ML Engineer",
                "start_date": "2020-06-01", "end_date": None,
                "duration_months": 72, "is_current": True,
                "industry": "Software", "company_size": "201-500",
                "description": "Built production retrieval systems with FAISS and NDCG evaluation.",
            }],
        )
        descs = [c["description"] for c in cand["career_history"]]
        embs = _embed(descs)
        score = s_role_fit(cand, embs, JD_INTENTS, CFG)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_ml_description_beats_marketing(self):
        """Top-K-mean ML career → higher role_fit than a marketing one."""
        ml_text = (
            "Built production embedding-based retrieval system using FAISS and sentence-transformers. "
            "Designed offline evaluation with NDCG and MRR. Shipped ranking model to 2M users."
        )
        mkt_text = (
            "Led demand-generation campaigns, owned content marketing and SEO strategy. "
            "Managed a team of 5 across performance marketing and marketing operations."
        )
        ml_cand = _make_candidate(
            career=[{
                "company": "TechCo", "title": "ML Engineer",
                "start_date": "2020-01-01", "end_date": None,
                "duration_months": 78, "is_current": True,
                "industry": "Software", "company_size": "201-500",
                "description": ml_text,
            }],
        )
        mkt_cand = _make_candidate(
            career=[{
                "company": "AdCo", "title": "Marketing Manager",
                "start_date": "2020-01-01", "end_date": None,
                "duration_months": 78, "is_current": True,
                "industry": "Marketing", "company_size": "51-200",
                "description": mkt_text,
            }],
        )
        ml_embs = _embed([c["description"] for c in ml_cand["career_history"]])
        mkt_embs = _embed([c["description"] for c in mkt_cand["career_history"]])
        ml_score = s_role_fit(ml_cand, ml_embs, JD_INTENTS, CFG)
        mkt_score = s_role_fit(mkt_cand, mkt_embs, JD_INTENTS, CFG)
        assert ml_score > mkt_score, (
            f"ML career should score higher than marketing: {ml_score:.3f} vs {mkt_score:.3f}"
        )

    def test_thin_desc_falls_back_to_title_prior(self):
        """All descriptions total < min_desc_chars → role-affinity title prior."""
        cand = _make_candidate(
            current_title="ML Engineer",  # role_affinity = 0.95
            career=[{
                "company": "X", "title": "ML Engineer",
                "start_date": "2024-01-01", "end_date": None,
                "duration_months": 12, "is_current": True,
                "industry": "Software", "company_size": "11-50",
                "description": "Hi.",  # 3 chars, well below min_desc_chars=40
            }],
        )
        embs = np.zeros((1, 384), dtype=np.float32)  # won't be used
        score = s_role_fit(cand, embs, JD_INTENTS, CFG)
        assert score == pytest.approx(0.95, abs=1e-6)  # title prior for "ML Engineer"

    def test_recent_outweighs_old(self):
        """A 2024 ML stint should outweigh a 2018 ML stint (recency decay).

        The test gives each candidate TWO descriptions — a relevant ML stint
        and an irrelevant marketing stint — so top-K-mean (K=2) actually
        pools both. The ML stint's recency weight differs between the two
        candidates (recent vs old), while the marketing stint's weight is
        roughly equal (both old-ish), so the weighted mean of the recent
        candidate's ML stint is higher.
        """
        ml_text = "Built production retrieval and ranking systems for real users."
        mkt_text = "Led demand-generation campaigns and SEO strategy."
        recent_cand = _make_candidate(
            career=[
                {
                    "company": "RecentCo", "title": "ML Engineer",
                    "start_date": "2024-01-01", "end_date": None,
                    "duration_months": 24, "is_current": True,
                    "industry": "Software", "company_size": "201-500",
                    "description": ml_text,
                },
                {
                    "company": "OldAdCo", "title": "Marketing Manager",
                    "start_date": "2017-01-01", "end_date": "2019-01-01",
                    "duration_months": 24, "is_current": False,
                    "industry": "Marketing", "company_size": "51-200",
                    "description": mkt_text,
                },
            ],
        )
        old_cand = _make_candidate(
            career=[
                {
                    "company": "OldCo", "title": "ML Engineer",
                    "start_date": "2013-01-01", "end_date": "2016-01-01",
                    "duration_months": 36, "is_current": False,
                    "industry": "Software", "company_size": "201-500",
                    "description": ml_text,
                },
                {
                    "company": "OldAdCo", "title": "Marketing Manager",
                    "start_date": "2017-01-01", "end_date": "2019-01-01",
                    "duration_months": 24, "is_current": False,
                    "industry": "Marketing", "company_size": "51-200",
                    "description": mkt_text,
                },
            ],
        )
        recent_embs = _embed([c["description"] for c in recent_cand["career_history"]])
        old_embs = _embed([c["description"] for c in old_cand["career_history"]])
        recent_score = s_role_fit(recent_cand, recent_embs, JD_INTENTS, CFG)
        old_score = s_role_fit(old_cand, old_embs, JD_INTENTS, CFG)
        assert recent_score > old_score, (
            f"Recent ML stint should score higher than old: {recent_score:.3f} vs {old_score:.3f}"
        )

    def test_50_sample_no_exceptions(self):
        """Every sample candidate → score in [0, 1], no exceptions."""
        assert SAMPLE, "Sample file missing"
        for c in SAMPLE:
            descs = [
                e.get("description") or ""
                for e in c.get("career_history", []) or []
            ]
            total_chars = sum(len(d) for d in descs)
            embs = _embed(descs) if total_chars >= 40 else None
            score = s_role_fit(c, embs, JD_INTENTS, CFG)
            assert 0.0 <= score <= 1.0, f"{c['candidate_id']}: score {score} out of [0,1]"


# ---------------------------------------------------------------------------
# s_skill (synonym collapse, GLM-v2 #A4)
# ---------------------------------------------------------------------------

class TestSkill:
    def test_canonical_skill_matches(self):
        cand = _make_candidate(skills=[
            {"name": "RAG", "proficiency": "advanced", "endorsements": 25, "duration_months": 18},
        ])
        s = s_skill(cand, CFG)
        assert 0.0 < s <= 1.0

    def test_synonym_collapses_to_canonical(self):
        """RAG and 'Retrieval-Augmented Generation' both → RAG, scored ONCE."""
        c1 = _make_candidate(skills=[
            {"name": "RAG", "proficiency": "advanced", "endorsements": 25, "duration_months": 18},
        ])
        c2 = _make_candidate(skills=[
            {"name": "Retrieval-Augmented Generation", "proficiency": "advanced",
             "endorsements": 25, "duration_months": 18},
        ])
        c3 = _make_candidate(skills=[
            {"name": "RAG", "proficiency": "advanced", "endorsements": 25, "duration_months": 18},
            {"name": "Retrieval-Augmented Generation", "proficiency": "advanced",
             "endorsements": 25, "duration_months": 18},
        ])
        s1 = s_skill(c1, CFG)
        s2 = s_skill(c2, CFG)
        s3 = s_skill(c3, CFG)
        # Synonym collapse: c3 should score like c1 (same canonical, counted once).
        assert s1 == pytest.approx(s2, abs=1e-6)
        assert s3 == pytest.approx(s1, abs=1e-6)  # no double-count

    def test_noise_skills_excluded(self):
        c = _make_candidate(skills=[
            {"name": "Photoshop", "proficiency": "expert", "endorsements": 50, "duration_months": 60},
        ])
        s = s_skill(c, CFG)
        assert s == pytest.approx(0.0, abs=1e-6)

    def test_platform_assessment_overrides(self):
        """A Redrob assessment score (0–100) overrides self-reported proficiency."""
        c_self = _make_candidate(skills=[
            {"name": "Python", "proficiency": "beginner", "endorsements": 0, "duration_months": 0},
        ])
        c_assessed = _make_candidate(
            skills=[
                {"name": "Python", "proficiency": "beginner", "endorsements": 0, "duration_months": 0},
            ],
            signals={
                **c_self["redrob_signals"],
                "skill_assessment_scores": {"Python": 95.0},
            },
        )
        s_self = s_skill(c_self, CFG)
        s_assessed = s_skill(c_assessed, CFG)
        assert s_assessed > s_self

    def test_endorsement_curve_caps(self):
        """100 endorsements should cap at the same score as endorse_floor (30)."""
        c_low = _make_candidate(skills=[
            {"name": "Python", "proficiency": "advanced", "endorsements": 30, "duration_months": 24},
        ])
        c_high = _make_candidate(skills=[
            {"name": "Python", "proficiency": "advanced", "endorsements": 100, "duration_months": 24},
        ])
        s_low = s_skill(c_low, CFG)
        s_high = s_skill(c_high, CFG)
        assert s_low == pytest.approx(s_high, abs=1e-6)  # both capped at 30


# ---------------------------------------------------------------------------
# s_exp_band (soft band)
# ---------------------------------------------------------------------------

class TestExperience:
    def test_ideal_band_returns_1(self):
        assert s_exp_band(7.0, CFG) == pytest.approx(1.0, abs=1e-6)
        assert s_exp_band(6.0, CFG) == pytest.approx(1.0, abs=1e-6)
        assert s_exp_band(8.0, CFG) == pytest.approx(1.0, abs=1e-6)

    def test_seven_years_better_than_one(self):
        assert s_exp_band(7.0, CFG) > s_exp_band(1.0, CFG)

    def test_acceptable_range_above_hard_min(self):
        """4 yrs (acceptable_min) > 1 yr (well below hard_min)."""
        assert s_exp_band(4.0, CFG) > s_exp_band(1.0, CFG)

    def test_over_qualified_still_above_zero(self):
        """14 yrs should still be > 0 (over-qualified but not zero)."""
        assert s_exp_band(14.0, CFG) > 0.0

    def test_below_hard_min_near_zero_but_not_zero(self):
        assert 0.0 <= s_exp_band(1.0, CFG) < 0.1

    def test_monotonic_within_acceptable(self):
        """Within the acceptable taper, more years → higher score."""
        assert s_exp_band(4.5, CFG) > s_exp_band(4.0, CFG)
        assert s_exp_band(5.5, CFG) > s_exp_band(4.5, CFG)

    def test_in_range(self):
        for yoe in [0.5, 2.0, 4.0, 6.0, 7.0, 8.0, 12.0, 15.0, 25.0]:
            s = s_exp_band(yoe, CFG)
            assert 0.0 <= s <= 1.0, f"yoe={yoe}: s={s}"


# ---------------------------------------------------------------------------
# s_education
# ---------------------------------------------------------------------------

class TestEducation:
    def test_tier_and_cgpa_combine(self):
        e = [
            {"institution": "IIT", "degree": "B.Tech", "field_of_study": "Computer Science",
             "start_year": 2016, "end_year": 2020, "grade": "9.0 CGPA", "tier": "tier_1"},
        ]
        s = s_education(e, CFG)
        assert 0.0 < s <= 1.0

    def test_higher_cgpa_higher_score(self):
        e_low = [
            {"institution": "X", "degree": "B.Tech", "field_of_study": "Computer Science",
             "start_year": 2016, "end_year": 2020, "grade": "6.0 CGPA", "tier": "tier_1"},
        ]
        e_high = [
            {"institution": "X", "degree": "B.Tech", "field_of_study": "Computer Science",
             "start_year": 2016, "end_year": 2020, "grade": "9.0 CGPA", "tier": "tier_1"},
        ]
        assert s_education(e_high, CFG) > s_education(e_low, CFG)

    def test_relevant_field_bonus(self):
        e_cs = [
            {"institution": "X", "degree": "B.Tech", "field_of_study": "Computer Science",
             "start_year": 2016, "end_year": 2020, "grade": "8.0 CGPA", "tier": "tier_2"},
        ]
        e_art = [
            {"institution": "X", "degree": "B.A.", "field_of_study": "Fine Arts",
             "start_year": 2016, "end_year": 2020, "grade": "8.0 CGPA", "tier": "tier_2"},
        ]
        assert s_education(e_cs, CFG) > s_education(e_art, CFG)

    def test_unknown_tier_neutral(self):
        e_known = [
            {"institution": "X", "degree": "B.Tech", "field_of_study": "Computer Science",
             "start_year": 2016, "end_year": 2020, "grade": "8.0 CGPA", "tier": "tier_3"},
        ]
        e_unknown = [
            {"institution": "X", "degree": "B.Tech", "field_of_study": "Computer Science",
             "start_year": 2016, "end_year": 2020, "grade": "8.0 CGPA", "tier": "unknown"},
        ]
        # unknown (0.30) is below tier_3 (0.50), so e_unknown < e_known.
        assert s_education(e_unknown, CFG) < s_education(e_known, CFG)

    def test_empty_education_returns_zero(self):
        assert s_education([], CFG) == 0.0

    def test_in_range(self):
        for edu in [
            [{"institution": "X", "degree": "B.Tech", "field_of_study": "Computer Science",
              "start_year": 2016, "end_year": 2020, "grade": g, "tier": t}
             for g, t in [("6.0 CGPA", "tier_2"), ("8.5 CGPA", "tier_1"),
                          ("9.5 CGPA", "tier_3"), ("First Class", "unknown")]],
        ]:
            s = s_education(edu, CFG)
            assert 0.0 <= s <= 1.0, f"edu={edu} → s={s}"

    def test_cgpa_parsing_handles_formats(self):
        for grade, expected in [("8.5 CGPA", True), ("74%", True), ("3.8 GPA", True),
                                 ("First Class", False), ("", False), (None, False)]:
            e = [{"institution": "X", "degree": "B.Tech", "field_of_study": "Computer Science",
                  "start_year": 2016, "end_year": 2020, "grade": grade, "tier": "tier_2"}]
            s = s_education(e, CFG)
            assert 0.0 <= s <= 1.0
            if expected:
                assert s > 0.0, f"grade={grade!r} should produce positive score, got {s}"


# ---------------------------------------------------------------------------
# m_behavior (multiplier in [0.5, 1.1])
# ---------------------------------------------------------------------------

class TestBehavior:
    def test_neutral_when_no_signals(self):
        assert m_behavior(None, CFG) == CFG["behavior"]["neutral_base"]
        assert m_behavior({}, CFG) == CFG["behavior"]["neutral_base"]

    def test_sentinels_yield_neutral_base(self):
        """All sentinel signals contribute nothing → multiplier = neutral_base.

        last_active_date is set to a date that produces a zero recency
        contribution (90–180 days ago → 0.0) and notice_period_days=60 is
        the neutral tier. So the only contributors are the sentinel
        signals, which must all evaluate to 0.
        """
        signals = {
            "last_active_date": "2026-02-20",  # ~4mo ago → recency tier = moderate (0.0)
            "recruiter_response_rate": -1.0,
            "interview_completion_rate": -1.0,
            "open_to_work_flag": False,
            "notice_period_days": 60,
            "avg_response_time_hours": -1.0,
            "saved_by_recruiters_30d": -1,
            "profile_completeness_score": -1.0,
        }
        m = m_behavior(signals, CFG)
        assert m == pytest.approx(CFG["behavior"]["neutral_base"], abs=1e-6)

    def test_never_below_min_multiplier(self):
        """Sentinel inputs → behavior ≥ min_multiplier (spec: never < min)."""
        # The most-negative possible: inactive + low response + low interview + no OOTW + >90d notice + slow response + 0 saved + low completeness.
        worst = {
            "last_active_date": "2020-01-01",  # way stale → -0.10
            "recruiter_response_rate": -1.0,
            "interview_completion_rate": -1.0,
            "open_to_work_flag": False,
            "notice_period_days": 120,         # >90 → -0.05
            "avg_response_time_hours": 100,    # >72 → -0.05
            "saved_by_recruiters_30d": -1,
            "profile_completeness_score": 10,  # <40 → -0.05
        }
        m = m_behavior(worst, CFG)
        assert m >= CFG["behavior"]["min_multiplier"]

    def test_never_above_max_multiplier(self):
        best = {
            "last_active_date": "2026-06-01",  # very active → +0.10
            "recruiter_response_rate": 1.0,    # +0.30
            "interview_completion_rate": 1.0,  # +0.20
            "open_to_work_flag": True,         # +0.10
            "notice_period_days": 10,          # sub-30 → +0.05
            "avg_response_time_hours": 1,      # <6h → +0.02
            "saved_by_recruiters_30d": 10,     # +0.03
            "profile_completeness_score": 100,  # 0
        }
        m = m_behavior(best, CFG)
        assert m <= CFG["behavior"]["max_multiplier"]

    def test_open_to_work_increases(self):
        # Use low-signal values so the off version stays below max_multiplier
        # (otherwise both clamp to 1.1 and the test is meaningless).
        s_off = {"open_to_work_flag": False, "last_active_date": "2025-12-01",
                 "recruiter_response_rate": 0.0, "interview_completion_rate": 0.0,
                 "notice_period_days": 60}
        s_on = {**s_off, "open_to_work_flag": True}
        assert m_behavior(s_on, CFG) > m_behavior(s_off, CFG)

    def test_recent_active_beats_stale(self):
        recent = {"last_active_date": "2026-06-01", "recruiter_response_rate": 0.5,
                  "interview_completion_rate": 0.5, "notice_period_days": 60}
        stale = {"last_active_date": "2024-01-01", "recruiter_response_rate": 0.5,
                 "interview_completion_rate": 0.5, "notice_period_days": 60}
        assert m_behavior(recent, CFG) > m_behavior(stale, CFG)

    def test_github_does_not_affect_behavior(self):
        """EXCEUTION_PLAN §2.5.g: github_activity_score is DROPPED from M_behavior.
        Setting it to -1 vs 100 must not change the multiplier."""
        base = {"last_active_date": "2026-06-01", "recruiter_response_rate": 0.5,
                "interview_completion_rate": 0.5, "notice_period_days": 60}
        s_low = {**base, "github_activity_score": -1.0}
        s_high = {**base, "github_activity_score": 100.0}
        assert m_behavior(s_low, CFG) == pytest.approx(m_behavior(s_high, CFG), abs=1e-9)


# ---------------------------------------------------------------------------
# s_location
# ---------------------------------------------------------------------------

class TestLocation:
    def test_preferred_city_noida(self):
        profile = {"location": "Noida, Uttar Pradesh", "current_title": "X"}
        signals = {"willing_to_relocate": False}
        assert s_location(profile, signals, CFG) == pytest.approx(
            CFG["location"]["preferred_score"], abs=1e-6
        )

    def test_preferred_city_substring(self):
        """Substring match (data is 'City, Region')."""
        profile = {"location": "Pune, Maharashtra"}
        signals = {}
        assert s_location(profile, signals, CFG) == pytest.approx(
            CFG["location"]["preferred_score"], abs=1e-6
        )

    def test_also_welcome_city(self):
        profile = {"location": "Hyderabad, Telangana"}
        signals = {}
        assert s_location(profile, signals, CFG) == pytest.approx(
            CFG["location"]["also_welcome_score"], abs=1e-6
        )

    def test_other_india_with_willing(self):
        profile = {"location": "Jaipur, Rajasthan"}
        signals = {"willing_to_relocate": True}
        s = s_location(profile, signals, CFG)
        assert s == pytest.approx(CFG["location"]["willing_to_relocate_score"], abs=1e-6)

    def test_outside_india(self):
        profile = {"location": "Toronto, Canada"}
        signals = {"willing_to_relocate": False}
        s = s_location(profile, signals, CFG)
        assert s == pytest.approx(CFG["location"]["outside_india_score"], abs=1e-6)

    def test_empty_location(self):
        """No location + no willing_relocate → outside_india fallback."""
        profile = {"location": ""}
        signals = {}
        s = s_location(profile, signals, CFG)
        assert s == pytest.approx(CFG["location"]["outside_india_score"], abs=1e-6)

    def test_in_range(self):
        for loc in ["Noida, UP", "Pune, MH", "Hyderabad, TS", "Mumbai, MH",
                    "Jaipur, RJ", "Toronto, Canada", "", "Delhi NCR"]:
            profile = {"location": loc}
            signals = {"willing_to_relocate": True}
            s = s_location(profile, signals, CFG)
            assert 0.0 <= s <= 1.0, f"loc={loc!r}: s={s}"


# ---------------------------------------------------------------------------
# 50-sample smoke (no exceptions across all extractors)
# ---------------------------------------------------------------------------

class TestFiftySampleSmoke:
    def test_all_extractors_on_50_sample(self):
        """Every sample candidate → every extractor returns a float in its
        declared range, with no exceptions."""
        assert SAMPLE, "Sample file missing"
        for c in SAMPLE:
            profile = c.get("profile", {})
            signals = c.get("redrob_signals", {})
            descs = [e.get("description") or "" for e in c.get("career_history", []) or []]
            total_chars = sum(len(d) for d in descs)
            embs = _embed(descs) if total_chars >= 40 else None

            r = s_role_fit(c, embs, JD_INTENTS, CFG)
            assert 0.0 <= r <= 1.0, f"{c['candidate_id']}: s_role_fit={r}"

            sk = s_skill(c, CFG)
            assert 0.0 <= sk <= 1.0, f"{c['candidate_id']}: s_skill={sk}"

            ex = s_exp_band(profile.get("years_of_experience", 0), CFG)
            assert 0.0 <= ex <= 1.0, f"{c['candidate_id']}: s_exp_band={ex}"

            ed = s_education(c.get("education", []), CFG)
            assert 0.0 <= ed <= 1.0, f"{c['candidate_id']}: s_education={ed}"

            beh = m_behavior(signals, CFG)
            min_m = CFG["behavior"]["min_multiplier"]
            max_m = CFG["behavior"]["max_multiplier"]
            assert min_m <= beh <= max_m, f"{c['candidate_id']}: m_behavior={beh}"

            loc = s_location(profile, signals, CFG)
            assert 0.0 <= loc <= 1.0, f"{c['candidate_id']}: s_location={loc}"
