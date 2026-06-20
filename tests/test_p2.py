"""
test_p2.py — P2 exit criterion tests.

P2 exit criterion (PHASED_BUILD_PLAN §P2):
  - All sample honeypots in data/samples are flagged `is_honeypot == True`.
  - Zero false-kills on the clear-fit synthetic profiles.
  - A consulting-only synthetic → penalty ≈ 0.15; research-only → ≈ 0.20;
    stacked → 0.15 × √0.20 ≈ 0.067 (the §2.5.d softened combination).
  - The consulting_only exemption fires when a synthetic has any prior
    product-co stint (industry != IT Services).
  - langchain_only_junior does NOT fire on a senior who recently added
    LangChain but has pre-2022 ML experience.
  - Sentinels do not trigger any gate.

Note: `data/samples/sample_candidates.json` (50 real candidates) contains
NO structural impossibilities — the ~80 honeypots live in the full 100K
pool. The "all sample honeypots" test is vacuous on the 50-sample; the
exit-criterion spirit is met by the **synthetic** obviously-impossible
profiles (which deterministically exercise the detector) plus a zero-
false-positive check on the 50-sample.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config_loader import load_config
from src.disqualifiers import compute_penalty
from src.honeypot import detect_honeypot

CFG = load_config()

SAMPLE_PATH = Path("data/samples/sample_candidates.json")
SAMPLE = json.loads(SAMPLE_PATH.read_text(encoding="utf-8")) if SAMPLE_PATH.exists() else []


# ---------------------------------------------------------------------------
# Synthetic-candidate builders
# ---------------------------------------------------------------------------

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
) -> dict:
    """Build a candidate dict with sensible defaults; override per test."""
    return {
        "candidate_id": candidate_id,
        "profile": {
            "anonymized_name": "Test Candidate",
            "headline": "Test headline",
            "summary": "Test summary",
            "location": "Pune, Maharashtra",
            "country": "India",
            "years_of_experience": yoe,
            "current_title": current_title,
            "current_company": current_company,
            "current_company_size": current_company_size,
            "current_industry": current_industry,
        },
        "career_history": career if career is not None else [],
        "education": education if education is not None else [
            {
                "institution": "Test University",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2016,
                "end_year": 2020,
                "grade": "8.5 CGPA",
                "tier": "tier_2",
            }
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


def _consulting_stint(
    company: str,
    *,
    duration_months: int = 48,
    is_current: bool = True,
    start_year: int = 2022,
    description: str | None = None,
) -> dict:
    """A career stint at a generic IT Services consulting firm (large band).

    `duration_months` is explicit (not derived) so the test can set it to
    match the candidate's `years_of_experience` and avoid triggering the
    honeypot experience-mismatch check. `description` defaults to a recent
    ML-relevant description (so no_recent_code does NOT fire); tests that
    need research_only to fire pass a non-production description.
    """
    return {
        "company": company,
        "title": "Consultant",
        "start_date": f"{start_year}-01-01",
        "end_date": None if is_current else f"{start_year + max(duration_months // 12, 1)}-01-01",
        "duration_months": duration_months,
        "is_current": is_current,
        "industry": "IT Services",
        "company_size": "10001+",
        "description": description if description is not None else (
            "Deployed Java/Spring services in production for enterprise clients. "
            "Project staffing, status reports."
        ),
    }


def _product_co_stint(
    company: str = "Acme",
    *,
    duration_months: int = 48,
    is_current: bool = True,
    start_year: int = 2022,
    description: str | None = None,
) -> dict:
    """A stint at a non-consulting product company (exemption trigger).

    `duration_months` is explicit so tests can match the candidate's yoe.
    Default description includes a production term so no_recent_code fires
    correctly for a recent product-co stint.
    """
    return {
        "company": company,
        "title": "ML Engineer",
        "start_date": f"{start_year}-01-01",
        "end_date": None if is_current else f"{start_year + max(duration_months // 12, 1)}-01-01",
        "duration_months": duration_months,
        "is_current": is_current,
        "industry": "Software",
        "company_size": "201-500",
        "description": description if description is not None else (
            "Deployed ML systems in production for real users. Shipped ranking features."
        ),
    }


# ---------------------------------------------------------------------------
# Honeypot detection
# ---------------------------------------------------------------------------

class TestHoneypotDetector:
    """detect_honeypot(candidate, cfg) -> (bool, reasons)."""

    def test_impossible_yoe_fires(self):
        """yoe=20 with one 10-month stint → mismatch > 12mo tolerance → fires."""
        cand = _make_candidate(
            yoe=20.0,
            career=[{
                "company": "TinyCorp",
                "title": "X",
                "start_date": "2025-01-01",
                "end_date": "2025-11-01",
                "duration_months": 10,
                "is_current": False,
                "industry": "Software",
                "company_size": "1-10",
                "description": "Short stint.",
            }],
        )
        is_hp, reasons = detect_honeypot(cand, CFG)
        assert is_hp is True
        assert "experience_mismatch" in reasons

    def test_many_expert_skills_with_zero_duration_fires(self):
        """3 expert skills with duration_months=0 → count > 2 → fires."""
        cand = _make_candidate(skills=[
            {"name": "Python", "proficiency": "expert", "endorsements": 10, "duration_months": 0},
            {"name": "TensorFlow", "proficiency": "expert", "endorsements": 5, "duration_months": 0},
            {"name": "PyTorch", "proficiency": "expert", "endorsements": 5, "duration_months": 0},
        ])
        is_hp, reasons = detect_honeypot(cand, CFG)
        assert is_hp is True
        assert "expert_with_zero_duration" in reasons

    def test_education_end_before_start_fires(self):
        """education[].end_year < start_year → fires."""
        cand = _make_candidate(education=[
            {"institution": "X", "degree": "BS", "field_of_study": "CS",
             "start_year": 2020, "end_year": 2016, "grade": "8.0", "tier": "tier_1"},
        ])
        is_hp, reasons = detect_honeypot(cand, CFG)
        assert is_hp is True
        assert "education_end_before_start" in reasons

    def test_clear_fit_synthetic_not_flagged(self):
        """A reasonable synthetic profile → not a honeypot."""
        cand = _make_candidate(
            yoe=6.0,
            career=[
                {
                    "company": "Acme",
                    "title": "ML Engineer",
                    "start_date": "2020-06-01",
                    "end_date": None,
                    "duration_months": 72,
                    "is_current": True,
                    "industry": "Software",
                    "company_size": "201-500",
                    "description": "Built production retrieval systems.",
                }
            ],
            skills=[
                {"name": "Python", "proficiency": "advanced", "endorsements": 12, "duration_months": 60},
                {"name": "PyTorch", "proficiency": "advanced", "endorsements": 8, "duration_months": 36},
            ],
        )
        is_hp, reasons = detect_honeypot(cand, CFG)
        assert is_hp is False
        assert reasons == []

    def test_50_sample_has_zero_false_positives(self):
        """All 50 real sample candidates must NOT be flagged as honeypots."""
        assert SAMPLE, "Sample file missing — cannot run this test"
        for c in SAMPLE:
            is_hp, reasons = detect_honeypot(c, CFG)
            assert is_hp is False, (
                f"False honeypot flag on {c['candidate_id']}: {reasons}"
            )


# ---------------------------------------------------------------------------
# Disqualifier gates
# ---------------------------------------------------------------------------

class TestDisqualifierGates:
    """compute_penalty(candidate, cfg, role_fit_text) -> (P, reasons)."""

    def test_clear_fit_returns_1(self):
        """A reasonable synthetic → no gates fire → P=1.0, no reasons."""
        cand = _make_candidate(
            yoe=6.0,
            career=[
                {
                    "company": "Acme",
                    "title": "ML Engineer",
                    "start_date": "2020-06-01",
                    "end_date": None,
                    "duration_months": 72,
                    "is_current": True,
                    "industry": "Software",
                    "company_size": "201-500",
                    "description": "Built production retrieval systems using FAISS and NDCG evaluation.",
                }
            ],
            skills=[
                {"name": "Python", "proficiency": "advanced", "endorsements": 12, "duration_months": 60},
                {"name": "PyTorch", "proficiency": "advanced", "endorsements": 8, "duration_months": 36},
            ],
        )
        p, reasons = compute_penalty(cand, CFG)
        assert p == pytest.approx(1.0, abs=1e-6)
        assert reasons == []

    def test_consulting_only_synthetic(self):
        """All-India-IT-Services career → consulting_only fires → P = score × p_scale."""
        cand = _make_candidate(
            yoe=8.0,
            career=[
                _consulting_stint("Infosys", duration_months=48, is_current=True, start_year=2019),
                _consulting_stint("TCS", duration_months=48, is_current=False, start_year=2015),
            ],
            current_industry="IT Services",
            current_company="Infosys",
            current_company_size="10001+",
        )
        p, reasons = compute_penalty(cand, CFG)
        assert "consulting_only" in reasons
        p_scale = CFG["penalties"]["p_scale"]
        assert p == pytest.approx(0.15 * p_scale, abs=1e-6)
        # And no honeypot
        assert "honeypot" not in reasons

    def test_research_only_synthetic(self):
        """All-academic career, no production terms, research framing → research_only fires.

        Note: a pure academic by definition has no production terms in their
        descriptions, which also means no_recent_code fires (no recent
        ML-relevant stint). The test asserts the §2.5.d STACKED result:
        min(research 0.20, no_recent 0.25) × √0.25 = 0.20 × 0.5 = 0.10.
        A clean research-only-at-0.20 is not achievable in isolation with
        a realistic career (the gates are not orthogonal by design).
        """
        cand = _make_candidate(
            yoe=6.0,
            career=[
                {
                    "company": "Test University",
                    "title": "Research Scientist",
                    "start_date": "2020-01-01",
                    "end_date": None,
                    "duration_months": 72,
                    "is_current": True,
                    "industry": "Research",
                    "company_size": "1001-5000",
                    # CRITICAL: must NOT contain any PRODUCTION_LEXICON term
                    # as a whole word (word-boundary match in _has_any_term).
                    # Pure academic phrasing only.
                    "description": "PhD thesis on statistical learning theory. Published papers, "
                                   "academic collaborations, university teaching duties, conference talks.",
                }
            ],
            current_title="Research Scientist",
            current_industry="Research",
            current_company="Test University",
            current_company_size="1001-5000",
        )
        p, reasons = compute_penalty(cand, CFG)
        assert "research_only" in reasons
        assert "no_recent_code" in reasons  # always co-fires for a pure academic
        # consulting_only should NOT fire (no stint in consulting_companies, and
        # industry="Research" is not in consulting_industries → not consulting).
        assert "consulting_only" not in reasons
        # Stacked (p_scale-aware): min(0.20, 0.25) × p_scale = 0.30 worst, then
        # × √(0.25 × p_scale) for the secondary. See §2.5.d combination.
        p_scale = CFG["penalties"]["p_scale"]
        worst = 0.20 * p_scale
        secondary = 0.25 * p_scale
        expected = worst * (secondary ** 0.5)
        assert p == pytest.approx(expected, abs=1e-3)

    def test_stacked_consulting_and_research_is_softened(self):
        """Three gates fire (consulting + research + no_recent_code) → §2.5.d softened.

        A research-only career with no production terms also triggers
        no_recent_code (no recent ML stint). All three stack:
          min(0.15, 0.20, 0.25) × √(0.20 × 0.25) = 0.15 × √0.05 ≈ 0.0335.
        This verifies the §2.5.d math: worst at full strength, ALL others
        softened geometrically together.
        """
        # All stints IT Services (consulting fires) + research framing + no production terms.
        cand = _make_candidate(
            yoe=8.0,
            career=[
                {
                    "company": "BigTech Research",
                    "title": "Research Scientist",
                    "start_date": "2018-01-01",
                    "end_date": None,
                    "duration_months": 96,
                    "is_current": True,
                    "industry": "IT Services",
                    "company_size": "10001+",
                    # No PRODUCTION_LEXICON terms — research framing only.
                    "description": "PhD researcher in industry research lab. Studies graph theory and "
                                   "combinatorial optimization. Publishes papers, academic collaborations, "
                                   "university teaching.",
                }
            ],
            current_title="Research Scientist",
            current_industry="IT Services",
            current_company="BigTech Research",
            current_company_size="10001+",
        )
        p, reasons = compute_penalty(cand, CFG)
        assert "consulting_only" in reasons
        assert "research_only" in reasons
        assert "no_recent_code" in reasons
        # §2.5.d: each gate score is scaled by p_scale, then worst-at-full,
        # others geometric. For 3 gates (scores 0.15, 0.20, 0.25) with
        # p_scale: min(scaled) × √(prod(other scaled)) = 0.225 × √(0.30×0.375).
        p_scale = CFG["penalties"]["p_scale"]
        worst = 0.15 * p_scale
        sec1 = 0.20 * p_scale
        sec2 = 0.25 * p_scale
        expected = worst * (sec1 * sec2) ** 0.5
        assert p == pytest.approx(expected, abs=1e-3), (
            f"Expected softened stacked penalty ≈ {expected:.4f}, got {p:.4f}"
        )

    def test_consulting_exemption_with_product_co_stint(self):
        """One product-co stint (industry=Software) → consulting_only does NOT fire → P=1.0.

        The product-co stint is RECENT (is_current, started recently) with
        production terms in its description, so no_recent_code also does NOT
        fire → clean P=1.0.
        """
        cand = _make_candidate(
            yoe=8.0,
            career=[
                # 7y consulting (old) + 1y product co (recent) → totals 8y.
                _consulting_stint("Infosys", duration_months=84, is_current=False, start_year=2015),
                _product_co_stint(
                    "Acme", duration_months=12, is_current=True, start_year=2025,
                ),
            ],
        )
        p, reasons = compute_penalty(cand, CFG)
        assert "consulting_only" not in reasons
        # research_only should also NOT fire (product-co stint exists, descriptions
        # have production terms, no research framing).
        assert "research_only" not in reasons
        # no_recent_code should NOT fire (recent product-co stint with production terms).
        assert "no_recent_code" not in reasons
        assert p == pytest.approx(1.0, abs=1e-6)

    def test_langchain_does_not_fire_on_senior_with_pre2022_ml(self):
        """
        Senior (yoe=12.3) with LangChain skill AND a pre-2022 career entry whose
        description contains production terms ("deployed ... in production").
        langchain_only_junior must NOT fire (the senior is genuinely
        experienced; pre-2022 ML production proves real ML depth).
        """
        cand = _make_candidate(
            yoe=12.3,  # 60 + 88 = 148 months total
            career=[
                {
                    "company": "OldCorp",
                    "title": "ML Engineer",
                    "start_date": "2014-01-01",
                    "end_date": "2019-01-01",
                    "duration_months": 60,
                    "is_current": False,
                    "industry": "Software",
                    "company_size": "201-500",
                    # Production terms → condition (2) fails → gate does not fire.
                    "description": "Deployed production ML inference service for real users. "
                                   "Shipped ranking system with NDCG evaluation.",
                },
                {
                    "company": "NewCorp",
                    "title": "Senior ML Engineer",
                    "start_date": "2019-02-01",
                    "end_date": None,
                    "duration_months": 88,
                    "is_current": True,
                    "industry": "Software",
                    "company_size": "201-500",
                    "description": "Recent LangChain prototyping, some retrieval work.",
                },
            ],
            skills=[
                # LangChain is the only ML skill in `skills` (the ML experience
                # is in career descriptions, which is exactly the
                # false-fire-prevention case).
                {"name": "LangChain", "proficiency": "intermediate", "endorsements": 2, "duration_months": 6},
            ],
        )
        p, reasons = compute_penalty(cand, CFG)
        assert "langchain_only_junior" not in reasons
        # consulting_only shouldn't fire (Software industry, not IT Services).
        assert "consulting_only" not in reasons
        assert p == pytest.approx(1.0, abs=1e-6)

    def test_langchain_does_fire_on_junior_with_no_other_ai(self):
        """
        Junior (yoe=1.4), only LangChain skill (6mo), recent career entry with
        production terms (so no_recent_code does NOT fire) but NO AI keywords
        in role_fit_text (so langchain condition (3) holds). All 3 conjunctive
        conditions hold → langchain_only_junior fires → P ≈ 0.40.
        """
        cand = _make_candidate(
            yoe=1.4,  # 17 months
            career=[
                {
                    "company": "StartupCo",
                    "title": "Junior Engineer",
                    "start_date": "2025-01-01",
                    "end_date": None,
                    "duration_months": 17,
                    "is_current": True,
                    "industry": "Software",
                    "company_size": "11-50",
                    # Production terms ("deployed", "in production") so
                    # no_recent_code does NOT fire (recent + ML-relevant).
                    # But NO AI_SKILL_KEYWORDS ("machine learning", "embedding",
                    # "pytorch" etc.) so langchain condition (3) holds.
                    "description": "Deployed an internal web app in production for the team. "
                                   "Wrote Python scripts, attended standups.",
                }
            ],
            skills=[
                {"name": "LangChain", "proficiency": "beginner", "endorsements": 0, "duration_months": 6},
            ],
        )
        p, reasons = compute_penalty(cand, CFG)
        assert "langchain_only_junior" in reasons
        # no_recent_code should NOT fire (recent career with production terms).
        assert "no_recent_code" not in reasons
        p_scale = CFG["penalties"]["p_scale"]
        assert p == pytest.approx(0.40 * p_scale, abs=1e-6)

    def test_sentinels_do_not_trigger_any_gate(self):
        """
        A candidate with sentinel values across the board (github=-1,
        offer_acceptance=-1, skill_assessment_scores={}) must not trip any
        gate. The github signal is dropped per §2.5.g; the other sentinels
        aren't read by any gate. This is a regression guard.
        """
        cand = _make_candidate(
            yoe=6.0,
            career=[
                {
                    "company": "Acme",
                    "title": "ML Engineer",
                    "start_date": "2020-06-01",
                    "end_date": None,
                    "duration_months": 72,
                    "is_current": True,
                    "industry": "Software",
                    "company_size": "201-500",
                    "description": "Built production retrieval systems using FAISS and NDCG evaluation.",
                }
            ],
            skills=[
                {"name": "Python", "proficiency": "advanced", "endorsements": 12, "duration_months": 60},
            ],
            signals={
                "profile_completeness_score": 80,
                "signup_date": "2023-01-01",
                "last_active_date": "2026-06-01",
                "open_to_work_flag": True,
                "profile_views_received_30d": 10,
                "applications_submitted_30d": 0,
                "recruiter_response_rate": 0.8,
                "avg_response_time_hours": 4.0,
                "skill_assessment_scores": {},   # sentinel
                "connection_count": 100,
                "endorsements_received": 20,
                "notice_period_days": 30,
                "expected_salary_range_inr_lpa": {"min": 20, "max": 35},
                "preferred_work_mode": "hybrid",
                "willing_to_relocate": True,
                "github_activity_score": -1,       # sentinel
                "search_appearance_30d": 5,
                "saved_by_recruiters_30d": 2,
                "interview_completion_rate": 0.9,
                "offer_acceptance_rate": -1,       # sentinel
                "verified_email": True,
                "verified_phone": True,
                "linkedin_connected": True,
            },
        )
        p, reasons = compute_penalty(cand, CFG)
        assert p == pytest.approx(1.0, abs=1e-6)
        assert reasons == []


# ---------------------------------------------------------------------------
# Combination-formula sanity (the §2.5.d math, not the data)
# ---------------------------------------------------------------------------

class TestCombinationFormula:
    """
    The §2.5.d combination is `min(g) × √(others)`. These tests don't need
    realistic candidates — they directly exercise the math by constructing
    candidates that fire exactly the gates we want.
    """

    def test_single_gate_passes_through_unchanged(self):
        """One gate firing → P = that gate's score (no softening)."""
        cand = _make_candidate(
            yoe=8.0,
            career=[
                _consulting_stint("Infosys", duration_months=48, is_current=True, start_year=2019),
                _consulting_stint("TCS", duration_months=48, is_current=False, start_year=2015),
            ],
            current_industry="IT Services",
            current_company="Infosys",
            current_company_size="10001+",
        )
        p, reasons = compute_penalty(cand, CFG)
        assert "consulting_only" in reasons
        p_scale = CFG["penalties"]["p_scale"]
        assert p == pytest.approx(0.15 * p_scale, abs=1e-6)
        assert len(reasons) == 1

    def test_two_gates_soften_secondary(self):
        """Three gates fire (consulting + research + no_recent_code) → §2.5.d softened.

        Same as the stacked test in TestDisqualifierGates — the formula
        applies to ALL non-honeypot gates that fire, not just two. The
        test verifies the math with three active gates.
        """
        cand = _make_candidate(
            yoe=8.0,
            career=[
                {
                    "company": "BigTech Research",
                    "title": "Research Scientist",
                    "start_date": "2018-01-01",
                    "end_date": None,
                    "duration_months": 96,
                    "is_current": True,
                    "industry": "IT Services",
                    "company_size": "10001+",
                    # No PRODUCTION_LEXICON terms — research framing only.
                    "description": "PhD researcher in industry research lab. Studies graph theory and "
                                   "combinatorial optimization. Publishes papers, academic collaborations, "
                                   "university teaching.",
                }
            ],
            current_title="Research Scientist",
            current_industry="IT Services",
            current_company="BigTech Research",
            current_company_size="10001+",
        )
        p, reasons = compute_penalty(cand, CFG)
        assert "consulting_only" in reasons
        assert "research_only" in reasons
        # §2.5.d (p_scale-aware): min(scaled) × √(prod(other scaled)).
        p_scale = CFG["penalties"]["p_scale"]
        worst = 0.15 * p_scale
        sec1 = 0.20 * p_scale
        sec2 = 0.25 * p_scale
        expected = worst * (sec1 * sec2) ** 0.5
        assert p == pytest.approx(expected, abs=1e-3)

    def test_honeypot_overrides_all(self):
        """A honeypot returns 0.01 regardless of other gates — single arm."""
        # Impossible yoe + consulting career + research framing.
        cand = _make_candidate(
            yoe=25.0,  # impossible — yoe*12=300, career sum tiny
            career=[
                {
                    "company": "BigTech Research",
                    "title": "Research Scientist",
                    "start_date": "2024-01-01",
                    "end_date": None,
                    "duration_months": 12,
                    "is_current": True,
                    "industry": "IT Services",
                    "company_size": "10001+",
                    "description": "PhD researcher. Studies embedding spaces. Publishes papers.",
                }
            ],
            current_title="Research Scientist",
            current_industry="IT Services",
            current_company="BigTech Research",
            current_company_size="10001+",
        )
        p, reasons = compute_penalty(cand, CFG)
        assert "honeypot" in reasons
        assert p == pytest.approx(0.01, abs=1e-6)
        # Other gates should NOT appear in reasons (honeypot short-circuits).
        assert "consulting_only" not in reasons
        assert "research_only" not in reasons


# ---------------------------------------------------------------------------
# p_scale: the calibratable global severity scale (§2.5.e)
# ---------------------------------------------------------------------------

class TestPenaltyScale:
    """penalties.p_scale multiplies all non-honeypot gate severities."""

    def test_p_scale_halves_non_honeypot_penalty(self):
        """p_scale=0.5 → consulting_only fires at 0.075, not 0.15."""
        cfg = {**CFG, "penalties": {**CFG["penalties"], "p_scale": 0.5}}
        cand = _make_candidate(
            yoe=8.0,
            career=[
                _consulting_stint("Infosys", duration_months=48, is_current=True, start_year=2019),
                _consulting_stint("TCS", duration_months=48, is_current=False, start_year=2015),
            ],
            current_industry="IT Services",
            current_company="Infosys",
            current_company_size="10001+",
        )
        p, reasons = compute_penalty(cand, cfg)
        assert "consulting_only" in reasons
        assert p == pytest.approx(0.5 * 0.15, abs=1e-6)

    def test_p_scale_does_not_affect_honeypot(self):
        """p_scale changes non-honeypot gates only — honeypot stays at 0.01."""
        cfg = {**CFG, "penalties": {**CFG["penalties"], "p_scale": 0.01}}
        cand = _make_candidate(
            yoe=25.0,
            career=[{
                "company": "X", "title": "Y", "start_date": "2024-01-01",
                "end_date": "2024-06-01", "duration_months": 5, "is_current": False,
                "industry": "Software", "company_size": "1-10",
                "description": "Tiny stint.",
            }],
        )
        p, reasons = compute_penalty(cand, cfg)
        assert p == pytest.approx(0.01, abs=1e-6)
