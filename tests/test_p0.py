"""
test_p0.py — P0 exit criterion tests.

P0 exit criterion: loads sample candidates, validates schema, loads config.
Run with: pytest tests/test_p0.py -v
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.config_loader import load_config
from src.data_loader import (
    SchemaValidationError,
    _validate_candidate,
    load_candidates,
    load_candidates_json,
)

SAMPLE_JSON = Path("data/samples/sample_candidates.json")
CONFIG_PATH = Path("config/scoring_config.yaml")


# ---------------------------------------------------------------------------
# Config loader tests
# ---------------------------------------------------------------------------

class TestConfigLoader:
    def test_loads_default_config(self):
        config = load_config()
        assert isinstance(config, dict)

    def test_required_sections_present(self):
        config = load_config()
        for section in ["weights", "experience", "education", "skills",
                        "location", "behavior", "penalties", "role_affinity",
                        "honeypot_detection"]:
            assert section in config, f"Missing section: {section}"

    def test_weights_sum_to_one(self):
        config = load_config()
        total = sum(config["weights"].values())
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected ~1.0"

    def test_missing_config_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")


# ---------------------------------------------------------------------------
# Data loader tests — sample JSON
# ---------------------------------------------------------------------------

class TestDataLoaderSampleJson:
    def test_loads_sample_json(self):
        candidates = load_candidates_json(SAMPLE_JSON)
        assert isinstance(candidates, list)
        assert len(candidates) > 0

    def test_all_have_required_top_level_keys(self):
        candidates = load_candidates_json(SAMPLE_JSON)
        required = {"candidate_id", "profile", "career_history", "education",
                    "skills", "redrob_signals"}
        for c in candidates:
            missing = required - c.keys()
            assert not missing, f"{c.get('candidate_id')} missing: {missing}"

    def test_candidate_id_format(self):
        import re
        pattern = re.compile(r"^CAND_[0-9]{7}$")
        candidates = load_candidates_json(SAMPLE_JSON)
        for c in candidates:
            assert pattern.match(c["candidate_id"]), (
                f"Bad candidate_id: {c['candidate_id']}"
            )

    def test_career_history_non_empty(self):
        candidates = load_candidates_json(SAMPLE_JSON)
        for c in candidates:
            assert len(c["career_history"]) >= 1, (
                f"{c['candidate_id']} has empty career_history"
            )

    def test_redrob_signals_present(self):
        candidates = load_candidates_json(SAMPLE_JSON)
        for c in candidates:
            signals = c["redrob_signals"]
            assert "last_active_date" in signals
            assert "recruiter_response_rate" in signals
            assert "open_to_work_flag" in signals

    def test_sentinel_values_allowed(self):
        """github_activity_score=-1 and offer_acceptance_rate=-1 are valid sentinels."""
        candidates = load_candidates_json(SAMPLE_JSON)
        for c in candidates:
            signals = c["redrob_signals"]
            gas = signals.get("github_activity_score")
            oar = signals.get("offer_acceptance_rate")
            # Sentinels are -1; valid range is -1 to 100 / -1 to 1
            if gas is not None:
                assert gas >= -1, f"{c['candidate_id']} github_activity_score < -1"
            if oar is not None:
                assert oar >= -1, f"{c['candidate_id']} offer_acceptance_rate < -1"


# ---------------------------------------------------------------------------
# Data loader tests — JSONL (synthetic)
# ---------------------------------------------------------------------------

class TestDataLoaderJsonl:
    def _make_valid_candidate(self, cid: str = "CAND_0000001") -> dict:
        return {
            "candidate_id": cid,
            "profile": {
                "anonymized_name": "Test User",
                "headline": "ML Engineer",
                "summary": "Summary text.",
                "location": "Noida",
                "country": "India",
                "years_of_experience": 7.0,
                "current_title": "ML Engineer",
                "current_company": "Acme",
                "current_company_size": "51-200",
                "current_industry": "Software",
            },
            "career_history": [
                {
                    "company": "Acme",
                    "title": "ML Engineer",
                    "start_date": "2020-01-01",
                    "end_date": None,
                    "duration_months": 60,
                    "is_current": True,
                    "industry": "Software",
                    "company_size": "51-200",
                    "description": "Built ML pipelines.",
                }
            ],
            "education": [],
            "skills": [
                {"name": "Python", "proficiency": "expert", "endorsements": 50, "duration_months": 60}
            ],
            "redrob_signals": {
                "profile_completeness_score": 90.0,
                "signup_date": "2023-01-01",
                "last_active_date": "2026-06-01",
                "open_to_work_flag": True,
                "profile_views_received_30d": 10,
                "applications_submitted_30d": 2,
                "recruiter_response_rate": 0.8,
                "avg_response_time_hours": 24.0,
                "skill_assessment_scores": {},
                "connection_count": 200,
                "endorsements_received": 50,
                "notice_period_days": 30,
                "expected_salary_range_inr_lpa": {"min": 20.0, "max": 40.0},
                "preferred_work_mode": "hybrid",
                "willing_to_relocate": True,
                "github_activity_score": 75.0,
                "search_appearance_30d": 100,
                "saved_by_recruiters_30d": 5,
                "interview_completion_rate": 0.9,
                "offer_acceptance_rate": 0.7,
                "verified_email": True,
                "verified_phone": True,
                "linkedin_connected": True,
            },
        }

    def test_loads_valid_jsonl(self):
        candidate = self._make_valid_candidate()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(json.dumps(candidate) + "\n")
            tmp_path = f.name

        results = list(load_candidates(tmp_path))
        assert len(results) == 1
        assert results[0]["candidate_id"] == "CAND_0000001"

    def test_skips_invalid_jsonl_by_default(self):
        bad_line = '{"candidate_id": "BAD_ID", "profile": {}}\n'
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(bad_line)
            tmp_path = f.name

        # Should skip the bad record and return empty list
        results = list(load_candidates(tmp_path, skip_invalid=True))
        assert results == []

    def test_raises_on_invalid_jsonl_strict(self):
        bad_line = '{"candidate_id": "BAD_ID", "profile": {}}\n'
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(bad_line)
            tmp_path = f.name

        with pytest.raises(SchemaValidationError):
            list(load_candidates(tmp_path, skip_invalid=False))

    def test_multiple_candidates_jsonl(self):
        candidates = [
            self._make_valid_candidate(f"CAND_{str(i).zfill(7)}")
            for i in range(1, 6)
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            for c in candidates:
                f.write(json.dumps(c) + "\n")
            tmp_path = f.name

        results = list(load_candidates(tmp_path))
        assert len(results) == 5


# ---------------------------------------------------------------------------
# Schema validator unit tests
# ---------------------------------------------------------------------------

class TestSchemaValidator:
    def test_valid_candidate_no_errors(self):
        record = {
            "candidate_id": "CAND_0000001",
            "profile": {
                "anonymized_name": "A",
                "headline": "H",
                "summary": "S",
                "location": "L",
                "country": "India",
                "years_of_experience": 5.0,
                "current_title": "T",
                "current_company": "C",
                "current_company_size": "51-200",
                "current_industry": "Software",
            },
            "career_history": [
                {
                    "company": "C",
                    "title": "T",
                    "duration_months": 60,
                    "is_current": True,
                    "description": "D",
                }
            ],
            "education": [],
            "skills": [{"name": "Python", "proficiency": "expert", "endorsements": 10}],
            "redrob_signals": {
                "profile_completeness_score": 80,
                "signup_date": "2023-01-01",
                "last_active_date": "2026-01-01",
                "open_to_work_flag": True,
                "recruiter_response_rate": 0.5,
                "skill_assessment_scores": {},
                "notice_period_days": 30,
                "preferred_work_mode": "hybrid",
                "willing_to_relocate": True,
                "github_activity_score": -1,
                "interview_completion_rate": 0.8,
                "offer_acceptance_rate": -1,
            },
        }
        errors = _validate_candidate(record)
        assert errors == []

    def test_bad_candidate_id_flagged(self):
        record = {"candidate_id": "WRONG_FORMAT", "profile": {}, "career_history": [],
                  "education": [], "skills": [], "redrob_signals": {}}
        errors = _validate_candidate(record)
        assert any("candidate_id" in e for e in errors)

    def test_missing_top_level_key(self):
        record = {"candidate_id": "CAND_0000001"}  # Missing most keys
        errors = _validate_candidate(record)
        assert any("Missing top-level keys" in e for e in errors)
