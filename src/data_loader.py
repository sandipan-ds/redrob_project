"""
data_loader.py — Load and validate candidate JSONL / JSON files.

Supports:
  - JSONL (one candidate per line) — production format for 100K candidates
  - JSON array — sample/dev format used in data/samples/

P0 exit criterion: loads 100K JSONL, validates schema field presence.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Generator, Iterator

logger = logging.getLogger(__name__)

# Required top-level keys per candidate_schema.json
REQUIRED_TOP_LEVEL = {
    "candidate_id",
    "profile",
    "career_history",
    "education",
    "skills",
    "redrob_signals",
}

# Required profile sub-keys
REQUIRED_PROFILE_KEYS = {
    "anonymized_name",
    "headline",
    "summary",
    "location",
    "country",
    "years_of_experience",
    "current_title",
    "current_company",
    "current_company_size",
    "current_industry",
}

# Required redrob_signals sub-keys
REQUIRED_SIGNAL_KEYS = {
    "profile_completeness_score",
    "signup_date",
    "last_active_date",
    "open_to_work_flag",
    "recruiter_response_rate",
    "skill_assessment_scores",
    "notice_period_days",
    "preferred_work_mode",
    "willing_to_relocate",
    "github_activity_score",
    "interview_completion_rate",
    "offer_acceptance_rate",
}

CANDIDATE_ID_RE = re.compile(r"^CAND_[0-9]{7}$")


class SchemaValidationError(ValueError):
    """Raised when a candidate record fails schema validation."""


def _validate_candidate(record: dict, strict: bool = False) -> list[str]:
    """
    Validate a single candidate record against the schema.

    Returns a list of validation error strings (empty = valid).
    If strict=True, missing optional keys are also flagged.
    """
    errors: list[str] = []
    cid = record.get("candidate_id", "<missing>")

    # Top-level keys
    missing_top = REQUIRED_TOP_LEVEL - record.keys()
    if missing_top:
        errors.append(f"[{cid}] Missing top-level keys: {missing_top}")
        return errors  # Can't validate sub-keys if top-level is broken

    # candidate_id format
    if not CANDIDATE_ID_RE.match(str(cid)):
        errors.append(f"[{cid}] candidate_id does not match CAND_XXXXXXX pattern")

    # profile sub-keys
    profile = record.get("profile", {})
    missing_profile = REQUIRED_PROFILE_KEYS - profile.keys()
    if missing_profile:
        errors.append(f"[{cid}] Missing profile keys: {missing_profile}")

    # years_of_experience type
    yoe = profile.get("years_of_experience")
    if yoe is not None and not isinstance(yoe, (int, float)):
        errors.append(f"[{cid}] profile.years_of_experience must be numeric, got {type(yoe)}")

    # career_history is a non-empty list
    ch = record.get("career_history", [])
    if not isinstance(ch, list) or len(ch) == 0:
        errors.append(f"[{cid}] career_history must be a non-empty list")
    else:
        for i, entry in enumerate(ch):
            for key in ("company", "title", "duration_months", "is_current", "description"):
                if key not in entry:
                    errors.append(f"[{cid}] career_history[{i}] missing key: {key}")

    # education is a list (may be empty)
    edu = record.get("education", [])
    if not isinstance(edu, list):
        errors.append(f"[{cid}] education must be a list")

    # skills is a list (may be empty)
    skills = record.get("skills", [])
    if not isinstance(skills, list):
        errors.append(f"[{cid}] skills must be a list")
    else:
        for i, sk in enumerate(skills):
            for key in ("name", "proficiency", "endorsements"):
                if key not in sk:
                    errors.append(f"[{cid}] skills[{i}] missing key: {key}")

    # redrob_signals sub-keys
    signals = record.get("redrob_signals", {})
    missing_signals = REQUIRED_SIGNAL_KEYS - signals.keys()
    if missing_signals:
        errors.append(f"[{cid}] Missing redrob_signals keys: {missing_signals}")

    return errors


def iter_candidates_jsonl(
    path: str | Path,
    validate: bool = True,
    skip_invalid: bool = True,
) -> Generator[dict, None, None]:
    """
    Stream candidates from a JSONL file (one JSON object per line).

    Args:
        path: Path to the .jsonl file.
        validate: Run schema validation on each record.
        skip_invalid: If True, log and skip invalid records; if False, raise.

    Yields:
        Validated candidate dicts.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {path}")

    total = 0
    skipped = 0

    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Line %d: JSON parse error — %s", lineno, exc)
                skipped += 1
                continue

            if validate:
                errors = _validate_candidate(record)
                if errors:
                    for err in errors:
                        logger.warning("Line %d: %s", lineno, err)
                    if not skip_invalid:
                        raise SchemaValidationError(
                            f"Validation failed at line {lineno}: {errors[0]}"
                        )
                    skipped += 1
                    continue

            total += 1
            yield record

    logger.info(
        "Loaded %d candidates from %s (%d skipped due to errors)",
        total,
        path,
        skipped,
    )


def load_candidates_json(
    path: str | Path,
    validate: bool = True,
    skip_invalid: bool = True,
) -> list[dict]:
    """
    Load candidates from a JSON array file (dev/sample format).

    Returns a list of validated candidate dicts.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data)}")

    results: list[dict] = []
    skipped = 0

    for i, record in enumerate(data):
        if validate:
            errors = _validate_candidate(record)
            if errors:
                for err in errors:
                    logger.warning("Record %d: %s", i, err)
                if not skip_invalid:
                    raise SchemaValidationError(
                        f"Validation failed at record {i}: {errors[0]}"
                    )
                skipped += 1
                continue
        results.append(record)

    logger.info(
        "Loaded %d candidates from %s (%d skipped due to errors)",
        len(results),
        path,
        skipped,
    )
    return results


def load_candidates(
    path: str | Path,
    validate: bool = True,
    skip_invalid: bool = True,
) -> list[dict] | Iterator[dict]:
    """
    Auto-detect file format (.jsonl vs .json) and load candidates.

    For JSONL files, returns a generator (memory-efficient for 100K records).
    For JSON files, returns a list.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        return iter_candidates_jsonl(path, validate=validate, skip_invalid=skip_invalid)
    elif suffix == ".json":
        return load_candidates_json(path, validate=validate, skip_invalid=skip_invalid)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Expected .jsonl or .json")
