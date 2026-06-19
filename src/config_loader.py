"""
config_loader.py — Load and expose scoring_config.yaml as a typed config object.

The config is loaded once at startup and passed around as a plain dict.
All scoring modules read from this dict — no hardcoded magic numbers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "scoring_config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """
    Load scoring_config.yaml and return as a nested dict.

    Args:
        path: Path to the config file. Defaults to config/scoring_config.yaml.

    Returns:
        Parsed config dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(f"Scoring config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must be a YAML mapping, got {type(config)}")

    _validate_config(config)
    logger.info("Loaded scoring config from %s", config_path)
    return config


def _validate_config(config: dict) -> None:
    """Sanity-check the config structure. Raises ValueError on bad config."""
    required_sections = [
        "weights",
        "experience",
        "education",
        "skills",
        "location",
        "behavior",
        "penalties",
        "role_affinity",
        "honeypot_detection",
    ]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"scoring_config.yaml is missing required section: '{section}'")

    # Weights must sum to ~1.0
    weights = config["weights"]
    total = sum(weights.values())
    if not (0.99 <= total <= 1.01):
        raise ValueError(
            f"scoring_config.yaml weights must sum to 1.0, got {total:.4f}. "
            f"Weights: {weights}"
        )

    logger.debug("Config validation passed. Weight sum = %.4f", total)
