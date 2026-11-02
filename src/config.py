"""Project configuration with validated, user-overridable defaults."""

from __future__ import annotations

import json
import math
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "params_v0941.json"

DEFAULT_PARAMS: dict[str, Any] = {
    "version": "0.9.41",
    "lambda_shrink": 0.81,
    "rho": -0.135,
    "prior_00": 1.17,
    "prior_11": 1.15,
    "high_score_cap": 0.90,
    "model_weight_base": 0.54,
    "model_weight_min": 0.38,
    "model_weight_max": 0.62,
    "kelly_fraction": 0.45,
    "rec_conf_min": 0.56,
    "low_conf_skip": 0.47,
    "draw_protect": 1.08,
    "home_boost": {
        "PL": 1.02, "PD": 1.03, "SA": 1.01, "BL1": 0.97, "FL1": 1.00,
        "Eredivisie": 1.01, "SerieA": 1.01, "J1": 1.00, "KLeague": 0.99,
        "LaLiga": 1.03,
    },
    "base_home_xg": 1.45,
    "base_away_xg": 1.15,
    "allow_baseline_recommend": False,
    "max_goals": 10,
    "schedule_cache_ttl_seconds": 300,
    "odds_ttl_seconds": 900,
    "http_timeout_seconds": 8,
}

_BOUNDED = {
    "lambda_shrink": (0.0, 1.0),
    "model_weight_base": (0.0, 1.0),
    "model_weight_min": (0.0, 1.0),
    "model_weight_max": (0.0, 1.0),
    "kelly_fraction": (0.0, 1.0),
    "rec_conf_min": (0.0, 1.0),
    "low_conf_skip": (0.0, 1.0),
    "high_score_cap": (0.0, 1.0),
}


def validate_params(params: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(params, Mapping):
        raise ValueError("Parameters must be a JSON object.")
    unknown = set(params) - set(DEFAULT_PARAMS)
    if unknown:
        raise ValueError(f"Unknown parameter(s): {', '.join(sorted(unknown))}")
    merged = deepcopy(DEFAULT_PARAMS)
    for key, value in params.items():
        if key == "home_boost":
            if not isinstance(value, Mapping):
                raise ValueError("home_boost must be a JSON object.")
            merged[key].update(value)
        else:
            merged[key] = value
    if not isinstance(merged["version"], str) or not merged["version"].strip():
        raise ValueError("version must be a non-empty string.")
    if not isinstance(merged["allow_baseline_recommend"], bool):
        raise ValueError("allow_baseline_recommend must be a boolean.")
    for key, (minimum, maximum) in _BOUNDED.items():
        value = _finite_number(merged[key], key)
        if not minimum <= value <= maximum:
            raise ValueError(f"{key} must be between {minimum} and {maximum}.")
        merged[key] = value
    for key in ("rho", "base_home_xg", "base_away_xg", "draw_protect"):
        value = _finite_number(merged[key], key)
        if key == "rho":
            if not -1.0 <= value <= 1.0:
                raise ValueError("rho must be between -1 and 1.")
        elif key in {"base_home_xg", "base_away_xg"}:
            if value < 0:
                raise ValueError(f"{key} must be non-negative.")
        elif value <= 0:
            raise ValueError(f"{key} must be greater than zero.")
        merged[key] = value
    for key in ("prior_00", "prior_11"):
        merged[key] = _finite_number(merged[key], key)
        if merged[key] <= 0:
            raise ValueError(f"{key} must be greater than zero.")
    for key in ("schedule_cache_ttl_seconds", "odds_ttl_seconds", "http_timeout_seconds"):
        value = _finite_number(merged[key], key)
        if value <= 0:
            raise ValueError(f"{key} must be greater than zero.")
        merged[key] = value
    if isinstance(merged["max_goals"], bool) or not isinstance(merged["max_goals"], int):
        raise ValueError("max_goals must be an integer.")
    if not 1 <= merged["max_goals"] <= 30:
        raise ValueError("max_goals must be between 1 and 30.")
    if merged["model_weight_min"] > merged["model_weight_max"]:
        raise ValueError("model_weight_min cannot exceed model_weight_max.")
    if not merged["model_weight_min"] <= merged["model_weight_base"] <= merged["model_weight_max"]:
        raise ValueError("model_weight_base must be within its configured bounds.")
    for league, boost in merged["home_boost"].items():
        if not isinstance(league, str) or not league:
            raise ValueError("home_boost keys must be non-empty strings.")
        if _finite_number(boost, f"home_boost.{league}") <= 0:
            raise ValueError(f"home_boost.{league} must be greater than zero.")
        merged["home_boost"][league] = float(boost)
    return merged


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def load_params(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if path is None and not config_path.exists():
        return validate_params(DEFAULT_PARAMS)
    try:
        with config_path.open("r", encoding="utf-8") as file:
            loaded = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load configuration from {config_path}: {error}") from error
    return validate_params(loaded)


def save_params(params: Mapping[str, Any], path: str | os.PathLike[str] | None = None) -> Path:
    validated = validate_params(params)
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=config_path.parent, delete=False
        ) as file:
            temp_name = file.name
            json.dump(validated, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
        os.replace(temp_name, config_path)
    except OSError as error:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
        raise OSError(f"Could not save configuration to {config_path}: {error}") from error
    return config_path
