"""Expected-goals input validation and explicitly disclosed baseline fallback."""

from __future__ import annotations

import math
from typing import Any

from src.config import load_params


def get_expected_goals(
    match: dict[str, Any], params: dict[str, Any] | None = None
) -> tuple[float, float, str]:
    config = params or load_params()
    has_home = "home_xg" in match
    has_away = "away_xg" in match
    if has_home != has_away:
        raise ValueError("Provide both home_xg and away_xg, or neither.")
    if not has_home:
        return (
            float(config["base_home_xg"]),
            float(config["base_away_xg"]),
            "unvalidated_configured_baseline",
        )
    values = []
    for key, base_key in (("home_xg", "base_home_xg"), ("away_xg", "base_away_xg")):
        value = match[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a finite non-negative number.")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a finite non-negative number.")
        shrinkage = float(config["lambda_shrink"])
        baseline = float(config[base_key])
        values.append(shrinkage * value + (1.0 - shrinkage) * baseline)
    return values[0], values[1], "match_xg_shrunk_to_unvalidated_baseline"
