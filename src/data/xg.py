"""Expected-goals input validation and explicitly disclosed baseline fallback."""

from __future__ import annotations

import math
from typing import Any

from src.config import load_params, validate_params


def get_expected_goals(
    match: dict[str, Any], params: dict[str, Any] | None = None
) -> tuple[float, float, str]:
    if not isinstance(match, dict):
        raise ValueError("match must be a JSON object.")
    config = load_params() if params is None else validate_params(params)
    has_home = "home_xg" in match
    has_away = "away_xg" in match
    if has_home != has_away:
        raise ValueError("Provide both home_xg and away_xg, or neither.")

    league = match.get("league")
    home_boost = 1.0
    if isinstance(league, str):
        home_boost = float(config["home_boost"].get(league.strip(), 1.0))
    home_baseline = float(config["base_home_xg"]) * home_boost
    away_baseline = float(config["base_away_xg"])
    boost_applied = home_boost != 1.0
    boost_suffix = "_with_league_home_boost" if boost_applied else ""

    if not has_home:
        return (
            home_baseline,
            away_baseline,
            f"unvalidated_configured_baseline{boost_suffix}",
        )
    values = []
    shrinkage = float(config["lambda_shrink"])
    for key, baseline in (("home_xg", home_baseline), ("away_xg", away_baseline)):
        value = match[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a finite non-negative number.")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a finite non-negative number.")
        values.append(shrinkage * value + (1.0 - shrinkage) * baseline)
    return (
        values[0],
        values[1],
        f"match_xg_shrunk_to_unvalidated_baseline{boost_suffix}",
    )
