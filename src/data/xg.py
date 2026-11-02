"""Expected-goals input validation and explicitly disclosed baseline fallback."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from src.config import load_params, validate_params
from src.utils import parse_timestamp


def get_expected_goals(
    match: dict[str, Any],
    params: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> tuple[float, float, str, dict[str, Any]]:
    if not isinstance(match, dict):
        raise ValueError("match must be a JSON object.")
    config = load_params() if params is None else validate_params(params)
    xg_values: Mapping[str, Any] | None = None
    provenance: Mapping[str, Any] = {}
    if "prematch_xg" in match:
        if "home_xg" in match or "away_xg" in match:
            raise ValueError("Use canonical prematch_xg or flat home_xg/away_xg fields, not both.")
        raw = match["prematch_xg"]
        if not isinstance(raw, Mapping):
            raise ValueError("prematch_xg must be an object with home and away values.")
        xg_values = raw
        provenance = raw
    else:
        xg_values = match
        raw_metadata = match.get("xg_metadata", {})
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("xg_metadata must be an object.")
        provenance = raw_metadata
    xg_type = match.get("xg_type", provenance.get("type", "prematch"))
    if xg_type not in {"prematch", "postmatch_actual_xg"}:
        raise ValueError("xg_type must be 'prematch' or 'postmatch_actual_xg'.")
    if xg_type == "postmatch_actual_xg":
        raise ValueError("Post-match actual xG cannot be used as pre-match prediction input.")

    has_home = "home" in xg_values if "prematch_xg" in match else "home_xg" in xg_values
    has_away = "away" in xg_values if "prematch_xg" in match else "away_xg" in xg_values
    if has_home != has_away:
        raise ValueError("Provide both home_xg and away_xg, or neither.")
    if not has_home and ("source" in provenance or "generated_at" in provenance):
        raise ValueError("xG provenance requires both expected-goal values.")

    league = match.get("league")
    if league is not None and not isinstance(league, str):
        raise ValueError("league must be a string when provided.")
    home_boost = 1.0
    if isinstance(league, str):
        home_boost = float(config["home_boost"].get(league.strip(), 1.0))
    home_baseline = float(config["base_home_xg"]) * home_boost
    away_baseline = float(config["base_away_xg"])
    boost_applied = home_boost != 1.0
    boost_suffix = "_with_league_home_boost" if boost_applied else ""
    meta: dict[str, Any] = {
        "xg_type": "prematch",
        "source": "configured_baseline_unvalidated",
        "generated_at": None,
        "freshness": "not_applicable",
        "recommendation_eligible": False,
    }

    if not has_home:
        return (
            home_baseline,
            away_baseline,
            f"unvalidated_configured_baseline{boost_suffix}",
            meta,
        )
    values: list[float] = []
    shrinkage = float(config["lambda_shrink"])
    fields = (("home", "home_xg", home_baseline), ("away", "away_xg", away_baseline))
    for canonical_key, flat_key, baseline in fields:
        key = canonical_key if "prematch_xg" in match else flat_key
        value = xg_values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{flat_key} must be a finite non-negative number.")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{flat_key} must be a finite non-negative number.")
        values.append(shrinkage * value + (1.0 - shrinkage) * baseline)

    provider_source = provenance.get("source")
    generated_at_value = provenance.get("generated_at")
    if provider_source is None and generated_at_value is None:
        meta["source"] = "provided_prematch_xg_unverified"
        source = "prematch_xg_shrunk_unverified"
    else:
        if not isinstance(provider_source, str) or not provider_source.strip():
            raise ValueError("Provider xG requires a non-empty source.")
        provider_source = provider_source.strip()
        if provider_source not in config["xg_allowed_sources"]:
            raise ValueError(f"xG source is not in xg_allowed_sources: {provider_source}")
        if generated_at_value is None:
            raise ValueError("Provider xG requires generated_at.")
        if "kickoff" not in match:
            raise ValueError("Provider pre-match xG requires a timezone-aware kickoff.")
        generated_at = parse_timestamp(generated_at_value, "xG generated_at")
        kickoff = parse_timestamp(match["kickoff"], "kickoff")
        current_time = now or datetime.now(timezone.utc)
        if current_time.tzinfo is None:
            raise ValueError("now must include a timezone.")
        current_time = current_time.astimezone(timezone.utc)
        if kickoff <= current_time:
            raise ValueError("Provider pre-match xG cannot be used at or after kickoff.")
        if generated_at >= kickoff:
            raise ValueError("Provider pre-match xG generated_at must be earlier than kickoff.")
        if generated_at > current_time:
            raise ValueError("Provider xG generated_at cannot be in the future.")
        age_seconds = (current_time - generated_at).total_seconds()
        if age_seconds > float(config["xg_ttl_seconds"]):
            raise ValueError("Provider pre-match xG is stale.")
        source = (
            "prematch_xg_provider_shrunk"
            if shrinkage < 1
            else "prematch_xg_provider"
        )
        meta = {
            "xg_type": "prematch",
            "source": provider_source,
            "generated_at": generated_at.isoformat(),
            "freshness": "fresh",
            "age_seconds": round(age_seconds, 3),
            "recommendation_eligible": True,
        }
    return (
        values[0],
        values[1],
        f"{source}{boost_suffix}",
        meta,
    )
