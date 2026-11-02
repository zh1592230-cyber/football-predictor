"""Probability-level fusion of the score model and de-vigged market odds."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Mapping

from src.config import load_params, validate_params
from src.utils import parse_timestamp

OUTCOMES = ("home", "draw", "away")


def de_vig_1x2(odds: Mapping[str, float]) -> dict[str, float]:
    probabilities: dict[str, float] = {}
    for outcome in OUTCOMES:
        price = odds.get(outcome)
        if isinstance(price, bool) or not isinstance(price, (int, float)):
            raise ValueError(f"{outcome} odds must be a finite number greater than 1.")
        price = float(price)
        if not math.isfinite(price) or price <= 1:
            raise ValueError(f"{outcome} odds must be a finite number greater than 1.")
        probabilities[outcome] = 1.0 / price
    total = math.fsum(probabilities.values())
    return {outcome: probabilities[outcome] / total for outcome in OUTCOMES}


def _probabilities(values: Mapping[str, float], name: str) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping of home/draw/away probabilities.")
    result: dict[str, float] = {}
    for outcome in OUTCOMES:
        value = values.get(outcome)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name}.{outcome} must be finite and between 0 and 1.")
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name}.{outcome} must be finite and between 0 and 1.")
        result[outcome] = value
    total = math.fsum(result.values())
    if total <= 0:
        raise ValueError(f"{name} must sum to a positive value.")
    return {outcome: result[outcome] / total for outcome in OUTCOMES}


def fuse_probabilities(
    model_probabilities: Mapping[str, float],
    market_probabilities: Mapping[str, float] | None = None,
    market_updated_at: str | None = None,
    params: dict | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    config = load_params() if params is None else validate_params(params)
    model = _probabilities(model_probabilities, "model_probabilities")
    if market_probabilities is None:
        return {
            "probabilities": model,
            "model_weight": 1.0,
            "market_freshness": 0.0,
            "market_used": False,
        }
    market = _probabilities(market_probabilities, "market_probabilities")
    if market_updated_at is None:
        raise ValueError("market_updated_at is required when market probabilities are supplied.")
    updated_at = parse_timestamp(market_updated_at, "market_updated_at")
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        raise ValueError("now must include a timezone.")
    if updated_at > clock.astimezone(timezone.utc) + timedelta(seconds=60):
        raise ValueError("market_updated_at cannot be more than 60 seconds in the future.")
    age = max(0.0, (clock.astimezone(timezone.utc) - updated_at).total_seconds())
    ttl = float(config["odds_ttl_seconds"])
    freshness = max(0.0, min(1.0, 1.0 - age / ttl))
    minimum = float(config["model_weight_min"])
    maximum = float(config["model_weight_max"])
    base = float(config["model_weight_base"])
    model_weight = min(maximum, max(minimum, base))
    model_weight = model_weight + (1.0 - model_weight) * (1.0 - freshness)
    fused = {
        outcome: model_weight * model[outcome] + (1.0 - model_weight) * market[outcome]
        for outcome in OUTCOMES
    }
    total = math.fsum(fused.values())
    fused = {outcome: value / total for outcome, value in fused.items()}
    return {
        "probabilities": fused,
        "model_weight": model_weight,
        "market_freshness": freshness,
        "market_age_seconds": age,
        "market_used": True,
    }
