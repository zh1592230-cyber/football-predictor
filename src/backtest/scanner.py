"""Chronological validation scan for the model/market probability blend weight."""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

from src.backtest.settle import score_probabilities
from src.config import load_params, validate_params

_OUTCOMES = ("home", "draw", "away")


def _validate_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("Each scan record must be a JSON object.")
    required = ("match_id", "date", "model_probabilities", "market_probabilities", "actual_outcome")
    if any(key not in record for key in required):
        raise ValueError(f"Each scan record requires: {', '.join(required)}.")
    raw_date = record["date"]
    if isinstance(raw_date, str):
        try:
            match_date = date.fromisoformat(raw_date[:10]).isoformat()
        except ValueError as error:
            raise ValueError(f"Invalid record date: {raw_date}") from error
    elif isinstance(raw_date, datetime):
        match_date = raw_date.date().isoformat()
    elif isinstance(raw_date, date):
        match_date = raw_date.isoformat()
    else:
        raise ValueError("Each scan record date must be an ISO date or timestamp.")
    actual = record["actual_outcome"]
    if actual not in _OUTCOMES:
        raise ValueError("actual_outcome must be home, draw, or away.")
    model = _normalize(record["model_probabilities"], "model_probabilities")
    market = _normalize(record["market_probabilities"], "market_probabilities")
    if not isinstance(record["match_id"], str) or not record["match_id"]:
        raise ValueError("Each scan record requires a non-empty match_id.")
    return {**record, "date": match_date, "actual_outcome": actual,
            "model_probabilities": model, "market_probabilities": market}


def _normalize(values: Any, name: str) -> dict[str, float]:
    if not isinstance(values, dict):
        raise ValueError(f"{name} must be an object.")
    output: dict[str, float] = {}
    for outcome in _OUTCOMES:
        value = values.get(outcome)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name}.{outcome} must be a finite probability.")
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name}.{outcome} must be between 0 and 1.")
        output[outcome] = value
    total = math.fsum(output.values())
    if total <= 0:
        raise ValueError(f"{name} must have positive probability mass.")
    return {outcome: output[outcome] / total for outcome in _OUTCOMES}


def _mean_brier(records: list[dict[str, Any]], weight: float) -> float:
    losses = []
    for record in records:
        mixed = {
            outcome: weight * record["model_probabilities"][outcome]
            + (1.0 - weight) * record["market_probabilities"][outcome]
            for outcome in _OUTCOMES
        }
        losses.append(score_probabilities(mixed, record["actual_outcome"])["brier"])
    return math.fsum(losses) / len(losses)


def scan_model_weight(
    records: list[dict[str, Any]] | dict[str, Any], params: dict[str, Any] | None = None
) -> dict[str, Any]:
    if isinstance(records, dict):
        records = records.get("matches")
    if not isinstance(records, list):
        raise ValueError("Scan input must be a labeled prediction array or settlement report.")
    validated = [_validate_record(record) for record in records]
    validated.sort(key=lambda item: (item["date"], item["match_id"]))
    dates = sorted({item["date"] for item in validated})
    split = int(len(dates) * 0.7)
    if len(dates) < 3 or split < 2 or len(dates) - split < 1:
        raise ValueError("At least 3 distinct dates are required for chronological train/validation.")
    train_dates, validation_dates = set(dates[:split]), set(dates[split:])
    train = [item for item in validated if item["date"] in train_dates]
    validation = [item for item in validated if item["date"] in validation_dates]
    if len(train) < 10 or len(validation) < 5:
        raise ValueError(
            "Insufficient labeled history: require at least 10 training and 5 validation matches "
            "after the chronological split."
        )
    config = load_params() if params is None else validate_params(params)
    low, high = float(config["model_weight_min"]), float(config["model_weight_max"])
    steps = 24
    candidates = [low + (high - low) * index / steps for index in range(steps + 1)]
    scored = [
        {"model_weight": weight, "train_brier": _mean_brier(train, weight),
         "validation_brier": _mean_brier(validation, weight)}
        for weight in candidates
    ]
    best = min(scored, key=lambda item: (item["validation_brier"], item["model_weight"]))
    return {
        "method": "chronological 70/30 split; multiclass 1X2 Brier score",
        "records": len(validated),
        "training_matches": len(train),
        "validation_matches": len(validation),
        "training_dates": [min(train_dates), max(train_dates)],
        "validation_dates": [min(validation_dates), max(validation_dates)],
        "best_candidate": best,
        "candidates": scored,
        "applied_to_config": False,
    }
