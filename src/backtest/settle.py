"""Match prediction settlement and proper scoring rules."""

from __future__ import annotations

import math
from typing import Any

from src.utils import get_data_dir, save_json, validate_date

_OUTCOMES = ("home", "draw", "away")


def _outcome(home_goals: Any, away_goals: Any) -> str:
    for value, name in ((home_goals, "home_goals"), (away_goals, "away_goals")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
    if home_goals > away_goals:
        return "home"
    if home_goals < away_goals:
        return "away"
    return "draw"


def score_probabilities(probabilities: dict[str, float], actual: str) -> dict[str, float]:
    if actual not in _OUTCOMES:
        raise ValueError("actual_outcome must be home, draw, or away.")
    if not isinstance(probabilities, dict):
        raise ValueError("probabilities must be a home/draw/away object.")
    values: dict[str, float] = {}
    for outcome in _OUTCOMES:
        value = probabilities.get(outcome)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"probabilities.{outcome} must be finite and in [0, 1].")
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"probabilities.{outcome} must be finite and in [0, 1].")
        values[outcome] = value
    if not math.isclose(math.fsum(values.values()), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("1X2 probabilities must sum to 1.")
    brier = math.fsum((values[outcome] - (1.0 if outcome == actual else 0.0)) ** 2 for outcome in _OUTCOMES)
    log_loss = -math.log(max(values[actual], 1e-15))
    return {"brier": brier, "log_loss": log_loss}


def settle_predictions(
    predictions: list[dict[str, Any]], results: list[dict[str, Any]]
) -> dict[str, Any]:
    if not isinstance(predictions, list) or not isinstance(results, list):
        raise ValueError("predictions and results must be JSON arrays.")
    forecasts: dict[str, dict[str, Any]] = {}
    for prediction in predictions:
        if not isinstance(prediction, dict) or prediction.get("status") != "predicted":
            continue
        match_id = prediction.get("match_id")
        if not isinstance(match_id, str) or not match_id:
            raise ValueError("Each prediction requires a non-empty match_id.")
        if match_id in forecasts:
            raise ValueError(f"Duplicate prediction match_id: {match_id}")
        forecasts[match_id] = prediction
    actuals: dict[str, dict[str, Any]] = {}
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("Each result must be a JSON object.")
        match_id = result.get("match_id")
        if not isinstance(match_id, str) or not match_id:
            raise ValueError("Each result requires a non-empty match_id.")
        if match_id in actuals:
            raise ValueError(f"Duplicate result match_id: {match_id}")
        actuals[match_id] = result
    missing = forecasts.keys() - actuals.keys()
    extra = actuals.keys() - forecasts.keys()
    if missing or extra:
        raise ValueError(
            "Prediction/result match_id sets differ; "
            f"missing results={sorted(missing)}, unmatched results={sorted(extra)}."
        )
    settled: list[dict[str, Any]] = []
    for match_id, prediction in forecasts.items():
        result = actuals[match_id]
        actual = _outcome(result.get("home_goals"), result.get("away_goals"))
        scores = score_probabilities(prediction.get("probabilities"), actual)
        settled.append({
            "match_id": match_id,
            "date": prediction.get("kickoff") or prediction.get("date"),
            "actual_outcome": actual,
            "probabilities": prediction["probabilities"],
            "model_probabilities": prediction.get("model_probabilities"),
            "market_probabilities": prediction.get("market_probabilities"),
            "brier": scores["brier"],
            "log_loss": scores["log_loss"],
            "home_goals": result["home_goals"],
            "away_goals": result["away_goals"],
        })
    count = len(settled)
    if count == 0:
        raise ValueError("No predicted matches with final results were available to settle.")
    return {
        "matches": settled,
        "metrics": {
            "count": count,
            "mean_brier": math.fsum(item["brier"] for item in settled) / count,
            "mean_log_loss": math.fsum(item["log_loss"] for item in settled) / count,
        },
    }


def write_backtest(
    date: str | None,
    predictions: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    target_date = validate_date(date)
    report = settle_predictions(predictions, results)
    path = get_data_dir("reports", f"backtest_{target_date}.json")
    save_json(report, path)
    return report, str(path)
