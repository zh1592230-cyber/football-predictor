"""Descriptive metrics for settled multiclass 1X2 forecasts."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from src.backtest.settle import score_probabilities

OUTCOMES = ("home", "draw", "away")
MIN_DESCRIPTIVE_SAMPLE = 30


def _mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _calibration_bins(
    matches: list[dict[str, Any]],
    probability_field: str,
    n_bins: int,
) -> dict[str, list[dict[str, float | int]]]:
    result: dict[str, list[dict[str, float | int]]] = {}
    for outcome in OUTCOMES:
        buckets: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
        for match in matches:
            probabilities = match.get(probability_field)
            if not isinstance(probabilities, dict):
                raise ValueError(f"{probability_field} is required for calibration.")
            probability = probabilities[outcome]
            index = min(int(probability * n_bins), n_bins - 1)
            buckets[index].append((probability, int(match["actual_outcome"] == outcome)))
        result[outcome] = [
            {
                "bin_lower": index / n_bins,
                "bin_upper": (index + 1) / n_bins,
                "count": len(bucket),
                "mean_predicted": _mean([item[0] for item in bucket]) or 0.0,
                "observed_frequency": _mean([float(item[1]) for item in bucket]) or 0.0,
            }
            for index, bucket in enumerate(buckets)
            if bucket
        ]
    return result


def _evaluate(matches: list[dict[str, Any]], probability_field: str) -> dict[str, Any]:
    if not matches:
        return {"count": 0, "brier": None, "log_loss": None, "hit_rate": None}
    brier_scores: list[float] = []
    log_losses: list[float] = []
    hits: list[float] = []
    for match in matches:
        probabilities = match.get(probability_field)
        if not isinstance(probabilities, dict):
            raise ValueError(f"{probability_field} is required for evaluation.")
        try:
            scores = score_probabilities(probabilities, match["actual_outcome"])
        except ValueError as error:
            raise ValueError(f"{probability_field}: {error}") from error
        brier_scores.append(scores["brier"])
        log_losses.append(scores["log_loss"])
        hits.append(float(max(probabilities, key=probabilities.get) == match["actual_outcome"]))
    return {
        "count": len(matches),
        "brier": _mean(brier_scores),
        "log_loss": _mean(log_losses),
        "hit_rate": _mean(hits),
    }


def calculate_metrics(
    matches: list[dict[str, Any]],
    n_bins: int = 10,
) -> dict[str, Any]:
    if not isinstance(matches, list) or not matches:
        raise ValueError("At least one settled match is required for metrics.")
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")

    model_metrics = _evaluate(matches, "model_probabilities")
    final_metrics = _evaluate(matches, "probabilities")
    market_matches = [
        match for match in matches
        if match.get("market_probabilities") is not None
    ]
    market_metrics = _evaluate(market_matches, "market_probabilities")
    paired_model_metrics = _evaluate(market_matches, "model_probabilities")
    paired_delta = None
    if market_matches:
        paired_delta = {
            # Negative means the model scores lower (better) than the market baseline.
            "model_minus_market_brier": (
                paired_model_metrics["brier"] - market_metrics["brier"]
            ),
            "model_minus_market_log_loss": (
                paired_model_metrics["log_loss"] - market_metrics["log_loss"]
            ),
            "count": len(market_matches),
        }

    by_league: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for match in matches:
        by_league[str(match.get("league") or "unknown")].append(match)
    league_metrics = {
        league: _evaluate(group, "model_probabilities")
        for league, group in sorted(by_league.items())
    }

    warnings: list[str] = []
    if len(matches) < MIN_DESCRIPTIVE_SAMPLE:
        warnings.append(
            f"Only {len(matches)} matches: metrics are descriptive and not evidence of reliable "
            f"performance (recommended minimum for even a basic summary: {MIN_DESCRIPTIVE_SAMPLE})."
        )
    if market_matches and len(market_matches) < MIN_DESCRIPTIVE_SAMPLE:
        warnings.append(
            f"Only {len(market_matches)} matched market records: market comparison is descriptive only."
        )
    if not market_matches:
        warnings.append("No market probabilities; market baseline comparison is unavailable.")

    return {
        "sample_count": len(matches),
        "model": model_metrics,
        "final_forecast": final_metrics,
        "market_baseline": market_metrics,
        "model_vs_market_on_same_matches": paired_delta,
        "calibration": {
            "method": "one-vs-rest reliability bins, reported separately for each outcome",
            "n_bins": n_bins,
            "outcomes": _calibration_bins(matches, "model_probabilities", n_bins),
        },
        "league_breakdown": league_metrics,
        "warnings": warnings,
        "interpretation": "Descriptive historical scores only; no out-of-sample performance guarantee.",
    }
