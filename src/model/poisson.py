"""Independent Poisson score matrix with standard Dixon-Coles low-score correction."""

from __future__ import annotations

import math
from typing import Any


def _validate_lambda(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number.")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return value


def poisson_probabilities(lam: float, max_goals: int) -> list[float]:
    lam = _validate_lambda(lam, "lambda")
    if isinstance(max_goals, bool) or not isinstance(max_goals, int) or max_goals < 1:
        raise ValueError("max_goals must be a positive integer.")
    probabilities = [math.exp(-lam)]
    for goals in range(1, max_goals + 1):
        probabilities.append(probabilities[-1] * lam / goals)
    return probabilities


def score_matrix(
    lambda_home: float,
    lambda_away: float,
    rho: float = 0.0,
    max_goals: int = 10,
) -> dict[str, Any]:
    lambda_home = _validate_lambda(lambda_home, "lambda_home")
    lambda_away = _validate_lambda(lambda_away, "lambda_away")
    if isinstance(rho, bool) or not isinstance(rho, (int, float)) or not math.isfinite(float(rho)):
        raise ValueError("rho must be finite.")
    rho = float(rho)
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must be between -1 and 1.")
    home_probs = poisson_probabilities(lambda_home, max_goals)
    away_probs = poisson_probabilities(lambda_away, max_goals)
    matrix = [[home * away for away in away_probs] for home in home_probs]
    correction = {
        (0, 0): 1.0 - lambda_home * lambda_away * rho,
        (0, 1): 1.0 + lambda_home * rho,
        (1, 0): 1.0 + lambda_away * rho,
        (1, 1): 1.0 - rho,
    }
    for (home_goals, away_goals), tau in correction.items():
        if tau < 0 or not math.isfinite(tau):
            raise ValueError(
                f"Dixon-Coles correction is negative for {home_goals}-{away_goals}; "
                "adjust rho or expected goals."
            )
        if home_goals <= max_goals and away_goals <= max_goals:
            matrix[home_goals][away_goals] *= tau
    raw_total = math.fsum(math.fsum(row) for row in matrix)
    if not math.isfinite(raw_total) or raw_total <= 0:
        raise ValueError("Score matrix has no finite positive probability mass.")
    omitted_mass = max(0.0, min(1.0, 1.0 - math.fsum(home_probs) * math.fsum(away_probs)))
    normalized = [[probability / raw_total for probability in row] for row in matrix]
    return {
        "matrix": normalized,
        "truncated_mass": omitted_mass,
        "normalization_factor": raw_total,
    }


def outcome_probabilities(matrix: list[list[float]]) -> dict[str, float]:
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("Score matrix must be a non-empty square matrix.")
    home = math.fsum(value for i, row in enumerate(matrix) for j, value in enumerate(row) if i > j)
    draw = math.fsum(matrix[i][i] for i in range(len(matrix)))
    away = math.fsum(value for i, row in enumerate(matrix) for j, value in enumerate(row) if i < j)
    total = home + draw + away
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Outcome probabilities must have positive finite mass.")
    return {"home": home / total, "draw": draw / total, "away": away / total}
