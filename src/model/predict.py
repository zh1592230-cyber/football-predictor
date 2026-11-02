"""Pre-match prediction path and daily job orchestration."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from typing import Any

from src.config import load_params, validate_params
from src.data.nowgoal import fetch_nowgoal_odds
from src.data.schedule import fetch_jingcai_schedule
from src.data.xg import get_expected_goals
from src.model.fusion import de_vig_1x2, fuse_probabilities
from src.model.poisson import outcome_probabilities, score_matrix
from src.utils import get_data_dir, save_json, validate_date


def predict_match(
    match: dict[str, Any],
    odds: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_params() if params is None else validate_params(params)
    if not isinstance(match, dict):
        raise ValueError("match must be a JSON object.")
    for field in ("match_id", "home_team", "away_team", "status"):
        if not isinstance(match.get(field), str) or not match[field].strip():
            raise ValueError(f"match requires a non-empty {field}.")
    status = match["status"].strip().lower()
    if status == "live":
        return {"match_id": match["match_id"], "status": "skipped_live",
                "reason": "No score-, minute-, or red-card-conditioned live model is implemented."}
    if status not in {"scheduled", "finished", "postponed", "cancelled"}:
        raise ValueError(f"Unsupported match status: {match['status']}")
    if status != "scheduled":
        return {"match_id": match["match_id"], "status": "skipped",
                "reason": f"Pre-match prediction is not applicable to status {status!r}."}

    lambda_home, lambda_away, xg_source = get_expected_goals(match, config)
    score = score_matrix(lambda_home, lambda_away, config["rho"], config["max_goals"])
    matrix = score["matrix"]
    model_probs = outcome_probabilities(matrix)
    odds_status: str | None = None
    if odds is not None and not isinstance(odds, dict):
        raise ValueError("odds must be a JSON object.")
    if odds is not None and odds.get("cache_status") == "stale":
        odds_status = "stale_not_fused"
        market_probs = None
    else:
        market_probs = de_vig_1x2(odds) if odds is not None else None
    if odds is not None and market_probs is not None:
        fused = fuse_probabilities(
            model_probs, market_probs, odds.get("updated_at"), config
        )
        if float(fused["market_freshness"]) <= 0.0:
            odds_status = "stale_not_fused"
            market_probs = None
            fused = fuse_probabilities(model_probs, params=config)
        else:
            odds_status = str(odds.get("cache_status", "provided"))
    else:
        fused = fuse_probabilities(model_probs, params=config)
    probabilities = fused["probabilities"]
    if not isinstance(probabilities, Mapping):
        raise RuntimeError("Probability fusion returned an invalid probabilities object.")
    top_scores = sorted(
        (
            {"score": f"{home}-{away}", "probability": matrix[home][away]}
            for home in range(len(matrix))
            for away in range(len(matrix[home]))
        ),
        key=lambda item: item["probability"],
        reverse=True,
    )[:3]
    over_2_5 = math.fsum(
        probability
        for home, row in enumerate(matrix)
        for away, probability in enumerate(row)
        if home + away >= 3
    )
    confidence = max(probabilities.values())
    recommendation = (
        max(probabilities, key=probabilities.get)
        if confidence >= float(config["rec_conf_min"])
        and (
            xg_source.startswith("match_xg_shrunk_to_unvalidated_baseline")
            or config["allow_baseline_recommend"]
        )
        else None
    )
    result = {
        "match_id": match["match_id"],
        "home_team": match["home_team"],
        "away_team": match["away_team"],
        "status": "predicted",
        "league": match.get("league"),
        "kickoff": match.get("kickoff"),
        "xg_source": xg_source,
        "expected_goals": {"home": lambda_home, "away": lambda_away, "source": xg_source},
        "model_probabilities": model_probs,
        "market_probabilities": market_probs,
        "probabilities": probabilities,
        "model_weight": fused["model_weight"],
        "market_freshness": fused["market_freshness"],
        "market_used": fused["market_used"],
        "over_2_5_probability": over_2_5,
        "top_scores": top_scores,
        "truncated_score_mass": score["truncated_mass"],
        "recommendation": recommendation,
        "recommendation_policy": (
            "confidence_threshold_unvalidated"
            if recommendation is not None
            else "disabled_for_unvalidated_baseline"
            if xg_source.startswith("unvalidated_configured_baseline")
            and not config["allow_baseline_recommend"]
            else "below_confidence_threshold"
        ),
        "top_outcome_probability": confidence,
        "limitations": [
            "赛前 Poisson/DC 模型；不支持比分、比赛分钟或红牌条件化。",
            "无 match xG 时使用未校准配置基线，不代表球队实时实力。",
            "矩阵只列 0..max_goals，输出在截断矩阵内归一化。",
        ],
    }
    if odds is not None:
        result["odds_source"] = odds.get("source", "configured_odds_provider")
        if "updated_at" in odds:
            result["odds_updated_at"] = odds["updated_at"]
        result["odds_cache_status"] = odds_status or str(odds.get("cache_status", "provided"))
        if odds_status == "stale_not_fused":
            result["odds_status"] = odds_status
    return result


def run_daily(
    date: str | None = None,
    only_recommend: bool = False,
    leagues: list[str] | None = None,
    params: dict[str, Any] | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    target_date = validate_date(date)
    config = load_params() if params is None else validate_params(params)
    snapshot = fetch_jingcai_schedule(target_date, config, use_cache=use_cache)
    selected_leagues = {league.strip().casefold() for league in leagues or [] if league.strip()}
    matches = [
        match for match in snapshot["matches"]
        if not selected_leagues or str(match.get("league", "")).casefold() in selected_leagues
    ]
    predictions: list[dict[str, Any]] = []
    odds_configured = bool(os.environ.get("FOOTBALL_ODDS_URL"))
    for match in matches:
        odds: dict[str, Any] | None = None
        odds_error: str | None = None
        if odds_configured and match.get("status", "").lower() == "scheduled":
            try:
                odds = fetch_nowgoal_odds(match, config)
            except (RuntimeError, ValueError) as error:
                odds_error = str(error)
        prediction = predict_match(match, odds, config)
        prediction["date"] = target_date
        prediction["schedule_source"] = snapshot["source"]
        prediction["schedule_freshness"] = snapshot["freshness"]
        prediction["schedule_age_seconds"] = snapshot["age_seconds"]
        if snapshot["freshness"] in {"stale", "demo"}:
            prediction["recommendation"] = None
        if snapshot.get("warning"):
            prediction["schedule_warning"] = snapshot["warning"]
        if odds_error:
            prediction["odds_error"] = odds_error
        elif not odds_configured and match.get("status", "").lower() == "scheduled":
            prediction["odds_status"] = "not_configured"
        predictions.append(prediction)
    output_path = get_data_dir("daily", f"predictions_{target_date}.json")
    save_json(predictions, output_path)
    displayed = [
        prediction for prediction in predictions
        if not only_recommend or prediction.get("recommendation") is not None
    ]
    return {
        "date": target_date,
        "schedule": {key: value for key, value in snapshot.items() if key != "matches"},
        "predictions": predictions,
        "displayed": displayed,
        "output_path": str(output_path),
    }
