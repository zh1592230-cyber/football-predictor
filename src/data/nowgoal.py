"""Configurable 1X2 odds adapter; it is not an official NowGoal integration."""

from __future__ import annotations

import hashlib
import math
import os
from typing import Any

from src.config import BASE_DIR, load_params
from src.data.schedule import _http_json, _request_url
from src.utils import load_json, parse_timestamp, save_json, utc_now


def _validate_odds(data: Any, match_id: str) -> dict[str, Any]:
    if isinstance(data, dict) and isinstance(data.get("odds"), dict):
        data = data["odds"]
    if not isinstance(data, dict):
        raise ValueError("Odds response must be a JSON object.")
    result: dict[str, Any] = {"match_id": match_id}
    for key in ("home", "draw", "away"):
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} odds must be a finite decimal number greater than 1.")
        value = float(value)
        if not math.isfinite(value) or value <= 1:
            raise ValueError(f"{key} odds must be a finite decimal number greater than 1.")
        result[key] = value
    result["updated_at"] = parse_timestamp(data.get("updated_at"), "odds updated_at").isoformat()
    result["source"] = str(data.get("source") or "configured_odds_provider")
    if "asian_handicap" in data:
        handicap = data["asian_handicap"]
        if isinstance(handicap, bool) or not isinstance(handicap, (int, float, str)):
            raise ValueError("asian_handicap must be a number or a non-empty line string.")
        if isinstance(handicap, str):
            if not handicap.strip():
                raise ValueError("asian_handicap cannot be empty.")
            result["asian_handicap"] = handicap.strip()
        else:
            if not math.isfinite(float(handicap)):
                raise ValueError("asian_handicap must be finite.")
            result["asian_handicap"] = float(handicap)
    if "over_under" in data:
        market = data["over_under"]
        if not isinstance(market, dict):
            raise ValueError("over_under must be a JSON object.")
        line = market.get("line")
        if isinstance(line, bool) or not isinstance(line, (int, float)) or not math.isfinite(float(line)):
            raise ValueError("over_under.line must be finite.")
        validated_market: dict[str, float] = {"line": float(line)}
        for key in ("over", "under"):
            price = market.get(key)
            if isinstance(price, bool) or not isinstance(price, (int, float)):
                raise ValueError(f"over_under.{key} odds must be greater than 1.")
            price = float(price)
            if not math.isfinite(price) or price <= 1:
                raise ValueError(f"over_under.{key} odds must be greater than 1.")
            validated_market[key] = price
        result["over_under"] = validated_market
    return result


def fetch_nowgoal_odds(match: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
    match_id = match.get("match_id")
    if not isinstance(match_id, str) or not match_id.strip():
        raise ValueError("match requires a non-empty match_id.")
    match_id = match_id.strip()
    config = params or load_params()
    cache_key = hashlib.sha256(match_id.encode("utf-8")).hexdigest()
    cache_path = BASE_DIR / "data" / "cache" / f"odds_{cache_key}.json"
    now = utc_now()
    cache: dict[str, Any] | None = None
    if cache_path.exists():
        loaded = load_json(cache_path)
        if not isinstance(loaded, dict) or loaded.get("match_id") != match_id:
            raise ValueError(f"Invalid odds cache: {cache_path}")
        cached_odds = _validate_odds(loaded.get("odds"), match_id)
        fetched_at = parse_timestamp(loaded.get("fetched_at"), "odds cache fetched_at")
        age_seconds = max(0.0, (now - fetched_at).total_seconds())
        if (fetched_at - now).total_seconds() > 60:
            raise ValueError(f"Odds cache timestamp is in the future: {cache_path}")
        cache = {"odds": cached_odds, "fetched_at": fetched_at.isoformat(), "age_seconds": age_seconds}
        if age_seconds <= config["odds_ttl_seconds"]:
            return {**cached_odds, "cache_status": "fresh_cache", "fetched_at": fetched_at.isoformat(),
                    "age_seconds": round(age_seconds, 3)}

    provider_url = os.environ.get("FOOTBALL_ODDS_URL")
    if not provider_url:
        if cache is not None:
            return {**cache["odds"], "cache_status": "stale",
                    "fetched_at": cache["fetched_at"], "age_seconds": round(cache["age_seconds"], 3),
                    "warning": "No odds provider configured; stale cache returned."}
        raise RuntimeError("No odds provider configured. Set FOOTBALL_ODDS_URL to a permitted JSON API.")
    try:
        response = _http_json(
            _request_url(provider_url, "match_id", match_id),
            float(config["http_timeout_seconds"]),
        )
        odds = _validate_odds(response, match_id)
    except (RuntimeError, ValueError) as error:
        if cache is not None:
            return {**cache["odds"], "cache_status": "stale",
                    "fetched_at": cache["fetched_at"], "age_seconds": round(cache["age_seconds"], 3),
                    "warning": f"Provider refresh failed: {error}"}
        raise
    fetched_at = now.isoformat()
    save_json({"match_id": match_id, "fetched_at": fetched_at, "odds": odds}, cache_path)
    return {**odds, "cache_status": "fresh", "fetched_at": fetched_at, "age_seconds": 0.0}
