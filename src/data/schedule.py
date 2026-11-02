"""JSON schedule adapter; no provider credentials or fabricated fixtures included."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from src.config import BASE_DIR, load_params, validate_params
from src.utils import load_json, parse_timestamp, save_json, utc_now, validate_date

_STATUSES = {"scheduled", "live", "finished", "postponed", "cancelled"}
_STATUS_ALIASES = {
    "not_started": "scheduled",
    "not started": "scheduled",
    "in_progress": "live",
    "in-progress": "live",
    "inplay": "live",
    "in_play": "live",
    "in-play": "live",
    "ft": "finished",
    "full_time": "finished",
    "full-time": "finished",
}


def _validate_matches(value: Any, require_kickoff: bool = False) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        value = value.get("matches")
    if not isinstance(value, list):
        raise ValueError("Schedule response must be a JSON array or contain a matches array.")
    if not value:
        raise ValueError("Schedule source returned zero matches; verify the date and provider response.")
    matches: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"Schedule match at index {index} must be a JSON object.")
        match = dict(item)
        for field in ("match_id", "home_team", "away_team", "status"):
            if not isinstance(match.get(field), str) or not match[field].strip():
                raise ValueError(f"Schedule match at index {index} requires {field}.")
            match[field] = match[field].strip()
        raw_status = match["status"].casefold()
        match["status"] = _STATUS_ALIASES.get(raw_status, raw_status)
        if match["status"] not in _STATUSES:
            raise ValueError(f"Unsupported match status: {match['status']}")
        if match["match_id"] in seen_ids:
            raise ValueError(f"Duplicate match_id in schedule: {match['match_id']}")
        seen_ids.add(match["match_id"])
        if require_kickoff and "kickoff" not in match:
            raise ValueError(f"Schedule match at index {index} requires kickoff.")
        if "kickoff" in match:
            parse_timestamp(match["kickoff"], "kickoff")
        if "prematch_xg" in match:
            if "home_xg" in match or "away_xg" in match:
                raise ValueError(
                    f"Schedule match at index {index} cannot mix prematch_xg with flat xG fields."
                )
            prematch_xg = match["prematch_xg"]
            if not isinstance(prematch_xg, dict):
                raise ValueError(f"Schedule match at index {index} prematch_xg must be an object.")
            prematch_xg = dict(prematch_xg)
            match["prematch_xg"] = prematch_xg
            has_home_xg = "home" in prematch_xg
            has_away_xg = "away" in prematch_xg
            if has_home_xg != has_away_xg:
                raise ValueError(
                    f"Schedule match at index {index} prematch_xg requires both home and away."
                )
            if has_home_xg:
                prematch_xg["home"] = _finite_nonnegative(
                    prematch_xg["home"], "prematch_xg.home"
                )
                prematch_xg["away"] = _finite_nonnegative(
                    prematch_xg["away"], "prematch_xg.away"
                )
        has_home_xg = "home_xg" in match
        has_away_xg = "away_xg" in match
        if has_home_xg != has_away_xg:
            raise ValueError(f"Schedule match at index {index} must provide both home_xg and away_xg.")
        for field in ("home_xg", "away_xg"):
            if field in match:
                match[field] = _finite_nonnegative(match[field], field)
        matches.append(match)
    return matches


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite non-negative number.")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be a finite non-negative number.")
    return number


def _request_url(base_url: str, parameter: str, value: str) -> str:
    if "{" + parameter + "}" in base_url:
        return base_url.replace("{" + parameter + "}", value)
    parts = urlsplit(base_url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query[parameter] = value
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _http_json(url: str, timeout: float) -> Any:
    headers = {"Accept": "application/json", "User-Agent": "football-predictor/0.9.41"}
    token = os.environ.get("FOOTBALL_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(f"Provider returned HTTP {response.status}.")
            return json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Schedule provider request failed: {error}") from error


def _snapshot(matches: list[dict[str, Any]], source: str, fetched_at: str, freshness: str,
              age_seconds: float, warning: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "matches": matches,
        "source": source,
        "fetched_at": fetched_at,
        "freshness": freshness,
        "age_seconds": round(max(0.0, age_seconds), 3),
    }
    if warning:
        result["warning"] = warning
    return result


def fetch_jingcai_schedule(
    date: str | None = None,
    params: dict[str, Any] | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    target_date = validate_date(date)
    config = load_params() if params is None else validate_params(params)
    cache_path = BASE_DIR / "data" / "cache" / f"schedule_{target_date}.json"
    now = utc_now()
    demo_mode = os.environ.get("FOOTBALL_PREDICTOR_DEMO") == "1"
    if demo_mode:
        fixture_path = BASE_DIR / "data" / "samples" / f"schedule_{target_date}.json"
        fixture = load_json(fixture_path)
        matches = _validate_matches(fixture)
        return _snapshot(matches, "demo_fixture", now.isoformat(), "demo", 0.0)

    cache: dict[str, Any] | None = None
    if use_cache and cache_path.exists():
        loaded = load_json(cache_path)
        if not isinstance(loaded, dict) or loaded.get("date") != target_date:
            raise ValueError(f"Invalid schedule cache: {cache_path}")
        cached_at = parse_timestamp(loaded.get("fetched_at"), "schedule cache fetched_at")
        cached_matches = _validate_matches(loaded.get("matches"), require_kickoff=True)
        age_seconds = (now - cached_at).total_seconds()
        if age_seconds < -60:
            raise ValueError(f"Schedule cache timestamp is in the future: {cache_path}")
        cache = {
            "fetched_at": cached_at.isoformat(),
            "matches": cached_matches,
            "age_seconds": max(0.0, age_seconds),
        }
        if age_seconds <= config["schedule_cache_ttl_seconds"]:
            return _snapshot(cached_matches, "cache", cached_at.isoformat(), "fresh_cache", age_seconds)

    provider_url = os.environ.get("FOOTBALL_SCHEDULE_URL")
    if not provider_url:
        if cache is not None:
            return _snapshot(
                cache["matches"], "cache", cache["fetched_at"], "stale",
                cache["age_seconds"], "No schedule provider configured; stale cache returned.",
            )
        raise RuntimeError(
            "No schedule provider configured. Set FOOTBALL_SCHEDULE_URL or explicitly enable "
            "FOOTBALL_PREDICTOR_DEMO=1 with a dated fixture file."
        )
    try:
        response = _http_json(
            _request_url(provider_url, "date", target_date),
            float(config["http_timeout_seconds"]),
        )
        matches = _validate_matches(response, require_kickoff=True)
    except (RuntimeError, ValueError) as error:
        if cache is not None:
            return _snapshot(
                cache["matches"], "cache", cache["fetched_at"], "stale",
                cache["age_seconds"], f"Provider refresh failed: {error}",
            )
        raise
    fetched_at = utc_now().isoformat()
    save_json({"date": target_date, "fetched_at": fetched_at, "matches": matches}, cache_path)
    return _snapshot(matches, "provider", fetched_at, "fresh", 0.0)
