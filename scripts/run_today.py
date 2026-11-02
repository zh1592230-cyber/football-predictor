from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_params
from src.data.schedule import fetch_jingcai_schedule
from src.model.predict import run_daily
from src.utils import validate_date


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch a schedule and produce pre-match predictions. No live schedule source is "
            "bundled; configure FOOTBALL_SCHEDULE_URL. Explicit demo fixtures are not live data."
        )
    )
    parser.add_argument("--date", help="Target calendar date (YYYY-MM-DD); defaults to local today.")
    parser.add_argument("--recommend", action="store_true", help="Display only matches meeting the configured confidence threshold.")
    parser.add_argument("--leagues", help="Comma-separated league codes/names to include.")
    parser.add_argument("--no-cache", action="store_true", help="Bypass the schedule cache and request the configured source.")
    parser.add_argument(
        "--test-schedule",
        action="store_true",
        help="Check schedule retrieval only; exits non-zero on errors, stale data, or zero matches.",
    )
    args = parser.parse_args()
    leagues = [item.strip() for item in args.leagues.split(",") if item.strip()] if args.leagues else None
    if args.leagues is not None and not leagues:
        parser.error("--leagues must contain at least one non-empty league name.")
    try:
        target_date = validate_date(args.date)
        params = load_params()
        if args.test_schedule:
            snapshot = fetch_jingcai_schedule(target_date, params, use_cache=not args.no_cache)
            matches = snapshot["matches"]
            status = "ok"
            exit_code = 0
            if snapshot["freshness"] == "demo":
                status, exit_code = "demo", 1
            elif snapshot["freshness"] == "stale":
                status, exit_code = "stale", 1
            elif not matches:
                status, exit_code = "empty", 1
            print(json.dumps({
                "status": status,
                "date": target_date,
                "source": snapshot["source"],
                "freshness": snapshot["freshness"],
                "fetched_at": snapshot["fetched_at"],
                "age_seconds": snapshot["age_seconds"],
                "match_count": len(matches),
                "matches": matches,
                "warning": snapshot.get("warning"),
            }, ensure_ascii=False, indent=2))
            return exit_code
        result = run_daily(
            target_date, args.recommend, leagues, params, use_cache=not args.no_cache
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"run_today failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "date": result["date"],
        "schedule": result["schedule"],
        "predictions": result["displayed"],
        "saved_full_prediction_count": len(result["predictions"]),
        "output_path": result["output_path"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
