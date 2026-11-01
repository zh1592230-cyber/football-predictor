from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_params
from src.model.predict import run_daily


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch a schedule and produce pre-match predictions.")
    parser.add_argument("--date", help="Target calendar date (YYYY-MM-DD); defaults to local today.")
    parser.add_argument("--recommend", action="store_true", help="Display only matches meeting the configured confidence threshold.")
    parser.add_argument("--leagues", help="Comma-separated league codes/names to include.")
    args = parser.parse_args()
    leagues = [item.strip() for item in args.leagues.split(",") if item.strip()] if args.leagues else None
    try:
        result = run_daily(args.date, args.recommend, leagues, load_params())
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
