from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.settle import write_backtest
from src.utils import load_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Settle predictions against final scores and calculate proper scores.")
    parser.add_argument("--predictions", required=True, help="Daily prediction JSON array.")
    parser.add_argument("--results", required=True, help='JSON array of {"match_id","home_goals","away_goals"}.')
    parser.add_argument("--date", help="Report date (YYYY-MM-DD); defaults to local today.")
    args = parser.parse_args()
    try:
        report, report_path = write_backtest(
            args.date, load_json(args.predictions), load_json(args.results)
        )
    except (OSError, ValueError) as error:
        print(f"run_backtest failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"metrics": report["metrics"], "report_path": report_path}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
