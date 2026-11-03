from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.report import write_backtest_report
from src.utils import load_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Settle pre-kickoff predictions and write a descriptive JSON/Markdown report."
    )
    parser.add_argument("--predictions", required=True, help="Prediction JSON array with generated_at and kickoff.")
    parser.add_argument("--results", required=True, help="Final result JSON array keyed by match_id.")
    parser.add_argument("--date", help="Report date (YYYY-MM-DD); defaults to current UTC date.")
    parser.add_argument("--output-dir", help="Output parent directory; defaults to data/reports.")
    args = parser.parse_args()
    try:
        report, report_dir = write_backtest_report(
            load_json(args.predictions),
            load_json(args.results),
            report_date=args.date,
            output_dir=args.output_dir,
        )
    except (OSError, ValueError) as error:
        print(f"run_backtest_report failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "report_type": report["report_type"],
        "sample_count": report["metrics"]["sample_count"],
        "warnings": report["metrics"]["warnings"],
        "auto_param_update": report["auto_param_update"],
        "report_directory": str(report_dir),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
