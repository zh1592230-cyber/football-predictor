from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.scanner import scan_model_weight
from src.config import load_params
from src.utils import load_json, save_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Chronologically validate probability-fusion weights.")
    parser.add_argument("--input", required=True, help="Labeled predictions JSON array.")
    parser.add_argument("--output", help="Optional report path; config is never modified.")
    args = parser.parse_args()
    try:
        report = scan_model_weight(load_json(args.input), load_params())
        output = Path(args.output) if args.output else ROOT / "data" / "reports" / f"weight_scan_{date.today().isoformat()}.json"
        save_json(report, output)
    except (OSError, ValueError) as error:
        print(f"scan_params failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "best_candidate": report["best_candidate"],
        "training_matches": report["training_matches"],
        "validation_matches": report["validation_matches"],
        "applied_to_config": report["applied_to_config"],
        "report_path": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
