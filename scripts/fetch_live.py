from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_params
from src.data.nowgoal import fetch_nowgoal_odds


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch configured 1X2 odds for a match (not live score data).")
    parser.add_argument("--match-id", required=True)
    args = parser.parse_args()
    try:
        odds = fetch_nowgoal_odds({"match_id": args.match_id}, load_params())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"fetch_live failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(odds, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
