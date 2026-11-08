#!/usr/bin/env bash
# 本地每日竞彩（与 GitHub Actions 对齐）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

DAY="${1:-$(TZ=Asia/Shanghai date +%F)}"
mkdir -p data/daily data/logs data/reports

echo "==== $(date -Is) run_today date=$DAY ===="
python scripts/run_today.py --date "$DAY" --no-cache "$@" 2>&1 | tee "data/logs/run_today_${DAY}.log"
echo "==== done ===="
