#!/usr/bin/env bash
# Local manual evening (EOD) pipeline — NOT for cron.
#
#   1. Kite login (or verify token in .env)
#   2. scripts/trading/daily_run.py  (EOD cache → futures paper → equity ops → dashboard push)
#
# Usage (from anywhere):
#   scripts/trading/local/run_evening.sh
#   SKIP_LOGIN=1 scripts/trading/local/run_evening.sh   # token already valid
#
# Log: logs/evening_local_YYYYMMDD_HHMMSS.log
# Exit codes match daily_run.py (0 ok · 2 soft halt · 3 hard halt).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

log "=== Evening local run (repo: $ROOT) ==="
ensure_kite_token
run_with_log evening_local "$PY" scripts/trading/daily_run.py
