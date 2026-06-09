#!/usr/bin/env bash
# Local manual morning pipeline — NOT for cron.
#
#   1. Kite login (or verify token in .env)
#   2. scripts/trading/morning_run.py  (panel → picks → recommendations → dashboard push)
#
# Usage (from anywhere):
#   scripts/trading/local/run_morning.sh
#   SKIP_LOGIN=1 scripts/trading/local/run_morning.sh   # token already valid
#
# Log: logs/morning_local_YYYYMMDD_HHMMSS.log
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

log "=== Morning local run (repo: $ROOT) ==="
ensure_kite_token
run_with_log morning_local "$PY" scripts/trading/morning_run.py
