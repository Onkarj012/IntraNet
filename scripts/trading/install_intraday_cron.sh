#!/usr/bin/env bash
# Install the intraday equity automation as cron jobs (macOS / Linux).
# Idempotent: removes any prior intraday_daily entries before adding.
#
# Usage:
#   bash scripts/trading/install_intraday_cron.sh
#   bash scripts/trading/install_intraday_cron.sh --uninstall
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO/.venv/bin/python"
LOG="$REPO/logs"
mkdir -p "$LOG"

MARKER="# intranet_optinet-intraday"

if [[ "${1:-}" == "--uninstall" ]]; then
  crontab -l 2>/dev/null | grep -v "$MARKER" | crontab - || true
  echo "Removed intraday cron jobs."
  exit 0
fi

if [[ ! -x "$PY" ]]; then
  echo "ERROR: venv python not found at $PY" >&2
  exit 1
fi

# Build the new crontab: keep existing non-intraday lines, append ours.
TMP="$(mktemp)"
crontab -l 2>/dev/null | grep -v "$MARKER" > "$TMP" || true

cat >> "$TMP" <<EOF
# === intranet_optinet intraday automation (IST; cron uses system TZ) === $MARKER
30 8  * * 1-5 cd $REPO && $PY scripts/trading/intraday_daily.py --mode preopen  >> $LOG/intraday_preopen.log 2>&1 $MARKER
22 9  * * 1-5 cd $REPO && $PY scripts/trading/intraday_daily.py --mode postopen >> $LOG/intraday_postopen.log 2>&1 $MARKER
35 15 * * 1-5 cd $REPO && $PY scripts/trading/intraday_daily.py --mode eod      >> $LOG/intraday_eod.log 2>&1 $MARKER
0  10 1-7 * 6 cd $REPO && $PY scripts/trading/train_intraday.py --universe nifty500 --min-date 2021-01-01 >> $LOG/intraday_retrain.log 2>&1 $MARKER
EOF

crontab "$TMP"
rm -f "$TMP"

echo "Installed intraday cron jobs:"
crontab -l | grep "$MARKER" | sed 's/  *#.*//'
echo
echo "NOTE: cron uses the system timezone. If your machine is not on IST,"
echo "      adjust the hours above (08:30/09:22/15:35 IST) accordingly."
echo "Logs: $LOG/intraday_*.log"
echo "Uninstall: bash scripts/trading/install_intraday_cron.sh --uninstall"
