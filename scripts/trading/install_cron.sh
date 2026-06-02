#!/usr/bin/env bash
# Idempotent installer for the two daily cron jobs:
#   - morning_run.py  (recommendations)   default 08:15 IST Mon-Fri
#   - daily_run.py    (paper trading EOD)  default 18:00 IST Mon-Fri
#
#   scripts/trading/install_cron.sh            # install both
#   scripts/trading/install_cron.sh --remove   # uninstall both
#
# Backs up the current crontab and records the schedule in results/cron_status.json.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY="$(command -v python3)"

MORNING_SCHED="15 8 * * 1-5"
EVENING_SCHED="0 18 * * 1-5"

mkdir -p "$ROOT/logs" "$ROOT/results"
ts="$(date +%Y%m%d_%H%M%S)"
crontab -l 2>/dev/null > "$ROOT/logs/crontab.backup.$ts" || true

# Drop any existing entries for our two scripts (and prior CRON_TZ line)
existing="$(crontab -l 2>/dev/null | grep -vE 'morning_run\.py|daily_run\.py|^CRON_TZ=' || true)"

if [ "${1:-}" = "--remove" ]; then
  printf '%s\n' "$existing" | grep -v '^$' | crontab - || crontab -r 2>/dev/null || true
  rm -f "$ROOT/results/cron_status.json"
  echo "Removed cron entries. Backup: logs/crontab.backup.$ts"
  exit 0
fi

human() { # "M H * * D" -> "HH:MM IST · days"
  read -r m h _ _ d <<<"$1"
  local days="$d"; [ "$d" = "1-5" ] && days="Mon-Fri"
  printf '%02d:%02d IST · %s' "$h" "$m" "$days"
}

L1="$MORNING_SCHED  cd $ROOT && $PY scripts/trading/morning_run.py >> $ROOT/logs/morning_run.log 2>&1"
L2="$EVENING_SCHED  cd $ROOT && $PY scripts/trading/daily_run.py >> $ROOT/logs/daily_run.log 2>&1"
printf '%s\n%s\n%s\n%s\n' "CRON_TZ=Asia/Kolkata" "$existing" "$L1" "$L2" | grep -v '^$' | crontab -

cat > "$ROOT/results/cron_status.json" <<JSON
{
  "timezone": "Asia/Kolkata",
  "jobs": [
    { "name": "Morning · recommendations", "schedule": "$MORNING_SCHED", "human": "$(human "$MORNING_SCHED")", "entrypoint": "scripts/trading/morning_run.py", "log": "logs/morning_run.log" },
    { "name": "Evening · paper trading", "schedule": "$EVENING_SCHED", "human": "$(human "$EVENING_SCHED")", "entrypoint": "scripts/trading/daily_run.py", "log": "logs/daily_run.log" }
  ],
  "installed_at": "$(date +%Y-%m-%dT%H:%M:%S%z)"
}
JSON

echo "Installed cron jobs:"
echo "  morning  $(human "$MORNING_SCHED")  → morning_run.py (recommendations)"
echo "  evening  $(human "$EVENING_SCHED")  → daily_run.py (paper trading)"
echo "Backup:   logs/crontab.backup.$ts"
echo
echo "NOTE: refresh the Kite token each trading morning before the runs:"
echo "  $PY scripts/data/kite_login.py"
