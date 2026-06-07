# Shared helpers for local manual runs (source from run_*.sh; do not execute directly).
set -euo pipefail

_LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$_LOCAL_DIR/../../.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi

export TZ="${TZ:-Asia/Kolkata}"
mkdir -p "$ROOT/logs"

log() {
  printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"
}

# Use SKIP_LOGIN=1 to skip when KITE_ACCESS_TOKEN in .env is already valid.
ensure_kite_token() {
  if [[ "${SKIP_LOGIN:-0}" == "1" ]]; then
    log "SKIP_LOGIN=1 — not checking Kite token"
    return 0
  fi
  log "Checking Kite access token…"
  if "$PY" scripts/data/kite_login.py --check; then
    log "Kite token OK"
    return 0
  fi
  log "Token missing or expired — starting interactive login (browser may open)…"
  "$PY" scripts/data/kite_login.py
}

run_with_log() {
  local name="$1"
  shift
  local logfile="$ROOT/logs/${name}_$(date +%Y%m%d_%H%M%S).log"
  log "Logging to $logfile"
  set +e
  "$@" 2>&1 | tee "$logfile"
  local rc="${PIPESTATUS[0]}"
  set -e
  if [[ "$rc" -eq 0 ]]; then
    log "$name finished OK (exit 0)"
  else
    log "$name failed (exit $rc)"
  fi
  return "$rc"
}
