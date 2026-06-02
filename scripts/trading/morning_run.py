#!/usr/bin/env python3
"""Morning recommendation run — pre-market cron entrypoint.

Produces TODAY's actionable recommendations (entry / current / target / stop /
ROI / risk-reward / confidence) for both books, grounded in the validated
engines:
  - Equity: momentum z-score + 10-day volatility band (src/equity/momentum.py)
  - Futures: the engine's ±0.40% / -0.30% target/stop band (paper_trade.py)

Steps: update panel → generate picks → enrich → futures plan → write
results/recommendations.json → push to the hosted dashboard.

Cron (08:15 IST Mon-Fri, before NSE open):
    15 8 * * 1-5  cd /path/to/repo && .venv/bin/python scripts/trading/morning_run.py
"""
from __future__ import annotations

import glob
import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
PYTHON = str(ROOT / ".venv/bin/python") if (ROOT / ".venv/bin/python").exists() else sys.executable
REC_PATH = ROOT / "results/recommendations.json"
PANEL = ROOT / "cache/v8/daily_panel_nifty500_adj.parquet"

# Validated futures band (scripts/trading/paper_trade.py)
FUT_TARGET_PCT = 0.0040
FUT_STOP_PCT = 0.0030
LOT = 50
# Validated futures Phase-2 guard
VIX_CUT = 22.0
RET_CUT = -0.015
# Equity holding horizon (momentum.REBALANCE_DAYS)
EQ_HORIZON = 10


def run(label: str, cmd: list[str]) -> int:
    print(f"\n{'='*70}\n  {label}\n  {' '.join(cmd)}\n{'='*70}", flush=True)
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    print(f"  exit code: {rc}", flush=True)
    return rc


def latest_picks() -> dict | None:
    files = sorted(glob.glob(str(ROOT / "results/equity/picks/picks_*.json")))
    return json.loads(Path(files[-1]).read_text()) if files else None


def validated_winrate() -> float | None:
    try:
        fw = json.loads((ROOT / "results/router_v0/forward_walk_summary.json").read_text())
        return float(fw["phase3_no_guard"]["win_rate"])
    except Exception:
        return None


def enrich_equity(picks: list[dict]) -> list[dict]:
    """Add entry/current/target/stop/roi/rr/confidence to each pick.

    target/stop = volatility band over the 10-day hold (1.25σ stop; target
    1.25–3.0σ scaled by momentum strength). confidence = logistic(z-score)."""
    try:
        from equity import momentum as M
        close, _ = M.load_panel(PANEL)
        v63 = M.vol63(close)
        date = close.index[-1]
    except Exception as e:  # noqa: BLE001
        print(f"  equity enrich skipped: {e}", file=sys.stderr)
        return picks
    out = []
    for p in picks:
        sym, entry, score = p["symbol"], float(p["price"]), float(p.get("score", 0.0))
        try:
            current = float(close[sym].dropna().iloc[-1])
        except Exception:
            current = entry
        try:
            sigma_d = float(v63.loc[date, sym])
        except Exception:
            sigma_d = float("nan")
        if not (sigma_d == sigma_d) or sigma_d <= 0:  # NaN / non-positive guard
            sigma_d = 0.02
        sigma_h = sigma_d * math.sqrt(EQ_HORIZON)
        k_up = 1.25 + min(max(score, 0.0), 3.0) / 3.0 * 1.75   # 1.25σ … 3.0σ
        target = entry * (1 + k_up * sigma_h)
        stop = entry * (1 - 1.25 * sigma_h)
        roi = (target - entry) / entry
        rr = (target - entry) / (entry - stop) if entry > stop else None
        out.append({
            **p,
            "entry": round(entry, 2), "current": round(current, 2),
            "target": round(target, 2), "stop": round(stop, 2),
            "roi": round(roi, 4), "rr": round(rr, 2) if rr else None,
            "confidence": round(100 / (1 + math.exp(-score))),
            "horizon_days": EQ_HORIZON, "sigma_pct": round(sigma_d * 100, 2),
        })
    return out


def futures_plan() -> dict | None:
    try:
        vix = pd.read_csv(ROOT / "data/nifty_intraday/INDIA VIX_day.csv")
        vix_last = float(vix["close"].iloc[-1])
        as_of = str(pd.to_datetime(vix["date"].iloc[-1]).date())
        nif = pd.read_csv(ROOT / "data/nifty_intraday/NIFTY 50_minute.csv", usecols=["date", "close"])
        daily = nif.groupby(pd.to_datetime(nif["date"]).dt.date)["close"].last()
        entry = float(daily.iloc[-1])
        ret5d = float(daily.iloc[-1] / daily.iloc[-6] - 1) if len(daily) > 6 else None
    except Exception as e:  # noqa: BLE001
        print(f"  futures plan skipped: {e}", file=sys.stderr)
        return None
    tradeable = vix_last < VIX_CUT and (ret5d is None or ret5d > RET_CUT)
    wr = validated_winrate()
    target = entry * (1 + FUT_TARGET_PCT)
    stop = entry * (1 - FUT_STOP_PCT)
    if tradeable:
        note = (f"India VIX {vix_last:.1f} < {VIX_CUT:.0f} — the volatility guard permits "
                f"long-only intraday entries when the model's score fires today.")
    else:
        why = "elevated VIX" if vix_last >= VIX_CUT else "weak 5-day trend"
        note = f"{why} — Phase-2 guard blocks new entries; stay flat today."
    return {
        "stance": "long-only intraday", "as_of": as_of, "tradeable": tradeable, "note": note,
        "vix": round(vix_last, 2), "vix_cut": VIX_CUT,
        "ret_5d": round(ret5d, 4) if ret5d is not None else None, "ret_cut": RET_CUT,
        "entry": round(entry, 2), "current": round(entry, 2),
        "target": round(target, 2), "stop": round(stop, 2),
        "target_pct": FUT_TARGET_PCT, "stop_pct": FUT_STOP_PCT,
        "roi": FUT_TARGET_PCT, "rr": round(FUT_TARGET_PCT / FUT_STOP_PCT, 2),
        "size_hint": "1.0–1.5× lot (1.5× on top-percentile scores)", "lot": LOT,
        "win_rate": round(wr, 4) if wr is not None else None,
        "confidence": round((wr or 0.5) * 100) if tradeable else 0,
    }


def main() -> int:
    print(f"\n  morning_run  {pd.Timestamp.now(tz='Asia/Kolkata').isoformat(timespec='seconds')}")

    run("Step 1: Update adjusted EOD panel",
        [PYTHON, "scripts/data/equity_eod_panel.py", "--universe", "nifty500", "--update"])
    run("Step 2: Generate equity picks", [PYTHON, "scripts/trading/equity_picks.py"])

    picks = latest_picks()
    equity = None
    if picks:
        equity = {"state": picks.get("state"), "n_picks": picks.get("n_picks", 0),
                  "config": picks.get("config", {}),
                  "picks": enrich_equity(picks.get("picks", []))}

    rec = {
        "generated_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(timespec="seconds"),
        "as_of": picks.get("as_of") if picks else str(pd.Timestamp.now(tz="Asia/Kolkata").date()),
        "equity": equity,
        "futures": futures_plan(),
    }
    REC_PATH.parent.mkdir(parents=True, exist_ok=True)
    REC_PATH.write_text(json.dumps(rec, indent=2))
    print(f"\n  recommendations written → {REC_PATH}")

    run("Step 3: Push dashboard snapshot", [PYTHON, "scripts/trading/push_dashboard.py"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
