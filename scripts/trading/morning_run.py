#!/usr/bin/env python3
"""Morning recommendation run — pre-market cron entrypoint (yfinance only).

Produces TODAY's actionable intraday recommendations:
  - Equity: V8 5-specialist ensemble (momentum + reversal + breakout +
            sentiment + macro), calibrated p_up, barrier-aligned levels
            (+1.5% target / -1.0% stop), expected value after costs.
  - Futures: validated ±0.40%/−0.30% band (unchanged).

No Kite login required. All data from yfinance.

Steps:
  1. Update OHLCV panel (yfinance)
  2. Run equity picks (momentum screener — candidacy gate)
  2.5. [--post-open only] Fetch real 5-min ORB bars, re-score ensemble
  3. Score candidates with intraday ensemble
  4. Build futures plan
  5. Write results/recommendations.json → push dashboard

Cron (IST Mon-Fri):
    Pre-open (08:30) — candidacy + pre-open features:
        30 8 * * 1-5  cd /repo && .venv/bin/python scripts/trading/morning_run.py
    Post-open (09:22) — re-score with real first 5-min bar:
        22 9 * * 1-5  cd /repo && .venv/bin/python scripts/trading/morning_run.py --post-open
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

PYTHON = str(ROOT / ".venv/bin/python") if (ROOT / ".venv/bin/python").exists() else sys.executable
REC_PATH = ROOT / "results/recommendations.json"
PANEL = ROOT / "cache/v8/daily_panel_nifty500_adj.parquet"
MODEL_DIR = ROOT / "models/v8_intraday"

FUT_TARGET_PCT = 0.0040
FUT_STOP_PCT = 0.0030
VIX_CUT = 22.0
RET_CUT = -0.015
LOT = 50


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


def enrich_equity(picks_json: dict, *, post_open: bool = False) -> list[dict]:
    """
    Score momentum-screened picks with the intraday ensemble.

    Returns enriched pick list with: p_up, confidence, entry, target, stop,
    roi, rr, expected_value, regime, drivers (per-specialist probs).
    Falls back to ATR-based levels if ensemble models not trained yet.
    """
    raw_picks = picks_json.get("picks", [])
    if not raw_picks:
        return []

    symbols = [p["symbol"] for p in raw_picks]

    # --- Path 1: trained ensemble ---
    if MODEL_DIR.exists() and any(MODEL_DIR.glob("*_signal.pkl")):
        try:
            from equity.intraday_model import score_universe
            from equity.google_news import fetch_overnight_news
            from equity.universe import get_symbol_metadata

            date = pd.Timestamp(picks_json.get("as_of", str(pd.Timestamp.now().date())))

            # Fetch overnight Google News for all candidates
            print("  Fetching overnight Google News...")
            _meta_csv = str(ROOT / "data/sentiment/ind_nifty500list.csv")
            news_df = fetch_overnight_news(symbols, hours_back=18,
                                           universe_meta_csv=_meta_csv)
            print(f"  News fetched: {len(news_df)} articles for {news_df['symbol'].nunique() if not news_df.empty else 0} symbols")

            picks = score_universe(date, symbols, model_dir=MODEL_DIR,
                                   live_news=False, n_picks=20,
                                   news_df=news_df, post_open=post_open)
            if picks:
                print(f"  Ensemble scored {len(symbols)} symbols → {len(picks)} picks")
                return picks
            print("  Ensemble returned 0 picks above threshold; falling back to ATR levels")
        except Exception as e:
            print(f"  Ensemble scoring failed ({e}); falling back to ATR levels", file=sys.stderr)

    # --- Path 2: ATR-based fallback (pre-training or ensemble error) ---
    print("  Using ATR-based intraday levels (ensemble not trained yet)")
    return _atr_enrich(raw_picks)


def _atr_enrich(raw_picks: list[dict]) -> list[dict]:
    """
    Compute realistic intraday levels using ATR(14) × 0.5.
    Targets ~0.5–1.5% per day — not the old √10 multi-week bands.
    """
    import math
    try:
        from equity.momentum import load_ohlcv
        ohlcv = load_ohlcv(PANEL)
        close = ohlcv["close"]
        high = ohlcv.get("high")
        low = ohlcv.get("low")
    except Exception:
        close = high = low = None

    out = []
    for p in raw_picks:
        sym = p["symbol"]
        entry = float(p["price"])
        score = float(p.get("score", 0.0))

        # ATR-based range unit
        atr_pct = 0.012  # fallback: 1.2% typical daily range
        if close is not None and sym in close.columns and high is not None and low is not None:
            try:
                c = close[sym].dropna().iloc[-20:]
                h = high[sym].reindex(c.index)
                lo = low[sym].reindex(c.index)
                prev_c = c.shift(1)
                tr = pd.concat([h - lo, (h - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
                atr_pct = float(tr.mean()) / entry
            except Exception:
                pass

        k_t = 0.5 + min(max(score, 0), 3.0) / 3.0 * 0.5   # 0.5–1.0 × ATR
        target = entry * (1 + k_t * atr_pct)
        stop = entry * (1 - 0.5 * atr_pct)
        roi = (target - entry) / entry
        rr = (target - entry) / max(entry - stop, 1e-6)
        # Logistic confidence bounded at 85% max (not the unbounded "100%" bug)
        conf = min(int(round(85 / (1 + math.exp(-score + 1)))), 85)

        out.append({
            **p,
            "direction": "LONG",
            "entry": round(entry, 2),
            "current": round(entry, 2),
            "target": round(target, 2),
            "stop": round(stop, 2),
            "roi": round(roi, 4),
            "rr": round(rr, 2),
            "confidence": conf,
            "horizon": "intraday",
            "horizon_days": 1,
            "sigma_pct": round(atr_pct * 100, 2),
            "expected_value": round(0.5 * roi - 0.5 * abs((stop - entry) / entry) - 0.0018, 4),
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
        ret5d = float(daily.iloc[-1] / daily.iloc[-6] - 1) if len(daily) >= 6 else None
    except Exception as e:
        print(f"  futures plan skipped: {e}", file=sys.stderr)
        return None
    tradeable = vix_last < VIX_CUT and (ret5d is None or ret5d > RET_CUT)
    wr = validated_winrate()
    target = entry * (1 + FUT_TARGET_PCT)
    stop = entry * (1 - FUT_STOP_PCT)
    note = (f"India VIX {vix_last:.1f} < {VIX_CUT:.0f} — guard permits long-only intraday entries."
            if tradeable else
            f"{'elevated VIX' if vix_last >= VIX_CUT else 'weak 5d trend'} — Phase-2 guard blocks entries.")
    return {
        "stance": "long-only intraday", "as_of": as_of, "tradeable": tradeable, "note": note,
        "vix": round(vix_last, 2), "vix_cut": VIX_CUT,
        "ret_5d": round(ret5d, 4) if ret5d is not None else None, "ret_cut": RET_CUT,
        "entry": round(entry, 2), "current": round(entry, 2),
        "target": round(target, 2), "stop": round(stop, 2),
        "target_pct": FUT_TARGET_PCT, "stop_pct": FUT_STOP_PCT,
        "roi": FUT_TARGET_PCT, "rr": round(FUT_TARGET_PCT / FUT_STOP_PCT, 2),
        "size_hint": "1.0–1.5× lot", "lot": LOT,
        "win_rate": round(wr, 4) if wr is not None else None,
        "confidence": round((wr or 0.5) * 100) if tradeable else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--post-open", action="store_true",
                        help="Re-score with real 5-min ORB bars (run at 09:22 IST)")
    args = parser.parse_args()
    post_open = args.post_open

    now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
    print(f"\n  morning_run  {now_ist.isoformat(timespec='seconds')}  post_open={post_open}")

    # Guard: --post-open requires it to be >= 09:20 IST
    if post_open:
        open_time = now_ist.normalize().replace(hour=9, minute=20)
        if now_ist < open_time:
            print("  WARNING: --post-open requested but it is before 09:20 IST — skipping ORB enrichment")
            post_open = False

    # Step 1: Update OHLCV panel (yfinance, no Kite)
    rc = run("Step 1: Update OHLCV panel",
             [PYTHON, "scripts/data/equity_eod_panel.py", "--universe", "nifty500", "--update"])
    if rc != 0:
        print("  WARNING: panel update had errors — continuing with existing data")

    # Step 2: Generate momentum-screened candidates
    rc = run("Step 2: Generate equity picks (candidacy gate)",
             [PYTHON, "scripts/trading/equity_picks.py"])
    if rc != 0:
        return rc

    picks = latest_picks()
    equity = None
    if picks:
        # Step 2.5 (post-open only): ORB enrichment is wired inside enrich_equity
        # via score_universe(post_open=True) → enrich_with_opening_bars()
        if post_open:
            print("\n  Step 2.5: Fetching real 5-min ORB bars (post-09:20 run)")
        enriched = enrich_equity(picks, post_open=post_open)
        equity = {
            "state": picks.get("state"),
            "n_picks": len(enriched),
            "config": picks.get("config", {}),
            "picks": enriched,
            "model": ("v8_intraday_ensemble+orb" if post_open else "v8_intraday_ensemble")
                     if MODEL_DIR.exists() and any(MODEL_DIR.glob("*_signal.pkl"))
                     else "atr_fallback",
        }

    rec = {
        "generated_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(timespec="seconds"),
        "as_of": picks.get("as_of") if picks else str(pd.Timestamp.now(tz="Asia/Kolkata").date()),
        "equity": equity,
        "futures": futures_plan(),
    }
    REC_PATH.parent.mkdir(parents=True, exist_ok=True)
    REC_PATH.write_text(json.dumps(rec, indent=2))
    print(f"\n  recommendations written → {REC_PATH}")

    run("Step 3: Push dashboard snapshot",
        [PYTHON, "scripts/trading/push_dashboard.py"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
