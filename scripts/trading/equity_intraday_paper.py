#!/usr/bin/env python
"""
Equity intraday paper trading ledger.

Run at ~15:30 IST (after close) or next morning to record previous day's outcome.
Fetches actual OHLC from yfinance, simulates the production strategy, appends to ledger.

Usage:
    python scripts/trading/equity_intraday_paper.py             # reconcile yesterday
    python scripts/trading/equity_intraday_paper.py --date 2026-06-06
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

LEDGER = ROOT / "results/equity/intraday_paper_ledger.csv"
HALT   = ROOT / "results/equity/INTRADAY_HALTED"
K_STOP = 2.0; COST = 0.0018; STARTING_CAP = 100_000


def last_trading_date() -> pd.Timestamp:
    """Return the most recent past NSE trading day (skip weekends)."""
    d = pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None)
    while d.dayofweek >= 5:
        d -= pd.Timedelta(days=1)
    return d - pd.Timedelta(days=1) if d.dayofweek >= 5 else d - pd.Timedelta(days=1)


def load_ledger() -> pd.DataFrame:
    if LEDGER.exists():
        return pd.read_csv(LEDGER, parse_dates=["date"])
    cols = ["date","symbol","dir_p","regime","exposure","entry","stop",
            "high","low","close","pnl_pct","pnl_inr","outcome","capital_after"]
    return pd.DataFrame(columns=cols)


def fetch_ohlc(symbol: str, date: pd.Timestamp) -> dict | None:
    try:
        import yfinance as yf
        raw = yf.download(f"{symbol}.NS",
                          start=date.strftime("%Y-%m-%d"),
                          end=(date+pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                          interval="1d", auto_adjust=True, progress=False, threads=False)
        if raw is None or raw.empty: return None
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
        raw.columns = [c.lower() for c in raw.columns]
        r = raw.iloc[0]
        return {"open":float(r.open),"high":float(r.high),"low":float(r.low),"close":float(r.close)}
    except Exception:
        return None


def reconcile(date: pd.Timestamp) -> pd.DataFrame:
    """Reconcile recommendations for `date` against actual OHLC."""
    rec_path = ROOT / "results/recommendations.json"
    if not rec_path.exists():
        print("  No recommendations.json found — nothing to reconcile")
        return pd.DataFrame()
    rec = json.loads(rec_path.read_text())
    picks = (rec.get("equity") or {}).get("picks", [])
    if not picks:
        print("  No equity picks in recommendations.json")
        return pd.DataFrame()

    # Load ATR from panel for stop calculation
    try:
        from equity.momentum import load_ohlcv
        ohlcv = load_ohlcv(ROOT / "cache/v8/daily_panel_nifty500_adj.parquet")
        c_ = ohlcv["close"]; h_ = ohlcv.get("high",c_); lo_ = ohlcv.get("low",c_)
        prev = c_.shift(1)
        tr = pd.DataFrame(np.maximum(np.maximum((h_-lo_).values,(h_-prev).abs().values),
                                     (lo_-prev).abs().values),index=c_.index,columns=c_.columns)
        atr_panel = (tr.ewm(span=14,min_periods=7).mean()/c_.replace(0,np.nan)).shift(1).fillna(0.02)
    except Exception:
        atr_panel = None

    regime_path = ROOT / "models/v8_intraday/regime_history.parquet"
    exposure_path = ROOT / "models/v8_intraday/exposure_by_regime.json"
    regime_hist = pd.read_parquet(regime_path)["regime"] if regime_path.exists() else None
    exposure_map = json.loads(exposure_path.read_text()) if exposure_path.exists() else {}

    regime = "strong_trend_up"
    if regime_hist is not None and date in regime_hist.index:
        regime = regime_hist.loc[date]
    exposure = exposure_map.get(str(regime), 1.0)

    rows = []
    for pick in picks:
        sym = pick["symbol"]
        dir_p = pick.get("dir_p") or pick.get("p_up", 0.44)
        ohlc = fetch_ohlc(sym, date)
        if ohlc is None:
            print(f"  {sym}: no OHLC data for {date.date()}")
            continue
        entry = ohlc["open"]
        if entry <= 0: continue
        atr = 0.02
        if atr_panel is not None and sym in atr_panel.columns and date in atr_panel.index:
            atr = max(min(float(atr_panel.loc[date, sym]), 0.10), 0.005)
        stop_lvl = entry * (1 - K_STOP * atr)
        if ohlc["low"] <= stop_lvl:
            pnl_pct = -K_STOP * atr - COST; outcome = "STOP"
        else:
            pnl_pct = (ohlc["close"] - entry) / entry - COST
            outcome = "WIN" if pnl_pct > 0 else "LOSS"
        rows.append({"date": date.date(), "symbol": sym, "dir_p": round(dir_p, 3),
                     "regime": regime, "exposure": exposure,
                     "entry": round(entry, 2), "stop": round(stop_lvl, 2),
                     "high": round(ohlc["high"], 2), "low": round(ohlc["low"], 2),
                     "close": round(ohlc["close"], 2), "pnl_pct": round(pnl_pct, 5),
                     "pnl_inr": 0.0,  # filled below
                     "outcome": outcome, "capital_after": 0.0})
    return pd.DataFrame(rows)


def main():
    if HALT.exists():
        print(f"  INTRADAY HALTED ({HALT}). Remove to resume.")
        return 3

    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None)
    args = p.parse_args()

    date = pd.Timestamp(args.date) if args.date else last_trading_date()
    print(f"\n  equity_intraday_paper — reconciling {date.date()}")

    ledger = load_ledger()
    if not ledger.empty and str(date.date()) in ledger["date"].astype(str).values:
        print(f"  Already reconciled for {date.date()} — skipping")
        return 0

    new_rows = reconcile(date)
    if new_rows.empty:
        print("  No new rows to append"); return 0

    # Capital tracking
    cap = float(ledger["capital_after"].iloc[-1]) if not ledger.empty else STARTING_CAP
    n_picks = len(new_rows)
    exposure = float(new_rows["exposure"].iloc[0]) if n_picks else 1.0
    deployed = cap * exposure
    alloc = deployed / n_picks if n_picks else 0
    for idx in new_rows.index:
        pnl_inr = alloc * float(new_rows.at[idx, "pnl_pct"])
        new_rows.at[idx, "pnl_inr"] = round(pnl_inr, 2)
    day_pnl = new_rows["pnl_inr"].sum()
    cap += day_pnl
    new_rows["capital_after"] = round(cap, 2)

    # Append
    ledger = pd.concat([ledger, new_rows], ignore_index=True)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(LEDGER, index=False)

    wins = (new_rows["outcome"] == "WIN").sum()
    print(f"  Date: {date.date()} | {n_picks} picks | "
          f"Win/Loss/Stop: {wins}/{(new_rows.outcome=='LOSS').sum()}/{(new_rows.outcome=='STOP').sum()}")
    print(f"  Day P&L: ₹{day_pnl:+,.0f}  |  Capital: ₹{cap:,.0f}")
    print(f"  Ledger: {len(ledger)} total rows → {LEDGER}")

    # Halt check: drawdown > 15%
    eq = pd.Series([STARTING_CAP] + list(ledger.groupby("date")["pnl_inr"].sum().cumsum() + STARTING_CAP))
    dd = float((eq / eq.cummax() - 1).min())
    if dd <= -0.15:
        HALT.write_text(f"Hard halt: drawdown {dd:.1%}")
        print(f"  *** HARD HALT: drawdown {dd:.1%} *** → {HALT}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
