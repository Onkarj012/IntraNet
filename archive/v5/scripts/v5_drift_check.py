#!/usr/bin/env python
"""V5 v1 drift check — runs at 18:30 IST.

Compares today's feature distributions and decision metrics against the
backtest baseline. Writes a daily report and triggers a `paused_<date>.flag`
if any rolling-20d threshold is breached.

Thresholds (from spec §5.3):
- Rolling 5-day PnL < -₹40,000  → pause
- Rolling 20-day Sharpe < 0.5   → pause
- Rolling 20-day win rate < 55% → pause
- Feature drift > 2 std vs training → pause
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd

from optinet.v5_runtime import ledger as ld
from optinet.v5_runtime.runtime_config import FLAGS_DIR, ensure_dirs

log = logging.getLogger("v5_drift")
logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(message)s")

ROLLING_5D_PNL_FLOOR = -40_000.0
ROLLING_20D_SHARPE_FLOOR = 0.5
ROLLING_20D_WIN_FLOOR = 0.55
DRIFT_STD_LIMIT = 2.0
DRIFT_REPORT_DIR = REPO_ROOT / "results" / "optinet_v5" / "drift"

# Training-time means/stds for feature drift (frozen — pulled once from cache)
# At first run, we compute these from the 2020-2023 cache and store as JSON.
TRAINING_STATS_PATH = REPO_ROOT / "models" / "optinet_v5" / "feature_stats.json"


def _ensure_training_stats() -> dict:
    """Compute mean/std for gate features from the training cache, once."""
    if TRAINING_STATS_PATH.exists():
        return json.loads(TRAINING_STATS_PATH.read_text())

    log.info("computing training-time feature stats (first run)…")
    chain_dir = REPO_ROOT / "cache" / "optinet_v4" / "chain_features" / "index=NIFTY"
    pieces = []
    for year in (2020, 2021, 2022, 2023):
        p = chain_dir / f"year={year}" / "data.parquet"
        if p.exists():
            pieces.append(pd.read_parquet(p))
    if not pieces:
        return {}
    df = pd.concat(pieces, ignore_index=True)
    feature_cols = [
        "atm_iv", "skew_slope", "pcr_oi", "pcr_vol",
        "max_oi_total_dist_pct", "iv_rv_spread", "realized_vol_30m",
    ]
    stats = {c: {"mean": float(df[c].mean()), "std": float(df[c].std())}
             for c in feature_cols}
    TRAINING_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_STATS_PATH.write_text(json.dumps(stats, indent=2))
    log.info(f"wrote {TRAINING_STATS_PATH}")
    return stats


def feature_drift(*, today: date) -> dict:
    """Compare today's recorded feature_snapshots to training stats."""
    stats = _ensure_training_stats()
    if not stats:
        return {}
    df = ld.trades_on_date(today)
    if df.empty or "feature_snapshot" not in df.columns:
        return {}
    rows = []
    for s in df["feature_snapshot"].dropna():
        try:
            rows.append(json.loads(s))
        except Exception:
            pass
    if not rows:
        return {}
    snap_df = pd.DataFrame(rows)
    drifts = {}
    for c, m_s in stats.items():
        if c not in snap_df.columns:
            continue
        cur_mean = float(snap_df[c].mean())
        z = abs(cur_mean - m_s["mean"]) / max(m_s["std"], 1e-6)
        drifts[c] = {"today_mean": cur_mean,
                      "train_mean": m_s["mean"], "z": float(z)}
    return drifts


def rolling_pnl_metrics(*, today: date) -> dict:
    df = ld.load_ledger()
    if df.empty:
        return {}
    df = df.copy()
    df["entry_date"] = pd.to_datetime(df["entry_ts"]).dt.date
    df = df[df["status"].isin(["RECONCILED"])]
    if df.empty:
        return {}
    daily = (df.groupby("entry_date")["realized_pnl_inr"]
              .agg(total="sum", n="count",
                    win=lambda s: float((s > 0).mean()))).sort_index()

    def window(n_days: int) -> dict:
        cutoff = today - timedelta(days=n_days)
        sub = daily[daily.index >= cutoff]
        if sub.empty:
            return {}
        d = sub["total"]
        sharpe = (float(d.mean() / d.std() * np.sqrt(252))
                  if len(d) > 1 and d.std() > 0 else None)
        return {
            "n_days": int(len(sub)),
            "trades": int(sub["n"].sum()),
            "total_pnl": float(d.sum()),
            "win_rate": float(sub["win"].mean()),
            "sharpe": sharpe,
        }
    return {"5d": window(5), "20d": window(20)}


def main() -> int:
    ensure_dirs()
    today = date.today()
    DRIFT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"drift check for {today}")
    rolls = rolling_pnl_metrics(today=today)
    drifts = feature_drift(today=today)

    pause = []
    r5 = rolls.get("5d", {})
    r20 = rolls.get("20d", {})

    if r5 and r5.get("n_days", 0) >= 4:
        if (r5.get("total_pnl") or 0) < ROLLING_5D_PNL_FLOOR:
            pause.append(f"5d_pnl={r5['total_pnl']:.0f} < {ROLLING_5D_PNL_FLOOR}")
    if r20 and r20.get("n_days", 0) >= 15:
        sharpe = r20.get("sharpe") or 0
        win = r20.get("win_rate") or 0
        if sharpe < ROLLING_20D_SHARPE_FLOOR:
            pause.append(f"20d_sharpe={sharpe:.2f} < {ROLLING_20D_SHARPE_FLOOR}")
        if win < ROLLING_20D_WIN_FLOOR:
            pause.append(f"20d_win={win:.1%} < {ROLLING_20D_WIN_FLOOR:.0%}")
    breached_drift = [c for c, d in drifts.items() if d["z"] > DRIFT_STD_LIMIT]
    if breached_drift:
        pause.append(f"feature_drift > {DRIFT_STD_LIMIT}σ on {breached_drift}")

    out = {"today": str(today), "rolls": rolls, "drifts": drifts,
            "pause_reasons": pause}
    DRIFT_REPORT_DIR.joinpath(f"drift_{today}.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))

    if pause:
        flag = FLAGS_DIR / f"paused_{today}.flag"
        flag.write_text("\n".join(pause))
        log.warning(f"PAUSE TRIGGERED: {pause}; wrote {flag}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
