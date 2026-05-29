#!/usr/bin/env python3
"""Sprint #1 — Futures-first directional engine on NIFTY.

Honest test of whether intraday NIFTY futures direction has a learnable edge
in this codebase's feature set.

Pipeline:
 1. Generate barrier labels per minute using ACTUAL NIFTY futures minute bars:
    - LONG_LABEL  = 1 if fut hits +TARGET_PCT before -STOP_PCT within HORIZON_MIN
    - SHORT_LABEL = 1 if fut hits -TARGET_PCT before +STOP_PCT within HORIZON_MIN
    - else 0
    Default: TARGET=+0.40%, STOP=-0.30%, HORIZON=60 min
 2. Train two LightGBM binary classifiers (one per side) using V4-A chain +
    V5-B futures features + time-of-day. No target leakage by construction.
 3. Backtest realistically on 2024:
    - At each minute, score both classifiers
    - If long_score > thr and short_score < (1-thr) → take LONG futures
    - If short_score > thr and long_score < (1-thr) → take SHORT futures
    - Else NO_TRADE
    - Apply daily caps, daily loss halt, 14:55 cutoff
    - Simulate trade with target / stop / time-stop on actual minute bars
 4. Report headline metrics + slices (entry-time, weekday, month, side).

Train 2020-2022, validate 2023, test on 2024 blind window.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optinet.v5_futures import load_fut_day, discover_fut_days
from optinet.v4_chain import load_spot

# ---------------------------------------------------------------------------
# Config — frozen for the sprint
# ---------------------------------------------------------------------------

CHAIN_DIR = PROJECT_ROOT / "cache/optinet_v4/chain_features"
FUT_DIR   = PROJECT_ROOT / "cache/optinet_v5/futures_features"
DATA_ROOT = PROJECT_ROOT / "data/option_data"
OUT_DIR   = PROJECT_ROOT / "models/router_v0/futures"
RESULTS   = PROJECT_ROOT / "results/router_v0"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# Barrier label config
TARGET_PCT = 0.0040          # +0.40% reach for LONG
STOP_PCT   = 0.0030          # -0.30% before reaching target invalidates LONG
HORIZON_MIN = 60             # within 60 minutes
ENTRY_MIN_MOD = 30           # earliest minute index after open (minute 30 = 09:45)
HARD_CUTOFF = dtime(14, 55)  # no entries after this

# Trading config
LOT_SIZE_NIFTY = 50
COSTS_PER_LOT_INR = 80.0     # 2x brokerage + STT + GST round-trip approximation
SLIPPAGE_PER_LOT_INR = 25.0  # ~₹0.50/share NIFTY fut bid-ask
PER_TRADE_STOP_LOSS_INR = -3000.0
DAILY_LOSS_HALT_INR = -15000.0
MAX_TRADES_PER_DAY = 3

# Training splits
TRAIN_END = pd.Timestamp("2023-01-01")
VAL_END   = pd.Timestamp("2024-01-01")

# Features
CHAIN_FEATURES = [
    "atm_iv", "atm_call_iv", "atm_put_iv", "skew_slope",
    "pcr_oi", "pcr_vol", "total_oi", "total_vol", "chain_breadth",
    "max_oi_call_dist_pct", "max_oi_put_dist_pct", "max_oi_total_dist_pct",
    "forward_basis", "T_years", "atm_straddle_premium",
    "realized_vol_30m", "iv_rv_spread",
]
FUT_FEATURES = [
    "fut_basis", "fut_basis_change_30m",
    "fut_oi_change_1m", "fut_oi_change_5m", "fut_oi_change_30m",
    "fut_volume_oi_ratio", "fut_session_position",
    "fut_oi_x_long_buildup", "fut_oi_x_short_buildup",
    "fut_oi_x_short_cover", "fut_oi_x_long_unwind",
]
SIM_CONTEXT = ["minute_of_day", "hour_of_day"]
ALL_FEATURES = CHAIN_FEATURES + FUT_FEATURES + SIM_CONTEXT


# ---------------------------------------------------------------------------
# Step 1 — Build per-minute futures bars dataframe (for label generation
#          and for backtest entry/exit prices)
# ---------------------------------------------------------------------------

def load_all_fut_bars(symbol: str = "NIFTY") -> pd.DataFrame:
    """Load every futures minute bar for the symbol across all days/years."""
    days = discover_fut_days(DATA_ROOT, symbol)
    rows = []
    for d, p in days:
        try:
            df = load_fut_day(p, symbol)
            rows.append(df)
        except Exception as exc:
            print(f"  skip {d}: {exc}")
            continue
    full = pd.concat(rows, ignore_index=True)
    full["datetime"] = pd.to_datetime(full["datetime"])
    full = full.sort_values("datetime").reset_index(drop=True)
    full["trade_date"] = full["datetime"].dt.normalize()
    return full


def build_barrier_labels(fut: pd.DataFrame) -> pd.DataFrame:
    """For each minute, compute LONG_LABEL and SHORT_LABEL via barrier method.

    LONG_LABEL = 1 iff within HORIZON_MIN minutes the price reaches
    +TARGET_PCT (relative to the entry close) BEFORE reaching -STOP_PCT.
    Mirror for SHORT_LABEL.

    Vectorized per-day for speed (~30-60s on full 5-year dataset).
    """
    out = fut[["datetime", "trade_date", "fut_close"]].copy().reset_index(drop=True)
    long_label = np.zeros(len(out), dtype=np.int8)
    short_label = np.zeros(len(out), dtype=np.int8)

    # Iterate by day, vectorize within-day
    days = list(out.groupby("trade_date").groups.items())
    for ti, (d, idx) in enumerate(days):
        idx_arr = np.asarray(idx)
        prices = out["fut_close"].iloc[idx_arr].to_numpy(dtype=np.float64)
        n = len(prices)
        for i in range(n):
            entry = prices[i]
            end = min(n, i + 1 + HORIZON_MIN)
            if end <= i + 1:
                continue
            window = prices[i+1:end]
            # LONG side
            target_long = entry * (1 + TARGET_PCT)
            stop_long = entry * (1 - STOP_PCT)
            hit_target_long = np.where(window >= target_long)[0]
            hit_stop_long = np.where(window <= stop_long)[0]
            if hit_target_long.size:
                if not hit_stop_long.size or hit_target_long[0] < hit_stop_long[0]:
                    long_label[idx_arr[i]] = 1
            # SHORT side
            target_short = entry * (1 - TARGET_PCT)
            stop_short = entry * (1 + STOP_PCT)
            hit_target_short = np.where(window <= target_short)[0]
            hit_stop_short = np.where(window >= stop_short)[0]
            if hit_target_short.size:
                if not hit_stop_short.size or hit_target_short[0] < hit_stop_short[0]:
                    short_label[idx_arr[i]] = 1
        if (ti + 1) % 100 == 0:
            print(f"    labelled {ti+1}/{len(days)} days", flush=True)

    out["long_label"] = long_label
    out["short_label"] = short_label
    return out


# ---------------------------------------------------------------------------
# Step 2 — Train classifiers
# ---------------------------------------------------------------------------

def train_one_side(*, df: pd.DataFrame, label_col: str, label_name: str) -> tuple[lgb.Booster, dict]:
    train = df[df["trade_date"] < TRAIN_END]
    val   = df[(df["trade_date"] >= TRAIN_END) & (df["trade_date"] < VAL_END)]
    test  = df[df["trade_date"] >= VAL_END]

    X_tr = train[ALL_FEATURES]; y_tr = train[label_col]
    X_va = val[ALL_FEATURES];   y_va = val[label_col]
    X_te = test[ALL_FEATURES];  y_te = test[label_col]

    print(f"\n  --- training {label_name} side ---")
    print(f"    train={len(train):,}  val={len(val):,}  test={len(test):,}")
    print(f"    pos_rate train={y_tr.mean():.4f}  val={y_va.mean():.4f}  test={y_te.mean():.4f}")

    params = {
        "objective": "binary", "metric": "binary_logloss",
        "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
        "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 5,
        "lambda_l1": 0.1, "lambda_l2": 1.0,
        "is_unbalance": True, "n_jobs": -1, "verbosity": -1,
    }
    booster = lgb.train(
        params, lgb.Dataset(X_tr, label=y_tr),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(X_va, label=y_va)],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    val_auc = roc_auc_score(y_va, booster.predict(X_va))
    test_auc = roc_auc_score(y_te, booster.predict(X_te))
    print(f"    best_iter={booster.best_iteration}  val_auc={val_auc:.4f}  test_auc={test_auc:.4f}")

    out = OUT_DIR / f"futures_{label_name}.lgb"
    booster.save_model(str(out))
    print(f"    saved → {out}")

    # Feature importance top 5
    imp = pd.DataFrame({
        "feature": booster.feature_name(),
        "gain": booster.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False).head(8)
    print(f"    top features:")
    for _, r in imp.iterrows():
        print(f"      {r.feature:<30s} gain={r.gain:.0f}")

    return booster, {
        "val_auc": float(val_auc), "test_auc": float(test_auc),
        "best_iter": int(booster.best_iteration),
        "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
    }


# ---------------------------------------------------------------------------
# Step 3 — Realistic backtest
# ---------------------------------------------------------------------------

def simulate_futures_trade(side: str, entry_idx: int, day_bars: pd.DataFrame,
                            entry_px: float) -> dict:
    """Simulate one futures trade with target / stop / time-stop on actual bars.

    Returns dict with: side, entry_px, exit_px, exit_reason, pnl_inr, was_stopped.
    """
    sign = +1 if side == "LONG" else -1
    target_px = entry_px * (1 + sign * TARGET_PCT)
    stop_px = entry_px * (1 - sign * STOP_PCT)

    j_end = min(len(day_bars), entry_idx + 1 + HORIZON_MIN)
    walk = day_bars.iloc[entry_idx+1:j_end]

    exit_px = None
    exit_reason = None
    for _, r in walk.iterrows():
        # Use HIGH/LOW of bar to detect intra-bar barrier hits if available;
        # we only have close, so check closes
        c = r["fut_close"]
        if side == "LONG":
            if c >= target_px:
                exit_px = target_px; exit_reason = "TARGET"; break
            if c <= stop_px:
                exit_px = stop_px; exit_reason = "STOP"; break
        else:  # SHORT
            if c <= target_px:
                exit_px = target_px; exit_reason = "TARGET"; break
            if c >= stop_px:
                exit_px = stop_px; exit_reason = "STOP"; break

    if exit_px is None:
        # Time stop — exit at last bar
        if walk.empty:
            exit_px = entry_px
            exit_reason = "NO_BARS"
        else:
            exit_px = float(walk["fut_close"].iloc[-1])
            exit_reason = "TIME"

    gross_pnl = sign * (exit_px - entry_px) * LOT_SIZE_NIFTY
    costs = COSTS_PER_LOT_INR + SLIPPAGE_PER_LOT_INR
    net_pnl = gross_pnl - costs

    # Apply per-trade hard floor regardless of bar resolution
    was_stopped = (exit_reason == "STOP") or (net_pnl <= PER_TRADE_STOP_LOSS_INR)
    if net_pnl < PER_TRADE_STOP_LOSS_INR:
        net_pnl = PER_TRADE_STOP_LOSS_INR
        was_stopped = True

    return {
        "side": side,
        "entry_px": float(entry_px),
        "exit_px": float(exit_px),
        "exit_reason": exit_reason,
        "gross_pnl_inr": float(gross_pnl),
        "costs_inr": float(costs),
        "net_pnl_inr": float(net_pnl),
        "was_stopped": bool(was_stopped),
    }


def backtest(*, df: pd.DataFrame, fut_bars: pd.DataFrame,
              long_model: lgb.Booster, short_model: lgb.Booster,
              long_thr: float, short_thr: float,
              year: int = 2024) -> tuple[pd.DataFrame, dict]:
    """Walk through the test year minute-by-minute, take trades, simulate."""
    test = df[df["trade_date"].dt.year == year].copy()
    test["long_score"] = long_model.predict(test[ALL_FEATURES])
    test["short_score"] = short_model.predict(test[ALL_FEATURES])

    # Apply gates
    test["take_long"] = (test["long_score"] >= long_thr) & (test["short_score"] < (1 - long_thr))
    test["take_short"] = (test["short_score"] >= short_thr) & (test["long_score"] < (1 - short_thr))

    # Apply timing constraints
    test["mod"] = test["minute_of_day"] - 9 * 60 - 15  # minute index from open
    eligible = (test["mod"] >= ENTRY_MIN_MOD) & (test["datetime"].dt.time < HARD_CUTOFF)
    test = test[eligible & (test["take_long"] | test["take_short"])].copy()

    # Group by day, apply daily caps + daily loss halt + chronological-first selection
    test = test.sort_values(["trade_date", "datetime"]).reset_index(drop=True)
    fut_lookup = {d: g.reset_index(drop=True)
                   for d, g in fut_bars.groupby("trade_date")}

    results = []
    daily_pnl = {}
    daily_count = {}

    for _, row in test.iterrows():
        td = row["trade_date"]
        if daily_count.get(td, 0) >= MAX_TRADES_PER_DAY:
            continue
        if daily_pnl.get(td, 0.0) <= DAILY_LOSS_HALT_INR:
            continue
        side = "LONG" if row["take_long"] else "SHORT"
        day_bars = fut_lookup.get(td)
        if day_bars is None:
            continue
        entry_idx = day_bars.index[day_bars["datetime"] == row["datetime"]]
        if len(entry_idx) == 0:
            continue
        i = int(entry_idx[0])
        entry_px = float(day_bars["fut_close"].iat[i])
        sim = simulate_futures_trade(side, i, day_bars, entry_px)

        results.append({
            "trade_date": td, "datetime": row["datetime"],
            "side": side, **sim,
            "long_score": float(row["long_score"]),
            "short_score": float(row["short_score"]),
        })
        daily_pnl[td] = daily_pnl.get(td, 0.0) + sim["net_pnl_inr"]
        daily_count[td] = daily_count.get(td, 0) + 1

    trades = pd.DataFrame(results)
    if trades.empty:
        return trades, {"n_trades": 0}

    p = trades["net_pnl_inr"]
    daily = trades.groupby("trade_date")["net_pnl_inr"].sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else float("nan")
    cum = p.cumsum(); dd = float((cum - cum.cummax()).min())
    gw = p[p > 0].sum(); gl = abs(p[p < 0].sum())
    pf = gw / gl if gl > 0 else float("inf")

    summary = {
        "n_trades": int(len(trades)),
        "n_days": int(trades["trade_date"].nunique()),
        "trades_per_day": float(len(trades) / trades["trade_date"].nunique()),
        "win_rate": float((p > 0).mean()),
        "stop_rate": float(trades["was_stopped"].mean()),
        "mean_pnl_inr": float(p.mean()),
        "total_pnl_inr": float(p.sum()),
        "best_trade_inr": float(p.max()),
        "worst_trade_inr": float(p.min()),
        "best_day_inr": float(daily.max()),
        "worst_day_inr": float(daily.min()),
        "sharpe_daily_ann": float(sharpe),
        "profit_factor": float(pf),
        "max_drawdown_inr": float(dd),
        "n_long": int((trades["side"] == "LONG").sum()),
        "n_short": int((trades["side"] == "SHORT").sum()),
    }
    return trades, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--long_thr", type=float, default=0.55)
    ap.add_argument("--short_thr", type=float, default=0.55)
    ap.add_argument("--rebuild_labels", action="store_true",
                    help="Rebuild barrier labels from scratch (slow, ~5 min)")
    ap.add_argument("--rebuild_models", action="store_true",
                    help="Retrain classifiers (fast)")
    args = ap.parse_args()

    labels_path = OUT_DIR / "futures_barrier_labels.parquet"

    # ── 1. Load fut bars and build labels ──────────────────────────────────
    print("=" * 78)
    print("Sprint #1 — Futures-first directional engine")
    print(f"  Target +{TARGET_PCT*100:.2f}%   Stop -{STOP_PCT*100:.2f}%   "
          f"Horizon {HORIZON_MIN}min")
    print("=" * 78)

    print("\n[1/4] Loading futures bars …")
    t0 = time.time()
    fut_bars = load_all_fut_bars("NIFTY")
    print(f"  loaded {len(fut_bars):,} bars across "
          f"{fut_bars['trade_date'].nunique()} days "
          f"({time.time()-t0:.0f}s)")

    if args.rebuild_labels or not labels_path.exists():
        print("\n[2/4] Building barrier labels (this is the slow step)…")
        t0 = time.time()
        labels = build_barrier_labels(fut_bars)
        print(f"  built {len(labels):,} label rows ({time.time()-t0:.0f}s)")
        labels.to_parquet(labels_path)
        print(f"  saved → {labels_path}")
    else:
        print(f"\n[2/4] Loading cached labels from {labels_path}")
        labels = pd.read_parquet(labels_path)

    print(f"  long pos rate:  {labels['long_label'].mean():.4f}")
    print(f"  short pos rate: {labels['short_label'].mean():.4f}")

    # ── 2. Merge with feature set ─────────────────────────────────────────
    print("\n[3/4] Loading features …")
    t0 = time.time()
    chain = pd.concat(
        [pd.read_parquet(f) for f in sorted(CHAIN_DIR.rglob("data.parquet"))
         if "index=NIFTY" in str(f)], ignore_index=True)
    chain["datetime"] = pd.to_datetime(chain["datetime"])
    fut = pd.concat(
        [pd.read_parquet(f) for f in sorted(FUT_DIR.rglob("data.parquet"))
         if "index=NIFTY" in str(f)], ignore_index=True)
    fut["datetime"] = pd.to_datetime(fut["datetime"])
    print(f"  chain: {len(chain):,}, fut: {len(fut):,} ({time.time()-t0:.0f}s)")

    feats = chain[["datetime"] + CHAIN_FEATURES].merge(
        fut[["datetime"] + FUT_FEATURES], on="datetime", how="inner")
    feats["minute_of_day"] = feats["datetime"].dt.hour * 60 + feats["datetime"].dt.minute
    feats["hour_of_day"] = feats["datetime"].dt.hour

    df = labels.merge(feats, on="datetime", how="inner")
    df = df.dropna(subset=ALL_FEATURES + ["long_label", "short_label"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    print(f"  merged & cleaned: {len(df):,} rows")

    # Skip very-early-session minutes from training (lag features incomplete)
    df["mod"] = df["minute_of_day"] - 9 * 60 - 15
    df = df[df["mod"] >= ENTRY_MIN_MOD]
    print(f"  after early-cutoff filter: {len(df):,}")

    # ── 3. Train two classifiers ──────────────────────────────────────────
    if (args.rebuild_models
        or not (OUT_DIR / "futures_long.lgb").exists()
        or not (OUT_DIR / "futures_short.lgb").exists()):
        long_model, long_meta = train_one_side(
            df=df, label_col="long_label", label_name="long")
        short_model, short_meta = train_one_side(
            df=df, label_col="short_label", label_name="short")
        train_summary = {"long": long_meta, "short": short_meta}
        (OUT_DIR / "training_summary.json").write_text(
            json.dumps(train_summary, indent=2))
    else:
        print("\n[skip] using cached models in", OUT_DIR)
        long_model = lgb.Booster(model_file=str(OUT_DIR / "futures_long.lgb"))
        short_model = lgb.Booster(model_file=str(OUT_DIR / "futures_short.lgb"))

    # ── 4. Backtest ───────────────────────────────────────────────────────
    print("\n[4/4] Realistic backtest on 2024 …")
    t0 = time.time()
    trades, summary = backtest(
        df=df, fut_bars=fut_bars,
        long_model=long_model, short_model=short_model,
        long_thr=args.long_thr, short_thr=args.short_thr,
        year=2024)
    print(f"  done ({time.time()-t0:.0f}s)")

    # Output
    print("\n" + "=" * 78)
    print("  Sprint #1 — Headline (2024 blind window, NIFTY futures only)")
    print("=" * 78)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<22s}: {v:>+12.2f}")
        else:
            print(f"  {k:<22s}: {v:>12}")

    if not trades.empty:
        print("\n--- by side ---")
        for s, sub in trades.groupby("side"):
            p = sub["net_pnl_inr"]
            print(f"  {s:<6} n={len(sub):>3d}  win={(p>0).mean()*100:>5.1f}%  "
                  f"mean=₹{p.mean():>+6.0f}  total=₹{p.sum():>+10,.0f}")

        print("\n--- by exit reason ---")
        print(trades.groupby("exit_reason").agg(
            n=("net_pnl_inr", "count"),
            mean=("net_pnl_inr", "mean"),
            total=("net_pnl_inr", "sum"),
        ).round(0).to_string())

        print("\n--- by month ---")
        trades["month"] = pd.to_datetime(trades["datetime"]).dt.to_period("M").astype(str)
        print(trades.groupby("month").agg(
            n=("net_pnl_inr", "count"),
            win=("net_pnl_inr", lambda s: (s > 0).mean()),
            total=("net_pnl_inr", "sum"),
        ).round(2).to_string())

        print("\n--- by entry-time hour ---")
        trades["entry_hour"] = pd.to_datetime(trades["datetime"]).dt.hour
        print(trades.groupby("entry_hour").agg(
            n=("net_pnl_inr", "count"),
            win=("net_pnl_inr", lambda s: (s > 0).mean()),
            total=("net_pnl_inr", "sum"),
        ).round(2).to_string())

        print("\n--- by weekday ---")
        trades["weekday"] = pd.to_datetime(trades["datetime"]).dt.day_name()
        print(trades.groupby("weekday").agg(
            n=("net_pnl_inr", "count"),
            win=("net_pnl_inr", lambda s: (s > 0).mean()),
            total=("net_pnl_inr", "sum"),
        ).round(2).to_string())

    # Persist
    if not trades.empty:
        trades.to_parquet(RESULTS / "sprint1_futures_trades.parquet", index=False)
    (RESULTS / "sprint1_futures_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"\nSaved → {RESULTS}/sprint1_futures_*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
