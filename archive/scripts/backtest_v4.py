#!/usr/bin/env python3
"""V4-E: Regime-conditional backtest using V4-B vol forecast as trade signal.

Trade logic (no V4-D classifier — it was degenerate):
  Signal = predicted_rv_30m (from V4-B model) vs atm_iv

  sell-options: predicted_rv < atm_iv * SELL_THRESHOLD
                AND regime_pred in {range, vol-crush}
  buy-options : predicted_rv > atm_iv * BUY_THRESHOLD
                AND regime_pred == vol-expansion
  no-trade    : otherwise

PnL = BS re-priced straddle at t+30 (entry - exit) × lot - costs.
One trade per (index, minute) — no position sizing, fixed 1 lot.

Reports: overall, by direction, by regime, by index, monthly, equity curve.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CHAIN_DIR   = PROJECT_ROOT / "cache/optinet_v4/chain_features"
REGIME_FILE = PROJECT_ROOT / "results/optinet_v4/v4c_test_regimes.parquet"
VOL_MODEL   = PROJECT_ROOT / "models/optinet_v4/rv_30m_forward.lgb"
OUT_DIR     = PROJECT_ROOT / "results/optinet_v4"

LOT_SIZE = {"NIFTY": 50, "BANKNIFTY": 15}
BROKERAGE = 40.0
SLIPPAGE_PCT = 0.005
RISK_FREE = 0.065
TRADING_DAYS = 252
TRADING_MINUTES = 375

SELL_THRESHOLD = 0.80   # sell if predicted_rv < atm_iv × 0.80
BUY_THRESHOLD  = 1.20   # buy  if predicted_rv > atm_iv × 1.20

# V4-B feature columns (must match train_v4_volatility.py)
VOL_FEATURES = [
    "atm_iv", "atm_call_iv", "atm_put_iv", "skew_slope",
    "pcr_oi", "pcr_vol", "total_oi", "total_vol", "chain_breadth",
    "max_oi_call_dist_pct", "max_oi_put_dist_pct", "max_oi_total_dist_pct",
    "forward_basis", "T_years", "atm_straddle_premium",
    "realized_vol_30m", "iv_rv_spread",
    "minute_of_day", "minutes_to_close",
    "atm_iv_lag5", "atm_iv_lag15",
    "skew_slope_lag5", "skew_slope_lag15",
    "pcr_oi_lag5", "pcr_oi_lag15",
    "pcr_vol_lag5", "pcr_vol_lag15",
    "iv_rv_spread_lag5", "iv_rv_spread_lag15",
    "realized_vol_30m_lag5", "realized_vol_30m_lag15",
    "max_oi_total_dist_pct_lag5", "max_oi_total_dist_pct_lag15",
]


def bs_straddle_vec(S, K, T, r, sigma):
    sqrtT = np.sqrt(np.maximum(T, 1e-9))
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(np.maximum(S, 1e-9) / np.maximum(K, 1e-9))
              + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put  = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.where((T > 0) & (sigma > 0), call + put, 0.0)


def add_lags_and_time(df: pd.DataFrame) -> pd.DataFrame:
    df["trade_date"] = df["datetime"].dt.normalize()
    grp = df.groupby(["index", "trade_date"], sort=False)
    for col in ["atm_iv", "skew_slope", "pcr_oi", "pcr_vol", "realized_vol_30m",
                "iv_rv_spread", "max_oi_total_dist_pct"]:
        df[f"{col}_lag5"]  = grp[col].shift(5)
        df[f"{col}_lag15"] = grp[col].shift(15)
    df["minute_of_day"]    = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
    df["minutes_to_close"] = ((15 * 60 + 30) - df["minute_of_day"]).clip(lower=0)
    return df


def metrics(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {"label": label, "trades": 0}
    total_pnl = float(df["net_pnl"].sum())
    win_rate  = float(df["win"].mean())
    avg_pnl   = float(df["net_pnl"].mean())
    daily = df.groupby("trade_date")["net_pnl"].sum()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    cum = df["net_pnl"].cumsum()
    max_dd = float((cum - cum.cummax()).min())
    return {
        "label": label, "trades": int(len(df)),
        "win_rate": round(win_rate, 4),
        "avg_pnl_per_trade": round(avg_pnl, 2),
        "total_pnl": round(total_pnl, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 2),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading chain features (2024 only) …")
    test_df = pd.concat(
        [pd.read_parquet(f) for f in sorted(CHAIN_DIR.rglob("data.parquet"))
         if "year=2024" in str(f)],
        ignore_index=True)
    test_df["datetime"] = pd.to_datetime(test_df["datetime"])
    test_df = test_df.sort_values(["index", "datetime"]).reset_index(drop=True)
    print(f"  {len(test_df):,} rows")

    test_df = add_lags_and_time(test_df)

    # Forward 30-min spot return (for PnL calculation — NOT used as feature)
    grp = test_df.groupby(["index", "trade_date"], sort=False)
    test_df["fwd_ret_30m"] = grp["spot"].transform(lambda s: s.shift(-30) / s - 1.0)

    # V4-B: predict realized vol
    print("Running V4-B vol forecast …")
    vol_model = lgb.Booster(model_file=str(VOL_MODEL))
    valid_mask = test_df[VOL_FEATURES].notna().all(axis=1)
    test_df["pred_rv"] = np.nan
    test_df.loc[valid_mask, "pred_rv"] = vol_model.predict(
        test_df.loc[valid_mask, VOL_FEATURES])

    # V4-C: regime predictions
    print("Loading V4-C regime predictions …")
    reg = pd.read_parquet(REGIME_FILE)[["index", "datetime", "regime_pred"]]
    reg["datetime"] = pd.to_datetime(reg["datetime"])
    test_df = test_df.merge(reg, on=["index", "datetime"], how="left")
    test_df["regime_pred"] = test_df["regime_pred"].fillna(0).astype(int)

    # Signal generation — only top-2 per (index, day) by rv_iv_ratio extremity
    test_df["rv_iv_ratio"] = test_df["pred_rv"] / test_df["atm_iv"].replace(0, np.nan)

    base_valid = (
        test_df["atm_straddle_premium"].gt(0) &
        test_df["fwd_ret_30m"].notna() &
        test_df["T_years"].gt(30.0 / (TRADING_DAYS * TRADING_MINUTES)) &
        test_df["pred_rv"].notna()
    )

    # Sell candidates: low rv_iv_ratio in range/vol-crush regime
    sell_cands = test_df[
        base_valid &
        test_df["rv_iv_ratio"].lt(SELL_THRESHOLD) &
        test_df["regime_pred"].isin([0, 4])
    ].copy()
    # Keep top-2 lowest rv_iv_ratio per (index, trade_date)
    sell_cands = (sell_cands.sort_values("rv_iv_ratio")
                  .groupby(["index", "trade_date"], sort=False)
                  .head(2))

    # Buy candidates: high rv_iv_ratio in vol-expansion regime
    buy_cands = test_df[
        base_valid &
        test_df["rv_iv_ratio"].gt(BUY_THRESHOLD) &
        test_df["regime_pred"].eq(3)
    ].copy()
    buy_cands = (buy_cands.sort_values("rv_iv_ratio", ascending=False)
                 .groupby(["index", "trade_date"], sort=False)
                 .head(2))

    print(f"Signals after top-2/day filter: sell={len(sell_cands):,}  buy={len(buy_cands):,}")

    # Simulate trades
    dt_30m = 30.0 / (TRADING_DAYS * TRADING_MINUTES)
    results = []
    for direction, sub in [("short", sell_cands), ("long", buy_cands)]:
        if sub.empty:
            continue
        lot_arr = np.array([LOT_SIZE.get(i, 50) for i in sub["index"]])
        S0 = sub["spot"].to_numpy()
        K  = sub["atm_strike"].to_numpy()
        T0 = sub["T_years"].to_numpy()
        iv = sub["atm_iv"].to_numpy()
        entry_px = sub["atm_straddle_premium"].to_numpy() * 2.0
        fwd = sub["fwd_ret_30m"].to_numpy()

        S1 = S0 * (1.0 + fwd)
        T1 = np.maximum(T0 - dt_30m, 1e-6)
        exit_px = bs_straddle_vec(S1, K, T1, RISK_FREE, iv)

        cost = BROKERAGE + SLIPPAGE_PCT * (entry_px + exit_px) * lot_arr
        if direction == "short":
            gross = (entry_px - exit_px) * lot_arr
        else:
            gross = (exit_px - entry_px) * lot_arr
        net = gross - cost

        for i, (_, row) in enumerate(sub.iterrows()):
            results.append({
                "index": row["index"],
                "trade_date": row["trade_date"],
                "datetime": row["datetime"],
                "regime_pred": row["regime_pred"],
                "rv_iv_ratio": row["rv_iv_ratio"],
                "direction": direction,
                "entry_px": float(entry_px[i]),
                "exit_px": float(exit_px[i]),
                "gross_pnl": float(gross[i]),
                "net_pnl": float(net[i]),
                "win": bool(net[i] > 0),
                "lot": int(lot_arr[i]),
            })

    if not results:
        print("No trades generated.")
        return

    trades = pd.DataFrame(results).sort_values("datetime").reset_index(drop=True)
    print(f"Total trades: {len(trades):,}")
    print(f"Avg trades/day: {len(trades)/test_df['trade_date'].nunique():.1f}")

    print("\n=== Overall ===")
    ov = metrics(trades, "overall")
    for k, v in ov.items():
        print(f"  {k}: {v}")

    print("\n=== By direction ===")
    dir_results = {}
    for d in ["short", "long"]:
        m = metrics(trades[trades["direction"] == d], d)
        dir_results[d] = m
        print(f"  {d:>5s}: trades={m['trades']:>5}  win={m.get('win_rate','N/A')}  "
              f"avg=₹{m.get('avg_pnl_per_trade','N/A')}  "
              f"total=₹{m.get('total_pnl','N/A')}  sharpe={m.get('sharpe','N/A')}")

    print("\n=== By regime ===")
    regime_names = {0: "range", 3: "vol-expansion", 4: "vol-crush"}
    regime_results = {}
    for r_id, r_name in regime_names.items():
        m = metrics(trades[trades["regime_pred"] == r_id], r_name)
        regime_results[r_name] = m
        print(f"  {r_name:>14s}: trades={m['trades']:>5}  win={m.get('win_rate','N/A')}  "
              f"avg=₹{m.get('avg_pnl_per_trade','N/A')}  total=₹{m.get('total_pnl','N/A')}")

    print("\n=== By index ===")
    index_results = {}
    for idx in sorted(trades["index"].unique()):
        m = metrics(trades[trades["index"] == idx], idx)
        index_results[idx] = m
        print(f"  {idx:>10s}: trades={m['trades']:>5}  win={m.get('win_rate','N/A')}  "
              f"avg=₹{m.get('avg_pnl_per_trade','N/A')}  total=₹{m.get('total_pnl','N/A')}  "
              f"sharpe={m.get('sharpe','N/A')}")

    print("\n=== Monthly PnL ===")
    trades["month"] = trades["trade_date"].dt.to_period("M")
    monthly = trades.groupby("month")["net_pnl"].sum()
    print(monthly.round(0).to_string())

    print("\n=== Theta/gamma breakdown (short trades) ===")
    short_t = trades[trades["direction"] == "short"]
    if not short_t.empty:
        print(f"  avg entry straddle: ₹{short_t['entry_px'].mean():.2f}")
        print(f"  avg exit  straddle: ₹{short_t['exit_px'].mean():.2f}")
        print(f"  avg theta captured: ₹{(short_t['entry_px']-short_t['exit_px']).mean():.2f}")
        print(f"  avg cost          : ₹{(short_t['gross_pnl']-short_t['net_pnl']).mean():.2f}")

    daily_pnl = trades.groupby("trade_date")["net_pnl"].sum().reset_index()
    daily_pnl["cumulative_pnl"] = daily_pnl["net_pnl"].cumsum()
    daily_pnl.to_parquet(OUT_DIR / "v4e_equity_curve.parquet", index=False)
    trades.to_parquet(OUT_DIR / "v4e_trades.parquet", index=False)

    summary = {
        "sell_threshold": SELL_THRESHOLD,
        "buy_threshold": BUY_THRESHOLD,
        "overall": ov, "by_direction": dir_results,
        "by_regime": regime_results, "by_index": index_results,
        "avg_trades_per_day": round(len(trades) / test_df["trade_date"].nunique(), 2),
    }
    (OUT_DIR / "v4e_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved → {OUT_DIR/'v4e_summary.json'}")
    print(f"Saved → {OUT_DIR/'v4e_trades.parquet'}")
    print(f"Saved → {OUT_DIR/'v4e_equity_curve.parquet'}")


if __name__ == "__main__":
    main()
