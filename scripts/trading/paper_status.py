#!/usr/bin/env python3
"""Paper-trading status dashboard.

Reads the persistent ledger and reports:
  - All-time aggregate metrics
  - Trailing 30-day metrics
  - Trailing 5-day metrics
  - Equity curve (terminal-friendly)
  - Halt-condition alerts

Halt conditions (writes the kill-switch file when triggered):
  - Trailing 30-day Sharpe < 0           → SOFT HALT (recommend stop)
  - Cumulative drawdown > -₹150,000      → HARD HALT (must stop)
  - Trailing 5-day net PnL < -₹50,000    → SOFT HALT
  - 7+ consecutive losing days           → SOFT HALT
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = PROJECT_ROOT / "results/router_v0/paper_trading_ledger.csv"
KILL_SWITCH    = PROJECT_ROOT / "results/router_v0/PAPER_TRADING_HALTED"

# Halt thresholds
HARD_DD_HALT_INR        = -150_000.0
SOFT_30D_SHARPE_HALT    = 0.5    # Phase-4: alert when rolling 30d Sharpe < 0.5
SOFT_5D_PNL_HALT_INR    = -50_000.0
SOFT_CONSEC_LOSS_DAYS   = 7

SOURCE_LABELS = {
    "paper": "Variant A live",
    "paper_c": "Variant C live",
    "forward_walk": "Forward-walk bootstrap",
}


def metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n_trades": 0, "n_days": 0, "win_rate": 0.0,
                 "total_pnl_inr": 0.0, "mean_pnl_inr": 0.0,
                 "sharpe_daily_ann": 0.0, "profit_factor": 0.0,
                 "max_drawdown_inr": 0.0}
    p = trades["net_pnl_inr"]
    daily = trades.groupby("trade_date")["net_pnl_inr"].sum()
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)
               if len(daily) > 1 and daily.std() > 0 else 0.0)
    cum = p.cumsum()
    gw = p[p > 0].sum()
    gl = abs(p[p < 0].sum())
    pf = float(gw / gl) if gl > 0 else float("inf")
    return {
        "n_trades": int(len(trades)),
        "n_days": int(trades["trade_date"].nunique()),
        "win_rate": float((p > 0).mean()),
        "total_pnl_inr": float(p.sum()),
        "mean_pnl_inr": float(p.mean()),
        "sharpe_daily_ann": float(sharpe),
        "profit_factor": pf,
        "max_drawdown_inr": float((cum - cum.cummax()).min()),
    }


def print_metrics_block(label: str, m: dict) -> None:
    if m["n_trades"] == 0:
        print(f"  {label:<22s}  (no trades)")
        return
    print(f"  {label:<22s}  n={m['n_trades']:>4d}  days={m['n_days']:>3d}  "
          f"win={m['win_rate']*100:>5.1f}%  "
          f"PnL=₹{m['total_pnl_inr']:>+11,.0f}  "
          f"Sharpe={m['sharpe_daily_ann']:>+5.2f}  "
          f"PF={m['profit_factor']:>4.2f}  "
          f"DD=₹{m['max_drawdown_inr']:>+10,.0f}")


def equity_curve_ascii(trades: pd.DataFrame, width: int = 80) -> str:
    if trades.empty:
        return ""
    daily = trades.sort_values("trade_date").groupby(
        "trade_date")["net_pnl_inr"].sum().cumsum()
    if len(daily) < 2:
        return ""
    n = len(daily)
    bins = max(1, n // width)
    sampled = daily.iloc[::max(1, bins)]
    vmin, vmax = sampled.min(), sampled.max()
    rng = max(vmax - vmin, 1.0)
    height = 14
    rows = [["·"] * len(sampled) for _ in range(height)]
    for i, v in enumerate(sampled):
        h = int(round((v - vmin) / rng * (height - 1)))
        rows[height - 1 - h][i] = "█"
    block = "\n".join("".join(r) for r in rows)
    legend = (f"  ₹{vmax:>+12,.0f}  ─── max\n  "
                f"…\n  ₹{vmin:>+12,.0f}  ─── min")
    return f"{legend}\n\n  {block.replace(chr(10), chr(10) + '  ')}"


def trigger_halts(all_m: dict, t30_m: dict, t5_m: dict,
                    consec_loss_days: int) -> list[str]:
    halts = []
    if all_m["max_drawdown_inr"] <= HARD_DD_HALT_INR:
        halts.append(f"HARD HALT: cumulative drawdown "
                       f"₹{all_m['max_drawdown_inr']:+,.0f} ≤ "
                       f"₹{HARD_DD_HALT_INR:+,.0f}")
    if t30_m["n_trades"] >= 30 and t30_m["sharpe_daily_ann"] < SOFT_30D_SHARPE_HALT:
        halts.append(f"SOFT HALT: 30-day Sharpe "
                       f"{t30_m['sharpe_daily_ann']:+.2f} < "
                       f"{SOFT_30D_SHARPE_HALT:+.2f}")
    if t5_m["total_pnl_inr"] <= SOFT_5D_PNL_HALT_INR:
        halts.append(f"SOFT HALT: trailing 5-day PnL "
                       f"₹{t5_m['total_pnl_inr']:+,.0f} ≤ "
                       f"₹{SOFT_5D_PNL_HALT_INR:+,.0f}")
    if consec_loss_days >= SOFT_CONSEC_LOSS_DAYS:
        halts.append(f"SOFT HALT: {consec_loss_days} consecutive losing "
                       f"days ≥ {SOFT_CONSEC_LOSS_DAYS}")
    return halts


def consec_losing_days(trades: pd.DataFrame) -> int:
    if trades.empty:
        return 0
    daily = trades.sort_values("trade_date").groupby(
        "trade_date")["net_pnl_inr"].sum()
    n = 0
    for v in reversed(daily.tolist()):
        if v < 0:
            n += 1
        else:
            break
    return n


def reference_distribution(bootstrap: pd.DataFrame) -> dict:
    """Compute the rolling 30-day distribution of metrics on the
    forward-walk bootstrap rows. Used to judge whether live paper
    trading is tracking the validated expectation."""
    if bootstrap.empty:
        return {}
    daily = bootstrap.sort_values("trade_date").groupby(
        "trade_date")["net_pnl_inr"].agg(["sum", "count",
                                            lambda s: (s > 0).sum()])
    daily.columns = ["pnl", "n_trades", "n_wins"]
    daily = daily.reset_index().sort_values("trade_date")

    # 30-day rolling stats — window in calendar days
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    daily = daily.set_index("trade_date")
    rolling = daily.rolling("30D")
    pnl_30d   = rolling["pnl"].sum()
    n_30d     = rolling["n_trades"].sum()
    wins_30d  = rolling["n_wins"].sum()
    sharpe_30d = (rolling["pnl"].mean() / rolling["pnl"].std()
                   * np.sqrt(252))
    win_pct_30d = (wins_30d / n_30d.replace(0, np.nan)).fillna(0)

    # Use only windows with ≥ 30 trades (matches the live halt threshold)
    mask = n_30d >= 30
    if mask.sum() == 0:
        return {}
    sharpe_q = sharpe_30d[mask].quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    pnl_q    = pnl_30d[mask].quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    win_q    = win_pct_30d[mask].quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "sharpe_30d": sharpe_q.to_dict(),
        "pnl_30d":    pnl_q.to_dict(),
        "win_30d":    win_q.to_dict(),
        "n_windows":  int(mask.sum()),
    }


def tracking_check(t30: dict, ref: dict) -> list[str]:
    """Compare current 30-day metrics to the reference distribution.
    Returns a list of human-readable status lines."""
    lines = []
    if not ref:
        return ["(no reference distribution — no bootstrap rows in ledger)"]
    if t30["n_trades"] < 30:
        return [f"trailing 30d has only {t30['n_trades']} trades — "
                f"need ≥30 before tracking is meaningful"]

    def classify(value: float, q: dict) -> str:
        if value < q[0.10]:   return "🔻 BELOW p10  (worse than 90% of reference windows)"
        if value < q[0.25]:   return "🟡 below p25  (worse than 75% of reference windows)"
        if value <= q[0.75]:  return "🟢 in IQR    (within central 50% — tracking)"
        if value <= q[0.90]:  return "🟢 above p75  (better than 75% of reference windows)"
        return "✨ above p90  (better than 90% — possibly transient outlier)"

    sharpe = t30["sharpe_daily_ann"]
    pnl    = t30["total_pnl_inr"]
    win    = t30["win_rate"]
    sq, pq, wq = ref["sharpe_30d"], ref["pnl_30d"], ref["win_30d"]
    lines.append(
        f"  {'30d Sharpe':<18s} {sharpe:+5.2f}  "
        f"ref median {sq[0.50]:+.2f}, IQR [{sq[0.25]:+.2f}, {sq[0.75]:+.2f}]   "
        f"{classify(sharpe, sq)}")
    lines.append(
        f"  {'30d PnL':<18s} ₹{pnl:+,.0f}  "
        f"ref median ₹{pq[0.50]:+,.0f}, IQR [₹{pq[0.25]:+,.0f}, ₹{pq[0.75]:+,.0f}]   "
        f"{classify(pnl, pq)}")
    lines.append(
        f"  {'30d Win rate':<18s} {100*win:5.1f}%  "
        f"ref median {100*wq[0.50]:.1f}%, IQR [{100*wq[0.25]:.1f}%, {100*wq[0.75]:.1f}%]   "
        f"{classify(win, wq)}")
    return lines


def rolling_stability_report(trades: pd.DataFrame,
                              window_days: int = 30) -> list[str]:
    """Phase-4: compute rolling 30-day Sharpe + win-rate.

    Returns a list of human-readable lines. Flags any window where
    Sharpe < SOFT_30D_SHARPE_HALT (0.5) with a WARNING marker.
    """
    if trades.empty:
        return ["  (no trades for rolling stability report)"]

    daily = (trades.sort_values("trade_date")
             .groupby("trade_date")["net_pnl_inr"]
             .agg(pnl="sum",
                  n_trades="count",
                  n_wins=lambda s: (s > 0).sum())
             .reset_index())
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    daily = daily.set_index("trade_date").sort_index()

    lines = [f"\n  -- ROLLING {window_days}-DAY STABILITY (Phase-4) --"]
    lines.append(f"  Alert threshold: Sharpe < {SOFT_30D_SHARPE_HALT:.1f}")
    lines.append(f"  {'Window end':<12s}  {'n':>4s}  {'Win%':>6s}  "
                 f"{'PnL':>12s}  {'Sharpe':>7s}  Status")
    lines.append("  " + "-" * 60)

    # Slide a 30-calendar-day window ending at each trading day
    dates = daily.index.tolist()
    alerts = []
    for end_date in dates:
        start_date = end_date - pd.Timedelta(days=window_days)
        window = daily[(daily.index > start_date) & (daily.index <= end_date)]
        if len(window) < 5:
            continue
        pnl_series = window["pnl"]
        sharpe = (pnl_series.mean() / pnl_series.std() * np.sqrt(252)
                  if pnl_series.std() > 0 else 0.0)
        win_rate = float(window["n_wins"].sum() / window["n_trades"].sum())
        total_pnl = float(pnl_series.sum())
        n = int(window["n_trades"].sum())
        status = "OK"
        if sharpe < SOFT_30D_SHARPE_HALT:
            status = f"ALERT Sharpe={sharpe:+.2f} < {SOFT_30D_SHARPE_HALT:.1f}"
            alerts.append((end_date, sharpe))
        lines.append(f"  {end_date.date()!s:<12s}  {n:>4d}  "
                     f"{100*win_rate:>5.1f}%  "
                     f"Rs{total_pnl:>+10,.0f}  "
                     f"{sharpe:>+6.2f}  {status}")

    if alerts:
        lines.append(f"\n  *** {len(alerts)} window(s) below Sharpe {SOFT_30D_SHARPE_HALT:.1f} ***")
        for d, s in alerts[-3:]:
            lines.append(f"      {d.date()}  Sharpe={s:+.2f}")
    else:
        lines.append(f"\n  All rolling windows above Sharpe {SOFT_30D_SHARPE_HALT:.1f} -- stable")

    return lines


def date_range_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    return f"{df['trade_date'].min().date()}  ->  {df['trade_date'].max().date()}"


def trailing_windows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df
    today = df["trade_date"].max()
    return (
        df[df["trade_date"] > today - pd.Timedelta(days=30)],
        df[df["trade_date"] > today - pd.Timedelta(days=5)],
    )


def print_source_report(label: str, trades: pd.DataFrame,
                        ref: dict | None = None,
                        by_month: bool = False,
                        show_tracking: bool = False) -> None:
    print(f"\n  -- {label.upper()} --")
    if trades.empty:
        print("  (no rows)")
        return
    df_30, df_5 = trailing_windows(trades)
    print_metrics_block("ALL-TIME", metrics(trades))
    print_metrics_block("trailing 30-day", metrics(df_30))
    print_metrics_block("trailing 5-day", metrics(df_5))
    print(f"  consecutive losing days: {consec_losing_days(trades)}")
    if show_tracking and ref is not None:
        print("\n  tracking vs forward-walk reference")
        for line in tracking_check(metrics(df_30), ref):
            print(line)
    if by_month:
        print("\n  by month")
        month_df = trades.copy()
        month_df["month"] = month_df["trade_date"].dt.to_period("M").astype(str)
        for mo, sub in month_df.groupby("month"):
            print_metrics_block(mo, metrics(sub))


def print_variant_comparison(live: pd.DataFrame) -> None:
    print("\n  -- A vs C LIVE PAPER COMPARISON --")
    rows = [
        ("Variant A", live[live["source"] == "paper"]),
        ("Variant C", live[live["source"] == "paper_c"]),
    ]
    print("  variant      n  days   win%        PnL   Sharpe    PF        DD")
    for label, sub in rows:
        m = metrics(sub)
        if m["n_trades"] == 0:
            print(f"  {label:<9s}  (no trades)")
            continue
        print(f"  {label:<9s} {m['n_trades']:>3d} {m['n_days']:>5d} "
              f"{100*m['win_rate']:>6.1f}% ₹{m['total_pnl_inr']:>+10,.0f} "
              f"{m['sharpe_daily_ann']:>+7.2f} {m['profit_factor']:>5.2f} "
              f"₹{m['max_drawdown_inr']:>+9,.0f}")


def print_reference_summary(ref: dict) -> None:
    if not ref:
        return
    print(f"\n  reference distribution: {ref['n_windows']} 30-day windows from bootstrap")
    print(f"    median 30d Sharpe {ref['sharpe_30d'][0.5]:+.2f}, "
          f"IQR [{ref['sharpe_30d'][0.25]:+.2f}, {ref['sharpe_30d'][0.75]:+.2f}]")
    print(f"    median 30d PnL    ₹{ref['pnl_30d'][0.5]:+,.0f}, "
          f"IQR [₹{ref['pnl_30d'][0.25]:+,.0f}, ₹{ref['pnl_30d'][0.75]:+,.0f}]")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=str, default=str(DEFAULT_LEDGER))
    parser.add_argument("--include-bootstrap", action="store_true",
                          help="Include source='forward_walk' rows in stats")
    parser.add_argument("--write-halt", action="store_true",
                          help="Write the kill-switch file if any halt triggers")
    parser.add_argument("--by-month", action="store_true",
                          help="Show per-month breakdown")
    parser.add_argument("--rolling-stability", action="store_true",
                          help="Phase-4: show rolling 30-day Sharpe + win-rate monitor")
    args = parser.parse_args()

    ledger_path = Path(args.ledger)

    print("+" + "=" * 88 + "+")
    print("|  Paper Trading Status - OptiNet Router".ljust(89) + "|")
    print("+" + "=" * 88 + "+")

    if not ledger_path.exists():
        print(f"\n  ledger not found: {ledger_path}")
        print("  (run scripts/paper_trade_daily.py --auto)")
        return 1

    df = pd.read_csv(ledger_path)
    if df.empty:
        print(f"\n  ledger is empty")
        return 0

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)

    n_paper     = int((df["source"] == "paper").sum())
    n_paper_c   = int((df["source"] == "paper_c").sum())
    n_bootstrap = int((df["source"] == "forward_walk").sum())
    print(f"\n  ledger: {ledger_path}")
    print(f"          {len(df):,} total rows  "
          f"({n_paper:,} Variant A  +  {n_paper_c:,} Variant C  +  {n_bootstrap:,} bootstrap)")
    print(f"          {df['trade_date'].min().date()}  →  {df['trade_date'].max().date()}")

    # Always compute reference distribution from bootstrap rows (if present)
    bootstrap = df[df["source"] == "forward_walk"].copy()
    live = df[df["source"].isin(["paper", "paper_c"])].copy()
    live_a = df[df["source"] == "paper"].copy()
    live_c = df[df["source"] == "paper_c"].copy()
    ref = reference_distribution(bootstrap)

    print("\n  source ranges")
    for source, label in SOURCE_LABELS.items():
        print(f"    {label:<24s} {date_range_text(df[df['source'] == source])}")

    print_variant_comparison(live)

    if live.empty:
        print("\n  no live paper-trading rows yet")
        print_reference_summary(ref)
    else:
        print_source_report("Variant A live paper", live_a, ref, args.by_month, True)
        print_source_report("Variant C live paper", live_c, ref, args.by_month, True)

    if args.include_bootstrap:
        print_source_report("Forward-walk bootstrap reference", bootstrap, None, args.by_month, False)
        print_reference_summary(ref)

    # Equity curve for promoted Variant A only.
    curve = equity_curve_ascii(live_a)
    if curve:
        print("\n  -- VARIANT A LIVE CUMULATIVE PnL --")
        print(curve)

    # Halts are evaluated only on Variant A, the promoted paper strategy.
    halt_30, halt_5 = trailing_windows(live_a)
    all_m = metrics(live_a)
    t30_m = metrics(halt_30)
    t5_m = metrics(halt_5)
    consec = consec_losing_days(live_a)
    halts = trigger_halts(all_m, t30_m, t5_m, consec) if not live_a.empty else []
    print("\n  -- HALT CHECKS (Variant A live only) --")
    if not halts:
        print("  OK: all clear")
    else:
        for h in halts:
            print(f"  WARNING: {h}")
        if args.write_halt and not KILL_SWITCH.exists():
            KILL_SWITCH.write_text(
                "Paper trading halted by paper_trade_status.py\n"
                f"At: {pd.Timestamp.now().isoformat()}\n"
                "Reasons:\n  - " + "\n  - ".join(halts) + "\n"
                "\nDelete this file to resume after investigation.\n"
            )
            print(f"\n  -> kill-switch written to {KILL_SWITCH}")
            return 3

    # Phase-4: rolling stability dashboard
    if args.rolling_stability or halts:
        for line in rolling_stability_report(live_a):
            print(line)

    return 0 if not halts else (3 if any("HARD HALT" in h for h in halts) else 2)


if __name__ == "__main__":
    sys.exit(main())
