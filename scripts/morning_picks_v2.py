"""
Morning Picks V2 — Pre-market stock recommendations using LightGBM V2.

Loads all 6 models (H15/H30/H60 direction + magnitude), computes 625 features
on-the-fly from CSV data, applies regime filter, aggregates predictions across
horizons, and outputs ranked LONG/SHORT picks.

Usage:
    python scripts/morning_picks_v2.py \
        --model-dir runs/lgbm_v2/ \
        --horizon H60 \
        --top-n 5 \
        --dir-threshold 0.60 \
        --min-confidence 0.55
"""

import argparse
import csv
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intradaynet.features.per_bar_features import (
    PER_BAR_FEATURE_NAMES,
    compute_per_bar_features,
)
from intradaynet.features.session_features import (
    SESSION_FEATURE_NAMES,
    compute_session_features,
)
from intradaynet.features.sentiment_features import SentimentFeatureBuilder
from intradaynet.features.market_features import MarketFeatureBuilder
from intradaynet.regime import detect_regime, get_regime_adjustments, RegimeConfig
from intradaynet.flatten import flatten_window_for_lgbm

console = Console()
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("intradaynet.morning_picks_v2")

HORIZONS = [15, 30, 60]
SEQUENCE_LENGTH = 120

HORIZON_WEIGHTS = {
    "H15": 0.2,
    "H30": 0.3,
    "H60": 0.5,
}

HORIZON_COST_ADJUSTED = {
    "H15": 0.003,
    "H30": 0.003,
    "H60": 0.003,
}

NIFTY_SYMBOL = "^NSEI"
VIX_SYMBOL = "^INDIAVIX"


def _get_stock_symbols(data_dir: Path) -> list[str]:
    if not data_dir.exists():
        return []
    return sorted({p.stem.replace("_minute", "") for p in data_dir.glob("*_minute.csv")})


def _next_trading_day(ref_date: str) -> str:
    """Return the next trading day after ref_date (simple weekday skip)."""
    d = datetime.strptime(ref_date[:10], "%Y-%m-%d")
    for _ in range(7):
        d += timedelta(days=1)
        if d.weekday() < 5:
            return d.strftime("%Y-%m-%d")
    return ref_date


def _download_latest_data(symbols: list[str], data_dir: Path, target_date: str = None):
    """Download latest minute bars for each symbol via yfinance."""
    import yfinance as yf

    updated, failed = 0, 0
    end = target_date[:10] if target_date else datetime.now().strftime("%Y-%m-%d")

    for symbol in symbols:
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            df = ticker.history(start="2020-01-01", end=end, interval="1m", auto_adjust=True)
            if df.empty or len(df) < 50:
                failed += 1
                continue
            df = df.reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            if "datetime" in df.columns:
                df = df.rename(columns={"datetime": "date"})
            elif "timestamp" in df.columns:
                df = df.rename(columns={"timestamp": "date"})
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            csv_path = data_dir / f"{symbol}_minute.csv"
            if csv_path.exists():
                existing = pd.read_csv(csv_path, parse_dates=["date"], nrows=1)
                last_date = existing["date"].iloc[0]
                df = df[df["date"] > last_date.strftime("%Y-%m-%d %H:%M:%S")]
                if not df.empty:
                    df.to_csv(csv_path, mode="a", header=False, index=False)
            else:
                df.to_csv(csv_path, index=False)
            updated += 1
        except Exception:
            failed += 1
    return updated, failed


def _download_daily_closes(symbols: list[str], end_date: str = None) -> dict:
    """Fetch official closing prices from yfinance."""
    import yfinance as yf

    closes = {}
    end = end_date[:10] if end_date else datetime.now().strftime("%Y-%m-%d")
    for symbol in symbols[:50]:
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            hist = ticker.history(start="2020-01-01", end=end)
            if not hist.empty:
                closes[symbol] = {
                    str(d.date()): float(d.close)
                    for d in hist.itertuples()
                    if d.close > 0
                }
        except Exception:
            pass
    return closes


def compute_features_for_stock(
    symbol: str,
    data_dir: Path,
    sentiment_builder,
    market_builder,
    market_open: str = "09:15",
    market_close: str = "15:29",
    target_date: str = None,
    daily_closes: dict = None,
) -> dict | None:
    """Load CSV, compute all features, return last window + session + sentiment."""
    csv_path = data_dir / f"{symbol}_minute.csv"
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
        df.columns = df.columns.str.lower()

        df["_time"] = df.index.strftime("%H:%M")
        df = df[(df["_time"] >= market_open) & (df["_time"] < market_close)]
        df = df.drop(columns=["_time"])

        if len(df) < SEQUENCE_LENGTH:
            return None

        if target_date:
            try:
                td = pd.Timestamp(target_date[:10])
                df = df[df.index < td]
            except Exception:
                pass

        if len(df) < SEQUENCE_LENGTH:
            return None

        per_bar = compute_per_bar_features(df)
        session = compute_session_features(df)
        sentiment = sentiment_builder.get_features(symbol, session.index)

        dates = per_bar.index.date
        unique_dates = sorted(set(dates))
        if not unique_dates:
            return None

        ref_date = str(unique_dates[-1])

        per_bar_values = per_bar[PER_BAR_FEATURE_NAMES].values.astype(np.float32)
        close_values = df["close"].values.astype(np.float32)

        if len(per_bar_values) < SEQUENCE_LENGTH:
            return None

        window = per_bar_values[-SEQUENCE_LENGTH:]

        last_close = float(close_values[-1])
        if daily_closes and symbol in daily_closes:
            sym_daily = daily_closes[symbol]
            if ref_date in sym_daily:
                last_close = sym_daily[ref_date]

        if last_close <= 0:
            return None

        sess_values = session[SESSION_FEATURE_NAMES].values.astype(np.float32)
        sent_values = sentiment[SENTIMENT_FEATURE_NAMES].values.astype(np.float32)

        s_feat = sess_values[-1] if len(sess_values) > 0 else np.zeros(
            len(SESSION_FEATURE_NAMES), dtype=np.float32
        )
        se_feat = sent_values[-1] if len(sent_values) > 0 else np.zeros(
            len(SENTIMENT_FEATURE_NAMES), dtype=np.float32
        )

        return {
            "window": window,
            "session_feats": s_feat,
            "sentiment_feats": se_feat,
            "last_close": last_close,
            "ref_date": ref_date,
        }
    except Exception as e:
        logger.debug(f"Failed {symbol}: {e}")
        return None


def load_lgbm_models(model_dir: Path) -> tuple[dict, dict, list[str]]:
    """Load all 6 LightGBM booster models from directory."""
    import lightgbm as lgb

    dir_models, mag_models = {}, {}
    horizons = []
    for h in HORIZONS:
        hname = f"H{h}"
        dp = model_dir / f"dir_{hname}.lgb"
        mp = model_dir / f"mag_{hname}.lgb"
        if dp.exists():
            dir_models[hname] = lgb.Booster(model_file=str(dp))
        if mp.exists():
            mag_models[hname] = lgb.Booster(model_file=str(mp))
        if hname in dir_models:
            horizons.append(hname)

    return dir_models, mag_models, horizons


def infer_single_stock(
    window: np.ndarray,
    session_feats: np.ndarray,
    sentiment_feats: np.ndarray,
    dir_models: dict,
    mag_models: dict,
    horizons: list[str],
) -> dict:
    """Run inference across all horizons, aggregate results."""
    results = {}

    for hname in horizons:
        flat, _ = flatten_window_for_lgbm(
            window, session_feats, sentiment_feats, PER_BAR_FEATURE_NAMES
        )
        flat = np.nan_to_num(flat, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)
        X = flat.reshape(1, -1)

        dir_prob = float(dir_models[hname].predict(X)[0]) if hname in dir_models else 0.5
        magnitude = float(mag_models[hname].predict(X)[0]) if hname in mag_models else 0.0

        results[hname] = {
            "prob": dir_prob,
            "magnitude": magnitude,
            "confidence": abs(dir_prob - 0.5) * 2,
        }

    combined_prob = 0.0
    combined_mag = 0.0
    combined_conf = 0.0
    weight_sum = 0.0

    for hname in horizons:
        w = HORIZON_WEIGHTS.get(hname, 1.0 / len(horizons))
        combined_prob += w * results[hname]["prob"]
        combined_mag += w * results[hname]["magnitude"]
        combined_conf += w * results[hname]["confidence"]
        weight_sum += w

    if weight_sum > 0:
        combined_prob /= weight_sum
        combined_mag /= weight_sum
        combined_conf /= weight_sum

    results["combined"] = {
        "prob": combined_prob,
        "magnitude": combined_mag,
        "confidence": combined_conf,
    }

    return results


def get_regime_for_stock(
    sentiment_feats: np.ndarray,
    session_feats: np.ndarray,
) -> tuple[str, bool, str]:
    """Infer current market regime from sentiment/session features."""
    try:
        feat_dict = dict(zip(SENTIMENT_FEATURE_NAMES, sentiment_feats))
        vix = feat_dict.get("vix_level", 18.0)
        vix_change = feat_dict.get("vix_change", 0.0)
        nifty_ret = feat_dict.get("nifty_intraday_return", 0.0)
        overnight = session_feats[4] if len(session_feats) > 4 else 0.0
        gap_pct = (session_feats[5] * 100) if len(session_feats) > 5 else 0.0
        is_expiry = bool(session_feats[11]) if len(session_feats) > 11 else False

        nifty_returns_10d = np.array([nifty_ret] * 10)
        regime, should_trade, reason = detect_regime(
            vix=vix,
            vix_change=vix_change,
            nifty_returns_10d=nifty_returns_10d,
            gap_pct=gap_pct,
            is_expiry=is_expiry,
            config=RegimeConfig(),
        )
        return regime.value, should_trade, reason
    except Exception:
        return "calm_bull", True, "feature unavailable"


def parse_args():
    parser = argparse.ArgumentParser(description="Morning Picks V2 — LightGBM V2 pre-market recommendations")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="runs/lgbm_v2/",
        help="Directory containing LightGBM .lgb model files",
    )
    parser.add_argument(
        "--horizon",
        type=str,
        default="H60",
        choices=["H15", "H30", "H60", "ALL"],
        help="Prediction horizon (default: H60, use ALL to aggregate)",
    )
    parser.add_argument("--top-n", type=int, default=5, help="Top N picks per direction")
    parser.add_argument(
        "--dir-threshold",
        type=float,
        default=0.60,
        help="Direction probability threshold (default: 0.60)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.55,
        help="Minimum confidence filter (default: 0.55)",
    )
    parser.add_argument("--max-price", type=float, default=0, help="Max stock price (0=no filter)")
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=0.01,
        help="Stop-loss as fraction (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        nargs="?",
        const="default",
        default="default",
        help="Save picks to CSV. Use 'none' to disable.",
    )
    parser.add_argument("--no-download", action="store_true", help="Skip yfinance download")
    parser.add_argument(
        "--date",
        type=str,
        default="",
        help="Picks for this date (YYYY-MM-DD). Uses data up to the day before.",
    )
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=0,
        help="Limit stocks analyzed (0=all)",
    )
    parser.add_argument(
        "--no-regime-filter",
        action="store_true",
        help="Disable market regime filter",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    minute_data_dir = Path("nifty500")

    if not model_dir.exists():
        console.print(f"[red]Model directory not found: {model_dir}[/red]")
        return

    horizons = HORIZONS if args.horizon == "ALL" else [int(args.horizon[1:])]
    horizon_names = [f"H{h}" for h in horizons]

    console.print(
        Panel.fit(
            f"[bold cyan]IntradayNet — Morning Picks V2[/bold cyan]\n"
            f"[dim]Horizon: {args.horizon} | "
            f"Top {args.top_n} per direction\n"
            f"Dir threshold: {args.dir_threshold:.2f} | "
            f"Min confidence: {args.min_confidence:.2f} | "
            f"SL: {args.stop_loss:.1%}[/dim]",
            border_style="cyan",
        )
    )

    console.print("\n[bold]Step 1:[/bold] Loading LightGBM models...")
    dir_models, mag_models, available_horizons = load_lgbm_models(model_dir)

    if not available_horizons:
        console.print("[red]No LightGBM models found in directory.[/red]")
        return

    active_horizons = [h for h in horizon_names if h in available_horizons]
    if not active_horizons:
        active_horizons = available_horizons

    console.print(
        f"  Direction models: [green]{len(dir_models)}[/green] "
        f"({', '.join(sorted(dir_models.keys()))})"
    )
    console.print(
        f"  Magnitude models: [green]{len(mag_models)}[/green] "
        f"({', '.join(sorted(mag_models.keys()))})"
    )
    console.print(f"  Active horizons: [cyan]{active_horizons}[/cyan]")

    symbols = _get_stock_symbols(minute_data_dir)
    if args.max_stocks > 0:
        symbols = symbols[: args.max_stocks]
    console.print(f"\n  Stock universe: [green]{len(symbols)}[/green]")

    if not args.no_download:
        console.print(
            "\n[bold]Step 2:[/bold] Downloading latest data from yfinance..."
        )
        updated, failed = _download_latest_data(symbols, minute_data_dir, args.date)
        console.print(
            f"  Updated: [green]{updated}[/green] | Failed: [yellow]{failed}[/yellow]"
        )

        console.print("  Fetching official daily close prices...")
        daily_closes = _download_daily_closes(symbols, args.date)
        console.print(f"  Daily closes: [green]{len(daily_closes)}[/green] stocks")
    else:
        console.print("\n[dim]Skipping download (--no-download)[/dim]")
        daily_closes = {}

    console.print("\n[bold]Step 3:[/bold] Downloading macro data...")
    market_builder = MarketFeatureBuilder(cache_dir="market_data_cache")
    if not args.no_download:
        market_builder.download()

    sentiment_csv = Path("sentiment_data/daily_sentiment.csv")
    sentiment_builder = SentimentFeatureBuilder(
        str(sentiment_csv) if sentiment_csv.exists() else None,
        market_builder=market_builder,
    )

    console.print("\n[bold]Step 4:[/bold] Computing features & running inference...")

    all_picks = []
    skipped_price = 0
    skipped_data = 0
    skipped_regime = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Analyzing stocks...", total=len(symbols))

        for symbol in symbols:
            progress.update(
                task, description=f"Analyzing {symbol}..."
            )

            result = compute_features_for_stock(
                symbol=symbol,
                data_dir=minute_data_dir,
                sentiment_builder=sentiment_builder,
                market_builder=market_builder,
                target_date=args.date if args.date else None,
                daily_closes=daily_closes,
            )

            if result is None:
                skipped_data += 1
                progress.update(task, advance=1)
                continue

            if args.max_price > 0 and result["last_close"] > args.max_price:
                skipped_price += 1
                progress.update(task, advance=1)
                continue

            if not args.no_regime_filter:
                regime_val, should_trade, regime_reason = get_regime_for_stock(
                    result["sentiment_feats"], result["session_feats"]
                )
                if not should_trade:
                    skipped_regime += 1
                    progress.update(task, advance=1)
                    continue

            preds = infer_single_stock(
                window=result["window"],
                session_feats=result["session_feats"],
                sentiment_feats=result["sentiment_feats"],
                dir_models=dir_models,
                mag_models=mag_models,
                horizons=active_horizons,
            )

            combined = preds["combined"]

            regime_val, should_trade, regime_reason = get_regime_for_stock(
                result["sentiment_feats"], result["session_feats"]
            )

            all_picks.append(
                {
                    "symbol": symbol,
                    "last_close": result["last_close"],
                    "ref_date": result["ref_date"],
                    "prob": combined["prob"],
                    "magnitude": combined["magnitude"],
                    "confidence": combined["confidence"],
                    "regime": regime_val,
                    "regime_reason": regime_reason,
                    "horizon_details": {
                        h: {
                            "prob": preds[h]["prob"],
                            "magnitude": preds[h]["magnitude"],
                            "confidence": preds[h]["confidence"],
                        }
                        for h in active_horizons
                        if h in preds
                    },
                }
            )

            progress.update(task, advance=1)

    console.print(
        f"\n  Predictions: [green]{len(all_picks)}[/green] stocks"
    )
    if skipped_price > 0:
        console.print(
            f"  Filtered by price (> ₹{args.max_price:,.0f}): [yellow]{skipped_price}[/yellow]"
        )
    if skipped_data > 0:
        console.print(f"  Skipped (no data): [yellow]{skipped_data}[/yellow]")
    if skipped_regime > 0:
        console.print(
            f"  Filtered by regime: [yellow]{skipped_regime}[/yellow]"
        )

    longs, shorts = [], []
    for pick in all_picks:
        if pick["confidence"] < args.min_confidence:
            continue
        if pick["prob"] >= args.dir_threshold:
            pick["direction"] = "LONG"
            pick["score"] = pick["confidence"] * abs(pick["magnitude"])
            longs.append(pick)
        elif pick["prob"] <= (1 - args.dir_threshold):
            pick["direction"] = "SHORT"
            pick["score"] = pick["confidence"] * abs(pick["magnitude"])
            shorts.append(pick)

    longs.sort(key=lambda x: x["score"], reverse=True)
    shorts.sort(key=lambda x: x["score"], reverse=True)
    top_longs = longs[: args.top_n]
    top_shorts = shorts[: args.top_n]

    if args.date:
        picks_for_date = args.date
        try:
            pfd = datetime.strptime(args.date[:10], "%Y-%m-%d")
            picks_for_date = pfd.strftime("%Y-%m-%d (%A)")
        except ValueError:
            pass
    else:
        ref_date_display = (
            top_longs[0]["ref_date"]
            if top_longs
            else (
                top_shorts[0]["ref_date"]
                if top_shorts
                else (all_picks[0]["ref_date"] if all_picks else "N/A")
            )
        )
        picks_for_date = (
            _next_trading_day(ref_date_display)
            if ref_date_display != "N/A"
            else "N/A"
        )

    regime_val = (
        all_picks[0]["regime"] if all_picks else "unknown"
    )
    regime_reason = (
        all_picks[0]["regime_reason"] if all_picks else ""
    )

    console.print()
    console.print(
        Panel.fit(
            f"[bold green]📅 Picks for: {picks_for_date}[/bold green]\n"
            f"[dim]Based on data from: {all_picks[0]['ref_date'] if all_picks else 'N/A'}"
            f"[/dim]\n"
            f"[dim]Regime: {regime_val} — {regime_reason}[/dim]",
            border_style="green",
        )
    )
    console.print(
        f"  Qualified: [green]{len(longs)}[/green] LONG, [red]{len(shorts)}[/red] SHORT\n"
    )

    if top_longs:
        lt = Table(
            title=f"🟢 TOP {len(top_longs)} LONG PICKS",
            show_header=True,
            header_style="bold green",
        )
        lt.add_column("#", style="dim", width=3)
        lt.add_column("Stock", style="bold")
        lt.add_column("Close (₹)", justify="right", style="dim")
        lt.add_column("Entry (₹)", justify="right")
        lt.add_column("Target (₹)", justify="right", style="green")
        lt.add_column("SL (₹)", justify="right", style="red")
        lt.add_column("Move %", justify="right", style="green")
        lt.add_column("Conf", justify="right")
        lt.add_column("Score", justify="right", style="cyan")
        lt.add_column("Horizon", justify="right", style="dim")

        for i, pick in enumerate(top_longs):
            entry = pick["last_close"]
            mag = abs(pick["magnitude"])
            target = entry * (1 + mag)
            sl = entry * (1 - args.stop_loss)
            best_h = max(
                pick["horizon_details"].items(),
                key=lambda x: x[1]["confidence"],
            )[0]

            lt.add_row(
                str(i + 1),
                pick["symbol"],
                f"{pick['last_close']:,.2f}",
                f"{entry:,.2f}",
                f"{target:,.2f}",
                f"{sl:,.2f}",
                f"+{mag*100:.2f}%",
                f"{pick['confidence']:.2f}",
                f"{pick['score']:.4f}",
                best_h,
            )
        console.print(lt)
    else:
        console.print("[yellow]No LONG picks passed the filters.[/yellow]")

    console.print()

    if top_shorts:
        st = Table(
            title=f"🔴 TOP {len(top_shorts)} SHORT PICKS",
            show_header=True,
            header_style="bold red",
        )
        st.add_column("#", style="dim", width=3)
        st.add_column("Stock", style="bold")
        st.add_column("Close (₹)", justify="right", style="dim")
        st.add_column("Entry (₹)", justify="right")
        st.add_column("Target (₹)", justify="right", style="green")
        st.add_column("SL (₹)", justify="right", style="red")
        st.add_column("Move %", justify="right", style="green")
        st.add_column("Conf", justify="right")
        st.add_column("Score", justify="right", style="cyan")
        st.add_column("Horizon", justify="right", style="dim")

        for i, pick in enumerate(top_shorts):
            entry = pick["last_close"]
            mag = abs(pick["magnitude"])
            target = entry * (1 - mag)
            sl = entry * (1 + args.stop_loss)
            best_h = max(
                pick["horizon_details"].items(),
                key=lambda x: x[1]["confidence"],
            )[0]

            st.add_row(
                str(i + 1),
                pick["symbol"],
                f"{pick['last_close']:,.2f}",
                f"{entry:,.2f}",
                f"{target:,.2f}",
                f"{sl:,.2f}",
                f"+{mag*100:.2f}%",
                f"{pick['confidence']:.2f}",
                f"{pick['score']:.4f}",
                best_h,
            )
        console.print(st)
    else:
        console.print("[yellow]No SHORT picks passed the filters.[/yellow]")

    all_top = top_longs + top_shorts
    if all_top:
        avg_target = np.mean([abs(p["magnitude"]) * 100 for p in all_top])
        avg_conf = np.mean([p["confidence"] for p in all_top])
        rr_ratio = avg_target / (args.stop_loss * 100) if args.stop_loss > 0 else 0

        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Avg predicted move: [cyan]{avg_target:.2f}%[/cyan]")
        console.print(f"  Avg confidence: [cyan]{avg_conf:.2f}[/cyan]")
        console.print(
            f"  Risk:Reward: [cyan]1:{rr_ratio:.1f}[/cyan] (SL={args.stop_loss:.1%})"
        )
        console.print(
            f"  Market regime: [cyan]{regime_val}[/cyan] — {regime_reason}\n"
        )
        console.print(
            f"  [bold]Place orders at market open (9:15 AM)[/bold] on [bold]{picks_for_date}[/bold]\n"
        )

    if args.save_csv and args.save_csv.lower() != "none" and all_top:
        if args.save_csv == "default":
            rec_dir = Path("recommendations")
            rec_dir.mkdir(exist_ok=True, parents=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = rec_dir / f"picks_v2_{timestamp}.csv"
        else:
            csv_path = Path(args.save_csv)
            csv_path.parent.mkdir(exist_ok=True, parents=True)

        rows = []
        for pick in top_longs + top_shorts:
            entry = pick["last_close"]
            mag = abs(pick["magnitude"])
            if pick["direction"] == "LONG":
                target = entry * (1 + mag)
                sl = entry * (1 - args.stop_loss)
            else:
                target = entry * (1 - mag)
                sl = entry * (1 + args.stop_loss)

            best_h = max(
                pick["horizon_details"].items(),
                key=lambda x: x[1]["confidence"],
            )[0]

            rows.append(
                {
                    "picks_for_date": picks_for_date.split(" (")[0]
                    if picks_for_date != "N/A"
                        else "",
                    "stock": pick["symbol"],
                    "direction": pick["direction"],
                    "last_close": round(pick["last_close"], 2),
                    "entry_price": round(entry, 2),
                    "target_price": round(target, 2),
                    "stop_loss": round(sl, 2),
                    "predicted_move_pct": round(mag * 100, 2),
                    "confidence": round(pick["confidence"], 3),
                    "score": round(pick["score"], 4),
                    "ref_date": pick["ref_date"],
                    "horizon": best_h,
                    "model_type": "lgbm_v2",
                    "regime": pick["regime"],
                }
            )

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        console.print(f"[bold green]✓ Picks saved to {csv_path}[/bold green]\n")


if __name__ == "__main__":
    main()
