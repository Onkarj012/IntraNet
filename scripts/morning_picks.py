#!/usr/bin/env python3
"""
Morning Picks — Pre-market recommendation engine with LIVE data.

Run before market open → downloads latest data via yfinance →
computes features → generates 5-10 LONG + SHORT picks with entry/target/SL.
Place orders at market open and walk away.

Usage:
    python scripts/morning_picks.py --model runs/intraday/resnls/best_model.pt
    python scripts/morning_picks.py --model runs/lgbm/                              # LightGBM
    python scripts/morning_picks.py --model runs/intraday/resnls/best_model.pt --max-price 500 --top-n 5
    python scripts/morning_picks.py --model runs/intraday/resnls/best_model.pt --no-download  # skip yfinance
"""

import argparse
import sys
import csv
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from intradaynet.config import load_config
from intradaynet.features.per_bar_features import compute_per_bar_features, PER_BAR_FEATURE_NAMES
from intradaynet.features.session_features import compute_session_features, SESSION_FEATURE_NAMES
from intradaynet.features.sentiment_features import SentimentFeatureBuilder, SENTIMENT_FEATURE_NAMES

warnings.filterwarnings("ignore", category=FutureWarning)
console = Console()


# ── Utilities ────────────────────────────────────────────────────────────────

def _next_trading_day(date_str: str) -> str:
    """Get the next trading day (skip weekends). date_str: 'YYYY-MM-DD'."""
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return "next trading day"
    dt += timedelta(days=1)
    while dt.weekday() >= 5:
        dt += timedelta(days=1)
    return dt.strftime("%Y-%m-%d (%A)")


def _previous_trading_day(dt: datetime) -> datetime:
    """Get the previous trading day (skip weekends)."""
    prev_dt = dt - timedelta(days=1)
    while prev_dt.weekday() >= 5:
        prev_dt -= timedelta(days=1)
    return prev_dt


def _get_stock_symbols(minute_data_dir: Path) -> list:
    """Get list of stock symbols from minute data CSVs."""
    return sorted([
        f.stem.replace("_minute", "")
        for f in minute_data_dir.glob("*_minute.csv")
    ])


# ── Live Data Download ───────────────────────────────────────────────────────

def _read_last_date_from_csv(csv_path: Path):
    """Efficiently read only the last timestamp from a minute CSV without parsing the whole file."""
    try:
        # Read the last ~4KB of the file to get the last few lines
        file_size = csv_path.stat().st_size
        with open(csv_path, "rb") as f:
            # Seek to near the end (last 4KB is plenty for a few CSV rows)
            f.seek(max(0, file_size - 4096))
            tail = f.read().decode("utf-8", errors="ignore")
        
        lines = tail.strip().split("\n")
        if len(lines) < 2:
            return None
        
        # Last non-empty line; first column is the date
        last_line = lines[-1].strip()
        if not last_line:
            last_line = lines[-2].strip() if len(lines) > 2 else None
        if not last_line:
            return None
        
        date_str = last_line.split(",")[0]
        return pd.Timestamp(date_str)
    except Exception:
        return None


def download_latest_data(symbols: list, minute_data_dir: Path, days_back: int = 7, target_date: str = None):
    """
    Download latest minute data from yfinance and APPEND to existing CSVs.

    yfinance limits: free tier gives 7 days of 1m data, 60 days of 2m.
    We download 7 days of 1m data for each stock.
    
    Returns:
        updated (int): Number of CSVs updated
        failed (int): Number of downloads failed
    """
    import yfinance as yf

    # If target_date is given, we download up to that date (inclusive)
    end_date_fetch = datetime.now()
    if target_date:
        try:
            # target_date is e.g. '2026-03-12', representing the date we want picks FOR.
            # So we need data up to the end of the PREVIOUS day (2026-03-11).
            # Which means end_date should be 2026-03-12 00:00:00 (since end is exclusive in yfinance)
            req_date = datetime.strptime(target_date[:10], "%Y-%m-%d")
            end_date_fetch = req_date
        except ValueError:
            pass

    updated = 0
    failed = 0

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(), console=console, transient=True,
    ) as progress:
        task = progress.add_task("Downloading latest data...", total=len(symbols))

        for symbol in symbols:
            progress.update(task, description=f"Downloading {symbol}...")
            csv_path = minute_data_dir / f"{symbol}_minute.csv"

            try:
                # Determine the last date in existing data (read only tail of file)
                last_date = None
                if csv_path.exists():
                    last_date = _read_last_date_from_csv(csv_path)

                # Download from yfinance (NSE symbol needs .NS suffix)
                yf_symbol = f"{symbol}.NS"
                ticker = yf.Ticker(yf_symbol)

                # yfinance 1m data: max 7 days
                end_date = end_date_fetch
                start_date = end_date - timedelta(days=days_back)

                # If we have recent data already, download starting from the previous trading day
                if last_date is not None:
                    last_dt = pd.Timestamp(last_date)
                    # Start from previous trading day before last date to handle overlap
                    start_date = max(start_date, _previous_trading_day(last_dt))

                df = ticker.history(
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    interval="1m",
                )

                if df.empty:
                    progress.update(task, advance=1)
                    failed += 1
                    continue

                # Format to match existing CSV structure
                df = df.reset_index()
                df = df.rename(columns={
                    "Datetime": "date", "Open": "open", "High": "high",
                    "Low": "low", "Close": "close", "Volume": "volume",
                })
                df = df[["date", "open", "high", "low", "close", "volume"]]
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

                if csv_path.exists() and last_date is not None:
                    # Append only new rows (avoid duplicates)
                    df = df[df["date"] > last_date]
                    if len(df) > 0:
                        df.to_csv(csv_path, mode="a", header=False, index=False)
                        updated += 1
                else:
                    df.to_csv(csv_path, index=False)
                    updated += 1

            except Exception:
                failed += 1

            progress.update(task, advance=1)

    return updated, failed


def download_daily_closes(symbols: list, end_date: str = None) -> dict:
    """
    Download official daily close prices from yfinance.
    The NSE closing price comes from the closing auction (15:30-15:40),
    which is NOT captured in 1-minute candle data (ends at 15:29).
    
    Returns:
        dict: {symbol: {date_str: close_price}}
    """
    import yfinance as yf

    if end_date is None:
        end_dt = datetime.now() + timedelta(days=1)
    else:
        try:
            end_dt = datetime.strptime(end_date[:10], "%Y-%m-%d") + timedelta(days=1)
        except ValueError:
            end_dt = datetime.now() + timedelta(days=1)

    start_dt = end_dt - timedelta(days=10)  # ~7 trading days back
    daily_closes = {}

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(), console=console, transient=True,
    ) as progress:
        task = progress.add_task("Fetching daily closes...", total=len(symbols))

        for symbol in symbols:
            progress.update(task, description=f"Daily close {symbol}...")
            try:
                ticker = yf.Ticker(f"{symbol}.NS")
                df = ticker.history(
                    start=start_dt.strftime("%Y-%m-%d"),
                    end=end_dt.strftime("%Y-%m-%d"),
                    interval="1d",
                )
                if not df.empty:
                    closes = {}
                    for idx, row in df.iterrows():
                        day_str = idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx)[:10]
                        closes[day_str] = float(row["Close"])
                    daily_closes[symbol] = closes
            except Exception:
                pass
            progress.update(task, advance=1)

    return daily_closes


# ── Feature Computation (on-the-fly) ─────────────────────────────────────────

def compute_features_for_stock(symbol: str, minute_data_dir: Path,
                               sentiment_builder, market_open: str,
                            market_close: str, seq_length: int,
                            target_date: str = None, daily_closes: dict = None):
    """
    Compute features for a single stock from its minute CSV.
    Returns (per_bar_window, session_feats, sentiment_feats, last_close, ref_date) or None.
    """
    csv_path = minute_data_dir / f"{symbol}_minute.csv"
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
        df.columns = df.columns.str.lower()

        # Filter to market hours
        df["_time"] = df.index.strftime("%H:%M")
        df = df[(df["_time"] >= market_open) & (df["_time"] < market_close)]
        df = df.drop(columns=["_time"])

        if len(df) < seq_length:
            return None
            
        # If target_date is provided, filter dataframe up to the PREVIOUS day of target_date
        # Because if we are running morning picks FOR target_date (e.g. 2026-03-13), 
        # we can only see data up to 2026-03-12 end of day.
        if target_date:
            try:
                td = pd.Timestamp(target_date[:10])
                df = df[df.index < td]
            except Exception:
                pass

        if len(df) < seq_length:
            return None

        # Compute features
        per_bar = compute_per_bar_features(df)
        session = compute_session_features(df)
        sentiment = sentiment_builder.get_features(symbol, session.index)

        # Get the last available date
        dates = per_bar.index.date
        unique_dates = sorted(set(dates))
        if not unique_dates:
            return None

        ref_date = str(unique_dates[-1])

        # Get last seq_length bars
        per_bar_values = per_bar[PER_BAR_FEATURE_NAMES].values.astype(np.float32)
        close_values = df["close"].values.astype(np.float32)

        if len(per_bar_values) < seq_length:
            return None

        window = per_bar_values[-seq_length:]

        # Use the official daily close (from closing auction) if available,
        # otherwise fall back to last minute bar close.
        # NSE closing auction (15:30-15:40) determines the official close price
        # which can differ significantly from the 15:29 minute bar close.
        last_close = float(close_values[-1])  # fallback: last minute bar
        if daily_closes and symbol in daily_closes:
            sym_daily = daily_closes[symbol]
            if ref_date in sym_daily:
                last_close = sym_daily[ref_date]

        if last_close <= 0:
            return None

        # Get session + sentiment for the last date
        sess_values = session[SESSION_FEATURE_NAMES].values.astype(np.float32)
        sent_values = sentiment[SENTIMENT_FEATURE_NAMES].values.astype(np.float32)

        s_feat = sess_values[-1] if len(sess_values) > 0 else np.zeros(len(SESSION_FEATURE_NAMES), dtype=np.float32)
        se_feat = sent_values[-1] if len(sent_values) > 0 else np.zeros(len(SENTIMENT_FEATURE_NAMES), dtype=np.float32)

        return {
            "window": window,
            "session_feats": s_feat,
            "sentiment_feats": se_feat,
            "last_close": last_close,
            "ref_date": ref_date,
        }

    except Exception:
        return None


# ── Model Loading ────────────────────────────────────────────────────────────

def load_model(checkpoint_path, cfg):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_type = checkpoint.get("model_type", "tcn_attention")

    # Detect sentiment feature count from checkpoint weights
    # (backward compat: old models trained with 14, new with 24)
    state_dict = checkpoint["model_state_dict"]
    ckpt_sent_features = cfg.model.num_sentiment_features
    for key, val in state_dict.items():
        if "sentiment_proj" in key and "weight" in key and val.dim() == 2:
            ckpt_sent_features = val.shape[1]
            break

    if model_type == "resnls":
        from intradaynet.models.resnls_intraday import IntradayResNLS
        model = IntradayResNLS(
            num_per_bar_features=cfg.model.num_per_bar_features,
            num_context_features=cfg.model.num_context_features,
            num_sentiment_features=ckpt_sent_features,
            hidden_dim=64, lstm_layers=2,
            num_horizons=len(cfg.horizons), dropout=cfg.model.dropout,
        )
    elif model_type == "compact_cnn":
        from intradaynet.models.compact_cnn import CompactCNN
        model = CompactCNN(
            num_per_bar_features=cfg.model.num_per_bar_features,
            num_context_features=cfg.model.num_context_features,
            num_sentiment_features=ckpt_sent_features,
            num_horizons=len(cfg.horizons), dropout=cfg.model.dropout,
        )
    elif model_type == "lightweight_gru":
        from intradaynet.models.lightweight_gru import LightweightGRU
        model = LightweightGRU(
            num_per_bar_features=cfg.model.num_per_bar_features,
            num_context_features=cfg.model.num_context_features,
            num_sentiment_features=ckpt_sent_features,
            hidden_dim=48,
            num_horizons=len(cfg.horizons), dropout=cfg.model.dropout,
        )
    elif model_type == "mlp_mixer":
        from intradaynet.models.mlp_mixer import IntradayMLPMixer
        model = IntradayMLPMixer(
            num_per_bar_features=cfg.model.num_per_bar_features,
            num_context_features=cfg.model.num_context_features,
            num_sentiment_features=ckpt_sent_features,
            patch_size=15, hidden_dim=64, num_mixer_blocks=3,
            num_horizons=len(cfg.horizons), dropout=cfg.model.dropout,
        )
    else:
        from intradaynet.models.tcn_attention import IntradayTCNAttention
        model = IntradayTCNAttention(
            num_per_bar_features=cfg.model.num_per_bar_features,
            num_context_features=cfg.model.num_context_features,
            num_sentiment_features=ckpt_sent_features,
            hidden_dim=cfg.model.hidden_dim,
            tcn_channels=cfg.model.tcn.channels,
            kernel_size=cfg.model.tcn.kernel_size,
            dilation_base=cfg.model.tcn.dilation_base,
            attn_heads=cfg.model.attn_heads,
            num_horizons=len(cfg.horizons), dropout=cfg.model.dropout,
        )

    model.load_state_dict(state_dict)
    model.eval()

    if ckpt_sent_features != cfg.model.num_sentiment_features:
        console.print(f"  [yellow]Note: checkpoint trained with {ckpt_sent_features} sentiment features "
                       f"(current config: {cfg.model.num_sentiment_features}). "
                       f"Retrain to use all features.[/yellow]")

    return model, model_type, checkpoint.get("epoch", "?"), ckpt_sent_features


# ── LightGBM Support ─────────────────────────────────────────────────────────

def is_lgbm_model(model_path):
    """Check if model path is a LightGBM directory."""
    p = Path(model_path)
    return p.is_dir() and any(p.glob("*.lgb"))


def flatten_for_lgbm(window, session_feats, sentiment_feats):
    """Flatten features for a single stock for LightGBM inference."""
    per_bar = window[np.newaxis]  # (1, L, F)
    N, L, F = per_bar.shape
    agg = []
    for w in [5, 30, min(L, 120)]:
        wdata = per_bar[:, -w:, :]
        agg.append(np.nanmean(wdata, axis=1))
        agg.append(np.nanstd(wdata, axis=1))
        if w > 1:
            agg.append((wdata[:, -1, :] - wdata[:, 0, :]) / w)
    agg.append(per_bar[:, -1, :])
    agg.append(np.nanmin(per_bar, axis=1))
    agg.append(np.nanmax(per_bar, axis=1))
    flat = np.concatenate(agg + [session_feats[np.newaxis], sentiment_feats[np.newaxis]], axis=1)
    return np.nan_to_num(flat, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)


def load_lgbm_models(model_dir, horizon_names):
    """Load LightGBM booster models from directory."""
    import lightgbm as lgb
    model_dir = Path(model_dir)
    dir_models, mag_models = {}, {}
    for hname in horizon_names:
        dp = model_dir / f"dir_{hname}.lgb"
        mp = model_dir / f"mag_{hname}.lgb"
        if dp.exists():
            dir_models[hname] = lgb.Booster(model_file=str(dp))
        if mp.exists():
            mag_models[hname] = lgb.Booster(model_file=str(mp))
    return dir_models, mag_models


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Morning Picks — Pre-market Recommendations")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to .pt file (PyTorch) or directory with .lgb files (LightGBM)")
    parser.add_argument("--config", type=str, default="configs/intraday_config.yaml")
    parser.add_argument("--horizon", type=str, default="H60",
                        help="Prediction horizon (H15, H30, H60, H375)")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of picks per direction (default: 5)")
    parser.add_argument("--dir-threshold", type=float, default=0.60,
                        help="Direction probability threshold (default: 0.60)")
    parser.add_argument("--min-confidence", type=float, default=0.55,
                        help="Minimum confidence filter (default: 0.55)")
    parser.add_argument("--max-price", type=float, default=0,
                        help="Max stock price filter (0 = no filter)")
    parser.add_argument("--stop-loss", type=float, default=0.01,
                        help="Stop-loss as fraction (default: 1%%)")
    parser.add_argument("--save-csv", type=str, nargs='?', const="default", default="default",
                        help="Save picks to CSV file. Defaults to recommendations/picks_YYYYMMDD_HHMMSS.csv. Pass 'none' to disable." )
    parser.add_argument("--no-download", action="store_true",
                        help="Skip yfinance download (use existing data only)")
    parser.add_argument("--date", type=str, default="",
                        help="Picks for this date (YYYY-MM-DD). E.g. --date 2026-03-12 uses data up to 2026-03-11. "
                             "If omitted, auto-detects next trading day from latest data.")
    parser.add_argument("--max-stocks", type=int, default=0,
                        help="Limit number of stocks to analyze (0 = all)")
    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config(args.config)
    seq_length = cfg.model.sequence_length
    minute_data_dir = Path(cfg.data.minute_data_dir)

    # Parse horizon
    horizon_map = {f"H{h}": (i, h) for i, h in enumerate(cfg.horizons)}
    if args.horizon not in horizon_map:
        console.print(f"[red]Unknown horizon: {args.horizon}. Choose from {list(horizon_map.keys())}[/red]")
        return
    horizon_idx, horizon_bars = horizon_map[args.horizon]

    # Header
    console.print(Panel.fit(
        f"[bold cyan]IntradayNet — Morning Picks[/bold cyan]\n"
        f"[dim]Horizon: {args.horizon} ({horizon_bars} mins) | "
        f"Top {args.top_n} per direction\n"
        f"Dir threshold: {args.dir_threshold:.2f} | Min confidence: {args.min_confidence:.2f} | "
        f"SL: {args.stop_loss:.1%}[/dim]",
        border_style="cyan",
    ))

    # Get stock symbols
    symbols = _get_stock_symbols(minute_data_dir)
    if args.max_stocks > 0:
        symbols = symbols[:args.max_stocks]
    console.print(f"  Total stocks: [green]{len(symbols)}[/green]")

    # ── Step 1: Download latest data ──
    if not args.no_download:
        console.print("\n[bold]Step 1:[/bold] Downloading latest stock + macro data from yfinance...")
        updated, failed = download_latest_data(symbols, minute_data_dir, target_date=args.date if args.date else None)
        console.print(f"  Stocks updated: [green]{updated}[/green] | Failed: [yellow]{failed}[/yellow]")

        # Download official daily closes (NSE closing auction prices)
        console.print("  [dim]Fetching official daily close prices (NSE closing auction)...[/dim]")
        daily_closes = download_daily_closes(symbols, end_date=args.date if args.date else None)
        console.print(f"  Daily closes fetched: [green]{len(daily_closes)}[/green] stocks")
    else:
        console.print("\n[dim]Skipping download (--no-download)[/dim]")
        daily_closes = {}  # Will fall back to minute bar close

    # ── Step 2: Load model ──
    console.print("\n[bold]Step 2:[/bold] Loading model...")
    use_lgbm = is_lgbm_model(args.model)

    if use_lgbm:
        horizon_names = [f"H{h}" for h in cfg.horizons]
        lgbm_dir_models, lgbm_mag_models = load_lgbm_models(args.model, horizon_names)
        console.print(f"  Model: [green]LightGBM[/green] ({len(lgbm_dir_models)} direction + {len(lgbm_mag_models)} magnitude models)")
        ckpt_sent_features = cfg.model.num_sentiment_features  # LightGBM uses all features
    else:
        model, model_type, epoch, ckpt_sent_features = load_model(args.model, cfg)
        console.print(f"  Model: [green]{model_type}[/green] (epoch {epoch})")

    # ── Step 3: Compute features & generate predictions ──
    console.print(f"\n[bold]Step 3:[/bold] Computing features & predictions...")

    # Initialize market feature builder (downloads VIX, NIFTY, crude, gold, etc.)
    from intradaynet.features.market_features import MarketFeatureBuilder
    market_builder = MarketFeatureBuilder(cache_dir="market_data_cache")
    if not args.no_download:
        console.print("  [dim]Downloading macro data (VIX, crude, gold, USD/INR, global indices)...[/dim]")
        market_builder.download()

    sentiment_builder = SentimentFeatureBuilder(cfg.data.sentiment_csv,
                                                market_builder=market_builder)

    all_picks = []
    skipped_price = 0
    skipped_data = 0

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(), console=console, transient=True,
    ) as progress:
        task = progress.add_task("Analyzing stocks...", total=len(symbols))

        for symbol in symbols:
            progress.update(task, description=f"Analyzing {symbol}...")

            result = compute_features_for_stock(
                symbol, minute_data_dir, sentiment_builder,
                cfg.data.market_open, cfg.data.market_close, seq_length,
                target_date=args.date if args.date else None,
                daily_closes=daily_closes,
            )

            if result is None:
                skipped_data += 1
                progress.update(task, advance=1)
                continue

            # Max price filter
            if args.max_price > 0 and result["last_close"] > args.max_price:
                skipped_price += 1
                progress.update(task, advance=1)
                continue

            # Run model inference
            if use_lgbm:
                X_flat = flatten_for_lgbm(
                    result["window"], result["session_feats"], result["sentiment_feats"]
                )
                hname = args.horizon
                prob = float(lgbm_dir_models[hname].predict(X_flat)[0]) if hname in lgbm_dir_models else 0.5
                magnitude = float(lgbm_mag_models[hname].predict(X_flat)[0]) if hname in lgbm_mag_models else 0.0
                confidence = abs(prob - 0.5) * 2  # proxy confidence
            else:
                per_bar_t = torch.from_numpy(result["window"][np.newaxis].astype(np.float32))
                context_t = torch.from_numpy(result["session_feats"][np.newaxis].astype(np.float32))
                sentiment_t = torch.from_numpy(result["sentiment_feats"][np.newaxis].astype(np.float32))

                # Truncate sentiment features for old checkpoints (14 vs 24)
                if sentiment_t.shape[-1] > ckpt_sent_features:
                    sentiment_t = sentiment_t[:, :ckpt_sent_features]

                with torch.no_grad():
                    preds = model(per_bar_t, context_t, sentiment_t)

                prob = float(torch.sigmoid(preds["direction_logits"][0, horizon_idx]).item())
                magnitude = float(preds["magnitudes"][0, horizon_idx].item())
                confidence = float(preds["confidences"][0, horizon_idx].item())

            all_picks.append({
                "symbol": symbol,
                "last_close": result["last_close"],
                "ref_date": result["ref_date"],
                "prob": prob,
                "magnitude": magnitude,
                "confidence": confidence,
            })

            progress.update(task, advance=1)

    console.print(f"  Predictions: [green]{len(all_picks)}[/green] stocks")
    if skipped_price > 0:
        console.print(f"  Filtered by price (> ₹{args.max_price:,.0f}): [yellow]{skipped_price}[/yellow]")
    if skipped_data > 0:
        console.print(f"  Skipped (no data): [yellow]{skipped_data}[/yellow]")

    # ── Step 4: Filter, rank, and display ──
    longs = []
    shorts = []

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
    top_longs = longs[:args.top_n]
    top_shorts = shorts[:args.top_n]

    # ── Date display ──
    if args.date:
        # User specified picks-for date explicitly
        picks_for_date = args.date
        try:
            pfd = datetime.strptime(args.date[:10], "%Y-%m-%d")
            picks_for_date = pfd.strftime("%Y-%m-%d (%A)")
        except ValueError:
            pass
        ref_date_display = top_longs[0]["ref_date"] if top_longs else (
            top_shorts[0]["ref_date"] if top_shorts else (
                all_picks[0]["ref_date"] if all_picks else "N/A"
            )
        )
    else:
        # Auto-detect from data
        ref_date_display = top_longs[0]["ref_date"] if top_longs else (
            top_shorts[0]["ref_date"] if top_shorts else (
                all_picks[0]["ref_date"] if all_picks else "N/A"
            )
        )
        picks_for_date = _next_trading_day(ref_date_display) if ref_date_display != "N/A" else "N/A"

    console.print()
    console.print(Panel.fit(
        f"[bold green]📅 Picks for: {picks_for_date}[/bold green]\n"
        f"[dim]Based on data from: {ref_date_display}[/dim]",
        border_style="green",
    ))
    console.print(f"  Qualified: [green]{len(longs)}[/green] LONG, [red]{len(shorts)}[/red] SHORT\n")

    # ── LONG Picks Table ──
    if top_longs:
        long_table = Table(
            title=f"🟢 TOP {len(top_longs)} LONG PICKS",
            show_header=True, header_style="bold green",
        )
        long_table.add_column("#", style="dim", width=3)
        long_table.add_column("Stock", style="bold")
        long_table.add_column("Last Close (₹)", justify="right", style="dim")
        long_table.add_column("Entry (₹)", justify="right")
        long_table.add_column("Target (₹)", justify="right", style="green")
        long_table.add_column("Stop Loss (₹)", justify="right", style="red")
        long_table.add_column("Target %", justify="right", style="green")
        long_table.add_column("SL %", justify="right", style="red")
        long_table.add_column("Confidence", justify="right")
        long_table.add_column("Score", justify="right", style="cyan")

        for i, pick in enumerate(top_longs):
            entry = pick["last_close"]
            mag = abs(pick["magnitude"])
            target = entry * (1 + mag)
            sl = entry * (1 - args.stop_loss)

            long_table.add_row(
                str(i + 1), pick["symbol"],
                f"{pick['last_close']:,.2f}",
                f"{entry:,.2f}", f"{target:,.2f}", f"{sl:,.2f}",
                f"+{mag*100:.2f}%", f"-{args.stop_loss*100:.1f}%",
                f"{pick['confidence']:.2f}", f"{pick['score']:.4f}",
            )
        console.print(long_table)
    else:
        console.print("[yellow]No LONG picks passed the filters.[/yellow]")

    console.print()

    # ── SHORT Picks Table ──
    if top_shorts:
        short_table = Table(
            title=f"🔴 TOP {len(top_shorts)} SHORT PICKS",
            show_header=True, header_style="bold red",
        )
        short_table.add_column("#", style="dim", width=3)
        short_table.add_column("Stock", style="bold")
        short_table.add_column("Last Close (₹)", justify="right", style="dim")
        short_table.add_column("Entry (₹)", justify="right")
        short_table.add_column("Target (₹)", justify="right", style="green")
        short_table.add_column("Stop Loss (₹)", justify="right", style="red")
        short_table.add_column("Target %", justify="right", style="green")
        short_table.add_column("SL %", justify="right", style="red")
        short_table.add_column("Confidence", justify="right")
        short_table.add_column("Score", justify="right", style="cyan")

        for i, pick in enumerate(top_shorts):
            entry = pick["last_close"]
            mag = abs(pick["magnitude"])
            target = entry * (1 - mag)
            sl = entry * (1 + args.stop_loss)

            short_table.add_row(
                str(i + 1), pick["symbol"],
                f"{pick['last_close']:,.2f}",
                f"{entry:,.2f}", f"{target:,.2f}", f"{sl:,.2f}",
                f"+{mag*100:.2f}%", f"-{args.stop_loss*100:.1f}%",
                f"{pick['confidence']:.2f}", f"{pick['score']:.4f}",
            )
        console.print(short_table)
    else:
        console.print("[yellow]No SHORT picks passed the filters.[/yellow]")

    # ── Summary ──
    all_top = top_longs + top_shorts
    if all_top:
        avg_target = np.mean([abs(p["magnitude"]) * 100 for p in all_top])
        avg_conf = np.mean([p["confidence"] for p in all_top])
        rr_ratio = avg_target / (args.stop_loss * 100) if args.stop_loss > 0 else 0

        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Avg predicted move: [cyan]{avg_target:.2f}%[/cyan]")
        console.print(f"  Avg confidence: [cyan]{avg_conf:.2f}[/cyan]")
        console.print(f"  Risk:Reward ratio: [cyan]1:{rr_ratio:.1f}[/cyan] (SL={args.stop_loss:.1%})")
        console.print(f"  Place orders at [bold]market open (9:15 AM)[/bold] on [bold]{picks_for_date}[/bold], "
                       f"set target + SL, and walk away.\n")

    # ── Save CSV ──
    if args.save_csv and args.save_csv.lower() != "none" and all_top:
        if args.save_csv == "default":
            rec_dir = PROJECT_ROOT / "recommendations"
            rec_dir.mkdir(exist_ok=True, parents=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = rec_dir / f"picks_{timestamp}.csv"
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

            rows.append({
                "picks_for_date": picks_for_date.split(" (")[0] if picks_for_date != "N/A" else "",
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
                "horizon": args.horizon,
                "model_type": "lgbm" if use_lgbm else model_type,
            })

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        console.print(f"[bold green]✓ Picks saved to {csv_path}[/bold green]\n")


if __name__ == "__main__":
    main()
