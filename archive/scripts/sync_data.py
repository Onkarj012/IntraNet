#!/usr/bin/env python3
"""
Data Sync Script — download latest minute bars for all NSE stocks.

Usage:
    python scripts/sync_data.py                          # sync all stocks
    python scripts/sync_data.py --max-stocks 50          # test with 50 stocks
    python scripts/sync_data.py --dry-run                # show what would be synced
    python scripts/sync_data.py --full                   # re-download full history (slow)

Syncs:
    1. Minute bars for all stocks in nifty500/ directory
    2. NIFTY 50 index data
    3. India VIX data
    4. Macro data: crude oil, gold, USD/INR, US 10Y yield, DXY
    5. Global indices: S&P 500, NASDAQ, Dow Jones
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intradaynet.universe import get_universe

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("intradaynet.sync")

NSE_SUFFIX = ".NS"
DATA_DIR = PROJECT_ROOT / "data/nifty500"
MARKET_DATA_DIR = PROJECT_ROOT / "market_data_cache"
MARKET_DATA_DIR.mkdir(exist_ok=True, parents=True)

START_DATE = "2020-01-01"

NIFTY_SYMBOLS = {
    "^NSEI": "nifty50",
    "^NSEBANK": "banknifty",
    "^INDIAVIX": "india_vix",
}

MACRO_SYMBOLS = {
    "CL=F": "crude_oil",
    "GC=F": "gold",
    "INR=X": "usdinr",
    "^TNX": "us_10y_yield",
    "DXY": "dxy",
    "^GSPC": "sp500",
    "^IXIC": "nasdaq",
    "^DJI": "dow_jones",
    "^VIX": "us_vix",
    "BTC-INR": "btc_inr",
}

BATCH_SIZE = 50


def _get_stock_symbols(data_dir: Path) -> list[str]:
    if not data_dir.exists():
        return []
    return sorted(
        {p.stem.replace("_minute", "") for p in data_dir.glob("*_minute.csv")}
    )


def _get_last_date(csv_path: Path) -> str | None:
    """Get the last date in a CSV file (last row's date column)."""
    try:
        df = pd.read_csv(csv_path, usecols=["date"], nrows=5)
        if df.empty:
            return None
        date_col = df["date"].iloc[-1]
        return str(date_col)[:10]
    except Exception:
        return None


def _download_symbol_history(
    symbol: str,
    suffix: str,
    start_date: str,
    end_date: str,
    interval: str = "1m",
) -> pd.DataFrame | None:
    """Download history for a single symbol from yfinance.

    For 1m interval, Yahoo only allows ~8 days per request,
    so we paginate in 7-day chunks and concatenate.
    """
    import yfinance as yf

    ticker_str = f"{symbol}{suffix}" if suffix else symbol

    try:
        if interval == "1m":
            return _download_1m_paginated(ticker_str, start_date, end_date)
        else:
            ticker = yf.Ticker(ticker_str)
            df = ticker.history(start=start_date, end=end_date, interval=interval, auto_adjust=True)
            if df.empty or len(df) < 5:
                return None
            return _normalize_yf_dataframe(df)
    except Exception as e:
        logger.debug(f"Failed {symbol}: {e}")
        return None


def _download_1m_paginated(ticker_str: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """Download 1-minute data in 7-day chunks to respect Yahoo's ~8-day limit."""
    import yfinance as yf

    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    chunks = []
    chunk_start = start_dt
    while chunk_start < end_dt:
        chunk_end = min(chunk_start + timedelta(days=7), end_dt)
        if chunk_end <= chunk_start:
            break
        try:
            ticker = yf.Ticker(ticker_str)
            df = ticker.history(
                start=chunk_start.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
                interval="1m",
                auto_adjust=True,
            )
            if not df.empty and len(df) >= 5:
                df = _normalize_yf_dataframe(df)
                chunks.append(df)
        except Exception:
            pass
        chunk_start = chunk_end + timedelta(days=1)

    if not chunks:
        return None
    result = pd.concat(chunks).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return result if len(result) >= 5 else None


def _normalize_yf_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance DataFrame columns."""
    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    if "datetime" in df.columns:
        df = df.rename(columns={"datetime": "date"})
    elif "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "date"})
    elif "date" not in df.columns and df.columns[0].lower() in ("open", "date"):
        df = df.rename(columns={df.columns[0]: "date"})

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def _append_to_csv(df: pd.DataFrame, csv_path: Path) -> int:
    """Append new rows to CSV (no duplicates). Returns number of rows appended."""
    if df is None or df.empty:
        return 0

    if not csv_path.exists():
        df.to_csv(csv_path, index=False)
        return len(df)

    try:
        existing = pd.read_csv(csv_path, parse_dates=["date"], nrows=1)
        last_date = existing["date"].iloc[0]
        cutoff = last_date.strftime("%Y-%m-%d %H:%M:%S")
        new_rows = df[df["date"] > cutoff]
        if new_rows.empty:
            return 0
        new_rows.to_csv(csv_path, mode="a", header=False, index=False)
        return len(new_rows)
    except Exception:
        df.to_csv(csv_path, mode="a", header=False, index=False)
        return len(df)


def sync_stocks(
    symbols: list[str],
    data_dir: Path,
    start_date: str,
    end_date: str,
    dry_run: bool = False,
    batch_size: int = BATCH_SIZE,
) -> tuple[int, int, int]:
    """Sync all stock CSVs. Returns (updated, skipped, failed)."""
    import yfinance as yf

    updated, skipped, failed = 0, 0, 0
    total = len(symbols)

    for i, symbol in enumerate(symbols):
        csv_path = data_dir / f"{symbol}_minute.csv"
        status_text = ""

        if csv_path.exists():
            last_date = _get_last_date(csv_path)
            if last_date:
                if not dry_run:
                    df = _download_symbol_history(symbol, NSE_SUFFIX, last_date, end_date)
                    n = _append_to_csv(df, csv_path)
                    if n > 0:
                        updated += 1
                        status_text = f"updated +{n}"
                    else:
                        skipped += 1
                        status_text = "up to date"
                else:
                    skipped += 1
            else:
                if not dry_run:
                    df = _download_symbol_history(symbol, NSE_SUFFIX, start_date, end_date)
                    n = _append_to_csv(df, csv_path)
                    if n > 0:
                        updated += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1
        else:
            if not dry_run:
                df = _download_symbol_history(symbol, NSE_SUFFIX, start_date, end_date)
                n = _append_to_csv(df, csv_path)
                if n > 0:
                    updated += 1
                    status_text = f"created ({n} rows)"
                else:
                    failed += 1
                    status_text = "no data"
            else:
                skipped += 1

        console.print(f"  [{i+1}/{total}] {symbol} {status_text}")

    return updated, skipped, failed


def sync_market_data(
    end_date: str,
    dry_run: bool = False,
) -> dict[str, int]:
    """Sync index and macro data. Returns dict of symbol -> rows synced."""
    results = {}
    all_symbols = {**NIFTY_SYMBOLS, **MACRO_SYMBOLS}

    for symbol, name in all_symbols.items():
        out_path = MARKET_DATA_DIR / f"{name}.csv"
        suffix = NSE_SUFFIX if symbol.startswith("^") or "NS" in symbol else ""

        last_date = _get_last_date(out_path) if out_path.exists() else None
        fetch_from = last_date if last_date else START_DATE

        if dry_run:
            console.print(f"  [yellow]DRY RUN:[/yellow] {symbol} ({name}) — would fetch from {fetch_from}")
            results[name] = 0
            continue

        console.print(f"  Syncing {symbol} ({name})...")
        interval = "1m" if "vix" in name.lower() else "1d"
        df = _download_symbol_history(symbol, suffix, fetch_from, end_date, interval)

        if df is None or df.empty:
            console.print(f"    [yellow]No data[/yellow]")
            results[name] = 0
            continue

        n = _append_to_csv(df, out_path)
        console.print(f"    [green]+{n} rows[/green] → {out_path.name}")
        results[name] = n

    return results


def sync_nifty500_universe(data_dir: Path) -> tuple[list[str], list[str]]:
    """Ensure we have the Nifty 500 universe list. Returns (symbols_to_add, symbols_to_remove)."""
    console.print("  [dim]Nifty 500 universe managed by nifty500/ directory.[/dim]")
    console.print(
        f"  [dim]To update universe: download latest list from NSE and add CSVs to {data_dir}[/dim]"
    )
    return [], []


def parse_args():
    parser = argparse.ArgumentParser(description="Sync latest market data from yfinance")
    parser.add_argument(
        "--universe",
        type=str,
        default=None,
        help="Seed download for a specific universe (nifty50, nifty100, nifty200, nifty500)",
    )
    parser.add_argument("--max-stocks", type=int, default=0, help="Limit stocks (0=all)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be synced")
    parser.add_argument("--full", action="store_true", help="Re-download full history (slow)")
    parser.add_argument(
        "--start-date",
        type=str,
        default=START_DATE,
        help=f"Start date for full sync (default: {START_DATE})",
    )
    parser.add_argument("--no-macro", action="store_true", help="Skip macro/index data")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers (yfinance handles threading internally)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    console.print(
        f"\n[bold cyan]IntradayNet Data Sync — {datetime.now().strftime('%Y-%m-%d %H:%M')}[/bold cyan]"
    )

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = args.start_date if args.full else "2020-01-01"

    if args.universe:
        symbols = get_universe(args.universe)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        console.print(f"  Seeding universe [cyan]{args.universe}[/cyan] with [green]{len(symbols)}[/green] stocks")
    else:
        symbols = _get_stock_symbols(DATA_DIR)
        if not symbols:
            console.print("[yellow]No stock CSVs found in nifty500/. Nothing to sync.[/yellow]")
            console.print("[yellow]Use --universe nifty100 to download data for the first time.[/yellow]")
            return

    console.print(f"  Stock universe: [green]{len(symbols)}[/green] stocks")
    console.print(f"  Date range: {start_date} → {end_date}")
    console.print(f"  Mode: [yellow]{'DRY RUN' if args.dry_run else 'LIVE'}[/yellow]\n")

    if args.max_stocks > 0:
        symbols = symbols[: args.max_stocks]
        console.print(f"  Limiting to [cyan]{args.max_stocks}[/cyan] stocks (test mode)\n")

    console.print("[bold]Syncing stock minute bars...[/bold]")
    u, s, f = sync_stocks(symbols, DATA_DIR, start_date, end_date, dry_run=args.dry_run)

    console.print(f"\n  Stock sync: [green]{u}[/green] updated, [cyan]{s}[/cyan] skipped, [red]{f}[/red] failed")

    if not args.no_macro:
        console.print("\n[bold]Syncing market data...[/bold]")
        results = sync_market_data(end_date, dry_run=args.dry_run)

        macro_table = Table(show_header=True, header_style="bold")
        macro_table.add_column("Symbol", style="bold")
        macro_table.add_column("File", style="dim")
        macro_table.add_column("New Rows", justify="right")

        for name, n_rows in results.items():
            out_path = MARKET_DATA_DIR / f"{name}.csv"
            status = f"[green]+{n_rows}[/green]" if n_rows > 0 else "[cyan]0[/cyan]"
            macro_table.add_row(name, str(out_path.name), status)

        console.print(macro_table)

    console.print(f"\n[bold green]✓ Sync complete[/bold green]")
    if args.dry_run:
        console.print("[yellow]  (dry run — no files were modified)[/yellow]")
    console.print()


if __name__ == "__main__":
    main()
