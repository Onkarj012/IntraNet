"""
OptiNet Dataset Builder
=======================
Downloads all 5 required datasets for OptiNet training:
  1. Index Spot OHLCV         — niftyindices.com (daily) + optional Kite (minute)
  2. Options EOD OHLCV + OI   — NSE F&O bhavcopy (ALL strikes, both CE/PE)
  3. Expiry Calendar          — derived from bhavcopy data
  4. Contract Metadata        — static table (lot sizes, costs)
  5. Macro / Context data     — yfinance

HOW TO RUN:
  pip install jugaad-data yfinance pandas tqdm requests
  python optinet_data_builder.py

  For Kite minute-level index data, set KITE_API_KEY and KITE_ACCESS_TOKEN.
  Get access_token from: https://kite.trade/docs/connect/v3/user/#response-attributes

NOTE: NSE bhavcopy download requires running from your home/office IP.
      Server IPs are blocked by Akamai. This script works fine on a laptop/desktop.
"""

import os
import io
import time
import zipfile
import logging
import calendar
from pathlib import Path
from datetime import date, timedelta
from urllib.parse import quote

import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm

# ─── Config ──────────────────────────────────────────────────────────────────

START_DATE  = date(2020, 1, 1)
END_DATE    = date(2025, 12, 31)
OUTPUT_DIR  = Path("optinet_data")

# Optional Kite credentials (for minute-level index OHLCV)
KITE_API_KEY     = os.getenv("KITE_API_KEY", "")
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "")

# NSE request delay — be gentle, 1 req/sec is safe
NSE_DELAY   = 1.2   # seconds between requests
MAX_RETRIES = 3
FO_UDIFF_CUTOFF = date(2024, 7, 8)

# ─── Setup ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "options").mkdir(exist_ok=True)
(OUTPUT_DIR / "index").mkdir(exist_ok=True)
(OUTPUT_DIR / "macro").mkdir(exist_ok=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def trading_days(start: date, end: date):
    """Yield all Mon-Fri dates in [start, end]."""
    day = start
    while day <= end:
        if day.weekday() < 5:   # Mon=0 … Fri=4
            yield day
        day += timedelta(days=1)


def make_nse_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com/",
        "DNT": "1",
    })
    return s


# ─── Dataset 2: Options EOD OHLCV + OI  (NSE F&O Bhavcopy) ──────────────────

def _fo_bhavcopy_url(dt: date) -> str:
    yyyy = dt.strftime("%Y")
    MMM  = dt.strftime("%b").upper()   # JAN, FEB …
    dd   = dt.strftime("%d")
    return (
        f"https://nsearchives.nseindia.com/content/historical/"
        f"DERIVATIVES/{yyyy}/{MMM}/fo{dd}{MMM}{yyyy}bhav.csv.zip"
    )


def _fo_udiff_url(dt: date) -> str:
    report_name = quote("F&O - UDiFF Common Bhavcopy Final (zip)", safe="")
    archives = (
        f'[{{"name":"{report_name}","type":"archives","category":"derivatives","section":"equity"}}]'
    )
    return (
        "https://www.nseindia.com/api/reports"
        f"?archives={archives}&date={dt.strftime('%d-%b-%Y')}&type=equity&mode=single"
    )


def _download_fo_bhavcopy_raw(dt: date, session: requests.Session) -> str | None:
    """Download and unzip F&O bhavcopy for one date. Returns CSV text or None."""
    url = _fo_bhavcopy_url(dt) if dt < FO_UDIFF_CUTOFF else _fo_udiff_url(dt)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=20)
            if r.status_code == 404:
                return None      # holiday / no data
            if r.status_code != 200:
                log.warning(f"  {dt} HTTP {r.status_code}, attempt {attempt}")
                time.sleep(NSE_DELAY * attempt)
                continue
            zf = zipfile.ZipFile(io.BytesIO(r.content))
            fname = zf.namelist()[0]
            return zf.open(fname).read().decode("utf-8")
        except zipfile.BadZipFile:
            log.warning(f"  {dt} bad zip / unexpected response, attempt {attempt}")
            time.sleep(NSE_DELAY)
            continue
        except Exception as e:
            log.warning(f"  {dt} error: {e}, attempt {attempt}")
            time.sleep(NSE_DELAY * attempt)
    return None


def _parse_fo_bhavcopy(csv_text: str, dt: date) -> pd.DataFrame | None:
    """Parse raw bhavcopy CSV and filter to NIFTY/BANKNIFTY options only."""
    try:
        df = pd.read_csv(io.StringIO(csv_text))
        # Strip whitespace from column names (old format has spaces)
        df.columns = df.columns.str.strip()

        if {"INSTRUMENT", "SYMBOL", "EXPIRY_DT", "STRIKE_PR", "OPTION_TYP"}.issubset(df.columns):
            mask = (
                (df["INSTRUMENT"] == "OPTIDX") &
                (df["SYMBOL"].isin(["NIFTY", "BANKNIFTY"]))
            )
            df = df[mask].copy()
            if df.empty:
                return None
            df = df.rename(columns={
                "SYMBOL":     "index_name",
                "EXPIRY_DT":  "expiry_date",
                "STRIKE_PR":  "strike_price",
                "OPTION_TYP": "option_type",
                "OPEN":       "open",
                "HIGH":       "high",
                "LOW":        "low",
                "CLOSE":      "close",
                "SETTLE_PR":  "settlement_price",
                "CONTRACTS":  "volume",
                "OPEN_INT":   "open_interest",
                "CHG_IN_OI":  "change_in_oi",
                "TIMESTAMP":  "date",
            })
            df["expiry_date"] = pd.to_datetime(df["expiry_date"], format="%d-%b-%Y", errors="coerce")
            df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
        elif {"FinInstrmTp", "TckrSymb", "XpryDt", "StrkPric", "OptnTp", "TradDt"}.issubset(df.columns):
            mask = (
                (df["FinInstrmTp"] == "IDO") &
                (df["TckrSymb"].isin(["NIFTY", "BANKNIFTY"]))
            )
            df = df[mask].copy()
            if df.empty:
                return None
            df = df.rename(columns={
                "TckrSymb":          "index_name",
                "TradDt":            "date",
                "XpryDt":            "expiry_date",
                "StrkPric":          "strike_price",
                "OptnTp":            "option_type",
                "OpnPric":           "open",
                "HghPric":           "high",
                "LwPric":            "low",
                "ClsPric":           "close",
                "SttlmPric":         "settlement_price",
                "TtlTradgVol":       "volume",
                "OpnIntrst":         "open_interest",
                "ChngInOpnIntrst":   "change_in_oi",
            })
            df["expiry_date"] = pd.to_datetime(df["expiry_date"], format="mixed", errors="coerce")
            df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
        else:
            log.error(f"  Parse error: unsupported F&O bhavcopy columns for {dt}")
            return None

        for col in [
            "strike_price", "open", "high", "low", "close",
            "settlement_price", "volume", "open_interest", "change_in_oi",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if df["date"].isna().all():
            df["date"] = pd.Timestamp(dt)

        df["days_to_expiry"] = (df["expiry_date"] - df["date"]).dt.days
        df["option_type"] = df["option_type"].astype(str).str.strip().replace({"CA": "CE", "PA": "PE"})

        cols = [
            "index_name", "date", "expiry_date", "strike_price", "option_type",
            "open", "high", "low", "close", "settlement_price",
            "volume", "open_interest", "change_in_oi", "days_to_expiry",
        ]
        return df[[c for c in cols if c in df.columns]]

    except Exception as e:
        log.error(f"  Parse error: {e}")
        return None


def download_options_eod(start: date = START_DATE, end: date = END_DATE):
    """Main function: download all NSE F&O bhavcopy dates, save per-year CSVs."""
    log.info("=" * 60)
    log.info("DATASET 2: Options EOD OHLCV + OI")
    log.info("=" * 60)

    session = make_nse_session()
    # Warm up the session with a real page load (helps with Akamai cookies)
    try:
        session.get("https://www.nseindia.com/all-reports", timeout=15)
        time.sleep(2)
    except Exception:
        pass

    all_days = list(trading_days(start, end))
    frames_by_year: dict[int, list[pd.DataFrame]] = {}
    cached_dates_by_year: dict[int, set[str]] = {}

    for year in range(start.year, end.year + 1):
        out_path = OUTPUT_DIR / "options" / f"options_eod_{year}.csv"
        if not out_path.exists():
            continue
        try:
            existing = pd.read_csv(out_path, usecols=["date"])
            existing_dates = (
                pd.to_datetime(existing["date"], format="mixed", errors="coerce")
                .dt.strftime("%Y-%m-%d")
                .dropna()
            )
            cached_dates_by_year[year] = set(existing_dates.tolist())
        except Exception as e:
            log.warning(f"  Could not read cached dates from {out_path.name}: {e}")

    skipped = 0
    downloaded = 0
    holiday = 0

    for dt in tqdm(all_days, desc="Options bhavcopy", unit="day"):
        year = dt.year
        if dt.isoformat() in cached_dates_by_year.get(year, set()):
            skipped += 1
            continue

        csv_text = _download_fo_bhavcopy_raw(dt, session)
        if csv_text is None:
            holiday += 1
            time.sleep(NSE_DELAY)
            continue

        df = _parse_fo_bhavcopy(csv_text, dt)
        if df is not None and not df.empty:
            frames_by_year.setdefault(year, []).append(df)
            cached_dates_by_year.setdefault(year, set()).add(dt.isoformat())
            downloaded += 1

            # Flush to disk every 20 days to avoid memory buildup
            if downloaded % 20 == 0:
                _flush_options(frames_by_year, OUTPUT_DIR / "options")
                frames_by_year.clear()

        time.sleep(NSE_DELAY)

    # Final flush
    _flush_options(frames_by_year, OUTPUT_DIR / "options")

    log.info(f"Options: {downloaded} days downloaded, {holiday} holidays/no-data, {skipped} already cached")


def _flush_options(frames_by_year: dict, out_dir: Path):
    for year, frames in frames_by_year.items():
        out_path = out_dir / f"options_eod_{year}.csv"
        new_df = pd.concat(frames, ignore_index=True)
        if out_path.exists():
            existing = pd.read_csv(out_path)
            new_df = pd.concat([existing, new_df], ignore_index=True)
            new_df = new_df.drop_duplicates(
                subset=["index_name", "date", "expiry_date", "strike_price", "option_type"]
            )
        new_df.sort_values(["index_name", "date", "expiry_date", "strike_price", "option_type"], inplace=True)
        new_df.to_csv(out_path, index=False)


# ─── Dataset 1: Index Spot OHLCV ─────────────────────────────────────────────

def _download_index_day(dt: date, session: requests.Session) -> pd.DataFrame | None:
    """Download one day's index snapshot from niftyindices.com"""
    dd   = dt.strftime("%d")
    mm   = dt.strftime("%m")
    yyyy = dt.strftime("%Y")
    url  = f"https://www.niftyindices.com/Daily_Snapshot/ind_close_all_{dd}{mm}{yyyy}.csv"
    try:
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            return None
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = df.columns.str.strip()
        # Filter to NIFTY 50 and Nifty Bank
        mask = df["Index Name"].isin(["Nifty 50", "Nifty Bank"])
        df = df[mask].copy()
        if df.empty:
            return None
        df = df.rename(columns={
            "Index Name":        "index_name",
            "Index Date":        "date",
            "Open Index Value":  "open",
            "High Index Value":  "high",
            "Low Index Value":   "low",
            "Closing Index Value": "close",
        })
        df["index_name"] = df["index_name"].replace({
            "Nifty 50":   "NIFTY",
            "Nifty Bank": "BANKNIFTY",
        })
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
        return df[["index_name", "date", "open", "high", "low", "close"]]
    except Exception as e:
        log.debug(f"  niftyindices {dt}: {e}")
        return None


def download_index_spot(start: date = START_DATE, end: date = END_DATE):
    """Download NIFTY + BANKNIFTY daily OHLCV from niftyindices.com"""
    log.info("=" * 60)
    log.info("DATASET 1: Index Spot OHLCV (daily)")
    log.info("=" * 60)

    out_path = OUTPUT_DIR / "index" / "index_spot_daily.csv"

    # If file exists, check what dates we already have
    existing_dates = set()
    if out_path.exists():
        existing = pd.read_csv(out_path)
        existing_dates = set(existing["date"].values)
        log.info(f"  Existing rows: {len(existing):,}  —  extending...")

    session = make_nse_session()
    session.headers.update({"Referer": "https://www.niftyindices.com/"})

    all_days = list(trading_days(start, end))
    frames = []

    for dt in tqdm(all_days, desc="Index spot", unit="day"):
        if dt.isoformat() in existing_dates:
            continue
        df = _download_index_day(dt, session)
        if df is not None:
            frames.append(df)
        time.sleep(0.5)   # niftyindices is more lenient

    if frames:
        new_df = pd.concat(frames, ignore_index=True)
        if out_path.exists():
            existing = pd.read_csv(out_path)
            new_df = pd.concat([existing, new_df], ignore_index=True)
            new_df = new_df.drop_duplicates(subset=["index_name", "date"])
        new_df["date"] = pd.to_datetime(new_df["date"])
        new_df.sort_values(["index_name", "date"], inplace=True)
        new_df.to_csv(out_path, index=False)
        log.info(f"  Saved {len(new_df):,} rows → {out_path}")
    else:
        log.info("  Nothing new to add.")


def download_index_minute_kite(start: date = START_DATE, end: date = END_DATE):
    """
    Download NIFTY + BANKNIFTY MINUTE data via Kite API.
    Requires KITE_API_KEY and KITE_ACCESS_TOKEN env vars.
    Instrument tokens: NIFTY=256265, BANKNIFTY=260105
    """
    if not KITE_API_KEY or not KITE_ACCESS_TOKEN:
        log.info("  Kite credentials not set — skipping minute index data.")
        log.info("  Set KITE_API_KEY and KITE_ACCESS_TOKEN env vars to enable.")
        return

    log.info("=" * 60)
    log.info("DATASET 1b: Index Spot OHLCV (minute, via Kite)")
    log.info("=" * 60)

    try:
        from kiteconnect import KiteConnect
    except ImportError:
        log.error("  kiteconnect not installed: pip install kiteconnect")
        return

    kite = KiteConnect(api_key=KITE_API_KEY)
    kite.set_access_token(KITE_ACCESS_TOKEN)

    instruments = {
        "NIFTY":     256265,
        "BANKNIFTY": 260105,
    }

    # Kite limit: 60 days per minute-data request
    CHUNK_DAYS = 55

    for index_name, token in instruments.items():
        out_path = OUTPUT_DIR / "index" / f"index_{index_name.lower()}_minute.csv"
        existing_dates = set()
        if out_path.exists():
            existing = pd.read_csv(out_path, usecols=["date"])
            existing_dates = set(existing["date"].str[:10].values)

        frames = []
        day = start
        while day <= end:
            chunk_end = min(day + timedelta(days=CHUNK_DAYS - 1), end)
            if day.isoformat()[:10] in existing_dates:
                day = chunk_end + timedelta(days=1)
                continue

            try:
                data = kite.historical_data(
                    token,
                    from_date=day,
                    to_date=chunk_end,
                    interval="minute",
                    oi=False,
                )
                if data:
                    df = pd.DataFrame(data)
                    df.columns = ["date", "open", "high", "low", "close", "volume"]
                    df["index_name"] = index_name
                    frames.append(df[["index_name", "date", "open", "high", "low", "close", "volume"]])
                    log.info(f"  {index_name} {day} → {chunk_end}: {len(data)} bars")
            except Exception as e:
                log.warning(f"  {index_name} {day}: {e}")

            day = chunk_end + timedelta(days=1)
            time.sleep(0.5)

        if frames:
            new_df = pd.concat(frames, ignore_index=True)
            if out_path.exists():
                existing = pd.read_csv(out_path)
                new_df = pd.concat([existing, new_df], ignore_index=True)
                new_df = new_df.drop_duplicates(subset=["index_name", "date"])
            new_df.sort_values(["index_name", "date"], inplace=True)
            new_df.to_csv(out_path, index=False)
            log.info(f"  Saved {len(new_df):,} minute bars → {out_path}")


# ─── Dataset 3: Expiry Calendar ──────────────────────────────────────────────

def build_expiry_calendar():
    """
    Derive expiry calendar from downloaded options data.
    Classifies expiries as weekly or monthly.
    Monthly = last Thursday of the month for NIFTY, last Wednesday for BANKNIFTY.
    """
    log.info("=" * 60)
    log.info("DATASET 3: Expiry Calendar")
    log.info("=" * 60)

    frames = []
    for f in sorted((OUTPUT_DIR / "options").glob("options_eod_*.csv")):
        df = pd.read_csv(f, usecols=["index_name", "date", "expiry_date"])
        frames.append(df)

    if not frames:
        log.warning("  No options data found — run download_options_eod() first.")
        return

    df = pd.concat(frames, ignore_index=True)
    df["date"]        = pd.to_datetime(df["date"])
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])

    # Unique (index_name, expiry_date) pairs
    expiries = (
        df.groupby(["index_name", "expiry_date"])
        .agg(first_seen=("date", "min"), last_seen=("date", "max"))
        .reset_index()
    )

    # Monthly = last Thursday of the month (for both NIFTY & BANKNIFTY historically)
    # Note: BANKNIFTY moved to monthly-only in Nov 2024
    def last_thursday(year, month):
        last_day = calendar.monthrange(year, month)[1]
        d = date(year, month, last_day)
        while d.weekday() != 3:   # 3 = Thursday
            d -= timedelta(days=1)
        return d

    monthly_thursdays = set()
    for yr in range(START_DATE.year, END_DATE.year + 1):
        for mo in range(1, 13):
            monthly_thursdays.add(pd.Timestamp(last_thursday(yr, mo)))

    expiries["expiry_type"] = expiries["expiry_date"].apply(
        lambda x: "monthly" if x in monthly_thursdays else "weekly"
    )

    # is_expiry_day and is_expiry_week helpers (join with all trading dates)
    all_trading = pd.DataFrame(
        {"date": [pd.Timestamp(d) for d in trading_days(START_DATE, END_DATE)]}
    )
    result_rows = []
    for _, row in expiries.iterrows():
        exp_date = row["expiry_date"]
        # Find the trading week of this expiry
        week_start = exp_date - timedelta(days=exp_date.weekday())  # Monday
        week_end   = week_start + timedelta(days=4)                 # Friday
        week_dates = all_trading[
            (all_trading["date"] >= week_start) &
            (all_trading["date"] <= week_end)
        ]["date"].tolist()

        result_rows.append({
            "index_name":    row["index_name"],
            "expiry_date":   exp_date.date().isoformat(),
            "expiry_type":   row["expiry_type"],
            "weekday":       exp_date.day_name(),
            "is_expiry_day_weekday": exp_date.weekday(),
        })

    cal = pd.DataFrame(result_rows)
    cal = cal.sort_values(["index_name", "expiry_date"])
    out_path = OUTPUT_DIR / "expiry_calendar.csv"
    cal.to_csv(out_path, index=False)
    log.info(f"  {len(cal)} expiries saved → {out_path}")
    log.info(f"  Weekly: {(cal.expiry_type=='weekly').sum()}, Monthly: {(cal.expiry_type=='monthly').sum()}")


# ─── Dataset 4: Contract Metadata / Costs ────────────────────────────────────

def build_contract_metadata():
    """
    Static costs table. Update lot_size if NSE changes it.
    Values current as of April 2025.
    """
    log.info("=" * 60)
    log.info("DATASET 4: Contract Metadata / Costs")
    log.info("=" * 60)

    data = [
        {
            "index_name":          "NIFTY",
            "lot_size":            25,          # from Jun 2024 (was 50 before)
            "tick_size":           0.05,
            "brokerage_per_lot":   40,           # Zerodha: ₹20 per order × 2 legs
            "exchange_charges_pct": 0.0005,      # NSE: 0.05% of turnover (options)
            "stt_pct":             0.0625,       # STT on sell side: 0.0625% of premium
            "stamp_duty_pct":      0.003,        # 0.003% of premium on buy
            "sebi_charges_pct":    0.000001,     # ₹1 per crore
            "gst_pct":             18,           # 18% on (brokerage + exchange)
            "slippage_ticks":      2,            # assumption: 2 ticks slippage
            "margin_required_approx_inr": 85000  # rough SPAN margin per lot
        },
        {
            "index_name":          "BANKNIFTY",
            "lot_size":            15,
            "tick_size":           0.05,
            "brokerage_per_lot":   40,
            "exchange_charges_pct": 0.0005,
            "stt_pct":             0.0625,
            "stamp_duty_pct":      0.003,
            "sebi_charges_pct":    0.000001,
            "gst_pct":             18,
            "slippage_ticks":      2,
            "margin_required_approx_inr": 55000
        },
    ]

    # Historical lot sizes (for P&L accuracy on old data)
    lot_size_history = [
        # NIFTY
        {"index_name": "NIFTY",     "effective_from": "2015-01-01", "effective_to": "2024-06-19", "lot_size": 75},
        {"index_name": "NIFTY",     "effective_from": "2024-06-20", "effective_to": "2099-12-31", "lot_size": 25},
        # BANKNIFTY
        {"index_name": "BANKNIFTY", "effective_from": "2015-01-01", "effective_to": "2020-11-26", "lot_size": 20},
        {"index_name": "BANKNIFTY", "effective_from": "2020-11-27", "effective_to": "2099-12-31", "lot_size": 15},
    ]

    df_meta = pd.DataFrame(data)
    df_hist = pd.DataFrame(lot_size_history)

    out_meta = OUTPUT_DIR / "contract_metadata.csv"
    out_hist = OUTPUT_DIR / "lot_size_history.csv"

    df_meta.to_csv(out_meta, index=False)
    df_hist.to_csv(out_hist, index=False)

    log.info(f"  Saved → {out_meta}")
    log.info(f"  Saved → {out_hist}")
    log.info("  ⚠ Review lot_size_history.csv — NSE changes lot sizes occasionally")


# ─── Dataset 5: Macro / Context Data ─────────────────────────────────────────

def download_macro(start: date = START_DATE, end: date = END_DATE):
    """Download macro context using yfinance."""
    log.info("=" * 60)
    log.info("DATASET 5: Macro / Context Data")
    log.info("=" * 60)

    tickers = {
        "india_vix":  "^INDIAVIX",
        "dxy":        "DX-Y.NYB",
        "usdinr":     "INR=X",
        "us_10y":     "^TNX",
        "crude":      "CL=F",
        "gold":       "GC=F",
        "sp500":      "^GSPC",
        "nasdaq":     "^IXIC",
        "dow":        "^DJI",
        "nikkei":     "^N225",
        "hang_seng":  "^HSI",
    }

    out_path = OUTPUT_DIR / "macro" / "macro_daily.csv"
    start_str = start.isoformat()
    end_str   = (end + timedelta(days=1)).isoformat()   # yfinance end is exclusive

    frames = {}
    for col_name, ticker in tickers.items():
        log.info(f"  Fetching {ticker} ({col_name})...")
        try:
            df = yf.download(ticker, start=start_str, end=end_str, progress=False, auto_adjust=True)
            if not df.empty:
                frames[col_name] = df["Close"].squeeze()
                log.info(f"    {len(df)} rows")
            else:
                log.warning(f"    Empty response for {ticker}")
        except Exception as e:
            log.warning(f"    {ticker} failed: {e}")
        time.sleep(0.3)

    if not frames:
        log.error("  No macro data downloaded.")
        return

    macro = pd.DataFrame(frames)
    macro.index.name = "date"
    macro = macro.reset_index()
    macro["date"] = pd.to_datetime(macro["date"])
    macro.sort_values("date", inplace=True)
    macro.to_csv(out_path, index=False)
    log.info(f"  Saved {len(macro)} rows → {out_path}")


# ─── Summary ─────────────────────────────────────────────────────────────────

def print_summary():
    log.info("")
    log.info("=" * 60)
    log.info("DOWNLOAD COMPLETE — file summary:")
    log.info("=" * 60)

    files = list(OUTPUT_DIR.rglob("*.csv"))
    total_rows = 0
    for f in sorted(files):
        try:
            n = sum(1 for _ in open(f)) - 1
            size_mb = f.stat().st_size / 1_048_576
            log.info(f"  {f.relative_to(OUTPUT_DIR)}  —  {n:>8,} rows  ({size_mb:.1f} MB)")
            total_rows += n
        except Exception:
            pass
    log.info(f"\n  TOTAL: {total_rows:,} rows across {len(files)} files")
    log.info(f"  Output dir: {OUTPUT_DIR.resolve()}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OptiNet Dataset Builder")
    parser.add_argument("--start",   default=START_DATE.isoformat(), help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     default=END_DATE.isoformat(),   help="End date YYYY-MM-DD")
    parser.add_argument("--only",    choices=["options","index","macro","calendar","meta","kite"],
                        help="Run only one dataset (default: all)")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    log.info(f"OptiNet Data Builder | {start} → {end}")
    log.info(f"Output: {OUTPUT_DIR.resolve()}")
    log.info("")

    if args.only == "options":
        download_options_eod(start, end)
    elif args.only == "index":
        download_index_spot(start, end)
    elif args.only == "kite":
        download_index_minute_kite(start, end)
    elif args.only == "macro":
        download_macro(start, end)
    elif args.only == "calendar":
        build_expiry_calendar()
    elif args.only == "meta":
        build_contract_metadata()
    else:
        # Full pipeline
        download_index_spot(start, end)
        download_index_minute_kite(start, end)
        download_options_eod(start, end)
        download_macro(start, end)
        build_contract_metadata()
        build_expiry_calendar()

    print_summary()
