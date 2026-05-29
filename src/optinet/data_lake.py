from __future__ import annotations

import io
import json
import time
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd
import requests

from optinet.config import INDEX_SPECS, canonical_index


SourceFormat = Literal["legacy", "udiff"]

CANONICAL_OPTION_COLUMNS = [
    "date",
    "symbol",
    "expiry_date",
    "strike_price",
    "option_type",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "open_interest",
    "change_in_oi",
    "settlement_price",
    "source_format",
]

OPTION_KEY_COLUMNS = ["date", "symbol", "expiry_date", "strike_price", "option_type"]
SUPPORTED_SYMBOLS = tuple(INDEX_SPECS.keys())
DEFAULT_UDIFF_CUTOFF = date(2024, 7, 8)

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}


@dataclass(frozen=True)
class DataLakeConfig:
    data_root: Path = Path("data")
    udiff_cutoff: date = DEFAULT_UDIFF_CUTOFF
    max_workers: int = 1
    retries: int = 3
    retry_sleep_seconds: float = 1.5
    request_timeout_seconds: int = 30

    @property
    def raw_root(self) -> Path:
        return self.data_root / "raw"

    @property
    def parquet_root(self) -> Path:
        return self.data_root / "parquet"

    @property
    def normalized_root(self) -> Path:
        return self.data_root / "normalized"

    @property
    def metadata_root(self) -> Path:
        return self.data_root / "metadata"


@dataclass
class DownloadResult:
    trade_date: date
    source_format: SourceFormat
    url: str
    path: Path
    status: Literal["downloaded", "exists", "missing", "failed", "skipped_weekend"]
    message: str = ""


@dataclass
class ValidationReport:
    row_count: int
    valid_row_count: int
    bad_row_count: int
    issues: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def iter_weekdays(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        if current.weekday() < 5:
            yield current
        current += timedelta(days=1)


def source_for_date(trade_date: date, cutoff: date = DEFAULT_UDIFF_CUTOFF) -> SourceFormat:
    return "legacy" if trade_date < cutoff else "udiff"


def legacy_filename(trade_date: date) -> str:
    return f"fo{trade_date:%d}{trade_date:%b}".upper().replace("FO", "fo", 1) + f"{trade_date:%Y}bhav.csv.zip"


def udiff_filename(trade_date: date) -> str:
    return f"BhavCopy_NSE_FO_0_0_0_{trade_date:%Y%m%d}_F_0000.csv.zip"


def bhavcopy_url(trade_date: date, source_format: SourceFormat) -> str:
    if source_format == "legacy":
        month = trade_date.strftime("%b").upper()
        return (
            "https://archives.nseindia.com/content/historical/DERIVATIVES/"
            f"{trade_date:%Y}/{month}"
            f"/{legacy_filename(trade_date)}"
        )
    return f"https://archives.nseindia.com/content/fo/{udiff_filename(trade_date)}"


def raw_bhavcopy_path(trade_date: date, config: DataLakeConfig) -> Path:
    source_format = source_for_date(trade_date, config.udiff_cutoff)
    filename = legacy_filename(trade_date) if source_format == "legacy" else udiff_filename(trade_date)
    return config.raw_root / source_format / f"{trade_date:%Y}" / filename


def ensure_data_lake_dirs(config: DataLakeConfig) -> None:
    for path in [
        config.raw_root / "legacy",
        config.raw_root / "udiff",
        config.raw_root / "index",
        config.normalized_root,
        config.parquet_root,
        config.metadata_root / "expiry",
        config.metadata_root / "contracts",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def download_bhavcopy_day(
    trade_date: date,
    config: DataLakeConfig,
    session: requests.Session | None = None,
) -> DownloadResult:
    if trade_date.weekday() >= 5:
        source_format = source_for_date(trade_date, config.udiff_cutoff)
        return DownloadResult(trade_date, source_format, "", raw_bhavcopy_path(trade_date, config), "skipped_weekend")

    ensure_data_lake_dirs(config)
    source_format = source_for_date(trade_date, config.udiff_cutoff)
    path = raw_bhavcopy_path(trade_date, config)
    url = bhavcopy_url(trade_date, source_format)
    if path.exists():
        return DownloadResult(trade_date, source_format, url, path, "exists", "raw file already present")

    client = session or requests.Session()
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error = ""
    for attempt in range(1, config.retries + 1):
        try:
            response = client.get(url, headers=NSE_HEADERS, timeout=config.request_timeout_seconds)
            if response.status_code == 200 and response.content:
                path.write_bytes(response.content)
                return DownloadResult(trade_date, source_format, url, path, "downloaded")
            if response.status_code in {404, 403}:
                return DownloadResult(trade_date, source_format, url, path, "missing", f"HTTP {response.status_code}")
            last_error = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)
        if attempt < config.retries:
            time.sleep(config.retry_sleep_seconds * attempt)
    return DownloadResult(trade_date, source_format, url, path, "failed", last_error)


def download_bhavcopies(start: date, end: date, config: DataLakeConfig) -> list[DownloadResult]:
    results: list[DownloadResult] = []
    with requests.Session() as session:
        for trade_date in iter_weekdays(start, end):
            results.append(download_bhavcopy_day(trade_date, config, session=session))
            time.sleep(0.35)
    return results


def read_zipped_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV file found inside {path}")
        with archive.open(csv_names[0]) as handle:
            return pd.read_csv(handle)


def parse_bhavcopy_file(path: str | Path, trade_date: date | None = None, source_format: SourceFormat | None = None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".zip":
        raw = read_zipped_csv(path)
    else:
        raw = pd.read_csv(path)
    if source_format is None:
        source_format = detect_source_format(raw)
    if trade_date is None:
        trade_date = infer_trade_date(path, raw, source_format)
    if source_format == "legacy":
        return parse_legacy_bhavcopy(raw, trade_date)
    return parse_udiff_bhavcopy(raw, trade_date)


def parse_bhavcopy_bytes(content: bytes | str, trade_date: date) -> pd.DataFrame:
    if isinstance(content, str):
        raw = pd.read_csv(io.StringIO(content))
    else:
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                csv_name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
                raw = pd.read_csv(archive.open(csv_name))
        except zipfile.BadZipFile:
            raw = pd.read_csv(io.BytesIO(content))
    source_format = detect_source_format(raw)
    return parse_legacy_bhavcopy(raw, trade_date) if source_format == "legacy" else parse_udiff_bhavcopy(raw, trade_date)


def detect_source_format(df: pd.DataFrame) -> SourceFormat:
    columns = {col.strip().lower() for col in df.columns}
    if {"tckrsymb", "traddt", "fininstrmtp"} & columns:
        return "udiff"
    return "legacy"


def infer_trade_date(path: Path, df: pd.DataFrame, source_format: SourceFormat) -> date:
    date_columns = ("TIMESTAMP", "TradDt", "BizDt", "date")
    for column in date_columns:
        if column in df.columns:
            parsed = pd.to_datetime(df[column], format="mixed", errors="coerce").dropna()
            if not parsed.empty:
                return parsed.iloc[0].date()
    digits = "".join(ch for ch in path.name if ch.isdigit())
    if source_format == "udiff" and len(digits) >= 8:
        return datetime.strptime(digits[-8:], "%Y%m%d").date()
    raise ValueError(f"Could not infer trade date for {path}")


def parse_legacy_bhavcopy(df: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    rename = {
        "TIMESTAMP": "date",
        "SYMBOL": "symbol",
        "EXPIRY_DT": "expiry_date",
        "STRIKE_PR": "strike_price",
        "OPTION_TYP": "option_type",
        "OPEN": "open",
        "HIGH": "high",
        "LOW": "low",
        "CLOSE": "close",
        "CONTRACTS": "volume",
        "OPEN_INT": "open_interest",
        "CHG_IN_OI": "change_in_oi",
        "SETTLE_PR": "settlement_price",
    }
    out = df.rename(columns={source: target for source, target in rename.items() if source in df.columns}).copy()
    out = out.loc[:, ~out.columns.duplicated()]
    if "date" not in out.columns:
        out["date"] = trade_date
    if "settlement_price" not in out.columns:
        out["settlement_price"] = out.get("close")
    out = out[_available_columns(out, CANONICAL_OPTION_COLUMNS)]
    out["source_format"] = "legacy"
    return normalize_options_frame(out)


def parse_udiff_bhavcopy(df: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    rename = {
        "TradDt": "date",
        "BizDt": "date",
        "TckrSymb": "symbol",
        "XpryDt": "expiry_date",
        "FininstrmActlXpryDt": "expiry_date",
        "StrkPric": "strike_price",
        "OptnTp": "option_type",
        "OpnPric": "open",
        "HghPric": "high",
        "LwPric": "low",
        "ClsPric": "close",
        "TtlTradgVol": "volume",
        "OpnIntrst": "open_interest",
        "ChngInOpnIntrst": "change_in_oi",
        "SttlmPric": "settlement_price",
        "UndrlygPric": "underlying_price",
        "NewBrdLotQty": "lot_size",
    }
    out = df.rename(columns={source: target for source, target in rename.items() if source in df.columns}).copy()
    out = out.loc[:, ~out.columns.duplicated()]
    if "date" not in out.columns:
        out["date"] = trade_date
    if "settlement_price" not in out.columns:
        out["settlement_price"] = out.get("close")
    if "FinInstrmTp" in df.columns:
        out = out[df["FinInstrmTp"].astype(str).str.upper().isin(["IDO", "OPTIDX"])]
    out["source_format"] = "udiff"
    cols = CANONICAL_OPTION_COLUMNS + ["underlying_price", "lot_size"]
    return normalize_options_frame(out[_available_columns(out, cols)])


def normalize_options_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in CANONICAL_OPTION_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    out["date"] = pd.to_datetime(out["date"], format="mixed", errors="coerce").dt.normalize()
    out["expiry_date"] = pd.to_datetime(out["expiry_date"], format="mixed", errors="coerce").dt.normalize()
    out["symbol"] = out["symbol"].map(canonical_index)
    out["option_type"] = out["option_type"].astype(str).str.upper().str.strip().str[0].map({"C": "CE", "P": "PE"}).fillna(
        out["option_type"].astype(str).str.upper().str.strip()
    )
    for column in [
        "strike_price",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
        "change_in_oi",
        "settlement_price",
        "underlying_price",
        "lot_size",
        "tick_size",
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out[out["symbol"].isin(SUPPORTED_SYMBOLS)]
    out = out[out["option_type"].isin(["CE", "PE"])]
    ordered = CANONICAL_OPTION_COLUMNS + [c for c in ["underlying_price", "expiry_type", "lot_size", "tick_size"] if c in out.columns]
    return out[ordered].dropna(subset=OPTION_KEY_COLUMNS).reset_index(drop=True)


def normalize_index_ohlc(df: pd.DataFrame, default_symbol: str | None = None) -> pd.DataFrame:
    aliases = {
        "index_name": ("index_name", "index", "symbol", "SYMBOL", "name"),
        "date": ("date", "Date", "TIMESTAMP", "timestamp"),
        "open": ("open", "Open", "OPEN"),
        "high": ("high", "High", "HIGH"),
        "low": ("low", "Low", "LOW"),
        "close": ("close", "Close", "CLOSE", "ltp", "LTP"),
        "volume": ("volume", "Volume", "VOLUME"),
    }
    out = _rename_aliases(df, aliases)
    if "index_name" not in out.columns:
        if default_symbol is None:
            raise ValueError("Index spot data requires index_name/symbol or a default_symbol")
        out["index_name"] = default_symbol
    if "volume" not in out.columns:
        out["volume"] = pd.NA
    out["index_name"] = out["index_name"].map(canonical_index)
    out["date"] = pd.to_datetime(out["date"], format="mixed", errors="coerce").dt.normalize()
    for column in ["open", "high", "low", "close", "volume"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out[["index_name", "date", "open", "high", "low", "close", "volume"]].dropna(
        subset=["index_name", "date", "open", "high", "low", "close"]
    )


def apply_contract_metadata(options: pd.DataFrame, contracts: pd.DataFrame) -> pd.DataFrame:
    if contracts.empty:
        return options
    meta = contracts.copy()
    meta["symbol"] = meta["symbol"].map(canonical_index)
    meta["effective_date"] = pd.to_datetime(meta["effective_date"], format="mixed", errors="coerce").dt.normalize()
    meta = meta.sort_values(["symbol", "effective_date"])
    frames = []
    for symbol, group in options.sort_values(["symbol", "date"]).groupby("symbol", sort=False):
        symbol_meta = meta[meta["symbol"] == symbol]
        if symbol_meta.empty:
            frames.append(group)
            continue
        enriched = pd.merge_asof(
            group.sort_values("date"),
            symbol_meta[["effective_date", "lot_size", "tick_size"]].sort_values("effective_date"),
            left_on="date",
            right_on="effective_date",
            direction="backward",
        ).drop(columns=["effective_date"])
        frames.append(enriched)
    return pd.concat(frames, ignore_index=True) if frames else options


def add_expiry_type(options: pd.DataFrame) -> pd.DataFrame:
    out = options.copy()
    monthly = out.groupby(["symbol", out["expiry_date"].dt.to_period("M")])["expiry_date"].transform("max")
    out["expiry_type"] = (out["expiry_date"] == monthly).map({True: "monthly", False: "weekly"})
    return out


def build_expiry_calendar(options: pd.DataFrame) -> pd.DataFrame:
    typed = add_expiry_type(options)
    calendar = typed[["symbol", "date", "expiry_date", "expiry_type"]].drop_duplicates().rename(columns={"symbol": "index_name"})
    calendar["is_expiry_day"] = calendar["date"] == calendar["expiry_date"]
    calendar["days_to_expiry"] = (calendar["expiry_date"] - calendar["date"]).dt.days
    return calendar.sort_values(["index_name", "date", "expiry_date"]).reset_index(drop=True)


def validate_options_frame(options: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, ValidationReport]:
    issues: dict[str, int] = {}
    bad_masks: list[pd.Series] = []

    missing = [column for column in CANONICAL_OPTION_COLUMNS if column not in options.columns]
    if missing:
        issues["missing_columns"] = len(missing)
        report = ValidationReport(len(options), 0, len(options), issues)
        bad = options.copy()
        bad["validation_error"] = f"missing columns: {', '.join(missing)}"
        return options.iloc[0:0].copy(), bad, report

    def add_issue(name: str, mask: pd.Series) -> None:
        count = int(mask.fillna(False).sum())
        if count:
            issues[name] = count
            bad_masks.append(mask.fillna(False))

    add_issue("invalid_dates", options["date"].isna() | options["expiry_date"].isna())
    add_issue("duplicate_rows", options.duplicated(OPTION_KEY_COLUMNS, keep=False))
    add_issue("invalid_numeric_values", options[["open", "high", "low", "close", "settlement_price"]].isna().any(axis=1))
    add_issue("invalid_price_range", options["high"] < options["low"])
    price_columns = ["open", "high", "low", "close", "settlement_price"]
    add_issue("negative_prices", options[price_columns].lt(0).any(axis=1))
    add_issue("invalid_strike", options["strike_price"].isna() | (options["strike_price"] <= 0))
    add_issue("invalid_option_type", ~options["option_type"].isin(["CE", "PE"]))
    add_issue("bad_expiry_dates", options["expiry_date"] < options["date"])
    add_issue("negative_open_interest", options["open_interest"].notna() & (options["open_interest"] < 0))
    add_issue("empty_open_interest", options["open_interest"].isna())
    # Suspicious OI change: |change_in_oi| > open_interest.
    # NSE settles all open positions on expiry day, so the previous day's OI is
    # closed out and the residual end-of-day OI is naturally smaller than the
    # day's change_in_oi. Exempt expiry-day rows to avoid false positives.
    is_expiry_day = options["date"] == options["expiry_date"]
    add_issue(
        "suspicious_open_interest_change",
        options["change_in_oi"].notna()
        & options["open_interest"].notna()
        & (options["change_in_oi"].abs() > options["open_interest"])
        & ~is_expiry_day.fillna(False),
    )
    if "volume" in options.columns:
        add_issue("negative_volume", options["volume"].notna() & (options["volume"] < 0))

    if bad_masks:
        combined = bad_masks[0].copy()
        for mask in bad_masks[1:]:
            combined |= mask
    else:
        combined = pd.Series(False, index=options.index)

    good = options.loc[~combined].copy()
    bad = options.loc[combined].copy()
    if not bad.empty:
        bad["validation_error"] = bad.apply(lambda row: _row_errors(row, options), axis=1)
    report = ValidationReport(len(options), len(good), len(bad), issues)
    return good, bad, report


def write_validation_artifacts(
    bad_rows: pd.DataFrame,
    report: ValidationReport,
    output_dir: str | Path,
    name: str,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{name}_report.json").write_text(report.to_json())
    if not bad_rows.empty:
        bad_rows.to_csv(output / f"{name}_bad_rows.csv", index=False)


def missing_trading_days(options: pd.DataFrame, start: date, end: date, holidays: Iterable[date] | None = None) -> list[date]:
    holidays = set(holidays or [])
    expected = {day for day in iter_weekdays(start, end) if day not in holidays}
    actual = {pd.Timestamp(day).date() for day in pd.to_datetime(options["date"], errors="coerce").dropna().unique()}
    return sorted(expected - actual)


def write_partitioned_parquet(options: pd.DataFrame, parquet_root: str | Path, overwrite: bool = False) -> list[Path]:
    root = Path(parquet_root)
    written: list[Path] = []
    if options.empty:
        return written
    frame = options.copy()
    frame["year"] = frame["date"].dt.year.astype("int64")
    for (symbol, year, trade_date), group in frame.groupby(["symbol", "year", "date"], sort=True):
        out_dir = root / f"symbol={symbol}" / f"year={year}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"options_{pd.Timestamp(trade_date):%Y%m%d}.parquet"
        if out_path.exists() and not overwrite:
            continue
        group.drop(columns=["year"]).to_parquet(out_path, index=False)
        written.append(out_path)
    return written


def ingest_raw_file(
    raw_path: str | Path,
    config: DataLakeConfig,
    trade_date: date | None = None,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, ValidationReport, list[Path]]:
    parsed = parse_bhavcopy_file(raw_path, trade_date=trade_date)
    parsed = add_expiry_type(parsed)
    good, bad, report = validate_options_frame(parsed)
    name = Path(raw_path).stem.replace(".csv", "")
    write_validation_artifacts(bad, report, config.normalized_root / "validation", name)
    written = write_partitioned_parquet(good, config.parquet_root, overwrite=overwrite)
    return good, report, written


def _available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _rename_aliases(df: pd.DataFrame, aliases: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    lookup = {column.lower(): column for column in df.columns}
    rename: dict[str, str] = {}
    for target, candidates in aliases.items():
        for candidate in candidates:
            source = lookup.get(candidate.lower())
            if source is not None:
                rename[source] = target
                break
    return df.rename(columns=rename)


def _row_errors(row: pd.Series, options: pd.DataFrame) -> str:
    errors: list[str] = []
    if row.name in options[options.duplicated(OPTION_KEY_COLUMNS, keep=False)].index:
        errors.append("duplicate_rows")
    if pd.isna(row["date"]) or pd.isna(row["expiry_date"]):
        errors.append("invalid_dates")
    if any(pd.isna(row[col]) for col in ["open", "high", "low", "close", "settlement_price"]):
        errors.append("invalid_numeric_values")
    if row["high"] < row["low"]:
        errors.append("invalid_price_range")
    if any(pd.notna(row[col]) and row[col] < 0 for col in ["open", "high", "low", "close", "settlement_price"]):
        errors.append("negative_prices")
    if pd.isna(row["strike_price"]) or row["strike_price"] <= 0:
        errors.append("invalid_strike")
    if row["option_type"] not in {"CE", "PE"}:
        errors.append("invalid_option_type")
    if row["expiry_date"] < row["date"]:
        errors.append("bad_expiry_dates")
    if pd.isna(row["open_interest"]):
        errors.append("empty_open_interest")
    elif row["open_interest"] < 0:
        errors.append("negative_open_interest")
    if (
        pd.notna(row["change_in_oi"])
        and pd.notna(row["open_interest"])
        and abs(row["change_in_oi"]) > row["open_interest"]
        and not (pd.notna(row["expiry_date"]) and row["date"] == row["expiry_date"])
    ):
        errors.append("suspicious_open_interest_change")
    if pd.notna(row["volume"]) and row["volume"] < 0:
        errors.append("negative_volume")
    return ",".join(errors)
