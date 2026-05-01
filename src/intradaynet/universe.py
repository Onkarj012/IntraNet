"""
Stock universe definitions for IntradayNet.

Defines NIFTY indices constituents for filtering and backtesting.
Using actual NIFTY100 as of 2024 for reproducibility.
"""

import csv
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

# NIFTY 100 constituents (as of early 2024)
# Top liquid stocks - reliable for backtesting
NIFTY100_SYMBOLS: List[str] = [
    "RELIANCE", "TCS", "HDFCBANK", "BHARTIARTL", "ICICIBANK",
    "INFY", "SBIN", "HINDUNILVR", "ITC", "BAJFINANCE",
    "LICI", "HCLTECH", "SUNPHARMA", "MARUTI", "KOTAKBANK",
    "AXISBANK", "TITAN", "ONGC", "NTPC", "ADANIENT",
    "ADANIPORTS", "POWERGRID", "TATAMOTORS", "ULTRACEMCO", "ASIANPAINT",
    "COALINDIA", "BAJAJFINSV", "M&M", "NESTLEIND", "ADANIGREEN",
    "WIPRO", "DLF", "TATASTEEL", "BAJAJ-AUTO", "GRASIM",
    "SBILIFE", "JSWSTEEL", "BRITANNIA", "TECHM", "TRENT",
    "HINDZINC", "IOC", "HAL", "DIVISLAB", "VBL",
    "PIDILITIND", "GODREJCP", "PNB", "INDUSINDBK", "DRREDDY",
    "ATGL", "SIEMENS", "LTIM", "SHRIRAMFIN", "ZOMATO",
    "LODHA", "HAVELLS", "BANKBARODA", "DABUR", "CIPLA",
    "AMBUJACEM", "TATAPOWER", "ICICIPRULI", "ABB", "CANBK",
    "CHOLAFIN", "BEL", "MARICO", "APOLLOHOSP", "UNITDSPR",
    "INDIGO", "TATACONSUM", "MCDOWELL-N", "BERGEPAINT", "NAUKRI",
    "IDBI", "JINDALSTEL", "PGHH", "BOSCHLTD", "SRF",
    "HINDALCO", "UPL", "POLYCAB", "MOTHERSON", "TORNTPHARM",
    "LUPIN", "INDUSTOWER", "PAYTM", "GAIL", "GODREJPROP",
    "HDFCLIFE", "COLPAL", "YESBANK", "NYKAA", "ADANIPOWER",
    "IRCTC", "PAGEIND", "PFC", "ASHOKLEY", "MAXHEALTH",
]

# NIFTY 50 - Most liquid
NIFTY50_SYMBOLS: List[str] = NIFTY100_SYMBOLS[:50]

# NIFTY 200 - Extended universe
NIFTY200_SYMBOLS: List[str] = NIFTY100_SYMBOLS + [
    "IRFC", "TVSMOTOR", "NHPC", "HEROMOTOCO", "DALBHARAT",
    "CONCOR", "BHEL", "ACC", "NMDC", "TATACOMM",
    "SUNTV", "MFSL", "ALKEM", "CGPOWER", "AUBANK",
    "PIIND", "ASTRAL", "LTTS", "LINDEINDIA", "PHOENIXLTD",
    "UNOMINDA", "ABCAPITAL", "IPCALAB", "BALKRISIND", "JUBLFOOD",
    "UBL", "HUDCO", "OBEROIRLTY", "SUZLON", "SOLARINDS",
    "MRF", "FEDERALBNK", "APOLLOTYRE", "MPHASIS", "FLUOROCHEM",
    "PERSISTENT", "PETRONET", "M&MFIN", "BHARATFORG", "AIAENG",
    "RECLTD", "KPITTECH", "IDEA", "CUMMINSIND", "TIINDIA",
    "KALYANKJIL", "POLICYBZR", "POONAWALLA", "L&TFH", "OIL",
    "SAIL", "SUNDARMFIN", "COROMANDEL", "DEVYANI", "ESCORTS",
    "BSE", "PFIZER", "SJVN", "BIOCON", "NAM-INDIA",
    "HONAUT", "INDHOTEL", "THREEPHIL", "DELHIVERY", "NAVINFLUOR",
    "DIXON", "RELAXO", "CREDITACC", "MEDANTA", "ZEEL",
    "VOLTAS", "AJANTPHARM", "SCHAEFFLER", "JKCEMENT", "LALPATHLAB",
    "WHIRLPOOL", "AWL", "BANDHANBNK", "AARTIIND", "KANSAINER",
    "SYNGENE", "EMAMILTD", "METROPOLIS", "APARINDS", "FORTIS",
    "KPRMILL", "MANYAVAR", "FIVESTAR", "GLAND", "BSOFT",
    "ITI", "TANLA", "360ONE", "RATNAMANI", "EIHOTEL",
    "KEI", "LAXMIMACH", "THERMAX", "NLCINDIA", "HAPPSTMNDS",
    "NATIONALUM", "SKFINDIA", "HUDCO", "CESC", "GSPL",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_nifty500_from_csv() -> List[str]:
    csv_path = _project_root() / "data" / "sentiment" / "ind_nifty500list.csv"
    if not csv_path.exists():
        return []

    symbols: list[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "Symbol" not in reader.fieldnames:
            return []
        for row in reader:
            symbol = (row.get("Symbol") or "").strip().upper()
            if not symbol:
                continue
            if symbol.endswith(".NS"):
                symbol = symbol[:-3]
            symbols.append(symbol)

    unique_symbols: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        if symbol in seen:
            continue
        seen.add(symbol)
        unique_symbols.append(symbol)
    return unique_symbols


@lru_cache(maxsize=1)
def get_universe_metadata(csv_path: str | None = None) -> pd.DataFrame:
    """Load canonical Nifty 500 metadata from CSV."""
    resolved_path = Path(csv_path) if csv_path else (_project_root() / "data" / "sentiment" / "ind_nifty500list.csv")
    if not resolved_path.exists():
        return pd.DataFrame(columns=["symbol", "industry", "company_name", "series", "isin"])

    df = pd.read_csv(resolved_path)
    rename_map = {
        "Symbol": "symbol",
        "Industry": "industry",
        "Company Name": "company_name",
        "Series": "series",
        "ISIN Code": "isin",
    }
    df = df.rename(columns=rename_map)
    for column in ["symbol", "industry", "company_name", "series", "isin"]:
        if column not in df.columns:
            df[column] = ""
    df["symbol"] = (
        df["symbol"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".NS", "", regex=False)
    )
    df["industry"] = df["industry"].fillna("").astype(str).str.strip()
    df["company_name"] = df["company_name"].fillna("").astype(str).str.strip()
    df["series"] = df["series"].fillna("").astype(str).str.strip()
    df["isin"] = df["isin"].fillna("").astype(str).str.strip()
    df = df[df["symbol"] != ""].drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)
    return df


@lru_cache(maxsize=1)
def get_symbol_to_industry_map(csv_path: str | None = None) -> Dict[str, str]:
    metadata = get_universe_metadata(csv_path)
    if metadata.empty:
        return {}
    return dict(zip(metadata["symbol"], metadata["industry"]))


@lru_cache(maxsize=1)
def get_canonical_industries(csv_path: str | None = None) -> Dict[str, str]:
    metadata = get_universe_metadata(csv_path)
    canonical: dict[str, str] = {}
    for industry in metadata.get("industry", pd.Series(dtype=str)).dropna().astype(str):
        normalized = normalize_industry_name(industry)
        if normalized:
            canonical[normalized] = industry
    return canonical


def normalize_industry_name(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("&", " and ").split())


def resolve_industry_filters(industry_filters: list[str] | None, csv_path: str | None = None) -> list[str]:
    if not industry_filters:
        return []

    canonical = get_canonical_industries(csv_path)
    resolved: list[str] = []
    for raw_value in industry_filters:
        for chunk in str(raw_value).split(","):
            normalized = normalize_industry_name(chunk)
            if not normalized:
                continue
            if normalized not in canonical:
                raise ValueError(
                    f"Unknown industry filter: {chunk.strip()}. "
                    f"Choose from {sorted(canonical.values())}"
                )
            resolved_value = canonical[normalized]
            if resolved_value not in resolved:
                resolved.append(resolved_value)
    return resolved


def filter_symbols_by_industry(
    symbols: list[str],
    industries: list[str] | None,
    csv_path: str | None = None,
) -> list[str]:
    if not industries:
        return list(symbols)
    allowed = set(resolve_industry_filters(industries, csv_path))
    metadata = get_universe_metadata(csv_path)
    if metadata.empty:
        return []
    allowed_symbols = set(metadata.loc[metadata["industry"].isin(allowed), "symbol"].tolist())
    return [symbol for symbol in symbols if symbol.upper() in allowed_symbols]


def get_symbol_metadata(symbol: str, csv_path: str | None = None) -> dict[str, str]:
    metadata = get_universe_metadata(csv_path)
    if metadata.empty:
        return {"symbol": symbol.upper(), "industry": "", "company_name": "", "series": "", "isin": ""}
    row = metadata.loc[metadata["symbol"] == symbol.upper()]
    if row.empty:
        return {"symbol": symbol.upper(), "industry": "", "company_name": "", "series": "", "isin": ""}
    record = row.iloc[0].to_dict()
    return {key: str(value) if value is not None else "" for key, value in record.items()}


def get_universe(name: str) -> List[str]:
    """Get list of symbols for a given universe."""
    normalized = name.lower()
    if normalized == "nifty500":
        csv_symbols = _load_nifty500_from_csv()
        if csv_symbols:
            return csv_symbols

        data_dir = _project_root() / "nifty500"
        if not data_dir.exists():
            raise ValueError("nifty500 data directory not found and ind_nifty500list.csv is missing")
        symbols = sorted(
            path.stem.replace("_minute", "")
            for path in data_dir.glob("*_minute.csv")
            if not path.name.startswith(".")
        )
        if not symbols:
            raise ValueError("No local nifty500 minute files found")
        return symbols

    universes = {
        "nifty50": NIFTY50_SYMBOLS,
        "nifty100": NIFTY100_SYMBOLS,
        "nifty200": NIFTY200_SYMBOLS,
    }
    if normalized not in universes:
        raise ValueError(f"Unknown universe: {name}. Choose from {[*universes.keys(), 'nifty500']}")
    return universes[normalized]


def is_in_universe(symbol: str, universe: str = "nifty100") -> bool:
    """Check if a symbol is in the given universe."""
    return symbol.upper() in [s.upper() for s in get_universe(universe)]
