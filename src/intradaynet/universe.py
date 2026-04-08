"""
Stock universe definitions for IntradayNet.

Defines NIFTY indices constituents for filtering and backtesting.
Using actual NIFTY100 as of 2024 for reproducibility.
"""

from typing import List, Dict, Optional

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


def get_universe(name: str) -> List[str]:
    """Get list of symbols for a given universe."""
    universes = {
        "nifty50": NIFTY50_SYMBOLS,
        "nifty100": NIFTY100_SYMBOLS,
        "nifty200": NIFTY200_SYMBOLS,
    }
    if name.lower() not in universes:
        raise ValueError(f"Unknown universe: {name}. Choose from {list(universes.keys())}")
    return universes[name.lower()]


def is_in_universe(symbol: str, universe: str = "nifty100") -> bool:
    """Check if a symbol is in the given universe."""
    return symbol.upper() in [s.upper() for s in get_universe(universe)]
