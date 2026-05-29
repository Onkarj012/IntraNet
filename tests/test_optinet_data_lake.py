from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from index_options.data_lake import (
    CANONICAL_OPTION_COLUMNS,
    DataLakeConfig,
    add_expiry_type,
    bhavcopy_url,
    build_expiry_calendar,
    ingest_raw_file,
    missing_trading_days,
    legacy_filename,
    normalize_index_ohlc,
    parse_bhavcopy_bytes,
    source_for_date,
    udiff_filename,
    validate_options_frame,
    write_partitioned_parquet,
)


LEGACY_CSV = """INSTRUMENT,SYMBOL,EXPIRY_DT,STRIKE_PR,OPTION_TYP,OPEN,HIGH,LOW,CLOSE,SETTLE_PR,CONTRACTS,OPEN_INT,CHG_IN_OI,TIMESTAMP
OPTIDX,NIFTY,25-JUL-2024,23350,CE,100,120,90,110,111,500,1000,50,05-JUL-2024
OPTIDX,BANKNIFTY,10-JUL-2024,42000,PE,200,250,180,230,231,700,2000,-20,05-JUL-2024
FUTIDX,NIFTY,25-JUL-2024,0,XX,0,0,0,0,0,0,0,0,05-JUL-2024
"""


UDIFF_CSV = """TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4
2025-01-02,2025-01-02,FO,NSE,IDO,1,,NIFTY,,2025-01-30,2025-01-30,23350,PE,NIFTY25JAN23350PE,140,150,120,147.3,147.3,130,23500,181.4,1200,20,250,0,0,F1,25,,,,,
2025-01-02,2025-01-02,FO,NSE,IDO,2,,BANKNIFTY,,2025-02-27,2025-02-27,55300,CE,BANKNIFTY25FEB55300CE,500,550,450,520,520,480,52500,510,2200,-10,320,0,0,F1,15,,,,,
2025-01-02,2025-01-02,FO,NSE,STO,3,,RELIANCE,,2025-01-30,2025-01-30,1200,CE,REL25JAN1200CE,10,15,8,12,12,11,1210,12,100,0,50,0,0,F1,250,,,,,
"""


def test_date_routing_and_filenames():
    assert source_for_date(date(2024, 7, 5)) == "legacy"
    assert source_for_date(date(2024, 7, 8)) == "udiff"
    assert legacy_filename(date(2020, 1, 1)) == "fo01JAN2020bhav.csv.zip"
    assert udiff_filename(date(2025, 5, 20)) == "BhavCopy_NSE_FO_0_0_0_20250520_F_0000.csv.zip"
    assert "DERIVATIVES/2020/JAN/fo01JAN2020bhav.csv.zip" in bhavcopy_url(date(2020, 1, 1), "legacy")
    assert "BhavCopy_NSE_FO_0_0_0_20250520_F_0000.csv.zip" in bhavcopy_url(date(2025, 5, 20), "udiff")


def test_legacy_and_udiff_parse_to_same_canonical_schema():
    legacy = parse_bhavcopy_bytes(LEGACY_CSV, date(2024, 7, 5))
    udiff = parse_bhavcopy_bytes(UDIFF_CSV, date(2025, 1, 2))

    assert CANONICAL_OPTION_COLUMNS == legacy.columns[: len(CANONICAL_OPTION_COLUMNS)].tolist()
    assert CANONICAL_OPTION_COLUMNS == udiff.columns[: len(CANONICAL_OPTION_COLUMNS)].tolist()
    assert set(legacy["symbol"]) == {"NIFTY", "BANKNIFTY"}
    assert set(udiff["symbol"]) == {"NIFTY", "BANKNIFTY"}
    assert set(legacy["source_format"]) == {"legacy"}
    assert set(udiff["source_format"]) == {"udiff"}


def test_validation_isolates_bad_rows():
    frame = parse_bhavcopy_bytes(LEGACY_CSV, date(2024, 7, 5))
    bad_extra = frame.iloc[[0]].copy()
    bad_extra["high"] = 1
    bad_extra["low"] = 2
    bad_extra["open_interest"] = -1
    mixed = pd.concat([frame, bad_extra], ignore_index=True)

    good, bad, report = validate_options_frame(mixed)

    assert report.bad_row_count >= 2
    assert "duplicate_rows" in report.issues
    assert "invalid_price_range" in report.issues
    assert "negative_open_interest" in report.issues
    assert len(good) < len(mixed)
    assert not bad.empty


def test_missing_trading_days_ignores_weekends_and_holidays():
    frame = parse_bhavcopy_bytes(LEGACY_CSV, date(2024, 7, 5))
    missing = missing_trading_days(frame, date(2024, 7, 4), date(2024, 7, 8), holidays=[date(2024, 7, 4)])

    assert missing == [date(2024, 7, 8)]


def test_index_normalization_and_expiry_calendar():
    index_raw = pd.DataFrame(
        {
            "SYMBOL": ["NIFTY 50", "NIFTY BANK"],
            "Date": ["2025-01-02", "2025-01-02"],
            "OPEN": [24000, 52000],
            "HIGH": [24100, 52100],
            "LOW": [23900, 51900],
            "CLOSE": [24050, 52050],
        }
    )
    index = normalize_index_ohlc(index_raw)
    options = add_expiry_type(parse_bhavcopy_bytes(UDIFF_CSV, date(2025, 1, 2)))
    calendar = build_expiry_calendar(options)

    assert set(index["index_name"]) == {"NIFTY", "BANKNIFTY"}
    assert {"index_name", "date", "expiry_date", "expiry_type", "is_expiry_day", "days_to_expiry"}.issubset(
        calendar.columns
    )
    assert calendar["days_to_expiry"].ge(0).all()


def test_partitioned_parquet_writer(tmp_path):
    frame = parse_bhavcopy_bytes(UDIFF_CSV, date(2025, 1, 2))
    written = write_partitioned_parquet(frame, tmp_path)

    assert len(written) == 2
    assert (tmp_path / "symbol=NIFTY" / "year=2025" / "options_20250102.parquet").exists()
    assert (tmp_path / "symbol=BANKNIFTY" / "year=2025" / "options_20250102.parquet").exists()


def test_ingest_raw_csv_writes_validation_and_parquet(tmp_path):
    raw = tmp_path / "BhavCopy_NSE_FO_0_0_0_20250102_F_0000.csv"
    raw.write_text(UDIFF_CSV)
    config = DataLakeConfig(data_root=tmp_path / "data")

    frame, report, written = ingest_raw_file(raw, config)

    assert len(frame) == 2
    assert report.bad_row_count == 0
    assert len(written) == 2
