"""
OptiNet FastAPI Backend
-----------------------
Self-contained server that:
  - Generates intraday equity recommendations (pre-open + post-open)
  - Runs EOD paper reconciliation
  - Schedules all three runs automatically (IST timezone)
  - Serves all data via REST endpoints

Directory layout (everything lives inside backend/):
  app.py              this file
  equity/             src/equity package (copied)
  models/             v8_intraday models
  market_data_cache/  macro CSV files
  data/sentiment/     ind_nifty500list.csv
  results/            generated artifacts
  logs/               run logs
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))   # makes `import equity` work

IST = ZoneInfo("Asia/Kolkata")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("optinet")

# ── scheduler ──────────────────────────────────────────────────────────────
_scheduled: dict[str, bool] = {}


async def _wait_until(h: int, m: int) -> None:
    """Sleep until next occurrence of HH:MM IST (skips weekends)."""
    while True:
        now = datetime.now(IST)
        target = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if target <= now:
            target = target.replace(day=now.day + 1)
        secs = (target - now).total_seconds()
        await asyncio.sleep(secs)
        # skip weekends (Sat=5, Sun=6)
        if datetime.now(IST).weekday() < 5:
            return


async def _scheduler_loop():
    """Run the three daily pipeline modes at their scheduled IST times."""
    log.info("Scheduler started (IST)")
    while True:
        now = datetime.now(IST)
        t = now.time()

        if time(8, 29) <= t <= time(8, 31):
            log.info("Scheduler: running preopen")
            await asyncio.to_thread(_run_pipeline, "preopen")
            await asyncio.sleep(120)   # don't re-trigger for 2 min

        elif time(9, 21) <= t <= time(9, 23):
            log.info("Scheduler: running postopen")
            await asyncio.to_thread(_run_pipeline, "postopen")
            await asyncio.sleep(120)

        elif time(15, 34) <= t <= time(15, 36):
            log.info("Scheduler: running eod")
            await asyncio.to_thread(_run_pipeline, "eod")
            await asyncio.sleep(120)

        else:
            await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_scheduler_loop())
    log.info("OptiNet backend started")
    yield


# ── pipeline runner ────────────────────────────────────────────────────────
def _run_pipeline(mode: str) -> dict:
    """Run morning_run.py or eod steps inline (no subprocess needed)."""
    from equity.momentum import load_ohlcv, load_panel, momentum_score, adv, select
    from equity.intraday_model import score_universe
    from equity.universe import get_symbol_metadata

    ts = datetime.now(IST).isoformat(timespec="seconds")
    log.info(f"Pipeline [{mode}] started at {ts}")

    try:
        if mode in ("preopen", "postopen"):
            result = _generate_recommendations(post_open=(mode == "postopen"))
        elif mode == "eod":
            result = _run_eod()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        _write_status(mode, ts, ok=True)
        log.info(f"Pipeline [{mode}] done")
        return result

    except Exception as e:
        log.exception(f"Pipeline [{mode}] failed: {e}")
        _write_status(mode, ts, ok=False, error=str(e))
        raise


def _generate_recommendations(post_open: bool = False) -> dict:
    """Core recommendation generation — mirrors morning_run.py logic."""
    from equity.momentum import load_ohlcv, load_panel, momentum_score, vol63, adv, select
    from equity.intraday_model import score_universe
    from equity.universe import get_symbol_metadata, get_universe

    log.info("Updating OHLCV panel via yfinance...")
    panel_path = ROOT / "cache/v8/daily_panel_nifty500_adj.parquet"
    panel_path.parent.mkdir(parents=True, exist_ok=True)

    # Update panel
    _update_panel(panel_path)

    # Load OHLCV
    ohlcv = load_ohlcv(str(panel_path))
    symbols = get_universe("nifty500")

    # Momentum screening
    scores = momentum_score(ohlcv, symbols)
    adv_map = adv(ohlcv, symbols)
    picks_symbols = select(scores, adv_map, top_n=20, min_adv=50_000_000)

    # Score with ensemble
    picks = score_universe(
        symbols=picks_symbols,
        ohlcv=ohlcv,
        model_dir=str(ROOT / "models"),
        macro_dir=str(ROOT / "market_data_cache"),
        sentiment_csv=str(ROOT / "data/sentiment/ind_nifty500list.csv"),
        post_open=post_open,
    )

    # Futures plan
    futures_plan = _futures_plan(ohlcv)

    rec = {
        "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
        "as_of": str(datetime.now(IST).date()),
        "equity": picks,
        "futures": futures_plan,
    }
    out = ROOT / "results/recommendations.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    log.info(f"Recommendations written → {out}")
    return rec


def _update_panel(panel_path: Path):
    """Pull latest OHLCV from yfinance and update the panel."""
    import yfinance as yf
    from equity.universe import get_universe

    symbols = get_universe("nifty500")
    yf_symbols = [f"{s}.NS" for s in symbols]

    try:
        raw = yf.download(yf_symbols, period="5d", auto_adjust=True, progress=False)
        if raw.empty:
            log.warning("yfinance returned empty data — skipping panel update")
            return

        # If panel exists, merge; otherwise create fresh
        if panel_path.exists():
            existing = pd.read_parquet(panel_path)
            # Merge new rows — simplified: just overwrite last 5 days
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in raw.columns.get_level_values(0):
                    new_slice = raw[col]
                    new_slice.columns = [s.replace(".NS", "") for s in new_slice.columns]
                    existing[col.lower()] = existing[col.lower()].combine_first(new_slice)
            existing.to_parquet(panel_path)
        else:
            log.warning("Panel not found — run full backfill first")
    except Exception as e:
        log.warning(f"Panel update failed: {e}")


def _futures_plan(ohlcv) -> dict:
    """Generate futures session plan from VIX and 5d return."""
    try:
        vix_path = ROOT / "market_data_cache/cboe_vix.csv"
        if vix_path.exists():
            vix_df = pd.read_csv(vix_path, index_col=0, parse_dates=True)
            vix = float(vix_df.iloc[-1, 0])
        else:
            vix = 15.0

        nifty_close = ohlcv["close"].get("NIFTY50", ohlcv["close"].median(axis=1))
        ret_5d = float(nifty_close.pct_change(5).iloc[-1]) if len(nifty_close) >= 6 else 0.0
        entry = float(nifty_close.iloc[-1]) if hasattr(nifty_close, "iloc") else 23000.0

        tradeable = vix < 22.0 and ret_5d > -0.015
        target = entry * 1.004
        stop = entry * 0.997

        return {
            "tradeable": tradeable,
            "note": "conditions met" if tradeable else (
                "high VIX" if vix >= 22 else "weak 5d trend — Phase-2 guard blocks entries."
            ),
            "vix": round(vix, 2),
            "vix_cut": 22.0,
            "ret_5d": round(ret_5d, 4),
            "ret_cut": -0.015,
            "entry": round(entry, 2),
            "current": round(entry, 2),
            "target": round(target, 2),
            "stop": round(stop, 2),
            "target_pct": 0.004,
            "stop_pct": 0.003,
            "roi": 0.004,
            "rr": 1.33,
            "size_hint": "1.0–1.5× lot",
            "lot": 50,
            "win_rate": 0.4991,
            "confidence": 49 if tradeable else 0,
        }
    except Exception as e:
        log.warning(f"Futures plan failed: {e}")
        return {"tradeable": False, "note": str(e)}


def _run_eod() -> dict:
    """EOD reconciliation — updates paper ledger."""
    from equity.intraday_features import IntradayFeatureAssembler  # noqa: F401
    ledger_path = ROOT / "results/equity/intraday_paper_ledger.csv"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    rec_path = ROOT / "results/recommendations.json"
    if not rec_path.exists():
        return {"status": "no recommendations to reconcile"}

    rec = json.loads(rec_path.read_text())
    picks = rec.get("equity", {}).get("picks", [])
    trade_date = rec.get("as_of", str(datetime.now(IST).date()))

    rows = []
    for p in picks:
        symbol = p["symbol"]
        try:
            import yfinance as yf
            ticker = yf.Ticker(f"{symbol}.NS")
            hist = ticker.history(period="2d")
            if hist.empty:
                continue
            today = hist.iloc[-1]
            entry = p["entry"]
            stop = p["stop"]
            close = float(today["Close"])
            high = float(today["High"])
            low = float(today["Low"])
            atr_pct = (high - low) / entry if entry else 0.02
            stop_lvl = entry * (1 - 2.0 * atr_pct)
            outcome = "STOP" if low <= stop else ("WIN" if close > entry else "LOSS")
            pnl_pct = (stop_lvl - entry) / entry if outcome == "STOP" else (close - entry) / entry
            pnl_pct -= 0.0018  # round-trip costs
            rows.append({
                "date": trade_date, "symbol": symbol,
                "dir_p": p.get("p_up", 0), "regime": p.get("regime", ""),
                "entry": round(entry, 2), "stop": round(stop, 2),
                "high": round(high, 2), "low": round(low, 2), "close": round(close, 2),
                "outcome": outcome, "pnl_pct": round(pnl_pct, 5),
                "pnl_inr": round(pnl_pct * 100000 / max(len(picks), 1), 2),
            })
        except Exception as e:
            log.warning(f"EOD reconcile failed for {symbol}: {e}")

    if rows:
        new_df = pd.DataFrame(rows)
        if ledger_path.exists():
            existing = pd.read_csv(ledger_path)
            existing = existing[existing["date"] != trade_date]
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(ledger_path, index=False)
        log.info(f"EOD: {len(rows)} trades reconciled → {ledger_path}")

    return {"reconciled": len(rows), "date": trade_date}


def _write_status(mode: str, ts: str, ok: bool, error: str = ""):
    status_path = ROOT / "results/intraday_daily_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    prior = {}
    if status_path.exists():
        try:
            prior = json.loads(status_path.read_text())
        except Exception:
            prior = {}
    runs = prior.get("runs", {})
    runs[mode] = {"mode": mode, "run_timestamp": ts, "ok": ok, "error": error}
    status_path.write_text(json.dumps(
        {"updated_at": datetime.now(IST).isoformat(timespec="seconds"), "runs": runs}, indent=2
    ))


# ── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(title="OptiNet", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise HTTPException(404, f"{path.name} not found — run the pipeline first")
    return json.loads(path.read_text())


def _read_csv(path: Path) -> str:
    if not path.exists():
        raise HTTPException(404, f"{path.name} not found")
    return path.read_text()


# ── endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "OptiNet backend"}


@app.get("/api/recommendations")
def recommendations():
    return _read_json(ROOT / "results/recommendations.json")


@app.get("/api/ledger")
def ledger():
    return PlainTextResponse(
        _read_csv(ROOT / "results/equity/intraday_paper_ledger.csv"),
        media_type="text/csv",
    )


@app.get("/api/status")
def status():
    return _read_json(ROOT / "results/intraday_daily_status.json")


@app.get("/api/walk-forward")
def walk_forward():
    return _read_json(ROOT / "results/equity/walk_forward_results.json")


@app.get("/api/regime")
def regime():
    return _read_json(ROOT / "models/exposure_by_regime.json")


@app.get("/api/train-meta")
def train_meta():
    return _read_json(ROOT / "models/train_meta.json")


@app.get("/api/drift")
def drift():
    """Compute rolling win-rate and Sharpe from paper ledger."""
    ledger_path = ROOT / "results/equity/intraday_paper_ledger.csv"
    if not ledger_path.exists():
        return {"trades": 0, "status": "no data"}
    df = pd.read_csv(ledger_path)
    recent = df.tail(20)
    win_rate = float((recent["outcome"] == "WIN").mean()) if len(recent) else 0.0
    pnl = df["pnl_pct"].values
    sharpe = (float(pnl.mean()) / float(pnl.std()) * (252 ** 0.5)) if len(pnl) > 1 and pnl.std() > 0 else 0.0
    return {
        "trades": len(df),
        "win_rate_20": round(win_rate, 3),
        "sharpe": round(sharpe, 2),
        "total_pnl_inr": round(float(df["pnl_inr"].sum()), 2),
        "status": "ok",
    }


@app.api_route("/api/run/{mode}", methods=["GET", "POST"])
async def run_mode(mode: str, background_tasks: BackgroundTasks):
    """Manually trigger preopen / postopen / eod pipeline."""
    if mode not in ("preopen", "postopen", "eod"):
        raise HTTPException(400, "mode must be preopen, postopen, or eod")
    background_tasks.add_task(asyncio.to_thread, _run_pipeline, mode)
    return {"status": "started", "mode": mode}
