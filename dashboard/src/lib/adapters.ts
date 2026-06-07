// Transforms repo artifacts into the shapes expected by the StockXpert UI.
import fs from "node:fs";
import path from "node:path";
import { srcText } from "./source";

async function readJson<T>(key: string): Promise<T | null> {
  const t = await srcText(key);
  if (t == null) return null;
  try {
    return JSON.parse(t) as T;
  } catch {
    return null;
  }
}

function parseCsv(text: string): Record<string, string>[] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else inQuotes = false;
      } else field += c;
    } else if (c === '"') inQuotes = true;
    else if (c === ",") {
      row.push(field);
      field = "";
    } else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field);
      field = "";
      if (row.length > 1 || row[0] !== "") rows.push(row);
      row = [];
    } else field += c;
  }
  if (field !== "" || row.length) {
    row.push(field);
    if (row.length > 1 || row[0] !== "") rows.push(row);
  }
  if (rows.length < 2) return [];
  const header = rows[0];
  return rows.slice(1).map((r) => {
    const o: Record<string, string> = {};
    header.forEach((h, i) => (o[h] = r[i] ?? ""));
    return o;
  });
}

function sampleStd(xs: number[]): number {
  const n = xs.length;
  if (n < 2) return 0;
  const m = xs.reduce((a, b) => a + b, 0) / n;
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (n - 1));
}

function repoRoot(): string {
  if (process.env.REPO_ROOT) return path.resolve(process.env.REPO_ROOT);
  return path.resolve(process.cwd(), "..");
}

function haltPath(kind: "equity" | "futures"): string {
  return kind === "equity"
    ? path.join(repoRoot(), "results/equity/INTRADAY_HALTED")
    : path.join(repoRoot(), "results/router_v0/PAPER_TRADING_HALTED");
}

function parseTestPeriod(test: string): { start: string; end: string } {
  const parts = test.split("→");
  if (parts.length === 2) return { start: parts[0].trim(), end: parts[1].trim() };
  return { start: test, end: test };
}

// ── Recommendations (Brief page) ──

export async function getBriefRecommendations() {
  const rec = await readJson<any>("results/recommendations.json");
  if (!rec) {
    return {
      generated_at: new Date().toISOString(),
      equity: { exposure: 0, picks: [] },
      futures: { tradeable: false, vix: 0, note: "No recommendations yet", stance: null, entry: null, target: null, stop: null, roi: null, rr: null, lot: null, confidence: null },
    };
  }

  const exposureMap = (await readJson<Record<string, number>>("models/v8_intraday/exposure_by_regime.json")) ?? {};
  const picks = (rec.equity?.picks ?? []).map((p: any) => ({
    symbol: p.symbol,
    direction: p.direction ?? "LONG",
    p_up: p.p_up ?? (p.confidence != null ? p.confidence / 100 : 0.5),
    entry: p.entry,
    target: p.target,
    stop: p.stop,
    rr: p.rr,
    regime: p.regime ?? "neutral",
    expected_value: p.expected_value ?? 0,
    drivers: p.drivers ?? { momentum: 0.5, reversal: 0.5, breakout: 0.5, sentiment: 0.5, macro: 0.5 },
  }));

  const regime = picks[0]?.regime ?? "choppy_reverting";
  const exposureFrac = exposureMap[regime] ?? 1;
  const exposure = Math.round(exposureFrac * 100);

  const fu = rec.futures ?? {};
  return {
    generated_at: rec.generated_at,
    equity: { exposure, picks },
    futures: {
      tradeable: !!fu.tradeable,
      entry: fu.entry ?? null,
      target: fu.target ?? null,
      stop: fu.stop ?? null,
      roi: fu.roi != null ? Math.round(fu.roi * 10000) / 100 : null,
      rr: fu.rr ?? null,
      lot: fu.lot ?? null,
      confidence: fu.confidence ?? null,
      vix: fu.vix ?? 0,
      stance: fu.stance ?? null,
      note: fu.note ?? null,
    },
  };
}

// ── Ledger CSV ──

export async function getLedgerCsv(): Promise<string> {
  const raw = await srcText("results/equity/intraday_paper_ledger.csv");
  if (!raw?.trim()) {
    return "date,symbol,direction,dir_p,entry,stop,high,low,close,outcome,pnl_inr,pnl_pct,regime,exposure,capital_after\n";
  }

  const rows = parseCsv(raw);
  if (!rows.length) return raw;

  const header = Object.keys(rows[0]);
  const hasDirection = header.includes("direction");
  const outHeader = hasDirection
    ? header
    : ["date", "symbol", "direction", ...header.filter((h) => h !== "date" && h !== "symbol")];

  const lines = [outHeader.join(",")];
  for (const r of rows) {
    const exposure = r.exposure != null ? Math.round(parseFloat(r.exposure) * 100) : 100;
    const vals = hasDirection
      ? header.map((h) => (h === "exposure" ? String(exposure) : r[h] ?? ""))
      : [
          r.date,
          r.symbol,
          "LONG",
          r.dir_p ?? "",
          r.entry ?? "",
          r.stop ?? "",
          r.high ?? "",
          r.low ?? "",
          r.close ?? "",
          r.outcome ?? "",
          r.pnl_inr ?? "",
          r.pnl_pct ?? "",
          r.regime ?? "",
          String(exposure),
          r.capital_after ?? "",
        ];
    lines.push(vals.join(","));
  }
  return lines.join("\n") + "\n";
}

// ── Pipeline status (Ops page) ──

export async function getPipelineStatus() {
  const intraday = await readJson<any>("results/intraday_daily_status.json");
  const daily = await readJson<any>("results/daily_run_status.json");

  let run: any = null;
  if (intraday?.runs) {
    const modes = ["preopen", "postopen", "eod"];
    for (const m of [...modes].reverse()) {
      if (intraday.runs[m]) {
        run = intraday.runs[m];
        break;
      }
    }
  }
  if (!run && daily) run = daily;

  if (!run) {
    return {
      last_run_mode: "unknown",
      timestamp: new Date().toISOString(),
      total_duration: 0,
      steps_passed: 0,
      steps_total: 0,
      steps: [],
    };
  }

  const steps = (run.steps ?? []).map((s: any) => ({
    label: s.label,
    return_code: s.return_code ?? (s.ok ? 0 : 1),
    duration: s.duration ?? 0,
    stdout: s.stdout ?? (s.ok ? "" : ""),
    stderr: s.stderr ?? "",
  }));

  const passed = steps.filter((s: { return_code: number }) => s.return_code === 0).length;

  return {
    last_run_mode: run.mode ?? "daily_run",
    timestamp: run.run_timestamp ?? run.timestamp ?? new Date().toISOString(),
    total_duration: steps.reduce((a: number, s: { duration: number }) => a + (s.duration ?? 0), 0),
    steps_passed: passed,
    steps_total: steps.length,
    steps,
  };
}

// ── Train meta ──

export async function getTrainMeta() {
  const meta = await readJson<any>("models/v8_intraday/train_meta.json");
  if (!meta) {
    return {
      trained_at: "",
      dir_auc: 0,
      meta_auc: 0,
      model_version: "unknown",
      retrain_threshold_days: 14,
    };
  }
  return {
    trained_at: meta.trained_at,
    dir_auc: meta.dir_auc ?? meta.val_auc ?? 0,
    meta_auc: meta.meta_auc ?? meta.barrier_auc ?? 0,
    training_samples: meta.train_rows,
    validation_samples: meta.val_rows,
    model_version: "v8_intraday",
    features_used: meta.feature_count,
    retrain_threshold_days: 14,
  };
}

// ── Drift monitor ──

export async function getDriftStatus() {
  const csv = await srcText("results/equity/intraday_paper_ledger.csv");
  const halts = (await readJson<{ futures: boolean; equity: boolean }>("@halts")) ?? {
    futures: fs.existsSync(haltPath("futures")),
    equity: fs.existsSync(haltPath("equity")),
  };

  let currentSharpe = 0;
  let winRate = 0;
  if (csv) {
    const rows = parseCsv(csv);
    const last = rows.slice(-20);
    const pnls = last.map((r) => parseFloat(r.pnl_pct)).filter((x) => !Number.isNaN(x));
    if (pnls.length) {
      winRate = (pnls.filter((x) => x > 0).length / pnls.length) * 100;
      const std = sampleStd(pnls);
      currentSharpe = std > 0 ? (pnls.reduce((a, b) => a + b, 0) / pnls.length / std) * Math.sqrt(252) : 0;
    }
  }

  const sharpeThreshold = 0.8;
  const winRateThreshold = 48;
  const driftOk = currentSharpe >= sharpeThreshold && winRate >= winRateThreshold;

  return {
    current_sharpe: Math.round(currentSharpe * 100) / 100,
    sharpe_threshold: sharpeThreshold,
    win_rate: Math.round(winRate * 10) / 10,
    win_rate_threshold: winRateThreshold,
    drift_level: driftOk ? "OK" : "WARN",
    equity_halt_active: halts.equity,
    futures_halt_active: halts.futures,
    halt_file_exists: halts.futures || halts.equity,
  };
}

export async function updateDriftStatus(body: Record<string, unknown>) {
  if (typeof body.equity_halt_active === "boolean") {
    const p = haltPath("equity");
    if (body.equity_halt_active) fs.writeFileSync(p, "manual halt via dashboard\n");
    else if (fs.existsSync(p)) fs.unlinkSync(p);
  }
  if (typeof body.futures_halt_active === "boolean") {
    const p = haltPath("futures");
    if (body.futures_halt_active) fs.writeFileSync(p, "manual halt via dashboard\n");
    else if (fs.existsSync(p)) fs.unlinkSync(p);
  }
  return getDriftStatus();
}

// ── Walk-forward (Performance page) ──

export async function getWalkForward() {
  const wf = await readJson<any>("results/equity/walk_forward_results.json");
  const fw = await readJson<any>("results/router_v0/forward_walk_summary.json");
  const liveRows = parseCsv((await srcText("results/router_v0/paper_trading_ledger.csv")) ?? "");
  const liveA = liveRows.filter((r) => r.source === "paper");

  if (!wf) {
    return {
      folds: [],
      aggregate: {
        dir_auc: 0, meta_auc: 0, sharpe: 0, pnl_inr: 0, win_rate: 0,
        positive_folds: 0, total_folds: 0, annual_return_pct: 0,
        annual_return_slippage_pct: 0, sharpe_after_slippage: 0,
      },
      futures: { pnl_inr: 0, sharpe: 0, win_rate: 0, profit_factor: 0, trades: 0, status: "UNKNOWN" },
    };
  }

  const folds = (wf.folds ?? []).map((f: any, i: number) => {
    const { start, end } = parseTestPeriod(f.test ?? "");
    return {
      fold: i + 1,
      period_start: start,
      period_end: end,
      dir_auc: f.auc ?? 0,
      meta_auc: f.auc ?? 0,
      sharpe: f.sharpe ?? 0,
      pnl_inr: f.pnl ?? 0,
      win_rate: (f.pct_prof ?? 0) * 100,
      result: (f.pnl ?? 0) > 0 ? "POSITIVE" : "NEGATIVE",
    };
  });

  const summary = wf.summary ?? {};
  const totalPnl = summary.total_pnl ?? folds.reduce((a: number, f: { pnl_inr: number }) => a + f.pnl_inr, 0);
  const aggSharpe = summary.mean_sharpe ?? 0;

  const phase3 = fw?.phase3_no_guard ?? {};
  const livePnl = liveA.reduce((a, r) => a + (Number(r.net_pnl_inr) || 0), 0);
  const liveWins = liveA.filter((r) => (Number(r.net_pnl_inr) || 0) > 0).length;

  return {
    folds,
    aggregate: {
      dir_auc: summary.mean_auc ?? 0,
      meta_auc: summary.mean_auc ?? 0,
      sharpe: aggSharpe,
      pnl_inr: totalPnl,
      win_rate: folds.length
        ? folds.reduce((a: number, f: { win_rate: number }) => a + f.win_rate, 0) / folds.length
        : 0,
      positive_folds: summary.positive_folds ?? folds.filter((f: { result: string }) => f.result === "POSITIVE").length,
      total_folds: summary.total_folds ?? folds.length,
      annual_return_pct: Math.round((totalPnl / 100_000) * 100 * 10) / 10,
      annual_return_slippage_pct: Math.round((totalPnl / 100_000) * 100 * 0.8 * 10) / 10,
      sharpe_after_slippage: Math.round(aggSharpe * 0.7 * 100) / 100,
    },
    futures: {
      pnl_inr: livePnl || phase3.total_pnl_inr || 0,
      sharpe: phase3.sharpe_daily_ann ?? 0,
      win_rate: liveA.length ? (liveWins / liveA.length) * 100 : (phase3.win_rate ?? 0) * 100,
      profit_factor: phase3.profit_factor ?? 0,
      trades: liveA.length || phase3.n_trades || 0,
      status: fs.existsSync(haltPath("futures")) ? "HARD_HALT" : "SOFT_HALT",
    },
  };
}

// ── Regime breakdown ──

export async function getRegimeBreakdown() {
  const csv = await srcText("results/equity/intraday_paper_ledger.csv");
  const exposureMap = (await readJson<Record<string, number>>("models/v8_intraday/exposure_by_regime.json")) ?? {};

  const regimeNames = [
    "trending_bull",
    "choppy_reverting",
    "choppy_breakout",
    "low_vol",
    "trending_bear",
    "crisis",
    "strong_trend_up",
    "low_vol_compression",
    "strong_trend_down",
    "high_vol_crisis",
  ];

  const stats: Record<string, { trades: number; wins: number; pnl: number; pnls: number[] }> = {};
  for (const r of regimeNames) stats[r] = { trades: 0, wins: 0, pnl: 0, pnls: [] };

  if (csv) {
    for (const row of parseCsv(csv)) {
      const regime = row.regime || "neutral";
      if (!stats[regime]) stats[regime] = { trades: 0, wins: 0, pnl: 0, pnls: [] };
      const pnl = parseFloat(row.pnl_inr) || 0;
      const pct = parseFloat(row.pnl_pct) || 0;
      stats[regime].trades++;
      if (pct > 0) stats[regime].wins++;
      stats[regime].pnl += pnl;
      stats[regime].pnls.push(pct);
    }
  }

  const allRegimes = new Set([...regimeNames, ...Object.keys(exposureMap), ...Object.keys(stats)]);

  const regimes = [...allRegimes].map((regime) => {
    const s = stats[regime] ?? { trades: 0, wins: 0, pnl: 0, pnls: [] };
    const winRate = s.trades ? (s.wins / s.trades) * 100 : 0;
    const avgPnl = s.trades ? Math.round(s.pnl / s.trades) : 0;
    const std = sampleStd(s.pnls);
    const sharpe = s.pnls.length > 1 && std > 0
      ? (s.pnls.reduce((a, b) => a + b, 0) / s.pnls.length / std) * Math.sqrt(252)
      : 0;
    const expFrac = exposureMap[regime] ?? 0;
    return {
      regime,
      trades: s.trades,
      win_rate: Math.round(winRate * 10) / 10,
      avg_pnl: avgPnl,
      sharpe: Math.round(sharpe * 100) / 100,
      exposure: Math.round(expFrac * 100),
    };
  });

  regimes.sort((a, b) => b.trades - a.trades);
  return { regimes };
}
