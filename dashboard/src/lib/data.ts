// Data layer. Reads raw artifacts via the pluggable source (local files in
// dev, Convex over HTTP when hosted) and computes the same metrics as
// scripts/trading/paper_status.py.
import { srcText } from "./source";
import type {
  Metrics,
  CurvePoint,
  FuturesPayload,
  EquityPayload,
  FuturesTrade,
  OpsPayload,
  OpsStep,
  RecommendationsPayload,
} from "./types";

async function readJson<T>(key: string): Promise<T | null> {
  const t = await srcText(key);
  if (t == null) return null;
  try {
    return JSON.parse(t) as T;
  } catch {
    return null;
  }
}

// ── Minimal RFC-4180 CSV parser (handles quoted fields + "" escapes) ──
function parseCsv(text: string): string[][] {
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
  return rows;
}

async function readCsvObjects(key: string): Promise<Record<string, string>[]> {
  const text = await srcText(key);
  if (!text) return [];
  const rows = parseCsv(text);
  if (rows.length < 2) return [];
  const header = rows[0];
  return rows.slice(1).map((r) => {
    const o: Record<string, string> = {};
    header.forEach((h, i) => (o[h] = r[i] ?? ""));
    return o;
  });
}

// ── Metrics (mirror paper_status.py) ──
function sampleStd(xs: number[]): number {
  const n = xs.length;
  if (n < 2) return 0;
  const m = xs.reduce((a, b) => a + b, 0) / n;
  const v = xs.reduce((a, b) => a + (b - m) ** 2, 0) / (n - 1);
  return Math.sqrt(v);
}

type Trade = { tradeDate: string; netPnl: number };

function dailySums(trades: Trade[]): Map<string, number> {
  const m = new Map<string, number>();
  for (const t of trades) m.set(t.tradeDate, (m.get(t.tradeDate) ?? 0) + t.netPnl);
  return m;
}

function metrics(trades: Trade[]): Metrics {
  if (trades.length === 0)
    return {
      nTrades: 0,
      nDays: 0,
      winRate: 0,
      totalPnl: 0,
      meanPnl: 0,
      sharpe: 0,
      profitFactor: 0,
      maxDrawdown: 0,
    };
  const pnls = trades.map((t) => t.netPnl);
  const total = pnls.reduce((a, b) => a + b, 0);
  const daily = [...dailySums(trades).values()];
  const dMean = daily.reduce((a, b) => a + b, 0) / daily.length;
  const dStd = sampleStd(daily);
  const sharpe = daily.length > 1 && dStd > 0 ? (dMean / dStd) * Math.sqrt(252) : 0;
  const gw = pnls.filter((x) => x > 0).reduce((a, b) => a + b, 0);
  const gl = Math.abs(pnls.filter((x) => x < 0).reduce((a, b) => a + b, 0));
  let cum = 0;
  let peak = 0;
  let maxDd = 0;
  for (const x of pnls) {
    cum += x;
    peak = Math.max(peak, cum);
    maxDd = Math.min(maxDd, cum - peak);
  }
  return {
    nTrades: trades.length,
    nDays: new Set(trades.map((t) => t.tradeDate)).size,
    winRate: pnls.filter((x) => x > 0).length / pnls.length,
    totalPnl: total,
    meanPnl: total / pnls.length,
    sharpe,
    // null = no losing trades (infinite PF); JSON cannot represent Infinity.
    profitFactor: gl > 0 ? gw / gl : gw > 0 ? null : 0,
    maxDrawdown: maxDd,
  };
}

function dailyCurve(trades: Trade[]): CurvePoint[] {
  const sums = dailySums(trades);
  const dates = [...sums.keys()].sort();
  let cum = 0;
  return dates.map((d) => {
    cum += sums.get(d)!;
    return { date: d, value: cum };
  });
}

function trailing(trades: Trade[], days: number): Trade[] {
  if (trades.length === 0) return [];
  const maxDate = trades.reduce((m, t) => (t.tradeDate > m ? t.tradeDate : m), "");
  const cutoff = new Date(maxDate);
  cutoff.setDate(cutoff.getDate() - days);
  const cutStr = cutoff.toISOString().slice(0, 10);
  return trades.filter((t) => t.tradeDate > cutStr);
}

function consecLosingDays(trades: Trade[]): number {
  const sums = dailySums(trades);
  const dates = [...sums.keys()].sort();
  let n = 0;
  for (let i = dates.length - 1; i >= 0; i--) {
    if (sums.get(dates[i])! < 0) n++;
    else break;
  }
  return n;
}

const HARD_DD = -150_000;
const SOFT_30D_SHARPE = 0.5;
const SOFT_5D_PNL = -50_000;
const SOFT_CONSEC = 7;

// ── Futures payload ──
export async function getFuturesPayload(): Promise<FuturesPayload> {
  const rows = await readCsvObjects("results/router_v0/paper_trading_ledger.csv");
  const toTrade = (r: Record<string, string>): Trade => ({
    tradeDate: r.trade_date,
    netPnl: Number(r.net_pnl_inr) || 0,
  });
  const all = rows.map(toTrade);
  const liveA = rows.filter((r) => r.source === "paper");
  const liveC = rows.filter((r) => r.source === "paper_c");
  const bootstrap = rows.filter((r) => r.source === "forward_walk");

  const liveATrades = liveA.map(toTrade);
  const t30 = trailing(liveATrades, 30);
  const t5 = trailing(liveATrades, 5);

  const allM = metrics(liveATrades);
  const m30 = metrics(t30);
  const m5 = metrics(t5);
  const consec = consecLosingDays(liveATrades);
  const hard: string[] = [];
  const soft: string[] = [];
  if (allM.maxDrawdown <= HARD_DD)
    hard.push(`Cumulative drawdown ${Math.round(allM.maxDrawdown)} ≤ ${HARD_DD}`);
  if (m30.nTrades >= 30 && m30.sharpe < SOFT_30D_SHARPE)
    soft.push(`30-day Sharpe ${m30.sharpe.toFixed(2)} < ${SOFT_30D_SHARPE}`);
  if (m5.totalPnl <= SOFT_5D_PNL)
    soft.push(`Trailing 5-day PnL ${Math.round(m5.totalPnl)} ≤ ${SOFT_5D_PNL}`);
  if (consec >= SOFT_CONSEC) soft.push(`${consec} consecutive losing days ≥ ${SOFT_CONSEC}`);

  const dates = rows.map((r) => r.trade_date).filter(Boolean).sort();

  const recentTrades: FuturesTrade[] = [...rows]
    .sort((a, b) =>
      (b.trade_date + (b.datetime_entry || "")).localeCompare(
        a.trade_date + (a.datetime_entry || ""),
      ),
    )
    .slice(0, 18)
    .map((r) => ({
      id: r.paper_trade_id,
      tradeDate: r.trade_date,
      entry: r.datetime_entry,
      exit: r.datetime_exit,
      side: r.side,
      entryPx: Number(r.entry_px) || 0,
      exitPx: Number(r.exit_px) || 0,
      netPnl: Number(r.net_pnl_inr) || 0,
      exitReason: r.exit_reason,
      regime: r.regime,
      longScore: Number(r.long_score) || 0,
      source: r.source,
    }));

  const fw = await readJson<Record<string, { [k: string]: number }>>(
    "results/router_v0/forward_walk_summary.json",
  );
  const v = fw?.phase3_no_guard ?? {};
  const validated: Metrics = {
    nTrades: v.n_trades ?? 0,
    nDays: v.n_days ?? 0,
    winRate: v.win_rate ?? 0,
    totalPnl: v.total_pnl_inr ?? 0,
    meanPnl: v.mean_pnl_inr ?? 0,
    sharpe: v.sharpe_daily_ann ?? 0,
    profitFactor: v.profit_factor ?? 0,
    maxDrawdown: v.max_drawdown_inr ?? 0,
  };

  const t1 = await readJson<any>("results/router_v0/tier1_validation.json");
  const tier1 = t1
    ? {
        overallPass: !!t1.overall_pass,
        proxyPass: !!t1.tier1a_proxy?.pass,
        costPass: !!t1.tier1b_cost?.pass,
        quintilePass: !!t1.tier1c_quintile?.pass,
        costRows: (t1.tier1b_cost?.rows ?? []).map((r: any) => ({
          cost: r.cost_inr,
          nTrades: r.n_trades,
          totalPnl: r.total_pnl_inr,
          winRate: r.win_rate,
          sharpe: r.sharpe_daily_ann,
          maxDrawdown: r.max_drawdown_inr,
        })),
      }
    : null;

  const fi = await readJson<any>("results/router_v0/feature_importance_long.json");
  const features = (fi?.top_20 ?? [])
    .filter((f: any) => f.gain_pct > 0)
    .slice(0, 10)
    .map((f: any) => ({ feature: f.feature, gainPct: f.gain_pct, splits: f.splits }));

  return {
    generatedAt: new Date().toISOString(),
    ledger: {
      totalRows: rows.length,
      dateMin: dates[0] ?? "",
      dateMax: dates[dates.length - 1] ?? "",
      nPaper: liveA.length,
      nPaperC: liveC.length,
      nBootstrap: bootstrap.length,
    },
    validated,
    liveA: { all: allM, t30: m30, t5: m5, consecLossDays: consec },
    liveC: metrics(liveC.map(toTrade)),
    variants: [
      { label: "Variant A (live)", metrics: allM },
      { label: "Variant C (live)", metrics: metrics(liveC.map(toTrade)) },
    ],
    equityCurve: dailyCurve(all),
    liveCurve: dailyCurve(liveATrades),
    recentTrades,
    halts: { hard, soft },
    tier1,
    features,
    health: await readLatestHealth(),
  };
}

async function readLatestHealth(): Promise<FuturesPayload["health"]> {
  const j = await readJson<any>("@health_latest");
  if (!j) return null;
  return {
    timestamp: j.timestamp,
    allOk: !!j.all_ok,
    checks: (j.checks ?? []).map((c: any) => ({ name: c.name, ok: !!c.ok, msg: c.msg })),
  };
}

// ── Equity payload ──
export async function getEquityPayload(): Promise<EquityPayload> {
  const rows = await readCsvObjects("results/equity/paper_ledger.csv");
  const recs = rows.map((r) => ({
    rebalanceDate: r.rebalance_date,
    exitDate: r.exit_date,
    state: r.state,
    nHoldings: Number(r.n_holdings) || 0,
    turnover: Number(r.turnover) || 0,
    netRet: Number(r.net_ret) || 0,
    equity: Number(r.equity) || 0,
    benchEquity: Number(r.bench_equity) || 0,
    holdings: r.holdings,
  }));

  const last = recs[recs.length - 1];
  const invested = recs.filter((r) => r.state === "invested");
  const wins = recs.filter((r) => r.netRet > 0).length;

  let latestHoldings: EquityPayload["latestHoldings"] = [];
  for (let i = recs.length - 1; i >= 0; i--) {
    const h = recs[i].holdings;
    if (h && h !== "{}") {
      try {
        const obj = JSON.parse(h) as Record<string, number>;
        latestHoldings = Object.entries(obj)
          .map(([symbol, weight]) => ({ symbol, weight }))
          .sort((a, b) => b.weight - a.weight);
      } catch {
        /* ignore */
      }
      break;
    }
  }

  const dates = recs.map((r) => r.rebalanceDate).filter(Boolean);

  return {
    generatedAt: new Date().toISOString(),
    latest: {
      date: last?.rebalanceDate ?? "",
      state: last?.state ?? "",
      nHoldings: last?.nHoldings ?? 0,
      equity: last?.equity ?? 0,
      benchEquity: last?.benchEquity ?? 0,
    },
    stats: {
      nRebalances: recs.length,
      dateMin: dates[0] ?? "",
      dateMax: dates[dates.length - 1] ?? "",
      totalReturn: (last?.equity ?? 1) - 1,
      benchReturn: (last?.benchEquity ?? 1) - 1,
      excessReturn: (last?.equity ?? 1) - (last?.benchEquity ?? 1),
      investedRate: recs.length ? invested.length / recs.length : 0,
      avgTurnover: invested.length
        ? invested.reduce((a, b) => a + b.turnover, 0) / invested.length
        : 0,
      winRate: recs.length ? wins / recs.length : 0,
    },
    strategyCurve: recs.map((r) => ({ date: r.rebalanceDate, value: r.equity })),
    benchCurve: recs.map((r) => ({ date: r.rebalanceDate, value: r.benchEquity })),
    recent: recs
      .slice(-14)
      .reverse()
      .map(({ holdings, ...rest }) => rest),
    latestHoldings,
  };
}

// ── Operations / cron monitoring payload ──
function istToday(): string {
  return new Intl.DateTimeFormat("en-CA", { timeZone: "Asia/Kolkata" }).format(new Date());
}

async function latestPicks(): Promise<OpsPayload["picks"]> {
  const j = await readJson<any>("@picks_latest");
  if (!j) return null;
  return {
    asOf: j.as_of,
    generatedAt: j.generated_at,
    state: j.state,
    nPicks: j.n_picks ?? (j.picks?.length ?? 0),
    items: (j.picks ?? []).map((x: any) => ({
      symbol: x.symbol,
      weight: x.weight,
      score: x.score,
      price: x.price,
      advCr: x.adv_cr,
    })),
  };
}

async function lastRunFromStatus(): Promise<OpsPayload["lastRun"]> {
  const j = await readJson<any>("results/daily_run_status.json");
  if (!j) return null;
  return {
    source: "daily_run",
    timestamp: j.run_timestamp,
    date: j.date,
    steps: (j.steps ?? []).map((s: any) => ({
      label: s.label,
      returnCode: s.return_code,
      ok: !!s.ok,
    })),
    exitCode: j.exit_code ?? 0,
    ranToday: j.date === istToday(),
  };
}

async function lastRunFromOpsReport(): Promise<OpsPayload["lastRun"]> {
  const j = await readJson<any>("@ops_latest");
  if (!j) return null;
  const steps: OpsStep[] = (j.steps ?? []).map((s: any) => ({
    label: s.label,
    returnCode: s.return_code,
    ok: s.return_code === 0,
  }));
  const date = String(j.run_timestamp ?? "").slice(0, 10);
  const exitCode = steps.reduce((m, s) => Math.max(m, s.returnCode), 0);
  return { source: "ops_report", timestamp: j.run_timestamp, date, steps, exitCode, ranToday: date === istToday() };
}

async function futuresToday(): Promise<OpsPayload["futuresToday"]> {
  const rows = (await readCsvObjects("results/router_v0/paper_trading_ledger.csv")).filter(
    (r) => r.source === "paper" || r.source === "paper_c",
  );
  if (!rows.length) return null;
  const maxDate = rows.reduce((m, r) => (r.trade_date > m ? r.trade_date : m), "");
  const today = rows.filter((r) => r.trade_date === maxDate);
  return {
    date: maxDate,
    nTrades: today.length,
    net: today.reduce((a, r) => a + (Number(r.net_pnl_inr) || 0), 0),
  };
}

export async function getOpsPayload(): Promise<OpsPayload> {
  const lastRun = (await lastRunFromStatus()) ?? (await lastRunFromOpsReport());
  const eodStep = lastRun?.steps.find((s) => /eod|cache/i.test(s.label));
  const halts = (await readJson<{ futures: boolean; equity: boolean }>("@halts")) ?? {
    futures: false,
    equity: false,
  };
  return {
    generatedAt: new Date().toISOString(),
    today: istToday(),
    cron: await readJson<OpsPayload["cron"]>("results/cron_status.json"),
    lastRun,
    halts,
    eodFetchedToday: !!(lastRun?.ranToday && eodStep?.ok),
    picks: await latestPicks(),
    futuresToday: await futuresToday(),
    health: await readLatestHealth(),
  };
}

// ── Recommendations payload (morning run) ──
export async function getRecommendationsPayload(): Promise<RecommendationsPayload> {
  const rec = await readJson<any>("results/recommendations.json");
  const mapEquity = (e: any) =>
    e
      ? {
          state: e.state,
          nPicks: e.n_picks ?? (e.picks?.length ?? 0),
          config: e.config ?? {},
          items: (e.picks ?? []).map((x: any) => ({
            symbol: x.symbol,
            weight: x.weight,
            score: x.score,
            price: x.price,
            advCr: x.adv_cr,
            entry: x.entry ?? null,
            current: x.current ?? null,
            target: x.target ?? null,
            stop: x.stop ?? null,
            roi: x.roi ?? null,
            rr: x.rr ?? null,
            confidence: x.confidence ?? null,
            horizonDays: x.horizon_days ?? null,
            sigmaPct: x.sigma_pct ?? null,
          })),
        }
      : null;

  const mapFutures = (fu: any) =>
    fu
      ? {
          stance: fu.stance,
          vix: fu.vix,
          vixCut: fu.vix_cut,
          ret5d: fu.ret_5d ?? null,
          retCut: fu.ret_cut,
          tradeable: !!fu.tradeable,
          asOf: fu.as_of,
          note: fu.note,
          entry: fu.entry,
          current: fu.current,
          target: fu.target,
          stop: fu.stop,
          roi: fu.roi,
          rr: fu.rr,
          targetPct: fu.target_pct,
          stopPct: fu.stop_pct,
          sizeHint: fu.size_hint,
          lot: fu.lot,
          winRate: fu.win_rate ?? null,
          confidence: fu.confidence,
        }
      : null;

  if (rec) {
    return {
      generatedAt: rec.generated_at,
      asOf: rec.as_of,
      hasData: true,
      equity: mapEquity(rec.equity),
      futures: mapFutures(rec.futures),
    };
  }

  // Fallback: no morning recommendations file yet — surface the latest picks.
  const picks = await readJson<any>("@picks_latest");
  return {
    generatedAt: picks?.generated_at ?? "",
    asOf: picks?.as_of ?? "",
    hasData: !!picks,
    equity: mapEquity(picks),
    futures: null,
  };
}
