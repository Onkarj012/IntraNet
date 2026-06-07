import { getFuturesPayload, getEquityPayload, getOpsPayload } from "@/lib/data";
import { inr, num, pct } from "@/lib/format";
import { Panel, SectionLabel, SectionHeading, StatusPill, HeroPanel } from "@/components/ui";
import KpiCard from "@/components/KpiCard";
import AreaChart from "@/components/AreaChart";
import BarChart from "@/components/BarChart";
import DataTable, { type Column } from "@/components/DataTable";
import AutoRefresh from "@/components/AutoRefresh";
import type { FuturesTrade, EquityRebalance } from "@/lib/types";

export const dynamic = "force-dynamic";

const sharpeTone = (v: number) => (v >= 1 ? "up" : v > 0 ? "warn" : "down");
const pnlTone = (v: number) => (v > 0 ? "up" : v < 0 ? "down" : "ink");
const mult = (v: number) => `${v.toFixed(2)}×`;

export default async function PaperPage() {
  const [d, eq, ops] = await Promise.all([
    getFuturesPayload(),
    getEquityPayload(),
    getOpsPayload(),
  ]);
  const liveEnd = d.liveCurve.at(-1)?.value ?? 0;
  const fHalt = d.halts.hard.length > 0 ? "hard" : d.halts.soft.length > 0 ? "soft" : "ok";

  const tradeCols: Column<FuturesTrade>[] = [
    { key: "date", header: "Date", render: (t) => t.tradeDate },
    { key: "src", header: "Book", render: (t) => <span className="text-ink-60">{t.source === "paper" ? "A" : t.source === "paper_c" ? "C" : "FW"}</span> },
    { key: "side", header: "Side", render: (t) => t.side },
    { key: "entry", header: "Entry", align: "right", render: (t) => t.entryPx.toFixed(1) },
    { key: "exit", header: "Exit", align: "right", render: (t) => (t.exitPx ? t.exitPx.toFixed(1) : "—") },
    { key: "reason", header: "Exit", render: (t) => <span className="text-ink-60">{t.exitReason || "—"}</span> },
    { key: "pnl", header: "Net PnL", align: "right", render: (t) => <span className={t.netPnl >= 0 ? "text-up" : "text-down"}>{inr(t.netPnl, { sign: true })}</span> },
  ];

  const rebalCols: Column<EquityRebalance>[] = [
    { key: "date", header: "Rebalance", render: (r) => r.rebalanceDate },
    { key: "state", header: "State", render: (r) => <span className={r.state === "invested" ? "text-up" : "text-warn"}>{r.state}</span> },
    { key: "n", header: "Holds", align: "right", render: (r) => r.nHoldings },
    { key: "ret", header: "Net ret", align: "right", render: (r) => <span className={r.netRet >= 0 ? "text-up" : "text-down"}>{pct(r.netRet, 2, { sign: true })}</span> },
    { key: "eq", header: "Equity", align: "right", render: (r) => mult(r.equity) },
  ];

  return (
    <div className="page-stack-lg">
      <HeroPanel>
        <div className="grid items-center gap-10 lg:grid-cols-2">
          <div>
            <SectionLabel>Live paper trading</SectionLabel>
            <h1 className="t-hero mt-4 text-ink">
              Paper trading,
              <br />
              <span className="text-ink-60">measured live.</span>
            </h1>
            <p className="mt-5 max-w-md text-[15px] leading-relaxed text-ink-60">
              Both books trade the day&rsquo;s recommendations against real session data. Tracked
              vs the validated reference; halts protect the run.
            </p>
            <div className="mt-7 flex flex-wrap items-center gap-3">
              <StatusPill tone={fHalt === "hard" ? "down" : fHalt === "soft" ? "warn" : "up"} pulse>
                Futures · {fHalt === "ok" ? "clear" : `${fHalt} halt`}
              </StatusPill>
              <StatusPill tone={ops.halts.equity ? "down" : "up"} dot={false}>
                Equity · {ops.halts.equity ? "halted" : "clear"}
              </StatusPill>
              <AutoRefresh generatedAt={d.generatedAt} />
            </div>
            {(d.halts.hard.length > 0 || d.halts.soft.length > 0) && (
              <div className="mt-4 flex flex-col gap-1.5 text-[12px]">
                {d.halts.hard.map((h) => <span key={h} className="text-down">✕ {h}</span>)}
                {d.halts.soft.map((h) => <span key={h} className="text-warn">⚠ {h}</span>)}
              </div>
            )}
          </div>

          <div className="card-raised p-5 sm:p-6">
            <div className="flex items-center justify-between">
              <SectionLabel>Futures · Variant A</SectionLabel>
              <span className="text-[12px] text-ink-60">{d.ledger.nPaper} trades</span>
            </div>
            <p className={`nums mt-3 text-[40px] font-bold leading-none ${liveEnd > 0 ? "text-up" : liveEnd < 0 ? "text-down" : "text-ink"}`}>
              {inr(liveEnd, { sign: true })}
            </p>
            <p className="nums mt-2 text-[12px] text-ink-60">cumulative net PnL since go-live</p>
            <div className="-mx-1 mt-4">
              <AreaChart series={[{ points: d.liveCurve, color: "#0099ff", fill: true, marker: true }]} height={120} baselineZero />
            </div>
            <div className="mt-5 grid grid-cols-3 gap-3 border-t border-hair pt-4">
              {[
                { l: "30d Sharpe", v: num(d.liveA.t30.sharpe), t: sharpeTone(d.liveA.t30.sharpe) },
                { l: "Win", v: pct(d.liveA.all.winRate, 0), t: "ink" as const },
                { l: "PF", v: num(d.liveA.all.profitFactor), t: "ink" as const },
              ].map((c) => (
                <div key={c.l}>
                  <SectionLabel>{c.l}</SectionLabel>
                  <p className={`nums mt-2 text-[18px] font-semibold ${c.t === "up" ? "text-up" : c.t === "down" ? "text-down" : c.t === "warn" ? "text-warn" : "text-ink"}`}>{c.v}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </HeroPanel>

      <section>
        <SectionHeading eyebrow="Futures · live book" title="Variant A — promoted strategy" description="Soft alerts flag drift but never freeze the run; only a hard drawdown halt writes the kill-switch." />
        <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
          <KpiCard label="All-time PnL" value={inr(d.liveA.all.totalPnl, { sign: true })} sub={`${d.liveA.all.nTrades} trades`} tone={pnlTone(d.liveA.all.totalPnl)} />
          <KpiCard label="30d Sharpe" value={num(d.liveA.t30.sharpe)} sub="alert if < 0.5" tone={sharpeTone(d.liveA.t30.sharpe)} />
          <KpiCard label="30d PnL" value={inr(d.liveA.t30.totalPnl, { sign: true })} sub={`${d.liveA.t30.nTrades} trades`} tone={pnlTone(d.liveA.t30.totalPnl)} />
          <KpiCard label="Win rate" value={pct(d.liveA.all.winRate)} sub="all-time live" />
          <KpiCard label="Consec. losses" value={`${d.liveA.consecLossDays}d`} sub="alert if ≥ 7" tone={d.liveA.consecLossDays >= 7 ? "down" : "ink"} />
        </div>
      </section>

      <section className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Panel className="p-6">
          <SectionLabel className="mb-5">Live book comparison · A vs C</SectionLabel>
          <DataTable
            columns={[
              { key: "v", header: "Book", render: (r: { label: string }) => r.label },
              { key: "n", header: "Trades", align: "right", render: (r: { metrics: { nTrades: number } }) => r.metrics.nTrades },
              { key: "win", header: "Win", align: "right", render: (r: { metrics: { winRate: number } }) => pct(r.metrics.winRate) },
              { key: "pnl", header: "PnL", align: "right", render: (r: { metrics: { totalPnl: number } }) => (
                <span className={r.metrics.totalPnl >= 0 ? "text-up" : "text-down"}>{inr(r.metrics.totalPnl, { sign: true })}</span>
              )},
              { key: "sh", header: "Sharpe", align: "right", render: (r: { metrics: { sharpe: number | null } }) => num(r.metrics.sharpe) },
              { key: "pf", header: "PF", align: "right", render: (r: { metrics: { profitFactor: number | null } }) => num(r.metrics.profitFactor) },
            ]}
            rows={d.variants}
          />
        </Panel>
        <Panel className="p-6">
          <SectionLabel className="mb-5">Recent futures trades</SectionLabel>
          <DataTable columns={tradeCols} rows={d.recentTrades} />
        </Panel>
      </section>

      <section>
        <SectionHeading eyebrow="Equity · live book" title="IntradayNet paper book" description={`${eq.stats.nRebalances} rebalances · ${eq.stats.dateMin} → ${eq.stats.dateMax}`} />
        <div className="mb-6 grid grid-cols-2 gap-4 md:grid-cols-4">
          <KpiCard label="Equity" value={mult(eq.latest.equity)} sub={`state: ${eq.latest.state}`} tone="up" accent />
          <KpiCard label="Benchmark" value={mult(eq.latest.benchEquity)} sub="Nifty 500" />
          <KpiCard label="Excess" value={`${eq.stats.excessReturn >= 0 ? "+" : "−"}${Math.abs(eq.stats.excessReturn).toFixed(2)}×`} sub="vs benchmark" tone={eq.stats.excessReturn >= 0 ? "up" : "down"} />
          <KpiCard label="Period win" value={pct(eq.stats.winRate, 0)} sub="net ret > 0" />
        </div>
        <Panel className="mb-6 p-6 sm:p-8">
          <div className="mb-4 flex items-center justify-between">
            <SectionLabel>Equity curve vs benchmark</SectionLabel>
            <div className="flex items-center gap-5 text-[12px]">
              <span className="flex items-center gap-2 text-ink"><span className="h-1.5 w-4 rounded-[4px] bg-accent" /> strategy</span>
              <span className="flex items-center gap-2 text-ink-60"><span className="h-1.5 w-4 rounded-[4px] bg-accent-soft" /> benchmark</span>
            </div>
          </div>
          <AreaChart
            series={[
              { points: eq.benchCurve, color: "rgba(0,153,255,0.45)", fill: false },
              { points: eq.strategyCurve, color: "#0099ff", fill: true, marker: true },
            ]}
            height={260}
          />
        </Panel>
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <Panel className="p-6">
            <div className="mb-5 flex items-center justify-between">
              <SectionLabel>Current holdings · weight</SectionLabel>
              <span className="text-[12px] text-ink-60">{eq.latestHoldings.length} names</span>
            </div>
            {eq.latestHoldings.length ? (
              <BarChart items={eq.latestHoldings.slice(0, 20).map((h) => ({ label: h.symbol, value: h.weight * 100, valueLabel: pct(h.weight, 1) }))} />
            ) : (
              <p className="text-[13px] text-muted">No invested holdings.</p>
            )}
          </Panel>
          <Panel className="p-6">
            <SectionLabel className="mb-5">Recent rebalances</SectionLabel>
            <DataTable columns={rebalCols} rows={eq.recent} />
          </Panel>
        </div>
      </section>
    </div>
  );
}
