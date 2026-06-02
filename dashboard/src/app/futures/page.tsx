import { getFuturesPayload } from "@/lib/data";
import { compactInr, num, pct } from "@/lib/format";
import { Panel, SectionLabel, SectionHeading, StatusPill } from "@/components/ui";
import KpiCard from "@/components/KpiCard";
import AreaChart from "@/components/AreaChart";
import BarChart from "@/components/BarChart";
import DataTable from "@/components/DataTable";
import AutoRefresh from "@/components/AutoRefresh";

export const dynamic = "force-dynamic";

export default async function FuturesPage() {
  const d = await getFuturesPayload();
  const curveEnd = d.equityCurve.at(-1)?.value ?? 0;

  return (
    <div className="space-y-16">
      {/* Hero */}
      <Panel variant="shell" radius="shell" glow diagonal className="p-6 sm:p-10">
        <div className="grid items-center gap-10 lg:grid-cols-[1.05fr_0.95fr]">
          <div>
            <SectionLabel>NIFTY futures · router_v0</SectionLabel>
            <h1 className="mt-4 text-[30px] font-bold leading-[1.06] tracking-[-0.03em] text-ink sm:text-[52px] sm:leading-[1.03]">
              The strategy,
              <br />
              <span className="text-ink-60">forward-walk validated.</span>
            </h1>
            <p className="mt-5 max-w-md text-[15px] font-normal leading-relaxed text-ink-60">
              Long-only NIFTY futures engine. This is the model & validation view — live paper
              results live on the Paper Trading tab.
            </p>
            <div className="mt-7 flex flex-wrap items-center gap-3">
              {d.tier1 && (
                <StatusPill tone={d.tier1.overallPass ? "up" : "down"} pulse={d.tier1.overallPass}>
                  {d.tier1.overallPass ? "TIER-1 VALIDATED" : "VALIDATION FAILED"}
                </StatusPill>
              )}
              <span className="text-[12px] text-ink-60">{d.ledger.dateMin} → {d.ledger.dateMax}</span>
              <AutoRefresh generatedAt={d.generatedAt} />
            </div>
          </div>

          <div className="surface rounded-lg p-5 sm:p-6">
            <div className="flex items-center justify-between">
              <SectionLabel>Track record · cumulative PnL</SectionLabel>
              <span className="text-[12px] text-ink-60">{d.validated.nTrades} trades</span>
            </div>
            <p className="nums mt-3 text-[40px] font-bold leading-none text-up">
              {compactInr(curveEnd, { sign: true })}
            </p>
            <p className="nums mt-2 text-[12px] text-ink-60">forward-walk bootstrap + live paper</p>
            <div className="-mx-1 mt-4">
              <AreaChart
                series={[{ points: d.equityCurve, color: "var(--color-accent)", fill: true, marker: true }]}
                height={120}
                baselineZero
              />
            </div>
            <div className="mt-5 grid grid-cols-3 gap-3 border-t border-hair pt-4">
              {[
                { l: "Sharpe", v: num(d.validated.sharpe) },
                { l: "PF", v: num(d.validated.profitFactor) },
                { l: "Win", v: pct(d.validated.winRate, 0) },
              ].map((c) => (
                <div key={c.l}>
                  <SectionLabel>{c.l}</SectionLabel>
                  <p className="nums mt-2 text-[18px] font-semibold text-ink">{c.v}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Panel>

      {/* Validated KPIs */}
      <section>
        <SectionHeading
          eyebrow="Backtest reference"
          title="Validated forward-walk performance"
          description="Locked metrics from the Nov 2024 → May 2026 forward walk — the bar the live book is measured against."
        />
        <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
          <KpiCard label="Net PnL" value={compactInr(d.validated.totalPnl, { sign: true })} sub={`${d.validated.nTrades} trades · ${d.validated.nDays} days`} tone="up" accent />
          <KpiCard label="Sharpe (ann)" value={num(d.validated.sharpe)} sub="daily, annualized" tone="up" />
          <KpiCard label="Profit factor" value={num(d.validated.profitFactor)} sub="gross win / loss" />
          <KpiCard label="Win rate" value={pct(d.validated.winRate)} sub="per trade" />
          <KpiCard label="Max drawdown" value={compactInr(d.validated.maxDrawdown)} sub="peak-to-trough" tone="down" />
        </div>
      </section>

      {/* Curve */}
      <section>
        <Panel variant="shell" radius="shell" glow className="p-6 sm:p-8">
          <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
            <div>
              <SectionLabel>Cumulative net PnL</SectionLabel>
              <p className="mt-2 text-[15px] font-normal text-ink-60">
                Forward-walk bootstrap + live paper · {d.equityCurve.length} trading days
              </p>
            </div>
            <span className="nums text-[28px] font-bold text-ink">{compactInr(curveEnd, { sign: true })}</span>
          </div>
          <AreaChart
            series={[{ points: d.equityCurve, color: "var(--color-accent)", fill: true, marker: true }]}
            height={280}
            baselineZero
          />
        </Panel>
      </section>

      {/* Tier-1 */}
      <section>
        <SectionHeading eyebrow="Pre-deployment" title="Tier-1 validation" right={d.tier1 && <StatusPill tone={d.tier1.overallPass ? "up" : "down"}>{d.tier1.overallPass ? "all pass" : "fail"}</StatusPill>} />
        <Panel radius="lg" className="p-6">
          {d.tier1 ? (
            <>
              <div className="mb-5 flex flex-wrap gap-2">
                <StatusPill tone={d.tier1.proxyPass ? "up" : "down"} dot={false}>Proxy {d.tier1.proxyPass ? "✓" : "✕"}</StatusPill>
                <StatusPill tone={d.tier1.costPass ? "up" : "down"} dot={false}>Cost sweep {d.tier1.costPass ? "✓" : "✕"}</StatusPill>
                <StatusPill tone={d.tier1.quintilePass ? "up" : "down"} dot={false}>Score quintiles {d.tier1.quintilePass ? "✓" : "✕"}</StatusPill>
              </div>
              <SectionLabel className="mb-3">Cost sensitivity · ₹/trade</SectionLabel>
              <DataTable
                columns={[
                  { key: "c", header: "Cost", render: (r: any) => `₹${r.cost}` },
                  { key: "pnl", header: "PnL", align: "right", render: (r: any) => compactInr(r.totalPnl, { sign: true }) },
                  { key: "win", header: "Win", align: "right", render: (r: any) => pct(r.winRate) },
                  { key: "sh", header: "Sharpe", align: "right", render: (r: any) => num(r.sharpe) },
                ]}
                rows={d.tier1.costRows}
              />
            </>
          ) : (
            <p className="text-[13px] text-muted">tier1_validation.json not found.</p>
          )}
        </Panel>
      </section>

      {/* Features + Health */}
      <section className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Panel radius="lg" className="p-6">
          <SectionLabel className="mb-6">Model feature importance · gain %</SectionLabel>
          {d.features.length ? (
            <BarChart items={d.features.map((f) => ({ label: f.feature, value: f.gainPct, valueLabel: `${f.gainPct.toFixed(1)}%` }))} />
          ) : (
            <p className="text-[13px] text-muted">feature_importance_long.json not found.</p>
          )}
        </Panel>

        <Panel radius="lg" className="p-6">
          <div className="mb-5 flex items-center justify-between">
            <SectionLabel>Data health</SectionLabel>
            {d.health && <StatusPill tone={d.health.allOk ? "up" : "down"}>{d.health.allOk ? "ok" : "issues"}</StatusPill>}
          </div>
          {d.health ? (
            <div className="flex flex-col">
              {d.health.checks.map((c) => (
                <div key={c.name} className="flex items-center justify-between border-b border-hair py-3 text-[13px] last:border-0">
                  <span className="text-ink-60">{c.name}</span>
                  <span className="flex items-center gap-2.5"><span className="text-ink">{c.msg}</span><span className={`h-1.5 w-1.5 rounded-pill ${c.ok ? "bg-up" : "bg-down"}`} /></span>
                </div>
              ))}
              <p className="pt-4 text-[11px] text-muted">last check · {d.health.timestamp}</p>
            </div>
          ) : (
            <p className="text-[13px] text-muted">no recent health_check_*.json.</p>
          )}
        </Panel>
      </section>
    </div>
  );
}
