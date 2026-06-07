import { getEquityPayload } from "@/lib/data";
import { pct } from "@/lib/format";
import { Panel, SectionLabel, SectionHeading, StatusPill, HeroPanel } from "@/components/ui";
import KpiCard from "@/components/KpiCard";
import AreaChart from "@/components/AreaChart";
import AutoRefresh from "@/components/AutoRefresh";

export const dynamic = "force-dynamic";

const mult = (v: number) => `${v.toFixed(2)}×`;
const signedMult = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}×`;

export default async function EquityPage() {
  const d = await getEquityPayload();
  const invested = d.latest.state === "invested";

  return (
    <div className="page-stack-lg">
      <HeroPanel>
        <div className="grid items-center gap-10 lg:grid-cols-2">
          <div>
            <SectionLabel>IntradayNet · NSE Nifty 500</SectionLabel>
            <h1 className="t-hero mt-4 text-ink">
              Daily long factor,
              <br />
              <span className="text-ink-60">compounding.</span>
            </h1>
            <p className="mt-5 max-w-md text-[15px] leading-relaxed text-ink-60">
              A diversified 20-name long book rebalanced through risk-on regimes, with a
              risk-off cash state. Tracked against the Nifty 500 benchmark since 2017.
            </p>
            <div className="mt-7 flex flex-wrap items-center gap-3">
              <StatusPill tone={invested ? "up" : "warn"} pulse>
                {invested ? "INVESTED" : d.latest.state.toUpperCase() || "—"}
              </StatusPill>
              <span className="text-[12px] text-ink-60">latest · {d.latest.date}</span>
              <AutoRefresh generatedAt={d.generatedAt} />
            </div>
          </div>

          <div className="card-raised p-5 sm:p-6">
            <div className="flex items-center justify-between">
              <SectionLabel>Strategy equity</SectionLabel>
              <span className="text-[12px] text-ink-60">{d.stats.nRebalances} rebalances</span>
            </div>
            <p className="nums mt-3 text-[40px] font-bold leading-none text-up">{mult(d.latest.equity)}</p>
            <p className="nums mt-2 text-[12px] text-ink-60">growth of ₹1 since {d.stats.dateMin}</p>
            <div className="-mx-1 mt-4">
              <AreaChart
                series={[
                  { points: d.benchCurve, color: "rgba(0,153,255,0.45)", fill: false },
                  { points: d.strategyCurve, color: "#0099ff", fill: true, marker: true },
                ]}
                height={120}
              />
            </div>
            <div className="mt-5 grid grid-cols-3 gap-3 border-t border-hair pt-4">
              <div>
                <SectionLabel>Benchmark</SectionLabel>
                <p className="nums mt-2 text-[18px] font-semibold text-ink-60">{mult(d.latest.benchEquity)}</p>
              </div>
              <div>
                <SectionLabel>Excess</SectionLabel>
                <p className={`nums mt-2 text-[18px] font-semibold ${d.stats.excessReturn >= 0 ? "text-up" : "text-down"}`}>
                  {signedMult(d.stats.excessReturn)}
                </p>
              </div>
              <div>
                <SectionLabel>Period win</SectionLabel>
                <p className="nums mt-2 text-[18px] font-semibold text-ink">{pct(d.stats.winRate, 0)}</p>
              </div>
            </div>
          </div>
        </div>
      </HeroPanel>

      <section>
        <SectionHeading
          eyebrow="Performance"
          title="Since-inception summary"
          description={`${d.stats.dateMin} → ${d.stats.dateMax}, ${d.stats.nRebalances} biweekly rebalances.`}
        />
        <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-6">
          <KpiCard label="Strategy equity" value={mult(d.latest.equity)} sub={`from ${d.stats.dateMin}`} tone="up" accent />
          <KpiCard label="Benchmark" value={mult(d.latest.benchEquity)} sub="Nifty 500" />
          <KpiCard label="Excess" value={signedMult(d.stats.excessReturn)} sub="vs benchmark" tone={d.stats.excessReturn >= 0 ? "up" : "down"} />
          <KpiCard label="Invested rate" value={pct(d.stats.investedRate, 0)} sub={`${d.stats.nRebalances} periods`} />
          <KpiCard label="Avg turnover" value={pct(d.stats.avgTurnover, 0)} sub="when invested" />
          <KpiCard label="Period win" value={pct(d.stats.winRate, 0)} sub="net ret > 0" />
        </div>
      </section>

      <section>
        <Panel className="p-6 sm:p-8">
          <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
            <div>
              <SectionLabel>Equity curve vs benchmark</SectionLabel>
              <p className="mt-2 text-[15px] text-ink-60">Growth of ₹1 · log-scale not applied</p>
            </div>
            <div className="flex items-center gap-5 text-[12px]">
              <span className="flex items-center gap-2 text-ink"><span className="h-1.5 w-4 rounded-[4px] bg-accent" /> strategy</span>
              <span className="flex items-center gap-2 text-ink-60"><span className="h-1.5 w-4 rounded-[4px] bg-accent-soft" /> benchmark</span>
            </div>
          </div>
          <AreaChart
            series={[
              { points: d.benchCurve, color: "rgba(0,153,255,0.45)", fill: false },
              { points: d.strategyCurve, color: "#0099ff", fill: true, marker: true },
            ]}
            height={300}
          />
        </Panel>
      </section>
    </div>
  );
}
