import Link from "next/link";
import { getRecommendationsPayload } from "@/lib/data";
import { num, pct } from "@/lib/format";
import { Panel, SectionLabel, SectionHeading, StatusPill } from "@/components/ui";
import DataTable, { type Column } from "@/components/DataTable";
import AutoRefresh from "@/components/AutoRefresh";
import type { RecPick } from "@/lib/types";

export const dynamic = "force-dynamic";

const price = (v: number | null) => (v == null ? "—" : `₹${v.toLocaleString("en-IN", { maximumFractionDigits: 1 })}`);
const confCls = (c: number | null) => (c == null ? "text-ink-60" : c >= 75 ? "text-up" : c >= 55 ? "text-warn" : "text-down");

export default async function RecommendationsPage() {
  const d = await getRecommendationsPayload();
  const f = d.futures;
  const eq = d.equity;

  const cols: Column<RecPick>[] = [
    { key: "sym", header: "Symbol", render: (r) => (
      <Link href={`/recommendations/${r.symbol}`} className="font-semibold text-ink hover:text-accent">{r.symbol}</Link>
    ) },
    { key: "entry", header: "Entry", align: "right", render: (r) => price(r.entry) },
    { key: "cur", header: "Current", align: "right", render: (r) => price(r.current) },
    { key: "tgt", header: "Target", align: "right", render: (r) => <span className="text-up">{price(r.target)}</span> },
    { key: "stop", header: "Stop", align: "right", render: (r) => <span className="text-down">{price(r.stop)}</span> },
    { key: "roi", header: "ROI", align: "right", render: (r) => <span className="text-up">{r.roi == null ? "—" : pct(r.roi, 1, { sign: true })}</span> },
    { key: "rr", header: "R:R", align: "right", render: (r) => (r.rr == null ? "—" : `${num(r.rr, 2)}`) },
    { key: "conf", header: "Conf", align: "right", render: (r) => <span className={confCls(r.confidence)}>{r.confidence == null ? "—" : `${r.confidence}%`}</span> },
    { key: "go", header: "", align: "right", render: (r) => <Link href={`/recommendations/${r.symbol}`} className="text-ink-60 hover:text-accent">›</Link> },
  ];

  return (
    <div className="space-y-16">
      {/* Hero */}
      <Panel variant="shell" radius="shell" glow diagonal className="p-6 sm:p-10">
        <div className="grid items-center gap-10 lg:grid-cols-[1.05fr_0.95fr]">
          <div>
            <SectionLabel>Today{d.asOf ? ` · as of ${d.asOf}` : ""}</SectionLabel>
            <h1 className="mt-4 text-[30px] font-bold leading-[1.06] tracking-[-0.03em] text-ink sm:text-[52px] sm:leading-[1.03]">
              Today&rsquo;s
              <br />
              <span className="text-ink-60">recommendations.</span>
            </h1>
            <p className="mt-5 max-w-md text-[15px] font-normal leading-relaxed text-ink-60">
              Each name with entry, target, stop, ROI, risk-reward and confidence. Tap any row for
              the full breakdown. The day then trades these — tracked on Paper Trading.
            </p>
            <div className="mt-7 flex flex-wrap items-center gap-3">
              {d.hasData ? <StatusPill tone="up" pulse>RECOMMENDATIONS READY</StatusPill> : <StatusPill tone="warn">AWAITING MORNING RUN</StatusPill>}
              {d.generatedAt && <span className="text-[12px] text-ink-60">generated {d.generatedAt}</span>}
              <AutoRefresh generatedAt={d.generatedAt || new Date().toISOString()} />
            </div>
          </div>

          {/* Futures trade card */}
          <Link href="/recommendations/futures" className="surface block rounded-lg p-5 transition-colors hover:border-hair-strong sm:p-6">
            <div className="flex items-center justify-between">
              <SectionLabel>NIFTY futures · session plan</SectionLabel>
              {f && <StatusPill tone={f.tradeable ? "up" : "down"} dot={false}>{f.tradeable ? "trade" : "flat"}</StatusPill>}
            </div>
            {f ? (
              <>
                <p className={`nums mt-3 text-[34px] font-bold leading-none ${f.tradeable ? "text-up" : "text-down"}`}>
                  {f.tradeable ? "Long-only" : "Stand aside"}
                </p>
                <div className="mt-5 grid grid-cols-3 gap-y-4 border-t border-hair pt-4">
                  {[
                    ["Entry", price(f.entry)],
                    ["Target", price(f.target)],
                    ["Stop", price(f.stop)],
                    ["ROI", pct(f.roi, 2, { sign: true })],
                    ["R:R", `${num(f.rr, 2)} : 1`],
                    ["Confidence", `${f.confidence}%`],
                  ].map(([k, v]) => (
                    <div key={k}>
                      <SectionLabel>{k}</SectionLabel>
                      <p className="nums mt-1.5 text-[15px] font-semibold text-ink">{v}</p>
                    </div>
                  ))}
                </div>
                <p className="mt-4 text-[12px] text-accent">Full analysis →</p>
              </>
            ) : (
              <p className="mt-4 text-[13px] text-muted">Futures plan not generated yet.</p>
            )}
          </Link>
        </div>
      </Panel>

      {/* Equity book */}
      <section>
        <SectionHeading
          eyebrow="Recommendation engine · equity"
          title="Today's stock recommendations"
          description="Top-20 momentum names with full trade levels. Targets/stops are 10-day volatility bands; confidence from momentum strength."
          right={
            eq ? (
              <div className="flex items-center gap-2">
                <StatusPill tone={eq.state === "invested" ? "up" : "warn"} dot={false}>{eq.state}</StatusPill>
                <span className="text-[12px] text-ink-60">{eq.nPicks} names</span>
              </div>
            ) : undefined
          }
        />
        <Panel variant="shell" radius="shell" className="p-6 sm:p-7">
          {eq && eq.items.length ? (
            <DataTable columns={cols} rows={eq.items} />
          ) : (
            <p className="text-[13px] text-muted">
              {eq ? "Risk-off — hold cash, no eligible names today." : "No equity picks yet. Run morning_run.py to generate them."}
            </p>
          )}
        </Panel>
      </section>
    </div>
  );
}
