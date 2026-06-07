import Link from "next/link";
import { getRecommendationsPayload } from "@/lib/data";
import { num, pct, priceFmt } from "@/lib/format";
import { Heading, Chip } from "@/components/ui";
import DataTable, { type Column } from "@/components/DataTable";
import AutoRefresh from "@/components/AutoRefresh";
import type { RecPick } from "@/lib/types";

export const dynamic = "force-dynamic";

function confColor(c: number | null) {
  if (c == null) return "var(--color-text-muted)";
  if (c >= 75) return "var(--color-accent-green)";
  if (c >= 55) return "var(--color-accent-amber)";
  return "var(--color-accent-pink)";
}

export default async function RecommendationsPage() {
  const d = await getRecommendationsPayload();
  const f = d.futures;
  const eq = d.equity;

  const cols: Column<RecPick>[] = [
    { key: "sym", header: "Symbol", render: (r) => (
      <Link href={`/recommendations/${encodeURIComponent(r.symbol)}`} className="sym-link">
        {r.symbol}
      </Link>
    )},
    { key: "entry", header: "Entry", align: "right", render: (r) => priceFmt(r.entry) },
    { key: "tgt", header: "Target", align: "right", render: (r) => <span className="text-up">{priceFmt(r.target)}</span> },
    { key: "stop", header: "Stop", align: "right", render: (r) => <span className="text-down">{priceFmt(r.stop)}</span> },
    { key: "roi", header: "ROI", align: "right", render: (r) => (
      <span className="text-up">{r.roi == null ? "—" : pct(r.roi, 1, { sign: true })}</span>
    )},
    { key: "rr", header: "R:R", align: "right", render: (r) => r.rr == null ? "—" : num(r.rr, 2) },
    { key: "conf", header: "Conf", align: "right", render: (r) => (
      <span style={{ color: confColor(r.confidence) }}>{r.confidence == null ? "—" : `${r.confidence}%`}</span>
    )},
    { key: "go", header: "", align: "right", render: (r) => (
      <Link href={`/recommendations/${encodeURIComponent(r.symbol)}`} className="text-[16px] text-muted no-underline hover:text-accent">
        →
      </Link>
    )},
  ];

  return (
    <div className="page-stack-lg">
      <section className="anim-fade-up">
        <div className="grid items-start gap-10 lg:grid-cols-2">
          <div>
            <p className="t-label mb-4">{d.asOf ? `Today · ${d.asOf}` : "Today"}</p>
            <h1 className="t-hero text-ink">Today&rsquo;s picks.</h1>
            <p className="t-body mt-4 max-w-md text-muted">
              Entry, target, stop and confidence for every name. Tap any row for the full
              breakdown. Results tracked live on Paper Trading.
            </p>
            <div className="mt-6 flex flex-wrap items-center gap-2.5">
              {d.hasData
                ? <Chip tone="green" dot pulse>Picks ready</Chip>
                : <Chip tone="amber" dot>Awaiting morning run</Chip>
              }
              {d.generatedAt && (
                <span className="text-[11px] text-muted">{d.generatedAt}</span>
              )}
              <AutoRefresh generatedAt={d.generatedAt} />
            </div>
          </div>

          {f && (
            <Link href="/recommendations/futures" className="no-underline">
              <div className="card p-5 transition-[border-color] duration-150 hover:border-hair-strong">
                <div className="mb-3 flex items-center justify-between">
                  <p className="t-label">NIFTY futures</p>
                  <Chip tone={f.tradeable ? "green" : "pink"} dot={false}>
                    {f.tradeable ? "tradeable" : "flat"}
                  </Chip>
                </div>
                <p className={`t-h3 nums mb-4 ${f.tradeable ? "text-up" : "text-down"}`}>
                  {f.tradeable ? "Long-only" : "Stand aside"}
                </p>
                <div className="grid grid-cols-3 gap-3 border-t border-hair pt-4">
                  {[
                    ["Entry",  priceFmt(f.entry)],
                    ["Target", priceFmt(f.target)],
                    ["Stop",   priceFmt(f.stop)],
                    ["ROI",    pct(f.roi, 2, { sign: true })],
                    ["R:R",    `${num(f.rr, 2)} : 1`],
                    ["Conf",   `${f.confidence}%`],
                  ].map(([k, v]) => (
                    <div key={k}>
                      <p className="t-label mb-1">{k}</p>
                      <p className="nums text-[14px] font-semibold text-ink">{v}</p>
                    </div>
                  ))}
                </div>
                <p className="mt-4 text-[12px] text-accent">Full analysis →</p>
              </div>
            </Link>
          )}
        </div>
      </section>

      <section>
        <Heading
          label="Equity · intraday"
          title="Stock picks for today"
          description="Top momentum names. Entry at open, 2×ATR stop, hold-to-close. Equal weight across regime-adjusted capital."
          right={eq ? (
            <div className="flex items-center gap-2.5">
              <Chip tone={eq.state === "invested" ? "green" : "amber"} dot={false}>{eq.state}</Chip>
              <span className="text-[12px] text-muted">{eq.nPicks} names</span>
            </div>
          ) : undefined}
        />
        <div className="card overflow-hidden p-4 sm:p-5">
          {eq && eq.items.length ? (
            <DataTable columns={cols} rows={eq.items} rowKey={(r) => r.symbol} />
          ) : (
            <p className="py-10 text-center text-[13px] text-muted">
              {eq ? "Risk-off — hold cash. No eligible names today." : "No picks yet. Run morning_run.py to generate them."}
            </p>
          )}
        </div>
      </section>
    </div>
  );
}
