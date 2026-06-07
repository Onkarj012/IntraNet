import Link from "next/link";
import { getRecommendationsPayload } from "@/lib/data";
import { num, pct, priceFmt } from "@/lib/format";
import { Panel, SectionLabel, StatusPill, StatRow } from "@/components/ui";
import KpiCard from "@/components/KpiCard";
import PriceLadder from "@/components/PriceLadder";

export const dynamic = "force-dynamic";

const confTone = (c: number) => (c >= 75 ? "up" : c >= 55 ? "warn" : "down");

function Methodology({ lines }: { lines: string[] }) {
  return (
    <Panel className="p-6">
      <SectionLabel className="mb-3">How this is derived</SectionLabel>
      <ul className="space-y-2 text-[13px] leading-relaxed text-ink-60">
        {lines.map((l, i) => (
          <li key={i} className="flex gap-2">
            <span className="text-accent">·</span>
            {l}
          </li>
        ))}
      </ul>
    </Panel>
  );
}

export default async function DetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id: rawId } = await params;
  const id = decodeURIComponent(rawId);
  const d = await getRecommendationsPayload();
  const back = (
    <Link href="/" className="t-small text-ink-60 no-underline hover:text-accent">
      ← all recommendations
    </Link>
  );

  if (id.toLowerCase() === "futures") {
    const f = d.futures;
    if (!f)
      return (
        <div className="space-y-6 pt-2">
          {back}
          <p className="text-[13px] text-muted">No futures plan generated yet.</p>
        </div>
      );
    const riskInr = (f.entry - f.stop) * f.lot;
    const rewardInr = (f.target - f.entry) * f.lot;
    return (
      <div className="page-stack pt-2">
        {back}
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <SectionLabel>NIFTY futures · session plan</SectionLabel>
            <h1 className="t-h2 mt-2 text-ink">NIFTY 50 Futures</h1>
          </div>
          <StatusPill tone={f.tradeable ? "up" : "down"}>{f.tradeable ? "tradeable today" : "stand aside"}</StatusPill>
        </div>

        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          <KpiCard label="Entry (ref)" value={priceFmt(f.entry)} sub="current NIFTY level" accent />
          <KpiCard label="Target" value={priceFmt(f.target)} sub={`+${pct(f.targetPct, 2)}`} tone="up" />
          <KpiCard label="Stop" value={priceFmt(f.stop)} sub={`−${pct(f.stopPct, 2)}`} tone="down" />
          <KpiCard label="Risk : Reward" value={`${num(f.rr, 2)} : 1`} sub="per the engine band" />
          <KpiCard label="ROI to target" value={pct(f.roi, 2, { sign: true })} sub="per lot move" tone="up" />
          <KpiCard label="Confidence" value={`${f.confidence}%`} sub="validated hit rate" tone={confTone(f.confidence)} />
          <KpiCard label="Reward / lot" value={`₹${Math.round(rewardInr).toLocaleString("en-IN")}`} sub={`${f.lot} qty`} tone="up" />
          <KpiCard label="Risk / lot" value={`₹${Math.round(riskInr).toLocaleString("en-IN")}`} sub={`${f.lot} qty`} tone="down" />
        </div>

        <Panel className="p-6 sm:p-8">
          <SectionLabel className="mb-1">Trade ladder</SectionLabel>
          <PriceLadder stop={f.stop} entry={f.entry} current={f.current} target={f.target} fmt={(v) => priceFmt(v)} />
        </Panel>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <Panel className="p-6">
            <SectionLabel className="mb-4">Session context</SectionLabel>
            <div className="flex flex-col">
              <StatRow label="Stance" value={f.stance} />
              <StatRow label="India VIX" value={`${num(f.vix, 1)} (guard < ${f.vixCut})`} />
              <StatRow
                label="5-day return"
                value={f.ret5d == null ? "—" : `${pct(f.ret5d, 2, { sign: true })} (guard > ${pct(f.retCut, 1)})`}
              />
              <StatRow label="Sizing" value={f.sizeHint} />
              <StatRow label="Validated win rate" value={f.winRate == null ? "—" : pct(f.winRate, 1)} />
            </div>
            <p className="mt-4 text-[13px] leading-relaxed text-ink-60">{f.note}</p>
          </Panel>
          <Methodology
            lines={[
              "Long-only intraday engine: entry fires on the model's score during the session — the entry above is the current reference level.",
              `Fixed band: target +${pct(f.targetPct, 2)}, stop −${pct(f.stopPct, 2)} → risk:reward ${num(f.rr, 2)}:1 (validated config).`,
              "Position size 1.0–1.5× lot, 1.5× only on top-percentile scores.",
              "Confidence = forward-walk validated win rate; positive expectancy comes from the asymmetric band (PF 1.39).",
              "Phase-2 guard gates the whole day: VIX below 22 and 5-day return above −1.5%.",
            ]}
          />
        </div>
      </div>
    );
  }

  const p = d.equity?.items.find((x) => x.symbol.toLowerCase() === id.toLowerCase());
  if (!p)
    return (
      <div className="space-y-6 pt-2">
        {back}
        <p className="text-[13px] text-muted">No recommendation found for “{id}”.</p>
      </div>
    );

  const enriched = p.entry != null && p.target != null && p.stop != null;
  const unreal = p.entry != null && p.current != null ? (p.current - p.entry) / p.entry : null;

  return (
    <div className="page-stack pt-2">
      {back}
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <SectionLabel>Equity · momentum book{d.asOf ? ` · ${d.asOf}` : ""}</SectionLabel>
          <h1 className="t-h2 mt-2 text-ink">{p.symbol}</h1>
        </div>
        {p.confidence != null && (
          <StatusPill tone={confTone(p.confidence)}>{p.confidence}% confidence</StatusPill>
        )}
      </div>

      {enriched ? (
        <>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <KpiCard label="Entry" value={priceFmt(p.entry)} sub="decision close" accent />
            <KpiCard
              label="Current"
              value={priceFmt(p.current)}
              sub={unreal == null ? "—" : `${pct(unreal, 2, { sign: true })} vs entry`}
              tone={unreal == null ? "ink" : unreal >= 0 ? "up" : "down"}
            />
            <KpiCard label="Target" value={priceFmt(p.target)} sub={`${pct(p.roi ?? 0, 1, { sign: true })} ROI`} tone="up" />
            <KpiCard label="Stop" value={priceFmt(p.stop)} sub={`${num(p.horizonDays ?? 10, 0)}-day hold`} tone="down" />
            <KpiCard label="Return on investment" value={pct(p.roi ?? 0, 1, { sign: true })} sub="entry → target" tone="up" />
            <KpiCard label="Risk : Reward" value={p.rr ? `${num(p.rr, 2)} : 1` : "—"} sub="reward / risk" />
            <KpiCard label="Confidence" value={`${p.confidence ?? "—"}%`} sub="momentum conviction" tone={p.confidence != null ? confTone(p.confidence) : "ink"} />
            <KpiCard label="Book weight" value={pct(p.weight, 1)} sub="inverse-vol" />
          </div>

          <Panel className="p-6 sm:p-8">
            <SectionLabel className="mb-1">Trade ladder</SectionLabel>
            <PriceLadder stop={p.stop!} entry={p.entry!} current={p.current} target={p.target!} fmt={(v) => priceFmt(v)} />
          </Panel>
        </>
      ) : (
        <p className="text-[13px] text-muted">
          Trade levels not generated yet — run the morning pipeline (morning_run.py) for full detail.
        </p>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Panel className="p-6">
          <SectionLabel className="mb-4">Signal context</SectionLabel>
          <div className="flex flex-col">
            <StatRow label="Momentum score (z)" value={num(p.score, 2)} />
            <StatRow label="Daily volatility" value={p.sigmaPct == null ? "—" : pct(p.sigmaPct / 100, 2)} />
            <StatRow label="Hold horizon" value={`${p.horizonDays ?? 10} trading days`} />
            <StatRow label="Book weight" value={pct(p.weight, 2)} />
            <StatRow label="Avg daily value" value={p.advCr == null ? "—" : `₹${p.advCr.toFixed(0)} cr`} />
          </div>
        </Panel>
        <Methodology
          lines={[
            "Selected as a top-20 name by the cross-sectional momentum z-score (4 horizons, 12-1 included).",
            `Target = entry × (1 + k·σ₁₀), stop = entry × (1 − 1.25·σ₁₀), where σ₁₀ is 10-day volatility (${p.sigmaPct ?? "—"}%/day scaled by √10).`,
            "k scales 1.25σ → 3.0σ with momentum strength, so stronger trends get wider targets.",
            "Confidence = logistic of the momentum z-score (stronger relative momentum → higher).",
            "Inverse-volatility weighted; held to the 10-day rebalance unless the market trend filter flips risk-off.",
          ]}
        />
      </div>
    </div>
  );
}
