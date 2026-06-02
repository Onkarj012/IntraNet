import { getOpsPayload } from "@/lib/data";
import { inr, num, pct } from "@/lib/format";
import { Panel, SectionLabel, SectionHeading, StatusPill } from "@/components/ui";
import KpiCard from "@/components/KpiCard";
import DataTable, { type Column } from "@/components/DataTable";
import AutoRefresh from "@/components/AutoRefresh";
import type { EquityPick } from "@/lib/types";

export const dynamic = "force-dynamic";

function exitState(code: number | undefined) {
  if (code === undefined) return { tone: "neutral" as const, label: "no run recorded" };
  if (code === 0) return { tone: "up" as const, label: "clean run" };
  if (code === 2) return { tone: "warn" as const, label: "soft alert" };
  if (code === 3) return { tone: "down" as const, label: "hard halt" };
  return { tone: "down" as const, label: `failed (rc ${code})` };
}

function Cmd({ children }: { children: React.ReactNode }) {
  return (
    <code className="flex items-center gap-1 rounded-input border border-hair bg-base px-3 py-2 font-mono text-[12px] text-ink">
      <span className="text-muted">$</span>
      {children}
      <span className="cursor-blink ml-0.5 text-accent">▋</span>
    </code>
  );
}

export default async function OpsPage() {
  const d = await getOpsPayload();
  const run = d.lastRun;
  const es = exitState(run?.exitCode);
  const anyHalt = d.halts.futures || d.halts.equity;

  const pickCols: Column<EquityPick & { i: number }>[] = [
    { key: "i", header: "#", render: (r) => <span className="text-muted">{r.i}</span> },
    { key: "sym", header: "Symbol", render: (r) => <span className="text-ink">{r.symbol}</span> },
    { key: "w", header: "Weight", align: "right", render: (r) => pct(r.weight, 1) },
    { key: "s", header: "Score", align: "right", render: (r) => num(r.score, 2) },
    { key: "p", header: "Price", align: "right", render: (r) => `₹${r.price.toLocaleString("en-IN")}` },
    { key: "adv", header: "ADV", align: "right", render: (r) => `₹${r.advCr.toFixed(0)}cr` },
  ];

  return (
    <div className="space-y-16">
      {/* ───────── Hero ───────── */}
      <Panel variant="shell" radius="shell" glow diagonal className="p-6 sm:p-10">
        <div className="grid items-center gap-10 lg:grid-cols-[1.05fr_0.95fr]">
          <div>
            <SectionLabel>Automation · cron</SectionLabel>
            <h1 className="mt-4 text-[30px] font-bold leading-[1.06] tracking-[-0.03em] text-ink sm:text-[52px] sm:leading-[1.03]">
              Operations
              <br />
              <span className="text-ink-60">on autopilot.</span>
            </h1>
            <p className="mt-5 max-w-md text-[15px] font-normal leading-relaxed text-ink-60">
              One cron entrypoint chains the EOD data cache, futures paper ops, and the equity
              momentum recommendations every trading evening. This is the watchtower.
            </p>
            <div className="mt-7 flex flex-wrap items-center gap-3">
              <StatusPill tone={es.tone} pulse={es.tone !== "neutral"}>
                {es.label}
              </StatusPill>
              {run && (
                <StatusPill tone={run.ranToday ? "up" : "warn"} dot={false}>
                  {run.ranToday ? "ran today" : `last run ${run.date}`}
                </StatusPill>
              )}
              <AutoRefresh generatedAt={d.generatedAt} />
            </div>
          </div>

          {/* schedule + last run card */}
          <div className="surface rounded-lg p-5 sm:p-6">
            <div className="flex items-center justify-between">
              <SectionLabel>Cron schedule</SectionLabel>
              {d.cron ? (
                <StatusPill tone="accent" dot={false}>installed</StatusPill>
              ) : (
                <StatusPill tone="warn" dot={false}>not installed</StatusPill>
              )}
            </div>
            <div className="mt-4 space-y-3">
              {(d.cron?.jobs ?? []).map((j) => (
                <div key={j.entrypoint} className="rounded-card border border-hair bg-raised/40 p-3.5">
                  <div className="flex items-center justify-between">
                    <span className="text-[13px] text-ink">{j.name}</span>
                    <span className="nums text-[13px] font-semibold text-ink">{j.human}</span>
                  </div>
                  <p className="mt-1.5 font-mono text-[11px] text-ink-60">{j.entrypoint}</p>
                </div>
              ))}
              {!d.cron && (
                <p className="text-[12px] text-muted">
                  Run <span className="font-mono">scripts/trading/install_cron.sh</span> to install.
                </p>
              )}
            </div>
            <div className="mt-5 space-y-2.5 border-t border-hair pt-4 text-[12px]">
              <div className="flex justify-between">
                <span className="text-muted">last paper run</span>
                <span className="nums text-ink">{run?.timestamp ?? "no run recorded"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted">recommendations</span>
                <span className="nums text-ink">{d.picks ? `as of ${d.picks.asOf}` : "—"}</span>
              </div>
            </div>
          </div>
        </div>
      </Panel>

      {/* ───────── Pipeline steps ───────── */}
      <section>
        <SectionHeading
          eyebrow="Daily run"
          title="Pipeline steps"
          description="Step exit codes from the most recent run. 0 clean · 2 soft alert (continues) · 3 hard halt (kill-switch)."
        />
        <Panel variant="shell" radius="shell" className="p-6 sm:p-7">
          {run ? (
            <div className="flex flex-col">
              {run.steps.map((s, i) => {
                const st = exitState(s.returnCode);
                return (
                  <div
                    key={i}
                    className="flex items-center justify-between border-b border-hair py-4 last:border-0"
                  >
                    <div className="flex items-center gap-3.5">
                      <span className="grid h-7 w-7 place-items-center rounded-input bg-raised text-[12px] font-semibold text-ink-60">
                        {i + 1}
                      </span>
                      <span className="text-[14px] text-ink">{s.label}</span>
                    </div>
                    <StatusPill tone={st.tone} dot={false}>
                      rc {s.returnCode} · {st.label}
                    </StatusPill>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="text-[13px] text-muted">
              No run recorded yet. The status file appears after the first cron run (or run{" "}
              <span className="font-mono">daily_run.py</span> manually).
            </p>
          )}
        </Panel>
      </section>

      {/* ───────── Kite session + Halts ───────── */}
      <section className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Panel radius="lg" className="p-6">
          <div className="mb-5 flex items-center justify-between">
            <SectionLabel>Kite session · each morning</SectionLabel>
            <StatusPill tone={d.eodFetchedToday ? "up" : "warn"} dot={false}>
              {d.eodFetchedToday ? "EOD fetched today" : "refresh required"}
            </StatusPill>
          </div>
          <p className="mb-4 text-[13px] leading-relaxed text-ink-60">
            The access token expires daily. Refresh it before the evening run, otherwise the EOD
            cache step fails and the whole pipeline aborts.
          </p>
          <Cmd>.venv/bin/python scripts/data/kite_login.py</Cmd>
        </Panel>

        <Panel radius="lg" className="p-6">
          <div className="mb-5 flex items-center justify-between">
            <SectionLabel>Kill-switch state</SectionLabel>
            <StatusPill tone={anyHalt ? "down" : "up"}>{anyHalt ? "halted" : "all clear"}</StatusPill>
          </div>
          <div className="flex flex-col gap-3">
            {[
              { label: "Futures paper", halted: d.halts.futures, file: "results/router_v0/PAPER_TRADING_HALTED" },
              { label: "Equity paper", halted: d.halts.equity, file: "results/equity/EQUITY_PAPER_HALTED" },
            ].map((h) => (
              <div key={h.label} className="rounded-card border border-hair bg-raised/40 p-3.5">
                <div className="flex items-center justify-between">
                  <span className="text-[13px] text-ink">{h.label}</span>
                  <span className={`flex items-center gap-2 text-[12px] ${h.halted ? "text-down" : "text-up"}`}>
                    {h.halted ? "HARD HALT" : "running"}
                    <span className={`h-1.5 w-1.5 rounded-pill ${h.halted ? "bg-down" : "bg-up"}`} />
                  </span>
                </div>
                {h.halted && (
                  <p className="mt-2 font-mono text-[11px] text-muted">resume: rm {h.file}</p>
                )}
              </div>
            ))}
          </div>
        </Panel>
      </section>

      {/* ───────── Today's futures activity ───────── */}
      <section>
        <SectionHeading eyebrow="Live testing · futures" title="Latest session activity" />
        {d.futuresToday ? (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <KpiCard label="Session date" value={d.futuresToday.date} sub="most recent paper trades" accent />
            <KpiCard label="Trades" value={`${d.futuresToday.nTrades}`} sub="Variant A + C" />
            <KpiCard
              label="Net PnL"
              value={inr(d.futuresToday.net, { sign: true })}
              sub="this session"
              tone={d.futuresToday.net > 0 ? "up" : d.futuresToday.net < 0 ? "down" : "ink"}
            />
          </div>
        ) : (
          <p className="text-[13px] text-muted">No live paper trades recorded yet.</p>
        )}
      </section>

      {/* ───────── Equity recommendations ───────── */}
      <section>
        <SectionHeading
          eyebrow="Recommendation engine"
          title="Today's equity picks"
          description="Momentum factor + liquidity & trend gates → inverse-vol weighted book. Identical to the validated backtest selection."
          right={
            d.picks ? (
              <div className="flex items-center gap-2">
                <StatusPill tone={d.picks.state === "invested" ? "up" : "warn"} dot={false}>
                  {d.picks.state}
                </StatusPill>
                <span className="text-[12px] text-ink-60">
                  as of {d.picks.asOf} · {d.picks.nPicks} names
                </span>
              </div>
            ) : undefined
          }
        />
        <Panel variant="shell" radius="shell" className="p-6 sm:p-7">
          {d.picks && d.picks.items.length ? (
            <DataTable columns={pickCols} rows={d.picks.items.map((it, i) => ({ ...it, i: i + 1 }))} />
          ) : (
            <p className="text-[13px] text-muted">
              {d.picks ? "Risk-off — hold cash, no eligible names." : "No picks file in results/equity/picks/ yet."}
            </p>
          )}
        </Panel>
      </section>
    </div>
  );
}
