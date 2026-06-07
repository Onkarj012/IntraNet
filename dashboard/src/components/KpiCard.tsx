type Tone = "ink" | "up" | "down" | "warn";

const toneColor: Record<Tone, string> = {
  ink:  "text-ink",
  up:   "text-up",
  down: "text-down",
  warn: "text-warn",
};

export default function KpiCard({
  label, value, sub, tone = "ink", accent = false,
}: {
  label: string; value: string; sub?: string;
  tone?: Tone; accent?: boolean;
}) {
  return (
    <div className="card p-5">
      {accent && (
        <div
          className="pointer-events-none absolute inset-x-0 top-0 h-px rounded-t-[10px] opacity-50"
          style={{ background: "linear-gradient(90deg, transparent, #0099ff, transparent)" }}
        />
      )}
      <p className="t-label mb-3">{label}</p>
      <p className={`nums text-[clamp(22px,3vw,28px)] font-bold leading-none ${toneColor[tone]}`}>
        {value}
      </p>
      {sub && <p className="mt-2 text-[12px] text-muted">{sub}</p>}
    </div>
  );
}
