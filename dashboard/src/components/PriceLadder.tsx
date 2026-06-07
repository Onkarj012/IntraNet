export default function PriceLadder({
  stop,
  entry,
  current,
  target,
  fmt = (v: number) => v.toFixed(2),
}: {
  stop: number;
  entry: number;
  current: number | null | undefined;
  target: number;
  fmt?: (v: number) => string;
}) {
  const cur = current ?? entry;
  const lo = Math.min(stop, entry, cur, target);
  const hi = Math.max(stop, entry, cur, target);
  const span = hi - lo || 1;
  const pos = (v: number) => `${((v - lo) / span) * 100}%`;

  const markers = [
    { key: "stop",   label: "stop",   value: stop,   color: "text-down" },
    { key: "entry",  label: "entry",  value: entry,  color: "text-ink" },
    { key: "target", label: "target", value: target, color: "text-up" },
  ] as const;

  return (
    <div className="pt-6">
      <div className="relative h-1 rounded-[4px] bg-raised">
        <div
          className="absolute inset-0 rounded-[4px]"
          style={{
            background: "linear-gradient(90deg, rgba(255,0,140,0.35) 0%, rgba(245,245,245,0.15) 50%, rgba(0,255,94,0.35) 100%)",
          }}
        />
        <div
          className="absolute -top-1 h-4 w-px -translate-x-1/2 bg-ink"
          style={{ left: pos(entry) }}
        />
        {current != null && (
          <div
            className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-[4px] border-2 border-bg bg-accent"
            style={{ left: pos(cur) }}
          />
        )}
      </div>
      <div className="relative mt-4 h-10 text-[11px]">
        {markers.map((m) => (
          <div key={m.key} className="absolute -translate-x-1/2 text-center" style={{ left: pos(m.value) }}>
            <p className={m.color}>{m.label}</p>
            <p className="nums text-ink-60">{fmt(m.value)}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
