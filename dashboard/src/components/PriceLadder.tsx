// Horizontal stop → entry → current → target ladder.
export default function PriceLadder({
  stop,
  entry,
  current,
  target,
  fmt = (v: number) => v.toFixed(2),
}: {
  stop: number;
  entry: number;
  current: number;
  target: number;
  fmt?: (v: number) => string;
}) {
  const lo = Math.min(stop, entry, current, target);
  const hi = Math.max(stop, entry, current, target);
  const span = hi - lo || 1;
  const pos = (v: number) => `${((v - lo) / span) * 100}%`;
  return (
    <div className="pt-6">
      <div className="relative h-2 rounded-pill bg-gradient-to-r from-down/60 via-mid to-up/60">
        {/* entry marker */}
        <div className="absolute -top-1 h-4 w-0.5 -translate-x-1/2 bg-ink" style={{ left: pos(entry) }} />
        {/* current marker */}
        <div
          className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-pill bg-accent ring-2 ring-bg"
          style={{ left: pos(current) }}
        />
      </div>
      <div className="relative mt-3 h-8 text-[11px]">
        <div className="absolute -translate-x-1/2 text-center" style={{ left: pos(stop) }}>
          <p className="text-down">stop</p>
          <p className="nums text-ink-60">{fmt(stop)}</p>
        </div>
        <div className="absolute -translate-x-1/2 text-center" style={{ left: pos(entry) }}>
          <p className="text-ink">entry</p>
          <p className="nums text-ink-60">{fmt(entry)}</p>
        </div>
        <div className="absolute -translate-x-1/2 text-center" style={{ left: pos(target) }}>
          <p className="text-up">target</p>
          <p className="nums text-ink-60">{fmt(target)}</p>
        </div>
      </div>
    </div>
  );
}
