export type BarItem = { label: string; value: number; valueLabel?: string };

export default function BarChart({
  items,
  unit = "",
}: {
  items: BarItem[];
  unit?: string;
}) {
  const max = Math.max(...items.map((i) => i.value), 0.0001);
  return (
    <div className="flex flex-col gap-3">
      {items.map((it, i) => (
        <div key={it.label} className="flex items-center gap-4">
          <div
            className="w-[120px] shrink-0 truncate text-[13px] text-ink-60"
            title={it.label}
          >
            {it.label}
          </div>
          <div className="relative h-2 flex-1 overflow-hidden rounded-pill bg-mid/35">
            <div
              className="bar-grow h-full rounded-pill"
              style={{
                width: `${(it.value / max) * 100}%`,
                background:
                  "linear-gradient(90deg, var(--color-accent-soft), var(--color-accent))",
                animationDelay: `${i * 40}ms`,
              }}
            />
          </div>
          <div className="nums w-[62px] shrink-0 text-right text-[13px] text-ink">
            {it.valueLabel ?? `${it.value.toFixed(1)}${unit}`}
          </div>
        </div>
      ))}
    </div>
  );
}
