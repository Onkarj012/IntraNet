export type BarItem = { label: string; value: number; valueLabel?: string };

export default function BarChart({ items, unit = "" }: { items: BarItem[]; unit?: string }) {
  const max = Math.max(...items.map((i) => i.value), 0.0001);
  return (
    <div className="flex flex-col gap-2.5">
      {items.map((it, i) => (
        <div key={it.label} className="flex items-center gap-3">
          <div
            className="w-[100px] shrink-0 truncate text-[12px] text-muted"
            title={it.label}
          >
            {it.label}
          </div>
          <div className="h-1 flex-1 overflow-hidden rounded-[2px] bg-hair">
            <div
              className="anim-bar-fill h-full rounded-[2px] bg-accent"
              style={{ width: `${(it.value / max) * 100}%`, animationDelay: `${i * 30}ms` }}
            />
          </div>
          <div className="nums w-[52px] shrink-0 text-right text-[12px] text-ink">
            {it.valueLabel ?? `${it.value.toFixed(1)}${unit}`}
          </div>
        </div>
      ))}
    </div>
  );
}
