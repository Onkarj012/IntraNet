import type { CurvePoint } from "@/lib/types";

export type Series = {
  points: CurvePoint[];
  color: string;
  fill?: boolean;
  label?: string;
  marker?: boolean;
};

const W = 1000;

// Catmull-Rom → cubic bezier for smooth, premium curves.
function smoothPath(pts: { x: number; y: number }[]): string {
  if (pts.length < 2) return pts.length ? `M${pts[0].x},${pts[0].y}` : "";
  let d = `M${pts[0].x.toFixed(2)},${pts[0].y.toFixed(2)}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i - 1] ?? pts[i];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[i + 2] ?? p2;
    const c1x = p1.x + (p2.x - p0.x) / 6;
    const c1y = p1.y + (p2.y - p0.y) / 6;
    const c2x = p2.x - (p3.x - p1.x) / 6;
    const c2y = p2.y - (p3.y - p1.y) / 6;
    d += ` C${c1x.toFixed(2)},${c1y.toFixed(2)} ${c2x.toFixed(2)},${c2y.toFixed(2)} ${p2.x.toFixed(2)},${p2.y.toFixed(2)}`;
  }
  return d;
}

export default function AreaChart({
  series,
  height = 240,
  baselineZero = false,
  smooth = true,
}: {
  series: Series[];
  height?: number;
  baselineZero?: boolean;
  smooth?: boolean;
}) {
  const all = series.flatMap((s) => s.points.map((p) => p.value));
  if (all.length === 0)
    return (
      <div className="grid place-items-center text-[12px] text-muted" style={{ height }}>
        no data
      </div>
    );

  let min = Math.min(...all);
  let max = Math.max(...all);
  if (baselineZero) {
    min = Math.min(min, 0);
    max = Math.max(max, 0);
  }
  const pad = (max - min) * 0.1 || 1;
  min -= pad;
  max += pad;
  const H = height;
  const y = (v: number) => H - ((v - min) / (max - min)) * H;
  const xOf = (i: number, n: number) => (n <= 1 ? 0 : (i / (n - 1)) * W);
  const zeroY = y(0);
  const showZero = baselineZero && 0 >= min && 0 <= max;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      className="edge-fade-x h-full w-full overflow-visible"
      style={{ height }}
    >
      <defs>
        {series.map((s, si) => (
          <linearGradient key={si} id={`fill-${si}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={s.color} stopOpacity="0.30" />
            <stop offset="60%" stopColor={s.color} stopOpacity="0.06" />
            <stop offset="100%" stopColor={s.color} stopOpacity="0" />
          </linearGradient>
        ))}
      </defs>

      {showZero && (
        <line
          x1="0"
          x2={W}
          y1={zeroY}
          y2={zeroY}
          stroke="rgba(255,255,255,0.14)"
          strokeWidth="1"
          strokeDasharray="3 6"
        />
      )}

      {series.map((s, si) => {
        const n = s.points.length;
        const pts = s.points.map((pt, i) => ({ x: xOf(i, n), y: y(pt.value) }));
        const line = smooth ? smoothPath(pts) : pts.map((p, i) => `${i ? "L" : "M"}${p.x},${p.y}`).join(" ");
        const base = showZero ? zeroY : H;
        const last = pts[pts.length - 1];
        return (
          <g key={si} className="reveal">
            {s.fill && <path d={`${line} L${W},${base} L0,${base} Z`} fill={`url(#fill-${si})`} />}
            <path
              d={line}
              fill="none"
              stroke={s.color}
              strokeWidth="2.25"
              strokeLinejoin="round"
              strokeLinecap="round"
              vectorEffect="non-scaling-stroke"
            />
            {s.marker && last && (
              <>
                <circle cx={last.x} cy={last.y} r="9" fill={s.color} opacity="0.18" />
                <circle cx={last.x} cy={last.y} r="3.5" fill={s.color} stroke="#0b0b0b" strokeWidth="1.5" />
              </>
            )}
          </g>
        );
      })}
    </svg>
  );
}
