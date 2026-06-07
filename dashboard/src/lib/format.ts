// Display formatters (client-safe — no Node imports).

export function priceFmt(
  v: number | null | undefined,
  opts: { maxFrac?: number } = {},
): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return `₹${v.toLocaleString("en-IN", { maximumFractionDigits: opts.maxFrac ?? 2 })}`;
}

export function inr(v: number, opts: { sign?: boolean } = {}): string {
  const sign = opts.sign && v > 0 ? "+" : "";
  const neg = v < 0;
  const abs = Math.abs(Math.round(v));
  const grouped = abs.toLocaleString("en-IN");
  return `${neg ? "−" : sign}₹${grouped}`;
}

export function compactInr(v: number, opts: { sign?: boolean } = {}): string {
  const sign = opts.sign && v > 0 ? "+" : v < 0 ? "−" : "";
  const a = Math.abs(v);
  if (a >= 1e7) return `${sign}₹${(a / 1e7).toFixed(2)}Cr`;
  if (a >= 1e5) return `${sign}₹${(a / 1e5).toFixed(2)}L`;
  if (a >= 1e3) return `${sign}₹${(a / 1e3).toFixed(1)}k`;
  return `${sign}₹${a.toFixed(0)}`;
}

export function pct(v: number, digits = 1, opts: { sign?: boolean } = {}): string {
  const sign = opts.sign && v > 0 ? "+" : "";
  return `${sign}${(v * 100).toFixed(digits)}%`;
}

export function num(
  v: number | null | undefined,
  digits = 2,
  opts: { sign?: boolean } = {},
): string {
  if (v == null || !Number.isFinite(v)) return "∞";
  const sign = opts.sign && v > 0 ? "+" : "";
  return `${sign}${v.toFixed(digits)}`;
}

// Direction → semantic color class for a delta value.
export function toneClass(v: number): string {
  if (v > 0) return "text-up";
  if (v < 0) return "text-down";
  return "text-muted";
}
