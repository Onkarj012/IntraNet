import { SectionLabel } from "./ui";

export default function KpiCard({
  label,
  value,
  sub,
  tone = "ink",
  accent = false,
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: "ink" | "up" | "down" | "warn";
  accent?: boolean;
}) {
  const valueColor =
    tone === "up"
      ? "text-up"
      : tone === "down"
        ? "text-down"
        : tone === "warn"
          ? "text-warn"
          : "text-ink";
  return (
    <div className="surface group relative overflow-hidden rounded-card p-5">
      {accent && (
        <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-accent to-transparent opacity-70" />
      )}
      <SectionLabel>{label}</SectionLabel>
      <p
        className={`nums mt-4 text-[24px] font-bold leading-none sm:text-[30px] ${valueColor}`}
      >
        {value}
      </p>
      {sub && <p className="nums mt-2.5 text-[12px] font-normal text-ink-60">{sub}</p>}
    </div>
  );
}
