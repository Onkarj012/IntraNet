import React from "react";

/* ── Label / eyebrow ── */
export function Label({ children, className }: { children: React.ReactNode; className?: string }) {
  return <p className={`t-label${className ? ` ${className}` : ""}`}>{children}</p>;
}
export const SectionLabel = Label;

/* ── Section heading block ── */
export function Heading({
  label, title, description, right,
}: {
  label?: string; title: string; description?: string; right?: React.ReactNode;
}) {
  return (
    <div className="mb-8 flex flex-wrap items-start justify-between gap-4">
      <div className="max-w-xl">
        {label && <Label>{label}</Label>}
        <h2 className="t-h2 mt-2.5 text-ink">{title}</h2>
        {description && <p className="t-body mt-2 text-muted">{description}</p>}
      </div>
      {right && <div className="shrink-0 pt-1">{right}</div>}
    </div>
  );
}
export function SectionHeading(props: {
  eyebrow: string; title: string; description?: string; right?: React.ReactNode;
}) {
  return <Heading label={props.eyebrow} title={props.title} description={props.description} right={props.right} />;
}

/* ── Card / Panel ── */
export function Card({
  children, className, raised = false, hero = false,
}: {
  children: React.ReactNode; className?: string; raised?: boolean; hero?: boolean;
}) {
  const base = hero ? "card-hero" : raised ? "card-raised" : "card";
  return <div className={`${base}${className ? ` ${className}` : ""}`}>{children}</div>;
}

export function Panel({
  children, className, raised, hero,
}: {
  children: React.ReactNode; className?: string;
  variant?: string; radius?: string; glow?: boolean; diagonal?: boolean;
  raised?: boolean; hero?: boolean;
}) {
  return <Card raised={raised} hero={hero} className={className}>{children}</Card>;
}

/* ── Hero section (two-column) ── */
export function HeroPanel({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <Panel hero className={`p-6 sm:p-10${className ? ` ${className}` : ""}`}>
      {children}
    </Panel>
  );
}

/* ── Status chip ── */
type ChipTone = "green" | "amber" | "blue" | "pink" | "neutral";

const CHIP: Record<ChipTone, { text: string; border: string; bg: string }> = {
  green:   { text: "#00ff5e", border: "rgba(0,255,94,0.20)",   bg: "rgba(0,255,94,0.06)"  },
  amber:   { text: "#ffb300", border: "rgba(255,179,0,0.20)",  bg: "rgba(255,179,0,0.06)" },
  blue:    { text: "#0099ff", border: "rgba(0,153,255,0.20)",  bg: "rgba(0,153,255,0.06)" },
  pink:    { text: "#ff008c", border: "rgba(255,0,140,0.20)",  bg: "rgba(255,0,140,0.06)" },
  neutral: { text: "#737373", border: "rgba(245,245,245,0.10)", bg: "transparent"          },
};

export function Chip({
  tone = "neutral", children, dot = false, pulse = false,
}: {
  tone?: ChipTone; children: React.ReactNode; dot?: boolean; pulse?: boolean;
}) {
  const c = CHIP[tone];
  return (
    <span
      className="chip"
      style={{ borderColor: c.border, background: c.bg, color: c.text }}
    >
      {dot && (
        <span
          className="inline-block h-[5px] w-[5px] shrink-0 rounded-full bg-current"
          style={pulse ? { animation: "dot-pulse 2s ease infinite" } : undefined}
        />
      )}
      {children}
    </span>
  );
}

export function StatusPill({
  tone = "neutral", children, dot = true, pulse = false,
}: {
  tone?: "up" | "down" | "warn" | "accent" | "neutral";
  children: React.ReactNode; dot?: boolean; pulse?: boolean;
}) {
  const map: Record<string, ChipTone> = {
    up: "green", down: "pink", warn: "amber", accent: "blue", neutral: "neutral",
  };
  return <Chip tone={map[tone] ?? "neutral"} dot={dot} pulse={pulse}>{children}</Chip>;
}

/* ── Key-value row ── */
export function StatRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between border-b border-hair py-3 text-[13px] last:border-0">
      <span className="text-muted">{label}</span>
      <span className="nums text-ink">{value}</span>
    </div>
  );
}

/* ── Terminal command block ── */
export function Cmd({ children }: { children: React.ReactNode }) {
  return (
    <code className="code-block flex items-center gap-1.5 text-[12px]">
      <span className="code-comment">$</span>
      <span className="code-plain">{children}</span>
      <span className="code-keyword cursor-blink">▋</span>
    </code>
  );
}

/* ── Back link ── */
export function BackLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a href={href} className="t-small text-ink-60 no-underline hover:text-accent">
      {children}
    </a>
  );
}
