export function SectionLabel({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return <p className={`eyebrow ${className}`}>{children}</p>;
}

export function SectionHeading({
  eyebrow,
  title,
  description,
  right,
}: {
  eyebrow: string;
  title: string;
  description?: string;
  right?: React.ReactNode;
}) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div className="max-w-2xl">
        <SectionLabel>{eyebrow}</SectionLabel>
        <h2 className="mt-3 text-[26px] font-bold leading-[1.1] tracking-[-0.02em] text-ink sm:text-[30px]">
          {title}
        </h2>
        {description && (
          <p className="mt-2.5 text-[15px] font-normal leading-snug text-ink-60">
            {description}
          </p>
        )}
      </div>
      {right && <div className="shrink-0">{right}</div>}
    </div>
  );
}

export function Panel({
  children,
  className = "",
  variant = "surface",
  radius = "lg",
  diagonal = false,
  glow = false,
}: {
  children: React.ReactNode;
  className?: string;
  variant?: "surface" | "shell";
  radius?: "card" | "lg" | "shell";
  diagonal?: boolean;
  glow?: boolean;
}) {
  const r =
    radius === "shell" ? "rounded-shell" : radius === "card" ? "rounded-card" : "rounded-lg";
  return (
    <section className={`relative overflow-hidden ${variant} ${r} ${className}`}>
      {glow && (
        <div className="glow-violet pointer-events-none absolute -top-24 left-1/2 -z-0 h-64 w-[80%] -translate-x-1/2" />
      )}
      {diagonal && (
        <div className="diagonal-lines pointer-events-none absolute inset-0 opacity-50" />
      )}
      <div className="relative z-10">{children}</div>
    </section>
  );
}

type Tone = "up" | "down" | "warn" | "accent" | "neutral";
const TONE: Record<Tone, string> = {
  up: "bg-up/10 text-up ring-up/20",
  down: "bg-down/10 text-down ring-down/20",
  warn: "bg-warn/10 text-warn ring-warn/20",
  accent: "bg-accent/12 text-violet ring-accent/25",
  neutral: "bg-raised text-ink-60 ring-hair",
};

export function StatusPill({
  tone = "neutral",
  children,
  dot = true,
  pulse = false,
}: {
  tone?: Tone;
  children: React.ReactNode;
  dot?: boolean;
  pulse?: boolean;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-pill px-3 py-1.5 text-[12px] font-semibold ring-1 ring-inset ${TONE[tone]}`}
    >
      {dot && (
        <span className={`h-1.5 w-1.5 rounded-pill bg-current ${pulse ? "animate-pulse" : ""}`} />
      )}
      {children}
    </span>
  );
}
