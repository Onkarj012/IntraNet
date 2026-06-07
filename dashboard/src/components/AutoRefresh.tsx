"use client";

import { useEffect, useState, useTransition } from "react";
import { useRouter } from "next/navigation";

export default function AutoRefresh({
  generatedAt,
  seconds = 30,
}: {
  generatedAt?: string | null;
  seconds?: number;
}) {
  const router = useRouter();
  const [pending, startTransition] = useTransition();
  const [auto, setAuto] = useState(true);
  const [, force] = useState(0);

  useEffect(() => {
    const tick = setInterval(() => force((n) => n + 1), 10_000);
    return () => clearInterval(tick);
  }, []);

  useEffect(() => {
    if (!auto) return;
    const id = setInterval(() => startTransition(() => router.refresh()), seconds * 1000);
    return () => clearInterval(id);
  }, [auto, seconds, router]);

  const ts = generatedAt ? new Date(generatedAt).getTime() : NaN;
  const agoLabel = Number.isFinite(ts)
    ? (() => {
        const ago = Math.max(0, Math.round((Date.now() - ts) / 1000));
        return ago < 60 ? `${ago}s ago` : `${Math.round(ago / 60)}m ago`;
      })()
    : "stale";

  return (
    <div className="flex flex-wrap items-center gap-2 text-[12px] text-ink-60">
      <span className="inline-flex items-center gap-1.5">
        <span className={`h-1.5 w-1.5 rounded-full ${pending ? "bg-warn" : "bg-up"} ${auto ? "animate-pulse" : ""}`} />
        updated {agoLabel}
      </span>
      <button
        type="button"
        onClick={() => setAuto((a) => !a)}
        className={`btn-ghost rounded-[4px] border px-2.5 py-1 ${auto ? "text-ink" : "text-muted"}`}
      >
        auto {auto ? "on" : "off"}
      </button>
      <button
        type="button"
        onClick={() => startTransition(() => router.refresh())}
        className="btn-ghost rounded-[4px] border px-2.5 py-1 text-ink"
      >
        refresh
      </button>
    </div>
  );
}
