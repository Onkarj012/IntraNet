import { CLOUD, probeConvex, type ConvexProbe } from "@/lib/source";
import { Panel } from "@/components/ui";

const MESSAGES: Record<Exclude<ConvexProbe, "ok" | "offline">, string> = {
  "no-secret":
    "Convex URL is configured but DASHBOARD_PUSH_SECRET is missing on this deployment. Add it in Vercel → Settings → Environment Variables (Production and Preview), matching Convex PUSH_SECRET.",
  unauthorized:
    "Convex rejected the read token (401). Ensure Vercel DASHBOARD_PUSH_SECRET equals Convex PUSH_SECRET on deployment next-magpie-347.",
  missing:
    "Convex is reachable but has no data yet. Run push_dashboard.py after morning_run / daily_run on your Mac.",
};

export default async function DataSourceBanner() {
  if (!CLOUD) return null;
  const status = await probeConvex();
  if (status === "ok" || status === "offline") return null;

  return (
    <Panel variant="shell" radius="lg" className="mb-6 border-warn/40 bg-warn/5 p-4">
      <p className="text-[13px] font-medium text-warn">Hosted data unavailable</p>
      <p className="mt-1 text-[13px] leading-relaxed text-ink-60">{MESSAGES[status]}</p>
    </Panel>
  );
}
