import { CLOUD, probeConvex, type ConvexProbe } from "@/lib/source";

const MSG: Record<Exclude<ConvexProbe, "ok" | "offline">, string> = {
  "no-secret":    "DASHBOARD_PUSH_SECRET missing — add it in Vercel environment variables.",
  unauthorized:   "Convex rejected auth token. Check DASHBOARD_PUSH_SECRET matches Convex PUSH_SECRET.",
  missing:        "No data yet. Run push_dashboard.py after morning_run on your machine.",
};

export default async function DataSourceBanner() {
  if (!CLOUD) return null;
  const status = await probeConvex();
  if (status === "ok" || status === "offline") return null;
  return (
    <div
      className="mb-6 rounded-[10px] border px-4 py-2.5"
      style={{ borderColor: "rgba(255,179,0,0.20)", background: "rgba(255,179,0,0.04)" }}
    >
      <span className="text-[12px] font-medium text-warn">⚠ </span>
      <span className="text-[12px] text-muted">{MSG[status]}</span>
    </div>
  );
}
