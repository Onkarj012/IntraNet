// Pluggable data source.
//   - Local dev (no CONVEX_HTTP_URL): read the repo's files/dirs directly.
//   - Hosted (CONVEX_HTTP_URL set):   read artifacts pushed to Convex over HTTP.
// Everything downstream (CSV parsing, metrics) is identical for both modes —
// only the raw-bytes source changes here.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const REPO_MARKER = "scripts/trading/daily_run.py";

function resolveRepoRoot(): string {
  if (process.env.REPO_ROOT) return path.resolve(process.env.REPO_ROOT);
  const candidates = [
    path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../.."),
    process.cwd(),
    path.resolve(process.cwd(), ".."),
  ];
  for (const root of candidates) {
    if (fs.existsSync(path.join(root, REPO_MARKER))) return root;
  }
  return candidates[0];
}

const REPO_ROOT = resolveRepoRoot();

// e.g. https://<deployment>.convex.site  (HTTP-action origin, not .convex.cloud)
const CONVEX = (
  process.env.CONVEX_HTTP_URL ?? process.env.NEXT_PUBLIC_CONVEX_SITE_URL
)?.replace(/\/$/, "");
// Server-only on Vercel (never NEXT_PUBLIC_* — do not expose to the browser).
const READ_SECRET =
  process.env.DASHBOARD_PUSH_SECRET?.trim() ||
  process.env.PUSH_SECRET?.trim() ||
  "";

export const CLOUD = !!CONVEX;

export type ConvexProbe = "ok" | "no-secret" | "unauthorized" | "missing" | "offline";

/** Quick health check for hosted misconfiguration (used in layout banner). */
export async function probeConvex(): Promise<ConvexProbe> {
  if (!CONVEX) return "offline";
  if (!READ_SECRET) return "no-secret";
  try {
    const r = await fetch(
      `${CONVEX}/file?key=${encodeURIComponent("results/recommendations.json")}`,
      { cache: "no-store", headers: { Authorization: `Bearer ${READ_SECRET}` } },
    );
    if (r.status === 401) return "unauthorized";
    if (!r.ok) return "offline";
    const t = await r.text();
    return t ? "ok" : "missing";
  } catch {
    return "offline";
  }
}

const p = (rel: string) => path.join(REPO_ROOT, rel);

function existsRel(rel: string): boolean {
  try {
    return fs.existsSync(p(rel));
  } catch {
    return false;
  }
}

function localLatest(dir: string, re: RegExp): string | null {
  try {
    const files = fs.readdirSync(p(dir)).filter((f) => re.test(f)).sort();
    if (!files.length) return null;
    return fs.readFileSync(path.join(p(dir), files[files.length - 1]), "utf8");
  } catch {
    return null;
  }
}

function localText(key: string): string | null {
  switch (key) {
    case "@health_latest":
      return localLatest("logs", /^health_check_.*\.json$/);
    case "@picks_latest":
      return localLatest("results/equity/picks", /^picks_.*\.json$/);
    case "@ops_latest":
      return localLatest("logs", /^paper_trade_ops_.*\.json$/);
    case "@halts":
      return JSON.stringify({
        futures: existsRel("results/router_v0/PAPER_TRADING_HALTED"),
        equity: existsRel("results/equity/EQUITY_PAPER_HALTED"),
      });
    default:
      try {
        return fs.readFileSync(p(key), "utf8");
      } catch {
        return null;
      }
  }
}

async function convexText(key: string): Promise<string | null> {
  if (!READ_SECRET) {
    if (process.env.NODE_ENV === "development") {
      console.warn(
        "[source] CONVEX_HTTP_URL is set but DASHBOARD_PUSH_SECRET (or PUSH_SECRET) is missing — hosted reads disabled",
      );
    }
    return null;
  }
  try {
    const r = await fetch(`${CONVEX}/file?key=${encodeURIComponent(key)}`, {
      cache: "no-store",
      headers: { Authorization: `Bearer ${READ_SECRET}` },
    });
    if (!r.ok) {
      if (process.env.NODE_ENV === "development" && r.status === 401) {
        console.warn(`[source] Convex GET ${key} → 401 (secret mismatch?)`);
      }
      return null;
    }
    const t = await r.text();
    return t === "" ? null : t;
  } catch {
    return null;
  }
}

/** Canonical keys the pipeline pushes / the dashboard reads. */
export const PUSH_KEYS = [
  "results/router_v0/paper_trading_ledger.csv",
  "results/router_v0/tier1_validation.json",
  "results/router_v0/forward_walk_summary.json",
  "results/router_v0/feature_importance_long.json",
  "results/equity/paper_ledger.csv",
  "results/daily_run_status.json",
  "results/cron_status.json",
  "results/recommendations.json",
  "@health_latest",
  "@picks_latest",
  "@ops_latest",
  "@halts",
];

export async function srcText(key: string): Promise<string | null> {
  return CONVEX ? convexText(key) : localText(key);
}
