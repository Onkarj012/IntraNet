// Pluggable data source.
//   - Local dev (no CONVEX_HTTP_URL): read the repo's files/dirs directly.
//   - Hosted (CONVEX_HTTP_URL set):   read artifacts pushed to Convex over HTTP.
// Everything downstream (CSV parsing, metrics) is identical for both modes —
// only the raw-bytes source changes here.
import fs from "node:fs";
import path from "node:path";

const REPO_ROOT = process.env.REPO_ROOT
  ? path.resolve(process.env.REPO_ROOT)
  : path.resolve(process.cwd(), "..");

// e.g. https://<deployment>.convex.site  (HTTP-action origin, not .convex.cloud)
const CONVEX = process.env.CONVEX_HTTP_URL?.replace(/\/$/, "");

export const CLOUD = !!CONVEX;

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
  try {
    const r = await fetch(`${CONVEX}/file?key=${encodeURIComponent(key)}`, {
      cache: "no-store",
    });
    if (!r.ok) return null;
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
