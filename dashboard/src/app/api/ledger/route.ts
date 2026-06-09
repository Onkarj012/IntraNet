import { NextResponse } from "next/server";
import { getLedgerCsv } from "@/lib/adapters";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export async function GET() {
  try {
    const csv = await getLedgerCsv();
    return new NextResponse(csv, { headers: { "Content-Type": "text/csv" } });
  } catch (e) {
    console.error("GET /api/ledger failed:", e);
    return NextResponse.json({ error: "failed to read ledger" }, { status: 500 });
  }
}
