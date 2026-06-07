import { NextResponse } from "next/server";
import { getWalkForward } from "@/lib/adapters";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export async function GET() {
  try {
    return NextResponse.json(await getWalkForward());
  } catch (e) {
    console.error("GET /api/walk-forward failed:", e);
    return NextResponse.json({ error: "failed to read walk-forward results" }, { status: 500 });
  }
}
