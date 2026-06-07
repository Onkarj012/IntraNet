import { NextResponse } from "next/server";
import { getRegimeBreakdown } from "@/lib/adapters";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export async function GET() {
  try {
    return NextResponse.json(await getRegimeBreakdown());
  } catch (e) {
    console.error("GET /api/regime failed:", e);
    return NextResponse.json({ error: "failed to read regime breakdown" }, { status: 500 });
  }
}
