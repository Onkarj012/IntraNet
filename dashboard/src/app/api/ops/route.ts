import { NextResponse } from "next/server";
import { getOpsPayload } from "@/lib/data";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export async function GET() {
  try {
    return NextResponse.json(await getOpsPayload());
  } catch (e) {
    console.error("GET /api/ops failed:", e);
    return NextResponse.json({ error: "failed to read ops data" }, { status: 500 });
  }
}
