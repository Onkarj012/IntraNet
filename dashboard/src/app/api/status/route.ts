import { NextResponse } from "next/server";
import { getPipelineStatus } from "@/lib/adapters";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export async function GET() {
  try {
    return NextResponse.json(await getPipelineStatus());
  } catch (e) {
    console.error("GET /api/status failed:", e);
    return NextResponse.json({ error: "failed to read status" }, { status: 500 });
  }
}
