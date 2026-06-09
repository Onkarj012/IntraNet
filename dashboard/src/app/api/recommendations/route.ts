import { NextResponse } from "next/server";
import { getBriefRecommendations } from "@/lib/adapters";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export async function GET() {
  try {
    return NextResponse.json(await getBriefRecommendations());
  } catch (e) {
    console.error("GET /api/recommendations failed:", e);
    return NextResponse.json({ error: "failed to read recommendations" }, { status: 500 });
  }
}
