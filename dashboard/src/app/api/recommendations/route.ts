import { NextResponse } from "next/server";
import { getRecommendationsPayload } from "@/lib/data";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export async function GET() {
  try {
    return NextResponse.json(await getRecommendationsPayload());
  } catch (e) {
    console.error("GET /api/recommendations failed:", e);
    return NextResponse.json({ error: "failed to read recommendations" }, { status: 500 });
  }
}
