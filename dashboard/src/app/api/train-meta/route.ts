import { NextResponse } from "next/server";
import { getTrainMeta } from "@/lib/adapters";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export async function GET() {
  try {
    return NextResponse.json(await getTrainMeta());
  } catch (e) {
    console.error("GET /api/train-meta failed:", e);
    return NextResponse.json({ error: "failed to read train meta" }, { status: 500 });
  }
}
