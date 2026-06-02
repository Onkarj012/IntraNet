import { NextResponse } from "next/server";
import { getFuturesPayload } from "@/lib/data";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export async function GET() {
  try {
    return NextResponse.json(await getFuturesPayload());
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "failed to read futures data" },
      { status: 500 },
    );
  }
}
