import { NextResponse } from "next/server";
import { getEquityPayload } from "@/lib/data";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export async function GET() {
  try {
    return NextResponse.json(await getEquityPayload());
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "failed to read equity data" },
      { status: 500 },
    );
  }
}
