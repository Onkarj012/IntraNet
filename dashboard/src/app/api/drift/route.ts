import { NextResponse } from "next/server";
import { getDriftStatus, updateDriftStatus } from "@/lib/adapters";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export async function GET() {
  try {
    return NextResponse.json(await getDriftStatus());
  } catch (e) {
    console.error("GET /api/drift failed:", e);
    return NextResponse.json({ error: "failed to read drift status" }, { status: 500 });
  }
}

export async function POST(req: Request) {
  try {
    const body = await req.json();
    return NextResponse.json(await updateDriftStatus(body));
  } catch (e) {
    console.error("POST /api/drift failed:", e);
    return NextResponse.json({ error: "failed to update drift status" }, { status: 500 });
  }
}
