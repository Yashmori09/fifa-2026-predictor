import { NextResponse } from "next/server";
import { getLiveData } from "@/lib/live-data";

// Same 60-second cache as the page — browser polling hits this route while a
// match is live, but we still don't hit football-data.org more than once a minute.
export const revalidate = 60;

export async function GET() {
  try {
    const data = await getLiveData();
    return NextResponse.json(data);
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "unknown" },
      { status: 502 }
    );
  }
}
