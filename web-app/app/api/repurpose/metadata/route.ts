"use server";

import { NextResponse } from "next/server";

function backendBase(): string {
  return (
    process.env.API_BASE ??
    process.env.NEXT_PUBLIC_API_BASE ??
    "http://localhost:8080"
  );
}

export async function GET() {
  try {
    const response = await fetch(`${backendBase()}/v1/metadata/scoring`, {
      cache: "no-store"
    });
    if (!response.ok) {
      const detail = await response.text();
      return NextResponse.json(
        { error: "Failed to load scoring metadata", detail },
        { status: 502 }
      );
    }
    const payload = await response.json();
    return NextResponse.json(payload);
  } catch (error) {
    return NextResponse.json(
      { error: "Metadata request failed", detail: (error as Error).message },
      { status: 500 }
    );
  }
}
