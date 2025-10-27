"use server";

import { getServerSession } from "next-auth";
import { NextResponse } from "next/server";
import { authOptions } from "../../auth/[...nextauth]/route";

function backendBase(): string {
  return (
    process.env.API_BASE ??
    process.env.NEXT_PUBLIC_API_BASE ??
    "http://localhost:8080"
  );
}

async function withSession() {
  const session = await getServerSession(authOptions);
  if (!session || !session.apiKey) {
    return null;
  }
  return session;
}

export async function PATCH(request: Request, { params }: { params: { id: string } }) {
  const session = await withSession();
  if (!session) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const payload = await request.json();
  const response = await fetch(`${backendBase()}/v1/workspaces/queries/${params.id}`, {
    method: "PATCH",
    headers: {
      "x-api-key": session.apiKey,
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload),
    cache: "no-store"
  });

  if (!response.ok) {
    const detail = await response.text();
    return NextResponse.json({ error: "Backend error", detail }, { status: response.status });
  }

  const data = await response.json();
  return NextResponse.json({ data });
}

export async function DELETE(request: Request, { params }: { params: { id: string } }) {
  const session = await withSession();
  if (!session) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const response = await fetch(`${backendBase()}/v1/workspaces/queries/${params.id}`, {
    method: "DELETE",
    headers: {
      "x-api-key": session.apiKey
    },
    cache: "no-store"
  });

  if (!response.ok) {
    const detail = await response.text();
    return NextResponse.json({ error: "Backend error", detail }, { status: response.status });
  }

  return new NextResponse(null, { status: 204 });
}
