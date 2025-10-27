"use server";

import { NextResponse } from "next/server";
import { z } from "zod";

const personaEventSchema = z.object({
  persona: z.string().min(1).max(64),
  disease_query: z.string().min(1).max(200).optional(),
  normalized_disease: z.string().max(200).nullable().optional(),
  override_keys: z.array(z.string().min(1)).optional(),
  override_weights: z.record(z.string(), z.number()).optional(),
  cached_backend: z.boolean().optional(),
  cached_bff: z.boolean().optional(),
  source: z.string().default("frontend"),
});

const ANALYTICS_ENDPOINT =
  process.env.PERSONA_ANALYTICS_ENDPOINT ?? process.env.NEXT_PUBLIC_PERSONA_ANALYTICS_ENDPOINT ?? "";
const ANALYTICS_API_KEY = process.env.PERSONA_ANALYTICS_API_KEY ?? "";

export async function POST(request: Request) {
  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const parsed = personaEventSchema.safeParse(payload);
  if (!parsed.success) {
    return NextResponse.json({ error: parsed.error.flatten().fieldErrors }, { status: 400 });
  }

  if (!ANALYTICS_ENDPOINT) {
    return new NextResponse(null, { status: 204 });
  }

  const eventPayload = {
    event: "persona_selection",
    persona: parsed.data.persona,
    disease_query: parsed.data.disease_query ?? null,
    normalized_disease: parsed.data.normalized_disease ?? null,
    override_keys: parsed.data.override_keys ?? null,
    override_weights: parsed.data.override_weights ?? null,
    cached_backend: parsed.data.cached_backend ?? false,
    cached_bff: parsed.data.cached_bff ?? false,
    source: parsed.data.source ?? "frontend",
    emitted_at: new Date().toISOString(),
  };

  try {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (ANALYTICS_API_KEY) {
      headers.Authorization = `Bearer ${ANALYTICS_API_KEY}`;
    }
    const response = await fetch(ANALYTICS_ENDPOINT, {
      method: "POST",
      headers,
      body: JSON.stringify(eventPayload),
      cache: "no-store",
    });
    if (!response.ok) {
      return NextResponse.json({ status: "upstream_error" }, { status: 202 });
    }
  } catch (error) {
    return NextResponse.json({ status: "network_error", detail: (error as Error).message }, { status: 202 });
  }

  return NextResponse.json({ status: "ok" }, { status: 200 });
}
