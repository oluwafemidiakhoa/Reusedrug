"use server";

import { LRUCache } from "lru-cache";
import { NextResponse } from "next/server";
import { z } from "zod";

const weightsSchema = z
  .record(z.string(), z.number().min(0))
  .optional()
  .transform((value) => {
    if (!value) {
      return undefined;
    }
    const entries = Object.entries(value).map(([key, raw]) => [key, Number(raw) || 0]);
    return Object.fromEntries(entries);
  });

const requestSchema = z.object({
  disease: z.string().min(3).max(200),
  persona: z.string().min(1).max(64).optional(),
  weights: weightsSchema,
  exclude_contraindicated: z.boolean().optional()
});

const cache = new LRUCache<string, Record<string, unknown>>({
  max: 100,
  ttl: 5 * 60 * 1000
});

type Bucket = {
  tokens: number;
  updatedAt: number;
};

const buckets = new Map<string, Bucket>();
const RATE_LIMIT_TOKENS = Number(process.env.RATE_LIMIT_TOKENS ?? "30");
const RATE_LIMIT_WINDOW = Number(process.env.RATE_LIMIT_WINDOW_MS ?? "60000");

function consumeToken(key: string): boolean {
  const now = Date.now();
  const bucket = buckets.get(key) ?? { tokens: RATE_LIMIT_TOKENS, updatedAt: now };
  const elapsed = now - bucket.updatedAt;
  if (elapsed > RATE_LIMIT_WINDOW) {
    bucket.tokens = RATE_LIMIT_TOKENS;
    bucket.updatedAt = now;
  }
  if (bucket.tokens <= 0) {
    buckets.set(key, bucket);
    return false;
  }
  bucket.tokens -= 1;
  bucket.updatedAt = now;
  buckets.set(key, bucket);
  return true;
}

function backendBase(): string {
  return (
    process.env.API_BASE ??
    process.env.NEXT_PUBLIC_API_BASE ??
    "http://localhost:8080"
  );
}

function cacheKey(
  disease: string,
  persona?: string,
  weights?: Record<string, number>,
  excludeContraindicated?: boolean
) {
  if (!persona && (!weights || Object.keys(weights).length === 0) && !excludeContraindicated) {
    return disease;
  }
  const orderedWeights = weights
    ? Object.fromEntries(Object.keys(weights).sort().map((key) => [key, Number(weights[key] ?? 0)]))
    : undefined;
  return JSON.stringify({
    disease,
    persona: persona ?? null,
    weights: orderedWeights ?? null,
    exclude_contraindicated: Boolean(excludeContraindicated)
  });
}

export async function GET() {
  try {
    const response = await fetch(`${backendBase()}/healthz`, { cache: "no-store" });
    const payload = await response.json();
    return NextResponse.json(payload, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      { status: "degraded", detail: (error as Error).message },
      { status: 503 }
    );
  }
}

export async function POST(request: Request) {
  const ip =
    request.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ?? "local";
  if (!consumeToken(ip)) {
    return NextResponse.json(
      { error: "Rate limit exceeded. Please retry later." },
      { status: 429 }
    );
  }

  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return NextResponse.json(
      { error: "Invalid JSON payload" },
      { status: 400 }
    );
  }

  const parsed = requestSchema.safeParse(payload);
  if (!parsed.success) {
    return NextResponse.json(
      { error: parsed.error.flatten().fieldErrors },
      { status: 400 }
    );
  }

  const { disease, persona, weights, exclude_contraindicated } = parsed.data;
  const key = cacheKey(disease, persona, weights, exclude_contraindicated);
  const cached = cache.get(key);
  if (cached) {
    return NextResponse.json({ data: cached, cached: true });
  }

  const response = await fetch(`${backendBase()}/v1/rank`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(
      Object.fromEntries(
        Object.entries({
          disease,
          persona,
          weights,
          exclude_contraindicated
        }).filter(([, value]) => value !== undefined && value !== null)
      )
    ),
    cache: "no-store"
  });

  if (!response.ok) {
    const errorText = await response.text();
    return NextResponse.json(
      { error: "Backend error", detail: errorText },
      { status: 502 }
    );
  }

  const data = (await response.json()) as Record<string, unknown>;
  cache.set(key, data);
  console.info("persona_selection", {
    persona: persona ?? "balanced",
    overrides: weights ? Object.keys(weights) : [],
    cached: false
  });
  return NextResponse.json({ data, cached: false });
}
