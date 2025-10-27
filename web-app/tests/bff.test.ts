import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const routePath = "../app/api/repurpose/route";

describe("BFF repurpose route", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.restoreAllMocks();
    process.env.API_BASE = "http://backend.test";
  });

  afterEach(() => {
    delete (globalThis as { fetch?: typeof fetch }).fetch;
  });

  it("rejects invalid payloads", async () => {
    const { POST } = await import(routePath);
    const response = await POST(
      new Request("http://localhost/api/repurpose", {
        method: "POST",
        body: JSON.stringify({ disease: "ab" })
      })
    );
    expect(response.status).toBe(400);
  });

  it("caches successful responses", async () => {
    const backendPayload = {
      candidates: [
        {
          drug_id: "CHEMBL1",
          name: "Example",
          score: {
            mechanism_fit: 0.1,
            network_proximity: 0.1,
            signature_reversal: 0.1,
            clinical_signal: 0.1,
            safety_penalty: -0.05,
            final_score: 0.35
          },
          evidence: []
        }
      ],
      warnings: [],
      cached: false
    };

    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(backendPayload), {
        status: 200,
        headers: { "Content-Type": "application/json" }
      })
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const { POST } = await import(routePath);

    const makeRequest = () =>
      POST(
        new Request("http://localhost/api/repurpose", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ disease: "influenza" })
        })
      );

    const first = await makeRequest();
    expect(first.status).toBe(200);
    const second = await makeRequest();
    expect(second.status).toBe(200);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
