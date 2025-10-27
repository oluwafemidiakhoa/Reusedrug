import { beforeEach, describe, expect, it, vi } from "vitest";

const routePath = "../app/api/analytics/persona/route";

describe("persona analytics route", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.restoreAllMocks();
    delete (globalThis as { fetch?: typeof fetch }).fetch;
    delete process.env.PERSONA_ANALYTICS_ENDPOINT;
    delete process.env.NEXT_PUBLIC_PERSONA_ANALYTICS_ENDPOINT;
    delete process.env.PERSONA_ANALYTICS_API_KEY;
  });

  it("returns 204 if analytics endpoint disabled", async () => {
    const { POST } = await import(routePath);
    const response = await POST(
      new Request("http://localhost/api/analytics/persona", {
        method: "POST",
        body: JSON.stringify({ persona: "balanced" })
      })
    );
    expect(response.status).toBe(204);
  });

  it("forwards payload to configured endpoint", async () => {
    process.env.PERSONA_ANALYTICS_ENDPOINT = "https://analytics.test/persona";
    process.env.PERSONA_ANALYTICS_API_KEY = "secret";

    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 200 }));
    global.fetch = fetchMock as typeof fetch;

    const { POST } = await import(routePath);

    const response = await POST(
      new Request("http://localhost/api/analytics/persona", {
        method: "POST",
        body: JSON.stringify({
          persona: "balanced",
          disease_query: "asthma"
        })
      })
    );

    expect(response.status).toBe(200);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [endpoint, options] = fetchMock.mock.calls[0];
    expect(endpoint).toBe("https://analytics.test/persona");
    expect(options?.method).toBe("POST");
    expect(options?.headers).toMatchObject({
      "Content-Type": "application/json",
      Authorization: "Bearer secret"
    });
  });
});
