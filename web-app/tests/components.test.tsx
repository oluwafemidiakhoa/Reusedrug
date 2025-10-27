import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import ResultsList from "@/components/ResultsList";

vi.mock("next-auth/react", () => ({
  useSession: () => ({ data: null, status: "unauthenticated" }),
  signIn: vi.fn(),
  signOut: vi.fn()
}));

describe("ResultsList", () => {
  const candidate = {
    drug_id: "CHEMBL1",
    name: "Example Drug",
    score: {
      mechanism_fit: 0.1,
      network_proximity: 0.1,
      signature_reversal: 0.1,
      clinical_signal: 0.1,
      safety_penalty: -0.05,
      final_score: 0.35
    },
    evidence: [
      { source: "Open Targets", url: "http://example.com", summary: "Example evidence" }
    ],
    narrative: {
      summary: "Example narrative",
      reasoning_steps: ["Mechanism supported by Open Targets"],
      citations: [{ source: "Open Targets", label: "Example evidence", url: "http://example.com" }]
    }
  };

  it("renders empty state when no query provided", () => {
    render(
      <ResultsList
        query=""
        loading={false}
        candidates={[]}
        warnings={[]}
        normalizedDisease={null}
        backendCached={false}
        bffCached={false}
        relatedConcepts={[]}
        graphOverview={null}
      />
    );
    expect(
      screen.getByText(/enter a disease to activate the workspace/i)
    ).toBeInTheDocument();
  });

  it("shows sign in button when unauthenticated", () => {
    render(
      <ResultsList
        query="influenza"
        loading={false}
        candidates={[candidate]}
        warnings={[]}
        normalizedDisease="Influenza"
        backendCached={false}
        bffCached={false}
        relatedConcepts={[]}
        graphOverview={null}
        scoringSummary={null}
      />
    );
    expect(screen.getByText(/sign in to save/i)).toBeInTheDocument();
  });

  it("expands evidence when toggled", () => {
    render(
      <ResultsList
        query="influenza"
        loading={false}
        candidates={[candidate]}
        warnings={[]}
        normalizedDisease="Influenza"
        backendCached={false}
        bffCached={false}
        relatedConcepts={[]}
        graphOverview={null}
        scoringSummary={null}
      />
    );
    fireEvent.click(screen.getByText(/inspect evidence/i));
    expect(screen.getAllByText(/Example evidence/i).length).toBeGreaterThan(0);
  });

  it("filters candidates based on evidence category", () => {
    const clinicalCandidate = {
      ...candidate,
      drug_id: "CHEMBL2",
      name: "Clinical Candidate",
      evidence: [
        { source: "ClinicalTrials.gov", url: "http://clinical.example", summary: "Phase 3" }
      ],
      narrative: {
        summary: "Clinical narrative",
        reasoning_steps: ["Late-stage clinical activity"],
        citations: [{ source: "ClinicalTrials.gov", label: "Phase 3", url: "http://clinical.example" }]
      }
    };

    render(
      <ResultsList
        query="influenza"
        loading={false}
        candidates={[candidate, clinicalCandidate]}
        warnings={[]}
        normalizedDisease="Influenza"
        backendCached={false}
        bffCached={false}
        relatedConcepts={[]}
        graphOverview={null}
        scoringSummary={null}
      />
    );

    const clinicalToggle = screen.getByRole("button", { name: /clinical signal/i });
    fireEvent.click(clinicalToggle);

    expect(screen.getByText(/Clinical Candidate/)).toBeInTheDocument();
    expect(screen.queryByText(/Example Drug/)).not.toBeInTheDocument();
  });

  it("renders scoring summary chip when data provided", () => {
    render(
      <ResultsList
        query="influenza"
        loading={false}
        candidates={[candidate]}
        warnings={[]}
        normalizedDisease="Influenza"
        backendCached={false}
        bffCached={false}
        relatedConcepts={[]}
        graphOverview={null}
        scoringSummary={{
          persona: "mechanism-first",
          weights: {
            mechanism: 0.5,
            network: 0.2,
            signature: 0.1,
            clinical: 0.1,
            safety: 0.1
          },
          delta_vs_default: {
            mechanism: 0.2,
            network: -0.05,
            signature: -0.1,
            clinical: -0.05,
            safety: 0
          },
          overrides: {
            mechanism: 0.5
          }
        }}
        personaLabel="Mechanism-first"
      />
    );

    expect(screen.getByText(/Mechanism-first/)).toBeInTheDocument();
    expect(screen.getByText(/Mechanism \+20%/)).toBeInTheDocument();
  });
});
