from __future__ import annotations

from fastapi import APIRouter
from opentelemetry import trace

from app.background import prefetcher
from app.metrics import record_persona_selection, record_rank_request, record_rank_warnings
from app.models import RankRequest, RankResponse, ScoringMetadata, ScoringPersona
from app.services import ranking
from app.services.scoring import get_default_weights, persona_definitions
from app import analytics

router = APIRouter(prefix="/v1", tags=["repurposing"])


@router.post("/rank", response_model=RankResponse)
async def rank_candidates(payload: RankRequest) -> RankResponse:
    overrides = payload.weights.as_dict() if payload.weights else None
    response = await ranking.compute_rank(
        payload.disease,
        persona=payload.persona,
        weight_overrides=overrides,
        exclude_contraindicated=payload.exclude_contraindicated,
    )

    scoring = response.scoring
    record_rank_request(
        {
            "normalized": "true" if (response.normalized_disease or "") else "false",
            "has_candidates": "true" if response.candidates else "false",
            "cached": "true" if response.cached else "false",
            "persona": (scoring.persona if scoring else "balanced"),
            "custom_weights": "true" if scoring and scoring.overrides else "false",
        }
    )
    record_rank_warnings(len(response.warnings))
    if scoring:
        record_persona_selection(scoring.persona, bool(scoring.overrides))
        await analytics.track_persona_event(
            persona=scoring.persona,
            overrides=scoring.overrides,
            disease_query=payload.disease,
            normalized_disease=response.normalized_disease,
            cached=response.cached,
        )
        span = trace.get_current_span()
        if span and span.is_recording():
            span.add_event(
                "persona.selection",
                {
                    "persona": scoring.persona,
                    "custom_weights": bool(scoring.overrides),
                    "override_keys": ",".join(sorted(scoring.overrides)) if scoring.overrides else "",
                },
            )

    prefetch_key = response.normalized_disease or payload.disease
    if not scoring or scoring.persona == "balanced":
        prefetcher.enqueue(prefetch_key)

    return response


@router.get("/metadata/scoring", response_model=ScoringMetadata)
async def get_scoring_metadata() -> ScoringMetadata:
    default_weights = get_default_weights()
    definitions = persona_definitions()
    personas = [
        ScoringPersona(
            name=item["name"],
            label=item["label"],
            description=item.get("description"),
            weights={key: round(value, 4) for key, value in item["weights"].items()},
        )
        for item in definitions
    ]
    return ScoringMetadata(
        default_persona="balanced",
        default_weights={key: round(value, 4) for key, value in default_weights.items()},
        personas=personas,
    )


