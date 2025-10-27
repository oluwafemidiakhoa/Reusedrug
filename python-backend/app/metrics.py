from __future__ import annotations

from __future__ import annotations

from typing import Mapping, Optional

from opentelemetry import metrics

_METER_NAME = "drug_repurposing.metrics"
_METER_VERSION = "0.1.0"

_rank_request_counter = None
_rank_warning_counter = None
_persona_selection_counter = None


def _get_rank_request_counter():
    global _rank_request_counter
    if _rank_request_counter is None:
        meter = metrics.get_meter(_METER_NAME, version=_METER_VERSION)
        _rank_request_counter = meter.create_counter(
            name="repurpose_rank_requests",
            description="Count of /v1/rank invocations",
            unit="1",
        )
    return _rank_request_counter


def _get_rank_warning_counter():
    global _rank_warning_counter
    if _rank_warning_counter is None:
        meter = metrics.get_meter(_METER_NAME, version=_METER_VERSION)
        _rank_warning_counter = meter.create_counter(
            name="repurpose_rank_warnings",
            description="Count of warnings emitted by /v1/rank",
            unit="1",
        )
    return _rank_warning_counter


def _get_persona_selection_counter():
    global _persona_selection_counter
    if _persona_selection_counter is None:
        meter = metrics.get_meter(_METER_NAME, version=_METER_VERSION)
        _persona_selection_counter = meter.create_counter(
            name="repurpose_persona_selections",
            description="Count of scoring persona selections on /v1/rank",
            unit="1",
        )
    return _persona_selection_counter


def record_rank_request(attributes: Optional[Mapping[str, str]] = None) -> None:
    counter = _get_rank_request_counter()
    counter.add(1, attributes=attributes or {})


def record_rank_warnings(count: int) -> None:
    if count <= 0:
        return
    counter = _get_rank_warning_counter()
    counter.add(count)


def record_persona_selection(persona: str, has_overrides: bool) -> None:
    counter = _get_persona_selection_counter()
    counter.add(
        1,
        attributes={
            "persona": persona,
            "custom_weights": "true" if has_overrides else "false",
        },
    )
