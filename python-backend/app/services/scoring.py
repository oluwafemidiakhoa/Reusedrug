from __future__ import annotations

import os
from statistics import mean
from typing import Dict, Iterable, Mapping, Optional, Tuple

from app.models import ScoreBreakdown
from app.services.graph import GraphInsights


_WEIGHT_KEYS: Tuple[str, ...] = ("mechanism", "network", "signature", "clinical", "safety")

_BASE_DEFAULT_WEIGHTS: Dict[str, float] = {
    "mechanism": 0.30,
    "network": 0.25,
    "signature": 0.20,
    "clinical": 0.15,
    "safety": 0.10,
}

_PERSONA_TEMPLATES: Dict[str, Dict[str, float]] = {
    "mechanism-first": {
        "mechanism": 0.5,
        "network": 0.2,
        "signature": 0.15,
        "clinical": 0.1,
        "safety": 0.05,
    },
    "clinical-first": {
        "mechanism": 0.18,
        "network": 0.18,
        "signature": 0.14,
        "clinical": 0.40,
        "safety": 0.10,
    },
}

_PERSONA_METADATA: Dict[str, Dict[str, Optional[str]]] = {
    "balanced": {
        "label": "Balanced",
        "description": "Default weighting across evidence dimensions.",
    },
    "mechanism-first": {
        "label": "Mechanism-first",
        "description": "Emphasizes mechanistic evidence and network proximity.",
    },
    "clinical-first": {
        "label": "Clinical-first",
        "description": "Prioritizes clinical trial signal and safety confidence.",
    },
}


def _normalize_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    sanitized: Dict[str, float] = {}
    for key in _WEIGHT_KEYS:
        value = weights.get(key, 0.0)
        try:
            sanitized[key] = max(float(value), 0.0)
        except (TypeError, ValueError):
            sanitized[key] = 0.0
    total = sum(sanitized.values())
    if total <= 0:
        raise ValueError("weight total must be positive")
    return {key: sanitized[key] / total for key in _WEIGHT_KEYS}


def get_default_weights() -> Dict[str, float]:
    overrides: Dict[str, float] = {}
    for key in _WEIGHT_KEYS:
        env_var = f"SCORE_WEIGHT_{key.upper()}"
        try:
            raw = os.getenv(env_var)
            if raw is None:
                continue
            overrides[key] = max(float(raw), 0.0)
        except (TypeError, ValueError):
            continue

    if overrides:
        try:
            return _normalize_weights({**_BASE_DEFAULT_WEIGHTS, **overrides})
        except ValueError:
            return _normalize_weights(_BASE_DEFAULT_WEIGHTS)
    return _normalize_weights(_BASE_DEFAULT_WEIGHTS)


def _resolve_persona(
    persona: Optional[str],
    default_weights: Dict[str, float],
) -> tuple[str, Dict[str, float]]:
    if not persona:
        return "balanced", dict(default_weights)
    key = persona.strip().lower()
    if key == "balanced":
        return "balanced", dict(default_weights)
    template = _PERSONA_TEMPLATES.get(key)
    if template:
        try:
            return key, _normalize_weights(template)
        except ValueError:
            return key, dict(default_weights)
    return "custom", dict(default_weights)


def _sanitize_overrides(overrides: Optional[Mapping[str, float]]) -> Dict[str, float]:
    if not overrides:
        return {}
    sanitized: Dict[str, float] = {}
    for key in _WEIGHT_KEYS:
        if key not in overrides:
            continue
        raw = overrides[key]
        if raw is None:
            continue
        try:
            sanitized[key] = max(float(raw), 0.0)
        except (TypeError, ValueError):
            continue
    return sanitized


def _apply_overrides(base: Dict[str, float], overrides: Dict[str, float]) -> Dict[str, float]:
    if not overrides:
        return dict(base)
    combined = {key: base.get(key, 0.0) for key in _WEIGHT_KEYS}
    combined.update(overrides)
    try:
        return _normalize_weights(combined)
    except ValueError:
        return dict(base)


def resolve_scoring_context(
    persona: Optional[str],
    overrides: Optional[Mapping[str, float]],
) -> tuple[str, Dict[str, float], Dict[str, float], Dict[str, float]]:
    default_weights = get_default_weights()
    resolved_persona, base_weights = _resolve_persona(persona, default_weights)
    sanitized_overrides = _sanitize_overrides(overrides)
    applied_weights = _apply_overrides(base_weights, sanitized_overrides)

    active_persona = resolved_persona
    if sanitized_overrides:
        active_persona = "custom"

    applied_overrides = {key: applied_weights[key] for key in sanitized_overrides.keys()}

    return active_persona, applied_weights, default_weights, applied_overrides


def persona_definitions() -> list[dict]:
    default_weights = get_default_weights()
    definitions = []
    for name, meta in _PERSONA_METADATA.items():
        if name == "balanced":
            weights = default_weights
        else:
            template = _PERSONA_TEMPLATES.get(name, {})
            try:
                weights = _normalize_weights(template)
            except ValueError:
                weights = default_weights
        definitions.append(
            {
                "name": name,
                "label": meta.get("label") or name.title(),
                "description": meta.get("description"),
                "weights": weights,
            }
        )
    return definitions

def _normalize(value: float, lower: float, upper: float) -> float:
    if upper == lower:
        return 0.0
    clamped = max(min(value, upper), lower)
    return (clamped - lower) / (upper - lower)


def _mechanism_fit(
    activity: Optional[Dict], moa_annotations: Optional[Iterable[Dict]]
) -> float:
    base = 0.05
    potency_val: Optional[float] = None

    if activity:
        potency = activity.get("potency")
        if potency is None:
            base = 0.1
        else:
            try:
                potency_val = float(potency)
            except (TypeError, ValueError):
                potency_val = None
            if potency_val is None:
                base = 0.1
            else:
                base = max(1 - _normalize(potency_val, 1e2, 1e6), 0.0)

    annotations = list(moa_annotations or [])
    if annotations:
        uniqueness = len(
            {ann.get("mechanism_of_action") for ann in annotations if ann.get("mechanism_of_action")}
        )
        base += min(0.2, 0.05 * uniqueness)

    return min(base, 1.0)


def _network_score(
    target_count: int,
    translator_paths: Optional[Iterable[Dict]],
    graph_metrics: Optional[GraphInsights],
) -> float:
    coverage = min(target_count / 10.0, 1.0) if target_count else 0.0
    paths = list(translator_paths or [])
    if paths:
        path_depths = []
        for path in paths:
            # Translator analyses->edge_bindings typically flatten edges by keys.
            edges = path.get("edge_bindings") if isinstance(path, dict) else path
            if isinstance(edges, dict):
                path_depths.append(sum(len(edges[key]) for key in edges))
        translator_signal = min((mean(path_depths) if path_depths else len(paths)) / 5.0, 1.0)
    else:
        translator_signal = 0.0
    graph_bonus = 0.0
    if graph_metrics:
        graph_bonus = min(graph_metrics.density * 0.5, 0.3)
        if graph_metrics.average_shortest_path:
            graph_bonus += max(0.0, (5 - graph_metrics.average_shortest_path) / 10)
        graph_bonus = min(graph_bonus, 0.5)
    return max(0.2, 0.5 * coverage + 0.3 * translator_signal + graph_bonus)


def _signature_score(signature_records: Optional[Iterable[Dict]]) -> float:
    if not signature_records:
        return 0.4
    scores = []
    for record in signature_records:
        for key in ("tau", "score", "similarity"):
            value = record.get(key)
            if value is None:
                continue
            try:
                scores.append(float(value))
            except (TypeError, ValueError):
                continue
    if not scores:
        return 0.5
    # Clamp tau-style scores [-1,1] -> [0,1]
    normalized_scores = [(score + 1) / 2 for score in scores]
    return max(0.1, min(mean(normalized_scores), 1.0))


def _clinical_signal(trials: list[dict]) -> float:
    if not trials:
        return 0.1
    phases = {"Phase 1": 0.3, "Phase 2": 0.6, "Phase 3": 0.9, "Phase 4": 1.0}
    statuses = {"Recruiting": 0.7, "Active, not recruiting": 0.6, "Completed": 1.0}
    best = 0.0
    for trial in trials:
        phase = trial.get("phase") or ""
        status = trial.get("status") or ""
        phase_score = phases.get(phase, 0.3)
        status_score = statuses.get(status, 0.4)
        best = max(best, 0.5 * phase_score + 0.5 * status_score)
    return best


def _safety_penalty(adverse_events: list[dict], sider_records: Optional[Iterable[Dict]]) -> float:
    penalty = 0.0
    if adverse_events:
        total_reports = sum(event.get("count", 0) or 0 for event in adverse_events)
        if total_reports > 0:
            penalty += min(total_reports / 1000.0, 1.0)

    entries = list(sider_records or [])
    if entries:
        serious = sum(1 for entry in entries if str(entry.get("seriousness", "")).lower() == "serious")
        penalty += min(serious / 10.0, 0.5)
        penalty += min((len(entries) - serious) / 20.0, 0.3)

    return -min(penalty, 1.0)


def score_candidate(
    drug_name: str,
    *,
    activity: Optional[Dict] = None,
    target_count: int = 0,
    trials: Optional[list[dict]] = None,
    adverse_events: Optional[list[dict]] = None,
    translator_paths: Optional[Iterable[Dict]] = None,
    moa_annotations: Optional[Iterable[Dict]] = None,
    signature_records: Optional[Iterable[Dict]] = None,
    sider_records: Optional[Iterable[Dict]] = None,
    graph_metrics: Optional[GraphInsights] = None,
    weights: Optional[Mapping[str, float]] = None,
) -> ScoreBreakdown:
    trials = trials or []
    adverse_events = adverse_events or []

    if weights is None:
        weights = get_default_weights()
    else:
        try:
            weights = _normalize_weights(weights)
        except ValueError:
            weights = get_default_weights()
    mech_raw = _mechanism_fit(activity, moa_annotations)
    network_raw = _network_score(target_count, translator_paths, graph_metrics)
    signature_raw = _signature_score(signature_records)
    clinical_raw = _clinical_signal(trials)
    penalty_raw = _safety_penalty(adverse_events, sider_records)

    mech = mech_raw * weights["mechanism"]
    network = network_raw * weights["network"]
    signature = signature_raw * weights["signature"]
    clinical = clinical_raw * weights["clinical"]
    penalty = penalty_raw * weights["safety"]

    raw_score = mech + network + signature + clinical + penalty
    final = max(min(raw_score, 1.0), 0.0)

    evidence_signals = [
        mech_raw > 0.1,
        network_raw > 0.1,
        signature_raw > 0.2,
        clinical_raw > 0.2,
        penalty_raw < 0.0,
        bool(graph_metrics),
    ]
    signal_strength = sum(1 for flag in evidence_signals if flag)
    margin = max(0.05, 0.25 - 0.03 * signal_strength)
    confidence_low = max(0.0, final - margin)
    confidence_high = min(1.0, final + margin)

    return ScoreBreakdown(
        mechanism_fit=round(mech, 4),
        network_proximity=round(network, 4),
        signature_reversal=round(signature, 4),
        clinical_signal=round(clinical, 4),
        safety_penalty=round(penalty, 4),
        final_score=round(final, 4),
        confidence_low=round(confidence_low, 4),
        confidence_high=round(confidence_high, 4),
        mechanism_raw=round(mech_raw, 4),
        network_raw=round(network_raw, 4),
        signature_raw=round(signature_raw, 4),
        clinical_raw=round(clinical_raw, 4),
        safety_raw=round(penalty_raw, 4),
    )
