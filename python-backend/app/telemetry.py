from __future__ import annotations

import logging
import os
from typing import Dict, Optional

from fastapi import FastAPI
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)


def _is_enabled() -> bool:
    return os.getenv("ENABLE_OTEL", "false").lower() in {"1", "true", "yes"}


def _exporter_headers() -> Optional[Dict[str, str]]:
    header_env = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
    if not header_env:
        return None
    headers: Dict[str, str] = {}
    for part in header_env.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        headers[key.strip()] = value.strip()
    return headers


def setup_telemetry(app: FastAPI) -> None:
    if not _is_enabled():
        return

    if getattr(app.state, "otel_instrumented", False):
        return

    service_name = os.getenv("OTEL_SERVICE_NAME", "drug-repurposing-backend")
    resource = Resource.create({"service.name": service_name})

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    if not endpoint:
        base_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4318")
        endpoint = f"{base_endpoint.rstrip('/')}/v1/traces"

    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)

    span_exporter = OTLPSpanExporter(
        endpoint=endpoint,
        headers=_exporter_headers(),
        timeout=float(os.getenv("OTEL_EXPORTER_TIMEOUT", "10")),
    )

    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))

    metrics_endpoint = os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")
    if not metrics_endpoint:
        base_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4318")
        metrics_endpoint = f"{base_endpoint.rstrip('/')}/v1/metrics"

    metric_exporter = OTLPMetricExporter(
        endpoint=metrics_endpoint,
        headers=_exporter_headers(),
        timeout=float(os.getenv("OTEL_EXPORTER_TIMEOUT", "10")),
    )

    metric_reader = PeriodicExportingMetricReader(metric_exporter)
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    FastAPIInstrumentor.instrument_app(app)

    try:
        HTTPXClientInstrumentor().instrument()
    except Exception as exc:  # noqa: BLE001
        logger.debug("httpx instrumentation skipped: %s", exc)

    try:
        LoggingInstrumentor().instrument(set_logging_format=False)
    except Exception as exc:  # noqa: BLE001
        logger.debug("logging instrumentation skipped: %s", exc)

    app.state.otel_instrumented = True
