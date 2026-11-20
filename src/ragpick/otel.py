"""OpenTelemetry trace exporter (``trace_exporter`` type ``otel``).

Replays each finished ragpick trace as OpenTelemetry spans, with the original start
and end times, attributes (prefixed ``ragpick.``) and error status. ragpick doesn't
depend on OpenTelemetry; install it yourself:

* ``pip install opentelemetry-api``: if your application already configures an
  OpenTelemetry SDK, spans flow into it (the API is a no-op otherwise);
* ``pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http`` and set
  ``endpoint``: ragpick sets up an OTLP/HTTP exporter itself (useful for the CLI and
  ``ragpick serve``).

::

    tracing:
      enabled: true
      exporters:
        - type: otel
          params: {service_name: support-bot, endpoint: "http://localhost:4318/v1/traces"}
"""

from __future__ import annotations

from typing import Any

from ragpick.components._optional import require
from ragpick.registry import register
from ragpick.tracing import Span

_SCALARS = (str, bool, int, float)


def otel_value(value: Any) -> Any:
    """Coerce a span attribute into a type OpenTelemetry accepts."""
    if isinstance(value, _SCALARS):
        return value
    if isinstance(value, list | tuple) and all(isinstance(v, str) for v in value):
        return list(value)
    if isinstance(value, list | tuple) and all(
        isinstance(v, int | float) and not isinstance(v, bool) for v in value
    ):
        return list(value)
    return str(value)


@register("trace_exporter", "otel")
class OTelExporter:
    """Exports ragpick traces through the OpenTelemetry API."""

    def __init__(
        self,
        service_name: str = "ragpick",
        endpoint: str | None = None,
        headers: dict[str, str] | None = None,
        tracer: Any | None = None,
        trace_api: Any | None = None,
    ) -> None:
        self._trace = trace_api or require(
            "opentelemetry.trace", "otel", "OTelExporter", "pip install opentelemetry-api"
        )
        if tracer is None and endpoint is not None:
            tracer = self._sdk_tracer(service_name, endpoint, headers or {})
        self._tracer = tracer or self._trace.get_tracer("ragpick")

    @staticmethod
    def _sdk_tracer(service_name: str, endpoint: str, headers: dict[str, str]) -> Any:
        what = "OTelExporter(endpoint=...)"
        hint = "pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http"
        sdk_trace = require("opentelemetry.sdk.trace", "otel", what, hint)
        export = require("opentelemetry.sdk.trace.export", "otel", what, hint)
        resources = require("opentelemetry.sdk.resources", "otel", what, hint)
        otlp = require("opentelemetry.exporter.otlp.proto.http.trace_exporter", "otel", what, hint)
        provider = sdk_trace.TracerProvider(
            resource=resources.Resource.create({"service.name": service_name})
        )
        provider.add_span_processor(
            export.BatchSpanProcessor(otlp.OTLPSpanExporter(endpoint=endpoint, headers=headers))
        )
        return provider.get_tracer("ragpick")

    def export(self, root: Span) -> None:
        """Emit ``root`` and its descendants as OpenTelemetry spans."""
        origin_ns = int((root.timestamp or 0.0) * 1e9)
        self._emit(root, origin_ns, None)

    def _emit(self, s: Span, origin_ns: int, parent: Any | None) -> None:
        start = origin_ns + int(s.start_ms * 1e6)
        end = start + int(s.duration_ms * 1e6)
        context = self._trace.set_span_in_context(parent) if parent is not None else None
        attributes = {f"ragpick.{k}": otel_value(v) for k, v in s.attributes.items()}
        otel_span = self._tracer.start_span(
            f"ragpick.{s.name}", context=context, start_time=start, attributes=attributes
        )
        if s.error:
            otel_span.set_status(self._trace.Status(self._trace.StatusCode.ERROR, s.error))
        for child in s.children:
            self._emit(child, origin_ns, otel_span)
        otel_span.end(end_time=end)
