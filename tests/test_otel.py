"""OpenTelemetry exporter with a fake OpenTelemetry API."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from ragpick.errors import MissingDependencyError
from ragpick.otel import OTelExporter, otel_value
from ragpick.registry import registry
from ragpick.tracing import Span


class FakeSpan:
    def __init__(
        self, name: str, context: Any, start_time: int, attributes: dict[str, Any]
    ) -> None:
        self.name = name
        self.context = context
        self.start_time = start_time
        self.attributes = attributes
        self.end_time: int | None = None
        self.status: Any = None

    def set_status(self, status: Any) -> None:
        self.status = status

    def end(self, end_time: int) -> None:
        self.end_time = end_time


class FakeTracer:
    def __init__(self) -> None:
        self.spans: list[FakeSpan] = []

    def start_span(
        self, name: str, context: Any = None, start_time: int = 0, attributes: Any = None
    ) -> FakeSpan:
        span = FakeSpan(name, context, start_time, attributes or {})
        self.spans.append(span)
        return span


def fake_trace_api(tracer: FakeTracer) -> Any:
    return SimpleNamespace(
        get_tracer=lambda name: tracer,
        set_span_in_context=lambda span: ("ctx", span.name),
        Status=lambda code, msg: (code, msg),
        StatusCode=SimpleNamespace(ERROR="ERROR"),
    )


ROOT = Span(
    name="query",
    timestamp=1_700_000_000.0,
    duration_ms=10.0,
    attributes={"groups": 1, "obj": {"a": 1}},
    children=[
        Span(
            name="retrieve",
            start_ms=1.0,
            duration_ms=4.0,
            attributes={"tags": ["a", "b"], "nums": [1, 2.5], "mixed": [1, "a"], "ok": True},
            children=[Span(name="grade", start_ms=2.0, duration_ms=1.0, error="ValueError: x")],
        )
    ],
)


def test_otel_value() -> None:
    assert otel_value("s") == "s"
    assert otel_value(3) == 3
    assert otel_value(["a"]) == ["a"]
    assert otel_value((1, 2.0)) == [1, 2.0]
    assert otel_value([True, 1]) == "[True, 1]"
    assert otel_value({"a": 1}) == "{'a': 1}"


def test_export_replays_spans_with_times_parents_and_errors() -> None:
    tracer = FakeTracer()
    exporter = OTelExporter(trace_api=fake_trace_api(tracer))
    exporter.export(ROOT)
    root, retrieve, grade = tracer.spans
    assert [s.name for s in tracer.spans] == ["ragpick.query", "ragpick.retrieve", "ragpick.grade"]
    origin = 1_700_000_000 * 10**9
    assert root.start_time == origin
    assert root.end_time == origin + 10 * 10**6
    assert retrieve.start_time == origin + 1 * 10**6
    assert retrieve.end_time == origin + 5 * 10**6
    assert root.context is None
    assert retrieve.context == ("ctx", "ragpick.query")
    assert grade.context == ("ctx", "ragpick.retrieve")
    assert root.attributes == {"ragpick.groups": 1, "ragpick.obj": "{'a': 1}"}
    assert retrieve.attributes["ragpick.tags"] == ["a", "b"]
    assert retrieve.attributes["ragpick.mixed"] == "[1, 'a']"
    assert grade.status == ("ERROR", "ValueError: x")
    assert root.status is None


def test_missing_opentelemetry_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "opentelemetry", None)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", None)
    with pytest.raises(MissingDependencyError, match="pip install opentelemetry-api"):
        OTelExporter()


def test_endpoint_sets_up_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, Any] = {}
    tracer = FakeTracer()

    class Provider:
        def __init__(self, resource: Any) -> None:
            created["resource"] = resource

        def add_span_processor(self, processor: Any) -> None:
            created["processor"] = processor

        def get_tracer(self, name: str) -> FakeTracer:
            return tracer

    def module(name: str, **attrs: Any) -> ModuleType:
        mod = ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        return mod

    modules = {
        "opentelemetry.trace": fake_trace_api(FakeTracer()),
        "opentelemetry.sdk.trace": module("opentelemetry.sdk.trace", TracerProvider=Provider),
        "opentelemetry.sdk.trace.export": module(
            "opentelemetry.sdk.trace.export", BatchSpanProcessor=lambda exp: ("batch", exp)
        ),
        "opentelemetry.sdk.resources": module(
            "opentelemetry.sdk.resources", Resource=SimpleNamespace(create=lambda attrs: attrs)
        ),
        "opentelemetry.exporter.otlp.proto.http.trace_exporter": module(
            "otlp", OTLPSpanExporter=lambda endpoint, headers: ("otlp", endpoint, headers)
        ),
    }
    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)
    exporter = OTelExporter(
        service_name="bot", endpoint="http://collector:4318/v1/traces", headers={"x": "1"}
    )
    assert created["resource"] == {"service.name": "bot"}
    assert created["processor"] == (
        "batch",
        ("otlp", "http://collector:4318/v1/traces", {"x": "1"}),
    )
    exporter.export(ROOT)
    assert len(tracer.spans) == 3
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.trace", None)
    with pytest.raises(MissingDependencyError, match="opentelemetry-sdk"):
        OTelExporter(endpoint="http://x")


def test_registered() -> None:
    assert registry.has("trace_exporter", "otel")
