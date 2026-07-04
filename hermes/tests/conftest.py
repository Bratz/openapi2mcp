"""Shared test plumbing: the FakeLLM seam and the no-real-client guard.

CLAUDE.md rule 3: unit/integration tests never call the live Anthropic API.
The autouse guard makes constructing a real client a hard failure.
"""

from __future__ import annotations

import threading
import time

import pytest

from hermes.schemas.models import (
    AgentResponse,
    ConsolidationResponse,
    Evidence,
    Extensions,
    Suggestion,
    UsageRecord,
)


@pytest.fixture(autouse=True)
def _no_real_anthropic_client(monkeypatch):
    """Any real Anthropic() construction inside the test suites is a bug."""
    try:
        import anthropic
    except ImportError:  # pragma: no cover
        yield
        return

    def _forbidden(self, *args, **kwargs):
        raise RuntimeError(
            "unit/integration tests must not construct a real Anthropic client "
            "(CLAUDE.md rule 3); inject a FakeLLM or fake client instead"
        )

    for cls_name in ("Anthropic", "AsyncAnthropic", "AnthropicBedrock", "AnthropicVertex"):
        cls = getattr(anthropic, cls_name, None)
        if cls is not None:
            monkeypatch.setattr(cls, "__init__", _forbidden)
    yield


def detected(smell_id: str, *, severity: str = "medium", confidence: float = 0.9) -> AgentResponse:
    """Convenience: a well-formed positive AgentResponse for tests."""
    return AgentResponse(
        smell_detected=True,
        justification=[
            f"The endpoint exhibits the {smell_id} smell based on concrete evidence "
            "observed in the reduced representation of this operation."
        ],
        suggestions=[
            Suggestion(
                action_title=f"[{smell_id}] - Improve documentation",
                description="Rework the affected fields so an autonomous agent can rely on them.",
            )
        ],
        extensions=Extensions(
            severity=severity,  # type: ignore[arg-type]
            confidence=confidence,
            evidence=[Evidence(location="summary", excerpt="...")],
        ),
    )


def not_detected() -> AgentResponse:
    return AgentResponse(smell_detected=False)


class FakeLLM:
    """Programmable in-memory LLM implementing the hermes.llm.LLM protocol.

    responses: dict mapping (erd_marker, smell_id) -> AgentResponse, where
    erd_marker is any substring of the ERD (typically the operation_id); or a
    dict mapping smell_id -> AgentResponse applied to every endpoint.
    """

    def __init__(self, responses=None, *, fail_after: int | None = None, delay: float = 0.0):
        self.responses = responses or {}
        for key, value in self.responses.items():
            if not isinstance(value, AgentResponse):
                raise TypeError(f"FakeLLM response for {key!r} must be an AgentResponse, got {type(value)}")
            if not (isinstance(key, str) or (isinstance(key, tuple) and len(key) == 2)):
                raise TypeError(f"FakeLLM key must be smell_id or (erd_marker, smell_id), got {key!r}")
        self.fail_after = fail_after
        self.delay = delay
        self.calls: list[tuple[str, str]] = []  # (smell_id, erd marker or "?")
        self.consolidations = 0
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0

    def detect(self, smell_id: str, erd_yaml: str) -> tuple[AgentResponse, UsageRecord]:
        with self._lock:
            self.calls.append((smell_id, erd_yaml[:40]))
            if self.fail_after is not None and len(self.calls) > self.fail_after:
                raise RuntimeError("FakeLLM: fail_after threshold reached")
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay:
                time.sleep(self.delay)
            response = self._lookup(smell_id, erd_yaml)
        finally:
            with self._lock:
                self.in_flight -= 1
        usage = UsageRecord(model="fake-model", input_tokens=1000, output_tokens=200)
        return response, usage

    def consolidate(self, endpoint_summary: str, findings_json: str):
        with self._lock:
            self.consolidations += 1
        return (
            ConsolidationResponse(edits=[], verdict="needs adaptation", note="fake"),
            UsageRecord(model="fake-consolidator", input_tokens=500, output_tokens=100),
        )

    def _lookup(self, smell_id: str, erd_yaml: str) -> AgentResponse:
        if isinstance(self.responses.get(smell_id), AgentResponse):
            return self.responses[smell_id]
        for key, response in self.responses.items():
            if isinstance(key, tuple):
                marker, key_smell = key
                # Match the ERD's own operation, not sibling-path context lines.
                if key_smell == smell_id and f"operation_id: {marker}\n" in erd_yaml:
                    return response
        return not_detected()
