"""Pydantic models: agent response contract + canonical Finding (docs/00-SPEC.md §4)."""

from __future__ import annotations

import hashlib
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SmellId = Literal[
    "LAZY", "BLOATED", "TANGLED", "FRAGMENTED", "EXCESS_STRUCTURED",
    "PATH_AND_METHOD", "INPUT", "RESPONSE", "SECURITY",
]

VERDICTS = ("agent-ready", "needs adaptation", "not agent-consumable")


class Evidence(BaseModel):
    location: str = Field(description="Dot-path relative to the operation, e.g. 'responses.200.description'")
    excerpt: str


class Suggestion(BaseModel):
    action_title: str = Field(description="Exact format: '[<SMELL>] - <short action title>'")
    description: str


class Extensions(BaseModel):
    """Dashboard-only fields — never used in eval scoring (DECISIONS.md)."""

    severity: Literal["low", "medium", "high"]
    confidence: float
    evidence: list[Evidence]


class AgentResponse(BaseModel):
    """What a smell agent returns for one (endpoint, smell) — the parse() target."""

    model_config = ConfigDict(extra="forbid")

    smell_detected: bool
    justification: list[str] = Field(default_factory=list)
    suggestions: list[Suggestion] = Field(default_factory=list)
    extensions: Extensions | None = None

    @model_validator(mode="after")
    def _detected_consistency(self) -> "AgentResponse":
        if self.smell_detected:
            if not self.justification:
                raise ValueError("smell_detected=true requires at least one justification bullet")
            if not self.suggestions:
                raise ValueError("smell_detected=true requires at least one suggestion")
            if self.extensions is None:
                raise ValueError("smell_detected=true requires the extensions block")
        else:
            if self.justification or self.suggestions or self.extensions is not None:
                raise ValueError("smell_detected=false must carry no justification/suggestions/extensions")
        return self


class EndpointInfo(BaseModel):
    path: str
    method: str
    operation_id: str | None
    tags: list[str] = Field(default_factory=list)


class DetectorInfo(BaseModel):
    model: str
    prompt_version: str
    truncation_applied: bool = False
    short_justification: bool = False


class Finding(BaseModel):
    """Canonical detection record (00-SPEC §4): paper core + extensions."""

    id: str
    run_id: str
    api_title: str
    endpoint: EndpointInfo
    smell: SmellId
    category: Literal["documentation", "rest"]
    justification: list[str]
    suggestions: list[Suggestion]
    extensions: Extensions
    detector: DetectorInfo


class EndpointVerdict(BaseModel):
    endpoint_key: str
    operation_id: str | None
    smells: list[SmellId]
    verdict: Literal["agent-ready", "needs adaptation", "not agent-consumable"]
    note: str = ""


class UsageRecord(BaseModel):
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False  # served from the hermes response cache — zero API cost
    estimated: bool = False  # includes estimated spend from failed attempts (SDK discards their usage)


class ConsolidationEdit(BaseModel):
    """One consolidator edit.

    APPLIER CONTRACT (M5): severity adjustments must be clamped to at most ONE
    step from the finding's stored severity (00-SPEC §8) — this model cannot
    see the original severity, so the edit-applier enforces the clamp.
    """

    finding_id: str
    action: Literal["keep", "drop_duplicate", "adjust_severity"]
    reason: str
    new_severity: Literal["low", "medium", "high"] | None = None

    @model_validator(mode="after")
    def _severity_iff_adjust(self) -> "ConsolidationEdit":
        if self.action == "adjust_severity" and self.new_severity is None:
            raise ValueError("adjust_severity requires new_severity")
        if self.action != "adjust_severity" and self.new_severity is not None:
            raise ValueError("new_severity only allowed with action=adjust_severity")
        return self


class ConsolidationResponse(BaseModel):
    """Consolidator output: may only keep/drop/adjust — never invent (00-SPEC §8)."""

    edits: list[ConsolidationEdit]
    verdict: Literal["agent-ready", "needs adaptation", "not agent-consumable"]
    note: str = ""


def finding_id(endpoint_key: str, smell_id: str) -> str:
    """Deterministic: one record max per (operation, smell); re-runs dedup naturally."""
    return "f_" + hashlib.sha1(f"{endpoint_key}|{smell_id}".encode()).hexdigest()[:16]


def build_finding(
    *,
    response: AgentResponse,
    run_id: str,
    api_title: str,
    endpoint: EndpointInfo,
    smell_id: str,
    category: str,
    model: str,
    prompt_version: str,
    truncation_applied: bool = False,
) -> Finding | None:
    """Post-validate an AgentResponse into a Finding (code-side rules, 00-SPEC §4).

    Returns None when the smell was not detected. Clamps confidence, repairs the
    action-title prefix, drops evidence-free extensions entries, flags short
    justifications.
    """
    if not response.smell_detected:
        return None
    ext = response.extensions
    assert ext is not None  # guaranteed by AgentResponse validator
    clamped = Extensions(
        severity=ext.severity,
        confidence=min(1.0, max(0.0, ext.confidence)),
        evidence=[e for e in ext.evidence if e.location.strip()],
    )
    prefix = f"[{smell_id}] - "
    suggestions = []
    for s in response.suggestions:
        if s.action_title.startswith(prefix):
            suggestions.append(s)
        else:
            # Strip any leading "[X] -"-style prefix (only at the start), keep the rest intact.
            body = re.sub(r"^\[[^\]]*\]\s*-?\s*", "", s.action_title).strip() or s.action_title
            suggestions.append(Suggestion(action_title=prefix + body, description=s.description))
    short = sum(len(j) for j in response.justification) < 120
    ep_key = f"{endpoint.method} {endpoint.path}"
    return Finding(
        id=finding_id(ep_key, smell_id),
        run_id=run_id,
        api_title=api_title,
        endpoint=endpoint,
        smell=smell_id,  # type: ignore[arg-type]
        category=category,  # type: ignore[arg-type]
        justification=response.justification,
        suggestions=suggestions,
        extensions=clamped,
        detector=DetectorInfo(
            model=model,
            prompt_version=prompt_version,
            truncation_applied=truncation_applied,
            short_justification=short,
        ),
    )
