"""Pure helpers used by graph nodes: verdict rules and consolidation-edit
application (docs/00-SPEC §8, DECISIONS M4/M5)."""

from __future__ import annotations

from hermes.schemas.models import ConsolidationResponse, EndpointVerdict, Finding

_SEVERITY_ORDER = ["low", "medium", "high"]


def rule_based_verdict(endpoint_key: str, operation_id: str | None, findings: list[Finding]) -> EndpointVerdict:
    """Deterministic fallback verdict (used for 0-1 findings, --no-consolidate,
    and consolidator failures): 0 smells → agent-ready; any high-severity or
    ≥3 smells → not agent-consumable; else needs adaptation."""
    smells = sorted({f.smell for f in findings})
    if not findings:
        verdict = "agent-ready"
    elif any(f.extensions.severity == "high" for f in findings) or len(smells) >= 3:
        verdict = "not agent-consumable"
    else:
        verdict = "needs adaptation"
    return EndpointVerdict(
        endpoint_key=endpoint_key,
        operation_id=operation_id,
        smells=smells,  # type: ignore[arg-type]
        verdict=verdict,  # type: ignore[arg-type]
        note="rule-based verdict",
    )


def apply_consolidation(
    endpoint_key: str,
    operation_id: str | None,
    findings: list[Finding],
    response: ConsolidationResponse,
) -> tuple[list[Finding], EndpointVerdict]:
    """Apply consolidator edits in code (00-SPEC §8): keep, drop exact duplicates,
    adjust severity CLAMPED to one step from the stored value. Unknown finding
    ids in edits are ignored (the consolidator may not invent)."""
    by_id = {f.id: f for f in findings}
    original_severity = {f.id: f.extensions.severity for f in findings}
    adjusted: set[str] = set()
    dropped: set[str] = set()
    for edit in response.edits:
        finding = by_id.get(edit.finding_id)
        if finding is None:
            continue  # invented/unknown id — ignore per the never-invent rule
        if edit.action == "drop_duplicate":
            dropped.add(finding.id)
        elif edit.action == "adjust_severity" and edit.new_severity:
            if finding.id in adjusted:
                continue  # repeat edits must not compound past the one-step clamp
            adjusted.add(finding.id)
            current = _SEVERITY_ORDER.index(original_severity[finding.id])
            target = _SEVERITY_ORDER.index(edit.new_severity)
            clamped = max(current - 1, min(current + 1, target))
            finding.extensions.severity = _SEVERITY_ORDER[clamped]  # type: ignore[assignment]
    kept = [f for f in findings if f.id not in dropped]
    smells = sorted({f.smell for f in kept})
    return kept, EndpointVerdict(
        endpoint_key=endpoint_key,
        operation_id=operation_id,
        smells=smells,  # type: ignore[arg-type]
        verdict=response.verdict,
        note=response.note or "consolidated",
    )
