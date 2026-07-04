"""Report rendering (docs/00-SPEC §7): the self-contained HTML dashboard and
the paper's Appendix-C per-endpoint markdown diagnostic report."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from jinja2 import Template
from markupsafe import Markup

from hermes.schemas.models import EndpointVerdict, Finding
from hermes.smells.catalog import SMELLS
from hermes.store import RunStore

_TEMPLATE_PATH = Path(__file__).parent / "template.html"


def render_run(store: RunStore) -> str:
    meta = store.load_run_meta()
    findings = store.load_findings()
    verdicts = store.load_verdicts()
    counts = meta.get("counts", {})
    config = meta.get("config", {})
    n_ops = counts.get("operations_scanned", 0) or 0

    smell_counter = Counter(f.smell for f in findings)
    smell_rollup = [
        {
            "name": SMELLS[sid].display_name,
            "count": smell_counter.get(sid, 0),
            "pct": round(100 * smell_counter.get(sid, 0) / n_ops, 1) if n_ops else 0,
        }
        for sid in SMELLS
    ]

    by_tag: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        for tag in f.endpoint.tags or ["(untagged)"]:
            by_tag[tag].append(f)
    tag_rollup = [
        {
            "tag": tag,
            "endpoints": len({f"{f.endpoint.method} {f.endpoint.path}" for f in group}),
            "count": len(group),
            "top": Counter(f.smell for f in group).most_common(1)[0][0],
        }
        for tag, group in sorted(by_tag.items())
    ]

    by_endpoint: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        by_endpoint[f"{f.endpoint.method} {f.endpoint.path}"].append(f)
    verdict_by_key = {v.endpoint_key: v for v in verdicts}
    worst = sorted(by_endpoint.items(), key=lambda kv: -len(kv[1]))[:10]
    worst_rows = [
        {
            "key": key,
            "smells": ", ".join(sorted({f.smell for f in group})),
            "verdict": getattr(verdict_by_key.get(key), "verdict", "?"),
        }
        for key, group in worst
    ]

    severity_rollup = Counter(f.extensions.severity for f in findings)
    template = Template(_TEMPLATE_PATH.read_text(encoding="utf-8"), autoescape=True)
    return template.render(
        api_title=config.get("spec_title", ""),
        run_id=store.run_id,
        generated_at=meta.get("generated_at", "?"),
        spec_version=config.get("spec_version", "?"),
        severity_rollup={"high": severity_rollup.get("high", 0),
                         "medium": severity_rollup.get("medium", 0),
                         "low": severity_rollup.get("low", 0)},
        generated_note=f"{n_ops} operations",
        detect_model=config.get("detect_model", "?"),
        consolidate_model=config.get("consolidate_model", "?"),
        consolidate=config.get("consolidate", False),
        counts={
            "operations_scanned": n_ops,
            "detections": counts.get("detections", len(findings)),
            "avg_smells_per_endpoint": counts.get("avg_smells_per_endpoint", 0),
            "endpoints_with_detections": counts.get("endpoints_with_detections", len(by_endpoint)),
            "detector_errors": counts.get("detector_errors", 0),
        },
        smell_rollup=smell_rollup,
        tag_rollup=tag_rollup,
        worst=worst_rows,
        # "</" must never appear literally inside a script block (XSS breakout).
        findings_json=_embed([f.model_dump() for f in findings]),
        verdicts_json=_embed([v.model_dump() for v in verdicts]),
        meta_json=_embed({
            "api_title": config.get("spec_title", ""),
            "detect_model": config.get("detect_model", "?"),
            "smell_display": {sid: s.display_name for sid, s in SMELLS.items()},
        }),
    )


def render_endpoint_md(store: RunStore, endpoint_key: str) -> str:
    """The paper's Appendix-C diagnostic report for one endpoint (00-SPEC §7.1)."""
    meta = store.load_run_meta()
    config = meta.get("config", {})
    findings = [f for f in store.load_findings() if f"{f.endpoint.method} {f.endpoint.path}" == endpoint_key]
    verdict: EndpointVerdict | None = next(
        (v for v in store.load_verdicts() if v.endpoint_key == endpoint_key), None
    )
    if verdict is None and not findings:
        known = sorted({v.endpoint_key for v in store.load_verdicts()})
        hint = ", ".join(known[:5]) + ("…" if len(known) > 5 else "")
        raise KeyError(f"endpoint {endpoint_key!r} not in this run (methods are upper-case; known: {hint})")
    method, _, path = endpoint_key.partition(" ")
    lines = [
        "## API Info",
        f"- Title: \"{config.get('spec_title', '')}\"",
        "## Endpoint Info",
        f"- Method: \"{method}\"",
        f"- Path: \"{path}\"",
        "## Model",
        f"- {config.get('detect_model', '?')}",
        "## Identified Smells",
        ", ".join(SMELLS[f.smell].display_name for f in findings) if findings else "None",
        "### Explanations",
    ]
    for f in findings:
        lines.append(f"### {SMELLS[f.smell].display_name}")
        lines.extend(f"- {j}" for j in f.justification)
    lines.append("## Improvement Suggestions")
    for f in findings:
        for s in f.suggestions:
            lines.append(f"{s.action_title} | {s.description}")
    if verdict is not None:
        lines.append("")
        lines.append(f"Verdict: {verdict.verdict} ({verdict.note})")
    return "\n".join(lines) + "\n"


def _embed(payload) -> Markup:
    # Markup: autoescape must not HTML-entity the JSON inside the script block;
    # the "</" replacement is the actual safety measure (script-breakout XSS).
    return Markup(json.dumps(payload).replace("</", "<\\/"))
