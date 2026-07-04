"""Report rendering. M5 ships a minimal-but-valid placeholder so `hermes scan`
always writes a usable report.html; the full dashboard (00-SPEC §7) lands in M6."""

from __future__ import annotations

import html
import json

from hermes.store import RunStore


def render_run(store: RunStore) -> str:
    meta = store.load_run_meta()
    findings = store.load_findings()
    counts = meta.get("counts", {})
    rows = "\n".join(
        f"<tr><td>{html.escape(f.endpoint.method)}</td><td>{html.escape(f.endpoint.path)}</td>"
        f"<td>{html.escape(f.smell)}</td><td>{html.escape(f.extensions.severity)}</td>"
        f"<td>{html.escape(f.justification[0] if f.justification else '')}</td></tr>"
        for f in findings
    )
    # "</" must not appear literally inside the script block: a spec/finding
    # containing "</script>" would otherwise break out into HTML (stored XSS).
    data = json.dumps([f.model_dump() for f in findings]).replace("</", "<\\/")
    return f"""<meta charset="utf-8">
<title>Hermes scan {html.escape(store.run_id)}</title>
<style>body{{font-family:sans-serif;margin:2rem}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left;font-size:14px}}</style>
<h1>Hermes scan — {html.escape(str(meta.get('config', {}).get('spec_title', '')))}</h1>
<p>run {html.escape(store.run_id)} · {counts.get('operations_scanned', 0)} operations ·
{counts.get('detections', 0)} detections · avg {counts.get('avg_smells_per_endpoint', 0)} smells/endpoint ·
{counts.get('detector_errors', 0)} errors</p>
<p><em>Placeholder report (M5) — full dashboard lands in M6.</em></p>
<table><tr><th>Method</th><th>Path</th><th>Smell</th><th>Severity</th><th>Justification</th></tr>
{rows}</table>
<script type="application/json" id="hermes-findings">{data}</script>
"""
