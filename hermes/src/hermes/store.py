"""Run persistence (docs/00-SPEC §6): runs/<run_id>/{findings.jsonl,
findings.raw.jsonl, endpoints.jsonl, run.json}. Appends are idempotent by id."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from hermes.schemas.models import EndpointVerdict, Finding, UsageRecord


class RunStore:
    def __init__(self, out_dir: Path | str, run_id: str):
        self.run_id = run_id
        self.dir = Path(out_dir) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.findings_path = self.dir / "findings.jsonl"
        self.raw_path = self.dir / "findings.raw.jsonl"
        self.endpoints_path = self.dir / "endpoints.jsonl"
        self.run_json_path = self.dir / "run.json"

    # -- writes ---------------------------------------------------------------
    # A run's artifacts are rewritten wholesale when it completes: re-running
    # the same run-id must leave findings, raw records, and verdicts CONSISTENT
    # with each other (per-id append would keep first-run finding bodies next to
    # second-run verdicts — see DECISIONS M5).

    def write_findings(self, findings: list[Finding]) -> int:
        seen: set[str] = set()
        with self.findings_path.open("w", encoding="utf-8") as fh:
            for finding in findings:
                if finding.id in seen:  # one record per (endpoint, smell) by construction
                    continue
                fh.write(finding.model_dump_json() + "\n")
                seen.add(finding.id)
        return len(seen)

    def write_raw(self, records: list[dict]) -> None:
        with self.raw_path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, sort_keys=True) + "\n")

    def write_verdicts(self, verdicts: list[EndpointVerdict]) -> None:
        # Verdicts are recomputed whole per run — overwrite, keyed file.
        with self.endpoints_path.open("w", encoding="utf-8") as fh:
            for v in verdicts:
                fh.write(v.model_dump_json() + "\n")

    def write_run_meta(
        self,
        *,
        config: dict,
        counts: dict,
        usage: list[UsageRecord],
        errors: list[dict],
        status: str,
    ) -> dict:
        rollup = {
            "calls": len(usage),
            "cached_replays": sum(1 for u in usage if u.cached),
            "fresh_calls": sum(1 for u in usage if not u.cached),
            "input_tokens": sum(u.input_tokens for u in usage),
            "output_tokens": sum(u.output_tokens for u in usage),
            "cache_read_input_tokens": sum(u.cache_read_input_tokens for u in usage),
            "cache_creation_input_tokens": sum(u.cache_creation_input_tokens for u in usage),
            "cost_usd": round(sum(u.cost_usd for u in usage), 4),
            "cost_includes_estimates": any(u.estimated for u in usage),
        }
        meta = {
            "run_id": self.run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": status,
            "config": config,
            "counts": counts,
            "usage": rollup,
            "detector_errors": errors,
        }
        self.run_json_path.write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
        return meta

    # -- reads (report/eval consumers) -----------------------------------------

    def load_findings(self) -> list[Finding]:
        return [Finding.model_validate(f) for f in self._read_jsonl(self.findings_path)]

    def load_verdicts(self) -> list[EndpointVerdict]:
        return [EndpointVerdict.model_validate(v) for v in self._read_jsonl(self.endpoints_path)]

    def load_run_meta(self) -> dict:
        if not self.run_json_path.exists():
            return {}
        return json.loads(self.run_json_path.read_text(encoding="utf-8"))

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out
