"""LangGraph scan orchestration (docs/01-ARCHITECTURE §3).

load_spec → reduce → plan ─(Send fan-out)→ detect → collect/consolidate → persist

Non-serializable collaborators (llm, cache, store, config) are bound into node
closures; graph state carries only data. Resume is cache-driven (DECISIONS M2/M5):
no LangGraph checkpointer — re-running a run_id replays completed pairs from the
response cache at zero API cost.
"""

from __future__ import annotations

import json
import operator
import threading
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from hermes.cache import ResponseCache, cache_key
from hermes.config import HermesConfig
from hermes.llm import LLM, DetectorFailure
from hermes.nodes import apply_consolidation, rule_based_verdict
from hermes.reducer import reduce_operation
from hermes.schemas.models import ConsolidationResponse, EndpointInfo, UsageRecord, build_finding
from hermes.smells.catalog import SMELLS
from hermes.smells.prompts import PROMPT_VERSIONS
from hermes.spec_loader import LoadedSpec, OperationRef, filter_operations
from hermes.store import RunStore


class BudgetExceeded(Exception):
    """Rolled-up cost passed HERMES_MAX_COST_USD; scan stops resumable (exit 3)."""


class ScanState(TypedDict, total=False):
    run_id: str
    api_title: str
    operations: list[dict]  # OperationRef dumps: path/method/operation_id/tags
    erds: dict[str, dict]  # endpoint_key -> {yaml, truncation_applied}
    tasks: list[dict]  # cache misses to detect: {endpoint_key, smell_id}
    findings: Annotated[list, operator.add]
    raw: Annotated[list, operator.add]
    usage: Annotated[list, operator.add]
    errors: Annotated[list, operator.add]
    verdicts: Annotated[list, operator.add]
    final_findings: list  # post-consolidation set; written once by collect


class _CostMeter:
    def __init__(self, limit: float | None):
        self._lock = threading.Lock()
        self._limit = limit
        self.total = 0.0

    def add(self, cost: float) -> None:
        with self._lock:
            self.total += cost
            if self._limit is not None and self.total > self._limit:
                raise BudgetExceeded(f"cost ${self.total:.2f} exceeded HERMES_MAX_COST_USD=${self._limit:.2f}")


def build_scan_graph(
    *,
    spec: LoadedSpec,
    operations: list[OperationRef],
    llm: LLM,
    cache: ResponseCache,
    config: HermesConfig,
    consolidate_enabled: bool = True,
):
    """Compile the scan graph. `operations` are pre-filtered (tags/sample/...)."""
    ops_by_key = {op.endpoint_key: op for op in operations}
    meter = _CostMeter(config.max_cost_usd)

    def node_reduce(state: ScanState) -> dict:
        erds = {}
        for op in operations:
            erd = reduce_operation(spec, op)
            erds[op.endpoint_key] = {"yaml": erd.yaml, "truncation_applied": erd.truncation_applied}
        return {
            "api_title": spec.title,
            "operations": [
                {"path": o.path, "method": o.method, "operation_id": o.operation_id, "tags": list(o.tags)}
                for o in operations
            ],
            "erds": erds,
        }

    def node_plan(state: ScanState) -> dict:
        """Split (endpoint × smell) into cache hits (immediate findings) and tasks."""
        tasks, findings, raw, usage = [], [], [], []
        for key, erd in state["erds"].items():
            op = ops_by_key[key]
            for smell_id in SMELLS:
                ck = cache_key(erd["yaml"], smell_id, PROMPT_VERSIONS[smell_id], config.detect_model)
                cached = cache.get(ck)
                if cached is None:
                    tasks.append({"endpoint_key": key, "smell_id": smell_id})
                    continue
                usage.append(UsageRecord(model=config.detect_model, cached=True))
                raw.append(_raw_record(state["run_id"], key, smell_id, cached, cached_replay=True))
                finding = _to_finding(state, op, smell_id, cached, erd["truncation_applied"], config)
                if finding is not None:
                    findings.append(finding)
        return {"tasks": tasks, "findings": findings, "raw": raw, "usage": usage}

    def fan_out(state: ScanState):
        if not state.get("tasks"):
            return "collect"
        return [
            Send("detect", {"task": t, "erd": state["erds"][t["endpoint_key"]], "run_id": state["run_id"],
                            "api_title": state["api_title"]})
            for t in state["tasks"]
        ]

    def node_detect(payload: dict) -> dict:
        task, erd = payload["task"], payload["erd"]
        key, smell_id = task["endpoint_key"], task["smell_id"]
        op = ops_by_key[key]
        try:
            response, usage = llm.detect(smell_id, erd["yaml"])
        except DetectorFailure as exc:
            error = {"endpoint_key": key, "smell_id": smell_id, "error": str(exc)}
            out: dict = {"errors": [error]}
            if exc.usage is not None:
                out["usage"] = [exc.usage]
                meter.add(exc.usage.cost_usd)
            return out
        # Cache BEFORE the budget check: a paid response must never be discarded
        # (resume would have to buy it again — see M5 review).
        ck = cache_key(erd["yaml"], smell_id, PROMPT_VERSIONS[smell_id], config.detect_model)
        cache.put(ck, response, config.detect_model)
        meter.add(usage.cost_usd)
        out = {
            "usage": [usage],
            "raw": [_raw_record(payload["run_id"], key, smell_id, response, cached_replay=False)],
        }
        finding = _to_finding_from(payload["run_id"], payload["api_title"], op, smell_id, response,
                                   erd["truncation_applied"], config)
        if finding is not None:
            out["findings"] = [finding]
        return out

    def node_collect(state: ScanState) -> dict:
        """Group findings per endpoint; consolidate (LLM) or rule-verdict."""
        by_endpoint: dict[str, list] = {op.endpoint_key: [] for op in operations}
        for finding in state.get("findings", []):
            by_endpoint[f"{finding.endpoint.method} {finding.endpoint.path}"].append(finding)
        verdicts, extra_usage, errors = [], [], []
        consolidated_findings: list = []
        for key, group in by_endpoint.items():
            op = ops_by_key[key]
            if len(group) >= 2 and consolidate_enabled:
                summary = f"{key} (operation_id={op.operation_id}, tags={list(op.tags)})"
                payload = json.dumps(
                    [{"id": f.id, "smell": f.smell, "severity": f.extensions.severity,
                      "justification": f.justification} for f in group],
                    indent=1,
                )
                ck = cache_key(payload, "_consolidator", "consolidator-v1", config.consolidate_model)
                cached = cache.get_as(ck, ConsolidationResponse)
                try:
                    if cached is not None:
                        response = cached
                        extra_usage.append(UsageRecord(model=config.consolidate_model, cached=True))
                    else:
                        response, usage = llm.consolidate(summary, payload)
                        cache.put(ck, response, config.consolidate_model)  # before budget check
                        meter.add(usage.cost_usd)
                        extra_usage.append(usage)
                    kept, verdict = apply_consolidation(key, op.operation_id, group, response)
                    consolidated_findings.extend(kept)
                    verdicts.append(verdict)
                    continue
                except DetectorFailure as exc:
                    errors.append({"endpoint_key": key, "smell_id": "_consolidator", "error": str(exc)})
                    if exc.usage is not None:
                        extra_usage.append(exc.usage)
            consolidated_findings.extend(group)
            verdicts.append(rule_based_verdict(key, op.operation_id, group))
        # findings channel already holds pre-consolidation findings; the
        # post-consolidation set goes to its own channel for persist.
        return {"verdicts": verdicts, "usage": extra_usage, "errors": errors,
                "final_findings": consolidated_findings}

    graph = StateGraph(ScanState)
    graph.add_node("reduce", node_reduce)
    graph.add_node("plan", node_plan)
    graph.add_node("detect", node_detect)
    graph.add_node("collect", node_collect)
    graph.add_edge(START, "reduce")
    graph.add_edge("reduce", "plan")
    graph.add_conditional_edges("plan", fan_out, ["detect", "collect"])
    graph.add_edge("detect", "collect")
    graph.add_edge("collect", END)
    return graph.compile(), meter


def run_scan(
    *,
    spec: LoadedSpec,
    llm: LLM,
    cache: ResponseCache,
    config: HermesConfig,
    store: RunStore,
    tags=None,
    path_globs=None,
    sample=None,
    max_endpoints=None,
    consolidate_enabled: bool = True,
) -> dict:
    """Execute a scan end-to-end and persist results. Returns run meta dict.

    Raises BudgetExceeded (resumable) if the cost ceiling is hit.
    """
    operations = filter_operations(
        spec.operations, tags=tags, path_globs=path_globs, sample=sample,
        seed=config.seed, max_endpoints=max_endpoints,
    )
    compiled, meter = build_scan_graph(
        spec=spec, operations=operations, llm=llm, cache=cache,
        config=config, consolidate_enabled=consolidate_enabled,
    )
    try:
        state = compiled.invoke(
            {"run_id": store.run_id},
            # No floor: max_concurrency=1 runs Sends strictly serially (verified
            # against langgraph 1.2 internals in the M5 review).
            config={"max_concurrency": config.concurrency},
        )
    except (BudgetExceeded, KeyboardInterrupt) as exc:
        # Leave a machine-readable record so the run dir is never empty and
        # tooling can tell "interrupted, resumable" from "never ran". Completed
        # pairs are safe in the response cache.
        status = "interrupted:budget" if isinstance(exc, BudgetExceeded) else "interrupted"
        store.write_run_meta(
            config=_config_echo(spec, config, consolidate_enabled),
            counts={"operations_scanned": len(operations), "interrupted_cost_usd": round(meter.total, 4)},
            usage=[],
            errors=[{"error": str(exc)}],
            status=status,
        )
        raise
    status = "completed"
    final_findings = state.get("final_findings", state.get("findings", []))
    store.write_findings(final_findings)
    store.write_raw(state.get("raw", []))
    store.write_verdicts(state.get("verdicts", []))
    meta = store.write_run_meta(
        config=_config_echo(spec, config, consolidate_enabled),
        counts={
            "operations_scanned": len(operations),
            "detections": len(final_findings),
            "endpoints_with_detections": len({f"{f.endpoint.method} {f.endpoint.path}" for f in final_findings}),
            "avg_smells_per_endpoint": round(len(final_findings) / len(operations), 2) if operations else 0.0,
            "detector_errors": len(state.get("errors", [])),
        },
        usage=state.get("usage", []),
        errors=state.get("errors", []),
        status=status,
    )
    return meta


def _config_echo(spec: LoadedSpec, config: HermesConfig, consolidate_enabled: bool) -> dict:
    """What run.json records about the run's configuration. NEVER include
    credentials or raw environment (docs/03-DEPLOYMENT secrets policy)."""
    return {
        "spec_title": spec.title,
        "spec_version": spec.version,
        "detect_model": config.detect_model,
        "consolidate_model": config.consolidate_model,
        "concurrency": config.concurrency,
        "seed": config.seed,
        "consolidate": consolidate_enabled,
    }


def _raw_record(run_id, endpoint_key, smell_id, response, *, cached_replay: bool) -> dict:
    return {
        "run_id": run_id,
        "endpoint_key": endpoint_key,
        "smell_id": smell_id,
        "cached_replay": cached_replay,
        "response": response.model_dump(),
    }


def _to_finding(state: ScanState, op: OperationRef, smell_id: str, response, truncation: bool, config):
    return _to_finding_from(state["run_id"], state["api_title"], op, smell_id, response, truncation, config)


def _to_finding_from(run_id, api_title, op: OperationRef, smell_id, response, truncation: bool, config):
    return build_finding(
        response=response,
        run_id=run_id,
        api_title=api_title,
        endpoint=EndpointInfo(path=op.path, method=op.method, operation_id=op.operation_id, tags=list(op.tags)),
        smell_id=smell_id,
        category=SMELLS[smell_id].category,
        model=config.detect_model,
        prompt_version=PROMPT_VERSIONS[smell_id],
        truncation_applied=truncation,
    )
