"""M5 unit tests: cache, store, verdict rules, consolidation application."""

from pathlib import Path

from hermes.cache import ResponseCache, cache_key
from hermes.nodes import apply_consolidation, rule_based_verdict
from hermes.schemas.models import (
    ConsolidationEdit,
    ConsolidationResponse,
    EndpointInfo,
    UsageRecord,
    build_finding,
)
from hermes.store import RunStore

from tests.conftest import detected, not_detected


def _finding(smell_id="LAZY", severity="medium", path="/x"):
    return build_finding(
        response=detected(smell_id, severity=severity),
        run_id="r1",
        api_title="T",
        endpoint=EndpointInfo(path=path, method="GET", operation_id="opx", tags=["T"]),
        smell_id=smell_id,
        category="documentation" if smell_id != "RESPONSE" else "rest",
        model="m",
        prompt_version="v1",
    )


# ---------- cache ----------


def test_cache_roundtrip_and_miss(tmp_path):
    cache = ResponseCache(tmp_path / "cache.db")
    key = cache_key("erd", "LAZY", "lazy-v1", "model-a")
    assert cache.get(key) is None
    cache.put(key, detected("LAZY"), "model-a")
    got = cache.get(key)
    assert got is not None and got.smell_detected
    cache.close()


def test_cache_key_sensitivity():
    base = cache_key("erd", "LAZY", "lazy-v1", "m")
    assert cache_key("erd2", "LAZY", "lazy-v1", "m") != base  # ERD change
    assert cache_key("erd", "INPUT", "lazy-v1", "m") != base  # smell change
    assert cache_key("erd", "LAZY", "lazy-v2", "m") != base  # prompt bump
    assert cache_key("erd", "LAZY", "lazy-v1", "m2") != base  # model change


def test_no_cache_mode_bypasses_reads_but_writes(tmp_path):
    db = tmp_path / "cache.db"
    key = cache_key("erd", "LAZY", "v", "m")
    writer = ResponseCache(db, read_enabled=False)
    writer.put(key, detected("LAZY"), "m")
    assert writer.get(key) is None  # reads bypassed
    writer.close()
    reader = ResponseCache(db)
    assert reader.get(key) is not None  # but the write landed
    reader.close()


# ---------- store ----------


def test_store_overwrite_semantics_and_rollup(tmp_path):
    store = RunStore(tmp_path, "r1")
    f = _finding()
    assert store.write_findings([f, f]) == 1  # dedup by id within a write
    assert store.write_findings([f]) == 1  # re-persist overwrites, no duplicates
    assert len(store.load_findings()) == 1
    store.write_raw([{"a": 1}])
    store.write_raw([{"a": 2}])  # overwrite: raw reflects the LAST run only
    assert store._read_jsonl(store.raw_path) == [{"a": 2}]
    usage = [
        UsageRecord(model="m", input_tokens=100, output_tokens=10, cost_usd=0.5),
        UsageRecord(model="m", cached=True),
        UsageRecord(model="m", input_tokens=50, output_tokens=5, cost_usd=0.25, estimated=True),
    ]
    meta = store.write_run_meta(config={}, counts={"x": 1}, usage=usage, errors=[], status="completed")
    assert meta["usage"]["calls"] == 3
    assert meta["usage"]["cached_replays"] == 1
    assert meta["usage"]["cost_usd"] == 0.75
    assert meta["usage"]["cost_includes_estimates"] is True
    assert store.load_run_meta()["status"] == "completed"


# ---------- verdict rules ----------


def test_rule_based_verdicts():
    assert rule_based_verdict("GET /x", "op", []).verdict == "agent-ready"
    one = [_finding(severity="medium")]
    assert rule_based_verdict("GET /x", "op", one).verdict == "needs adaptation"
    high = [_finding(severity="high")]
    assert rule_based_verdict("GET /x", "op", high).verdict == "not agent-consumable"
    three = [_finding("LAZY"), _finding("INPUT"), _finding("RESPONSE")]
    assert rule_based_verdict("GET /x", "op", three).verdict == "not agent-consumable"


def test_apply_consolidation_clamps_and_drops():
    f1, f2 = _finding("LAZY", severity="low"), _finding("INPUT", severity="medium")
    response = ConsolidationResponse(
        edits=[
            ConsolidationEdit(finding_id=f1.id, action="adjust_severity", reason="r", new_severity="high"),
            ConsolidationEdit(finding_id=f2.id, action="drop_duplicate", reason="dupe"),
            ConsolidationEdit(finding_id="f_invented", action="drop_duplicate", reason="hallucinated"),
        ],
        verdict="needs adaptation",
        note="n",
    )
    kept, verdict = apply_consolidation("GET /x", "opx", [f1, f2], response)
    assert [f.id for f in kept] == [f1.id]
    assert kept[0].extensions.severity == "medium"  # low -> high clamped to one step
    assert verdict.verdict == "needs adaptation"
    assert verdict.smells == ["LAZY"]


def test_repeat_severity_edits_do_not_compound():
    f1 = _finding("LAZY", severity="low")
    response = ConsolidationResponse(
        edits=[
            ConsolidationEdit(finding_id=f1.id, action="adjust_severity", reason="r", new_severity="high"),
            ConsolidationEdit(finding_id=f1.id, action="adjust_severity", reason="again", new_severity="high"),
        ],
        verdict="needs adaptation",
    )
    kept, _ = apply_consolidation("GET /x", "opx", [f1], response)
    assert kept[0].extensions.severity == "medium"  # still one step from ORIGINAL
