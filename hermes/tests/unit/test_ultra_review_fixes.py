"""Regression tests for the whole-codebase ultra review findings.

Each test pins one confirmed defect fix; see DECISIONS.md (ultra review entry)
for the finding inventory.
"""

import json
import sqlite3
from types import SimpleNamespace

import pytest

from hermes.cache import ResponseCache
from hermes.config import HermesConfig
from hermes.llm import AnthropicLLM, DetectorFailure
from hermes.reducer import reduce_operation
from hermes.schemas.models import ConsolidationResponse
from hermes.spec_loader import LoadedSpec, OperationRef, SpecError, load_spec

from tests.conftest import detected, not_detected


# ---------- llm.py: refusal path must not lose billed spend ----------


class RefusingClient:
    """parse() returns parsed_output=None (refusal) with real billed usage."""

    def __init__(self):
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **kwargs):
        usage = SimpleNamespace(input_tokens=1200, output_tokens=50,
                                cache_read_input_tokens=0, cache_creation_input_tokens=0)
        return SimpleNamespace(parsed_output=None, usage=usage, stop_reason="refusal")


def test_refusal_detector_failure_carries_real_usage():
    llm = AnthropicLLM(HermesConfig(), client=RefusingClient())
    with pytest.raises(DetectorFailure, match="no parseable output") as ei:
        llm.detect("LAZY", "endpoint: {}")
    usage = ei.value.usage
    assert usage is not None, "billed refusal must reach the budget meter (ultra review B1)"
    assert usage.input_tokens == 1200 and not usage.estimated
    assert usage.cost_usd == pytest.approx((1200 * 1 + 50 * 5) / 1e6)


# ---------- cache.py: stale/foreign blobs degrade to a miss ----------


def test_cache_schema_mismatch_is_a_miss_not_a_crash(tmp_path):
    cache = ResponseCache(tmp_path / "c.db")
    cache.put("k", detected("LAZY"), "m")
    # An AgentResponse blob does not validate as ConsolidationResponse — the
    # scan must treat it as a miss (re-buy), never crash (ultra review C2).
    assert cache.get_as("k", ConsolidationResponse) is None
    assert cache.get("k") is not None  # correct type still hits
    cache.close()


def test_cache_garbage_row_is_a_miss(tmp_path):
    cache = ResponseCache(tmp_path / "c.db")
    with cache._lock:
        cache._conn.execute(
            "INSERT INTO cache (key, response_json, model, created_at) VALUES (?, ?, ?, ?)",
            ("bad", '{"not": "an agent response"}', "m", "t"),
        )
        cache._conn.commit()
    assert cache.get("bad") is None
    cache.close()


def test_cache_put_survives_locked_db(tmp_path, monkeypatch):
    cache = ResponseCache(tmp_path / "c.db")

    class LockedConn:
        def execute(self, *a, **k):
            raise sqlite3.OperationalError("database is locked")

        def commit(self):  # pragma: no cover
            raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(cache, "_conn", LockedConn())
    cache.put("k", detected("LAZY"), "m")  # must not raise (ultra review C2)
    assert cache.get("k") is None


# ---------- reducer.py: hostile spec content ----------


def _spec_with(raw_extra: dict) -> tuple[LoadedSpec, OperationRef]:
    raw = {
        "swagger": "2.0",
        "info": {"title": "Hostile", "version": "1"},
        "paths": {"/x": {"get": {"operationId": "getX", "responses": {"200": {"description": "ok"}}}}},
    }
    raw.update(raw_extra)
    op = OperationRef(path="/x", method="GET", operation_id="getX", tags=())
    return LoadedSpec(version="swagger2", title="Hostile", raw=raw, operations=[op]), op


@pytest.mark.parametrize("hostile", [
    {"securityDefinitions": 5},
    {"securityDefinitions": "x"},
    {"securityDefinitions": [1]},
    {"tags": 5},
    {"tags": True},
])
def test_reducer_survives_hostile_top_level_values(hostile):
    spec, op = _spec_with(hostile)
    erd = reduce_operation(spec, op)  # must not raise (ultra review R1/R2)
    assert "getX" in erd.yaml


def test_reducer_caps_unbounded_inline_nesting():
    deep: dict = {"type": "object"}
    for _ in range(5000):
        deep = {"type": "object", "properties": {"inner": deep}}
    spec, op = _spec_with({})
    spec.raw["paths"]["/x"]["get"]["parameters"] = [
        {"name": "b", "in": "body", "schema": deep}
    ]
    erd = reduce_operation(spec, op)  # RecursionError before the fix (ultra review R3)
    assert "$truncated" in erd.yaml


def test_oas3_hostile_security_schemes():
    raw = {
        "openapi": "3.0.0",
        "info": {"title": "Hostile", "version": "1"},
        "paths": {"/x": {"get": {"operationId": "getX", "responses": {"200": {"description": "ok"}}}}},
        "components": {"securitySchemes": "nope"},
    }
    op = OperationRef(path="/x", method="GET", operation_id="getX", tags=())
    spec = LoadedSpec(version="oas3", title="Hostile", raw=raw, operations=[op])
    assert "getX" in reduce_operation(spec, op).yaml


# ---------- spec_loader.py: parser recursion maps to SpecError ----------


def test_deeply_nested_json_is_spec_error_not_recursion_error(tmp_path):
    hostile = tmp_path / "deep.json"
    hostile.write_text("[" * 100_000 + "]" * 100_000, encoding="utf-8")
    with pytest.raises(SpecError, match="could not parse"):
        load_spec(hostile)  # bare RecursionError before the fix (ultra review R4)


# ---------- report: script-block embedding ----------


def test_embed_escapes_every_angle_bracket():
    from hermes.report.render import _embed

    payload = {"title": '<!--<script x="</script>"'}
    out = str(_embed(payload))
    assert "<" not in out, "any literal '<' inside a script block is a breakout vector (ultra review R6)"
    assert json.loads(out) == payload  # < escapes keep the JSON equivalent


# ---------- eval harness: flake shield + staleness ----------


def test_swap_smell_failed_retry_keys_keep_base_prediction():
    from hermes.eval.harness import _swap_smell

    gold = {"op1": set(), "op2": {"LAZY"}}
    base = {"op1": {"LAZY"}, "op2": {"LAZY"}}  # op1 is a base-run false positive
    retry = {"op1": set(), "op2": {"LAZY"}}  # op1 absent because the retry call FAILED
    # Without failure tracking the FP would be silently "cured" by an API flake.
    assert _swap_smell(base, retry, "LAZY", gold) is not None
    # With it, the failed key keeps the base prediction; nothing improves -> no swap.
    assert _swap_smell(base, retry, "LAZY", gold, {("op1", "LAZY")}) is None


def _write_recordings(path, lines):
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")


def _record(op, smell, prompt_version=None):
    from hermes.smells.prompts import PROMPT_VERSIONS

    return {"operation_id": op, "smell_id": smell,
            "prompt_version": prompt_version or PROMPT_VERSIONS[smell],
            "response": not_detected().model_dump()}


def test_replay_llm_unknown_smell_is_stale_not_keyerror(tmp_path):
    from hermes.eval.harness import ReplayLLM, StaleRecordingsError

    recordings = tmp_path / "responses.jsonl"
    _write_recordings(recordings, [{"operation_id": "op1", "smell_id": "NOT_A_SMELL",
                                    "prompt_version": "x-v1", "response": not_detected().model_dump()}])
    with pytest.raises(StaleRecordingsError, match="unknown smell"):
        ReplayLLM(recordings)  # raw KeyError before the fix (ultra review E2)


def test_replay_llm_detect_model_mismatch_is_stale(tmp_path):
    from hermes.eval.harness import ReplayLLM, StaleRecordingsError

    recordings = tmp_path / "responses.jsonl"
    _write_recordings(recordings, [{"_meta": True, "detect_model": "old-model"},
                                   _record("op1", "LAZY")])
    with pytest.raises(StaleRecordingsError, match="detect model"):
        ReplayLLM(recordings, expected_detect_model="new-model")
    # Matching or unspecified expectations replay fine.
    assert ReplayLLM(recordings, expected_detect_model="old-model").detect_model == "old-model"
    assert ReplayLLM(recordings).detect_model == "old-model"
