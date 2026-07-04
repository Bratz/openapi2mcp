"""M5 integration tests: graph e2e with FakeLLM, resume via cache, consolidation, CLI.

Replaces the M0 placeholder smoke test.
"""

import json
from pathlib import Path

import pytest

from hermes.cache import ResponseCache
from hermes.config import HermesConfig
from hermes.graph import run_scan
from hermes.spec_loader import load_spec
from hermes.store import RunStore

from tests.conftest import FakeLLM, detected

GOLDEN = Path(__file__).parent.parent / "fixtures" / "golden"


@pytest.fixture()
def golden_spec():
    return load_spec(GOLDEN / "seeded_spec.yaml")


@pytest.fixture()
def env(tmp_path):
    config = HermesConfig(out_dir=tmp_path / "runs", concurrency=4)
    cache = ResponseCache(config.cache_db)
    store = RunStore(config.out_dir, "r_test")
    yield config, cache, store
    cache.close()


def test_graph_end_to_end(golden_spec, env):
    config, cache, store = env
    llm = FakeLLM({
        ("getAccountBalance", "LAZY"): detected("LAZY", severity="high"),
        ("createRefund", "LAZY"): detected("LAZY"),
        ("createRefund", "SECURITY"): detected("SECURITY"),
    })
    meta = run_scan(spec=golden_spec, llm=llm, cache=cache, config=config, store=store,
                    tags=None, consolidate_enabled=True)
    assert meta["counts"]["operations_scanned"] == 40
    assert meta["counts"]["detections"] == 3
    assert meta["counts"]["detector_errors"] == 0
    # 40 ops x 9 smells fresh calls + 1 consolidation (createRefund has 2 findings)
    assert meta["usage"]["fresh_calls"] == 40 * 9 + 1
    assert llm.consolidations == 1
    findings = store.load_findings()
    assert {f.smell for f in findings} == {"LAZY", "SECURITY"}
    verdicts = store.load_verdicts()
    assert len(verdicts) == 40
    by_op = {v.operation_id: v for v in verdicts}
    assert by_op["createRefund"].verdict == "needs adaptation"  # FakeLLM consolidator verdict
    assert by_op["listAccounts"].verdict == "agent-ready"
    # report placeholder renders
    from hermes.report.render import render_run
    html = render_run(store)
    assert "hermes-findings" in html and "LAZY" in html


def test_resume_via_cache_after_crash(golden_spec, env):
    config, cache, store = env
    crashing = FakeLLM({("getAccountBalance", "LAZY"): detected("LAZY")}, fail_after=50)
    with pytest.raises(RuntimeError, match="fail_after"):
        run_scan(spec=golden_spec, llm=crashing, cache=cache, config=config, store=store)
    assert len(crashing.calls) >= 50

    fresh = FakeLLM({("getAccountBalance", "LAZY"): detected("LAZY")})
    meta = run_scan(spec=golden_spec, llm=fresh, cache=cache, config=config, store=store)
    total_tasks = 40 * 9
    # Run 2 only detects what run 1 didn't cache: proves cache-driven resume.
    assert len(fresh.calls) < total_tasks
    assert len(fresh.calls) + 50 >= total_tasks  # ...and covers the remainder
    assert meta["usage"]["cached_replays"] >= 40
    assert meta["counts"]["detections"] == 1


def test_no_consolidate_uses_rule_verdicts(golden_spec, env):
    config, cache, store = env
    llm = FakeLLM({
        ("createRefund", "LAZY"): detected("LAZY", severity="high"),
        ("createRefund", "SECURITY"): detected("SECURITY"),
    })
    run_scan(spec=golden_spec, llm=llm, cache=cache, config=config, store=store,
             consolidate_enabled=False)
    assert llm.consolidations == 0
    verdicts = {v.operation_id: v for v in store.load_verdicts()}
    assert verdicts["createRefund"].verdict == "not agent-consumable"  # high severity rule
    assert verdicts["createRefund"].note == "rule-based verdict"


def test_detector_failures_recorded_not_fatal(golden_spec, env):
    from hermes.llm import DetectorFailure
    from hermes.schemas.models import UsageRecord

    config, cache, store = env

    class FlakyLLM(FakeLLM):
        def detect(self, smell_id, erd_yaml):
            if smell_id == "SECURITY":
                raise DetectorFailure("boom", usage=UsageRecord(model="fake", cost_usd=0.01, estimated=True))
            return super().detect(smell_id, erd_yaml)

    llm = FlakyLLM({("getAccountBalance", "LAZY"): detected("LAZY")})
    meta = run_scan(spec=golden_spec, llm=llm, cache=cache, config=config, store=store)
    assert meta["counts"]["detector_errors"] == 40  # SECURITY failed on every op
    assert meta["counts"]["detections"] == 1
    assert meta["status"] == "completed"
    errors = meta["detector_errors"]
    assert all(e["smell_id"] == "SECURITY" for e in errors)


def test_budget_exceeded_raises(golden_spec, env):
    from hermes.graph import BudgetExceeded

    config, cache, store = env
    config.max_cost_usd = 0.005

    class CostlyLLM(FakeLLM):
        def detect(self, smell_id, erd_yaml):
            response, usage = super().detect(smell_id, erd_yaml)
            usage.cost_usd = 0.001
            return response, usage

    with pytest.raises(BudgetExceeded):
        run_scan(spec=golden_spec, llm=CostlyLLM(), cache=cache, config=config, store=store)


# ---------- ultra review regressions ----------


def test_consolidation_failure_spend_counts_toward_budget(golden_spec, env):
    """Ultra review B2: a failed consolidation's estimated spend must trip
    HERMES_MAX_COST_USD exactly like a failed detection's."""
    from hermes.graph import BudgetExceeded
    from hermes.llm import DetectorFailure
    from hermes.schemas.models import UsageRecord

    config, cache, store = env
    config.max_cost_usd = 0.5

    class FailingConsolidator(FakeLLM):
        def consolidate(self, endpoint_summary, findings_json):
            raise DetectorFailure("boom", usage=UsageRecord(model="fake", cost_usd=1.0, estimated=True))

    llm = FailingConsolidator({
        ("createRefund", "LAZY"): detected("LAZY"),
        ("createRefund", "SECURITY"): detected("SECURITY"),
    })
    with pytest.raises(BudgetExceeded):
        run_scan(spec=golden_spec, llm=llm, cache=cache, config=config, store=store)


def test_reducer_failure_isolated_to_operation(golden_spec, env, monkeypatch):
    """Ultra review R5: one hostile operation gets a _reducer error record;
    the other 39 still scan, and it gets no spurious clean verdict."""
    import hermes.graph as graph_mod

    config, cache, store = env
    real = graph_mod.reduce_operation

    def flaky(spec, op):
        if op.operation_id == "getAccountBalance":
            raise ValueError("hostile operation")
        return real(spec, op)

    monkeypatch.setattr(graph_mod, "reduce_operation", flaky)
    meta = run_scan(spec=golden_spec, llm=FakeLLM(), cache=cache, config=config, store=store)
    assert meta["status"] == "completed"
    reducer_errors = [e for e in meta["detector_errors"] if e["smell_id"] == "_reducer"]
    assert [e["endpoint_key"] for e in reducer_errors] == ["GET /accounts/{accountId}/balance"]
    assert len(store.load_verdicts()) == 39


def test_run_json_echoes_filters(golden_spec, env):
    """Ultra review C4: run.json records the endpoint filters so a --resume
    with a different scope is detectable."""
    config, cache, store = env
    meta = run_scan(spec=golden_spec, llm=FakeLLM(), cache=cache, config=config, store=store,
                    tags=["Accounts"])
    assert meta["config"]["filters"] == {
        "tags": ["Accounts"], "paths": None, "sample": None, "max_endpoints": None,
    }


def test_consolidation_replays_from_cache(golden_spec, env):
    """Ultra review F7.3: a re-run with identical findings replays the cached
    ConsolidationResponse (validated as the right model) instead of re-paying."""
    config, cache, store = env
    responses = {
        ("createRefund", "LAZY"): detected("LAZY"),
        ("createRefund", "SECURITY"): detected("SECURITY"),
    }
    first = FakeLLM(responses)
    run_scan(spec=golden_spec, llm=first, cache=cache, config=config, store=store)
    assert first.consolidations == 1
    second = FakeLLM(responses)
    meta = run_scan(spec=golden_spec, llm=second, cache=cache, config=config, store=store)
    assert second.consolidations == 0  # replayed from cache
    assert len(second.calls) == 0  # detections replayed too
    assert meta["counts"]["detections"] == 2
