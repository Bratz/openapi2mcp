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
