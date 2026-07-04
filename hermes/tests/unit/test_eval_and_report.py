"""M6 unit tests: multi-label metrics (hand-computed), harness plumbing,
dashboard + Appendix-C markdown rendering (docs/02-TEST-PLAN §4)."""

import json
from pathlib import Path

import pytest

from hermes.eval.metrics import compute_metrics, gate_failures
from hermes.schemas.models import EndpointInfo, build_finding
from hermes.store import RunStore

from tests.conftest import detected

# ---------- metrics: 3 endpoints, hand-computed ----------
# gold:  op1={LAZY, INPUT}  op2={}  op3={RESPONSE}
# pred:  op1={LAZY}         op2={SECURITY}  op3={RESPONSE}
#
# per-op jaccard: op1: |{LAZY}|/|{LAZY,INPUT}| = 1/2; op2: 0/1 = 0; op3: 1/1 = 1
#   -> mean = (0.5 + 0 + 1)/3 = 0.5
# tp=2 (LAZY@op1, RESPONSE@op3), fp=1 (SECURITY@op2), fn=1 (INPUT@op1)
#   f1_micro = 2*2/(2*2+1+1) = 4/6 = 0.6667
# per-smell f1: LAZY=1.0, INPUT=0.0, RESPONSE=1.0, SECURITY=0.0 (others inactive)
#   f1_macro = (1+0+1+0)/4 = 0.5
# hamming: wrong labels = 2 (INPUT missed, SECURITY spurious) over 3*9=27 -> 0.0741
# cardinality: (1-2) + (1-0) + (1-1) = 0 -> 0.0
# clean ops with predictions: op2 -> 1

GOLD = {"op1": {"LAZY", "INPUT"}, "op2": set(), "op3": {"RESPONSE"}}
PRED = {"op1": {"LAZY"}, "op2": {"SECURITY"}, "op3": {"RESPONSE"}}


def test_metrics_hand_computed():
    m = compute_metrics(PRED, GOLD)
    assert m.jaccard == pytest.approx(0.5)
    assert m.f1_micro == pytest.approx(2 / 3)
    assert m.f1_macro == pytest.approx(0.5)
    assert m.hamming_loss == pytest.approx(2 / 27)
    assert m.cardinality_diff == pytest.approx(0.0)
    assert m.clean_ops_with_predictions == 1
    assert m.per_smell["LAZY"].f1 == pytest.approx(1.0)
    assert m.per_smell["INPUT"].recall == pytest.approx(0.0)


def test_metrics_both_empty_jaccard_is_one():
    m = compute_metrics({"a": set()}, {"a": set()})
    assert m.jaccard == 1.0
    assert m.hamming_loss == 0.0


def test_metrics_id_mismatch_raises():
    with pytest.raises(ValueError):
        compute_metrics({"a": set()}, {"b": set()})


def test_gate_failures_boundaries():
    perfect = compute_metrics(GOLD, GOLD)
    assert gate_failures(perfect) == []
    bad = compute_metrics(PRED, GOLD)  # f1_micro 0.67 < 0.85, jaccard 0.5 < 0.75
    failures = gate_failures(bad)
    assert any("jaccard" in f for f in failures)
    assert any("f1_micro" in f for f in failures)


# ---------- harness: offline replay + staleness ----------


def _recording_line(op, smell, detected_flag, prompt_version=None):
    from hermes.smells.prompts import PROMPT_VERSIONS

    from tests.conftest import detected as make_detected, not_detected

    response = make_detected(smell) if detected_flag else not_detected()
    return json.dumps({
        "operation_id": op,
        "smell_id": smell,
        "prompt_version": prompt_version or PROMPT_VERSIONS[smell],
        "response": response.model_dump(),
    })


def test_replay_llm_staleness_check(tmp_path):
    from hermes.eval.harness import ReplayLLM, StaleRecordingsError

    recordings = tmp_path / "responses.jsonl"
    recordings.write_text(_recording_line("op1", "LAZY", True, prompt_version="lazy-v0") + "\n")
    with pytest.raises(StaleRecordingsError, match="stale"):
        ReplayLLM(recordings)
    with pytest.raises(FileNotFoundError):
        ReplayLLM(tmp_path / "missing.jsonl")


def test_replay_llm_lookup(tmp_path):
    from hermes.eval.harness import ReplayLLM

    recordings = tmp_path / "responses.jsonl"
    recordings.write_text(
        json.dumps({"_meta": True, "detect_model": "m"}) + "\n"
        + _recording_line("op1", "LAZY", True) + "\n"
        + _recording_line("op1", "INPUT", False) + "\n"
    )
    replay = ReplayLLM(recordings)
    assert replay.detect_model == "m"
    assert replay.lookup("op1", "LAZY").smell_detected
    assert not replay.lookup("op1", "INPUT").smell_detected
    assert replay.lookup("op2", "LAZY") is None


# ---------- report rendering ----------


@pytest.fixture()
def populated_store(tmp_path):
    store = RunStore(tmp_path, "r_render")
    findings = [
        build_finding(
            response=detected(smell, severity=sev),
            run_id="r_render", api_title="Atlas",
            endpoint=EndpointInfo(path=path, method=method, operation_id=op, tags=[tag]),
            smell_id=smell, category="documentation" if smell == "LAZY" else "rest",
            model="m", prompt_version="v",
        )
        for smell, sev, method, path, op, tag in [
            ("LAZY", "high", "GET", "/a", "opA", "T1"),
            ("SECURITY", "medium", "GET", "/a", "opA", "T1"),
            ("RESPONSE", "low", "POST", "/b</script><b>x", "opB", "T2"),  # hostile path
        ]
    ]
    store.write_findings(findings)
    from hermes.nodes import rule_based_verdict

    store.write_verdicts([
        rule_based_verdict("GET /a", "opA", findings[:2]),
        rule_based_verdict("POST /b</script><b>x", "opB", findings[2:]),
    ])
    store.write_run_meta(
        config={"spec_title": "Atlas", "detect_model": "m", "consolidate_model": "c", "consolidate": True},
        counts={"operations_scanned": 5, "detections": 3, "endpoints_with_detections": 2,
                "avg_smells_per_endpoint": 0.6, "detector_errors": 0},
        usage=[], errors=[], status="completed",
    )
    return store


def test_dashboard_renders_with_required_elements(populated_store):
    from hermes.report.render import render_run

    html = render_run(populated_store)
    for required in (
        "Atlas", "avg smells / endpoint (paper: 4.08)", "hermes-findings", "hermes-verdicts",
        "Per-tag rollup", "Top endpoints", "f-smell", "f-severity", "page-info",
    ):
        assert required in html, required
    # no external fetches: no http(s) URLs in src/href attributes
    import re

    assert not re.search(r'(src|href)\s*=\s*["\']https?://', html)
    # script-breakout safety: the hostile path must not appear with a live "</"
    assert "/b</script>" not in html


def test_dashboard_renders_empty_run(tmp_path):
    from hermes.report.render import render_run

    store = RunStore(tmp_path, "r_empty")
    store.write_findings([])
    store.write_verdicts([])
    store.write_run_meta(config={"spec_title": "Empty"}, counts={"operations_scanned": 0},
                         usage=[], errors=[], status="completed")
    html = render_run(store)
    assert "Empty" in html


def test_appendix_c_markdown_section_order(populated_store):
    from hermes.report.render import render_endpoint_md

    md = render_endpoint_md(populated_store, "GET /a")
    sections = ["## API Info", "## Endpoint Info", "## Model", "## Identified Smells",
                "### Explanations", "## Improvement Suggestions"]
    positions = [md.index(s) for s in sections]
    assert positions == sorted(positions)
    assert "Lazy, Security" in md
    assert "[LAZY] - " in md and "[SECURITY] - " in md
    assert 'Method: "GET"' in md and 'Path: "/a"' in md
    assert "Verdict: not agent-consumable" in md  # high severity present


def test_appendix_c_markdown_matches_handwritten_snapshot(populated_store):
    """02-TEST-PLAN §4: snapshot against a hand-written expected file."""
    from hermes.report.render import render_endpoint_md

    expected = (Path(__file__).parent.parent / "fixtures" / "snapshots" /
                "appendix_c_expected.md").read_text(encoding="utf-8")
    assert render_endpoint_md(populated_store, "GET /a") == expected


def test_appendix_c_unknown_endpoint_raises(populated_store):
    from hermes.report.render import render_endpoint_md

    with pytest.raises(KeyError, match="not in this run"):
        render_endpoint_md(populated_store, "get /a")  # lowercase method


def test_dashboard_embedded_json_roundtrips(populated_store):
    import re

    from hermes.report.render import render_run

    html = render_run(populated_store)
    block = re.search(r'<script type="application/json" id="hermes-findings">(.*?)</script>',
                      html, re.S).group(1)
    assert json.loads(block) == [f.model_dump() for f in populated_store.load_findings()]


def test_dashboard_renders_5000_findings(tmp_path):
    """00-SPEC §7: usable at 5,000 findings — render completes and carries all rows."""
    import time

    from hermes.report.render import render_run

    store = RunStore(tmp_path, "r_big")
    smells = ["LAZY", "INPUT", "RESPONSE", "SECURITY", "BLOATED"]
    findings = [
        build_finding(
            response=detected(smells[i % 5]),
            run_id="r_big", api_title="Big",
            endpoint=EndpointInfo(path=f"/p{i//5}", method="GET", operation_id=f"op{i//5}", tags=["T"]),
            smell_id=smells[i % 5],
            category="documentation" if smells[i % 5] in ("LAZY", "BLOATED") else "rest",
            model="m", prompt_version="v",
        )
        for i in range(5000)
    ]
    store.write_findings(findings)
    store.write_verdicts([])
    store.write_run_meta(config={"spec_title": "Big"}, counts={"operations_scanned": 1000},
                         usage=[], errors=[], status="completed")
    start = time.monotonic()
    html = render_run(store)
    assert time.monotonic() - start < 30
    import re
    block = re.search(r'id="hermes-findings">(.*?)</script>', html, re.S).group(1)
    assert len(json.loads(block)) == 5000
