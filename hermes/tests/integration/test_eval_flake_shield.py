"""Ultra review E1 regression: the live-eval flake shield must never write a
"green" baseline whose records contradict the reported metrics.

Scenario: the base run produces LAZY false positives on 3 clean operations
(clean-op gate fails, 3 > max 2); the per-smell retry then FAILS (API flake)
on exactly those keys and is correct everywhere else. Before the fix, the
absent retry keys were scored as confident negatives — gates flipped to PASS
while the base run's smell_detected=true records were committed as the green
baseline, so an offline replay of the recordings failed the very gates the
live run reported passing.
"""

import json
import re
from pathlib import Path

from hermes.eval.harness import load_cases, run_live_eval
from hermes.schemas.models import UsageRecord

from tests.conftest import detected, not_detected

GOLDEN = Path(__file__).parent.parent / "fixtures" / "golden"


class FlakyRetryLLM:
    """Detects exactly the gold labels, except LAZY false positives on
    `fp_ops` in the first pass; repeat calls for those keys raise."""

    def __init__(self, gold, fp_ops):
        self.gold = gold
        self.fp_ops = fp_ops
        self.seen: set[tuple[str, str]] = set()

    def detect(self, smell_id, erd_yaml):
        op = re.search(r"operation_id: (\S+)\n", erd_yaml).group(1)
        usage = UsageRecord(model="fake")
        key = (op, smell_id)
        if op in self.fp_ops and smell_id == "LAZY":
            if key in self.seen:  # the retry pass
                from hermes.llm import DetectorFailure

                raise DetectorFailure("simulated API flake on retry")
            self.seen.add(key)
            return detected("LAZY"), usage  # base-run false positive
        if smell_id in self.gold.get(op, set()):
            return detected(smell_id), usage
        return not_detected(), usage

    def consolidate(self, endpoint_summary, findings_json):  # pragma: no cover
        raise AssertionError("eval never consolidates")


def test_failed_retry_cannot_flip_gates_or_poison_baseline(tmp_path):
    _cases, gold = load_cases()
    clean_ops = sorted(op for op, labels in gold.items() if not labels)[:3]
    assert len(clean_ops) == 3
    llm = FlakyRetryLLM(gold, set(clean_ops))
    record_to = tmp_path / "responses.jsonl"

    result = run_live_eval(llm, detect_model="fake", record_to=record_to)

    # The false positives survive (retry failed -> base predictions kept), so
    # the clean-op gate still fails and the run goes to the FAILED side path.
    assert result["metrics"]["clean_ops_with_predictions"] == 3
    assert any("clean operations" in f for f in result["gate_failures"])
    assert result["recorded_to"].endswith("responses.failed.jsonl")
    assert not record_to.exists()  # green baseline never written

    # And the written records agree with the reported metrics: the FP keys
    # still say smell_detected=true (base record kept, prediction kept).
    written = [json.loads(line) for line in
               Path(result["recorded_to"]).read_text(encoding="utf-8").splitlines()]
    fp_records = [r for r in written if not r.get("_meta")
                  and r["operation_id"] in clean_ops and r["smell_id"] == "LAZY"]
    assert len(fp_records) == 3
    assert all(r["response"]["smell_detected"] for r in fp_records)
