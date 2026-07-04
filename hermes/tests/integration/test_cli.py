"""M5 CLI integration: estimate output, scan confirmation gate (docs/02-TEST-PLAN §4)."""

import subprocess
import sys
from pathlib import Path

GOLDEN = Path(__file__).parent.parent / "fixtures" / "golden" / "seeded_spec.yaml"


def _run(argv, **kwargs):
    return subprocess.run(
        [sys.executable, "-m", "hermes.cli", *argv], capture_output=True, text=True, **kwargs
    )


def test_estimate_reports_ops_times_nine():
    out = _run(["estimate", "--spec", str(GOLDEN)])
    assert out.returncode == 0
    assert "operations (after filters): 40" in out.stdout
    assert "detection calls: 360 (40 ops x 9 smells)" in out.stdout


def test_scan_without_yes_non_tty_exits_4(tmp_path):
    out = _run(["scan", "--spec", str(GOLDEN), "--out", str(tmp_path)], stdin=subprocess.DEVNULL)
    assert out.returncode == 4
    assert "refusing to spend without --yes" in out.stderr


def test_estimate_with_bad_env_exits_2(tmp_path):
    import os

    env = dict(os.environ, HERMES_CONCURRENCY="abc")
    out = subprocess.run(
        [sys.executable, "-m", "hermes.cli", "estimate", "--spec", str(GOLDEN)],
        capture_output=True, text=True, env=env,
    )
    assert out.returncode == 2
    assert "HERMES_CONCURRENCY" in out.stderr


def test_resume_without_run_id_exits_2(tmp_path):
    out = _run(["scan", "--spec", str(GOLDEN), "--out", str(tmp_path), "--resume", "--yes"])
    assert out.returncode == 2
    assert "--resume requires --run-id" in out.stderr


def test_scan_success_writes_all_artifacts_in_process(tmp_path, monkeypatch):
    """Drive cmd_scan through main() with a FakeLLM injected at the llm seam:
    proves scan writes findings.jsonl, endpoints.jsonl, run.json AND report.html
    (00-SPEC §6) and that no secret ever lands in the run dir."""
    import hermes.llm as llm_mod
    from hermes.cli import main
    from tests.conftest import FakeLLM, detected

    fake = FakeLLM({("getAccountBalance", "LAZY"): detected("LAZY")})
    monkeypatch.setattr(llm_mod, "AnthropicLLM", lambda config, client=None: fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-secret-should-never-leak")
    rc = main([
        "scan", "--spec", str(GOLDEN), "--out", str(tmp_path), "--run-id", "r_cli",
        "--max-endpoints", "3", "--yes",
    ])
    assert rc == 0
    run_dir = tmp_path / "r_cli"
    for artifact in ("findings.jsonl", "endpoints.jsonl", "run.json", "report.html"):
        assert (run_dir / artifact).exists(), artifact
    # secrets policy (03-DEPLOYMENT): no credential material in any run artifact
    for artifact in run_dir.iterdir():
        assert "sk-ant" not in artifact.read_text(encoding="utf-8"), artifact


def test_budget_exceeded_maps_to_exit_3(tmp_path, monkeypatch):
    import hermes.llm as llm_mod
    from hermes.cli import main
    from hermes.schemas.models import UsageRecord
    from tests.conftest import FakeLLM

    class CostlyLLM(FakeLLM):
        def detect(self, smell_id, erd_yaml):
            response, usage = super().detect(smell_id, erd_yaml)
            usage.cost_usd = 1.0
            return response, usage

    monkeypatch.setattr(llm_mod, "AnthropicLLM", lambda config, client=None: CostlyLLM())
    monkeypatch.setenv("HERMES_MAX_COST_USD", "1.5")
    rc = main(["scan", "--spec", str(GOLDEN), "--out", str(tmp_path), "--run-id", "r_budget", "--yes"])
    assert rc == 3
    # interrupted runs still leave a machine-readable run.json
    import json
    meta = json.loads((tmp_path / "r_budget" / "run.json").read_text())
    assert meta["status"] == "interrupted:budget"


def test_report_command_renders_html_and_md(tmp_path, monkeypatch):
    import hermes.llm as llm_mod
    from hermes.cli import main
    from tests.conftest import FakeLLM, detected

    fake = FakeLLM({("getAccountBalance", "LAZY"): detected("LAZY")})
    monkeypatch.setattr(llm_mod, "AnthropicLLM", lambda config, client=None: fake)
    assert main(["scan", "--spec", str(GOLDEN), "--out", str(tmp_path), "--run-id", "r_rep",
                 "--tags", "Accounts", "--yes"]) == 0
    run_dir = str(tmp_path / "r_rep")
    assert main(["report", "--run", run_dir]) == 0
    html = (tmp_path / "r_rep" / "report.html").read_text(encoding="utf-8")
    assert "Per-tag rollup" in html  # full dashboard, not the placeholder
    # Appendix-C markdown for a specific endpoint
    import contextlib, io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(["report", "--run", run_dir, "--endpoint",
                   "GET /accounts/{accountId}/balance", "--md"])
    assert rc == 0
    md = buf.getvalue()
    assert "## Identified Smells" in md and "Lazy" in md


def test_offline_eval_missing_recordings_exits_2(tmp_path, monkeypatch):
    from hermes.cli import main
    import hermes.eval.harness as harness

    monkeypatch.setattr(harness, "RECORDINGS", tmp_path / "none.jsonl")
    assert main(["eval"]) == 2
