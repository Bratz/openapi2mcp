"""M3 tests: smell catalog + Appendix-A prompt assembly (docs/02-TEST-PLAN.md §4)."""

import os
import re
from pathlib import Path

import pytest

from hermes.smells.catalog import DOCUMENTATION_SMELLS, REST_SMELLS, SMELLS
from hermes.smells.prompts import PROMPT_VERSIONS, assemble_messages, build_system_prompt

SNAPSHOTS = Path(__file__).parent.parent / "fixtures" / "snapshots" / "prompts"

EXPECTED_IDS = {
    "LAZY", "BLOATED", "TANGLED", "FRAGMENTED", "EXCESS_STRUCTURED",
    "PATH_AND_METHOD", "INPUT", "RESPONSE", "SECURITY",
}


def test_exactly_nine_smells_with_paper_ids():
    assert set(SMELLS) == EXPECTED_IDS
    assert len(DOCUMENTATION_SMELLS) == 5
    assert len(REST_SMELLS) == 4


def test_every_smell_is_fully_specified():
    assert set(PROMPT_VERSIONS) == set(SMELLS)  # no missing AND no stale/extra versions
    for smell in SMELLS.values():
        assert smell.definition.strip()
        assert smell.scoping_rule.strip()
        assert len(smell.occurs_when) >= 2, smell.id
        assert re.fullmatch(r"[a-z_]+-v\d+", PROMPT_VERSIONS[smell.id])


def test_display_names_match_paper_report_style():
    assert SMELLS["PATH_AND_METHOD"].display_name == "Path & Method"
    assert SMELLS["EXCESS_STRUCTURED"].display_name == "Excess Structured"


@pytest.mark.parametrize("smell_id", sorted(EXPECTED_IDS))
def test_appendix_a_section_order(smell_id):
    prompt = build_system_prompt(SMELLS[smell_id])
    sections = [
        "Role:",
        "Smell Definition",
        "This smell typically occurs when:",
        "Task:",
        "Classification Rules:",
        "Explanation and Improvement Rules",
    ]
    positions = [prompt.index(s) for s in sections]  # raises if any section missing
    assert positions == sorted(positions), f"sections out of order for {smell_id}"
    # per-smell scoping rule is inside Classification Rules
    assert SMELLS[smell_id].scoping_rule in prompt
    # action-title contract
    assert f"[{smell_id}] - " in prompt


def test_prompts_are_static(monkeypatch):
    # Byte-stable: identical across calls, no timestamps/years, and immune to
    # environment changes (cache keys depend on all three).
    baseline = {sid: assemble_messages(sid, "erd: placeholder") for sid in EXPECTED_IDS}
    monkeypatch.setenv("HERMES_DETECT_MODEL", "some-other-model")
    monkeypatch.setenv("LANG", "C")
    for smell_id in EXPECTED_IDS:
        again = assemble_messages(smell_id, "erd: placeholder")
        assert again == baseline[smell_id], f"{smell_id} prompt not env-stable"
        system = again["system"]
        for banned in ("{run_id}", "datetime", "os.environ"):
            assert banned not in system
        assert not re.search(r"\b20\d{2}\b", system), f"year-like content in {smell_id} prompt"
        assert not re.search(r"\d{4}-\d{2}-\d{2}", system), f"date in {smell_id} prompt"


def test_user_message_carries_erd_verbatim():
    msg = assemble_messages("LAZY", "endpoint:\n  method: GET\n")
    assert "endpoint:\n  method: GET\n" in msg["user"]
    assert msg["prompt_version"] == "lazy-v1"


@pytest.mark.parametrize("smell_id", sorted(EXPECTED_IDS))
def test_prompt_snapshots(smell_id):
    """Prompt drift must be visible in diffs (docs/04-BUILD-PLAN.md M3).

    Any intentional change: re-record with HERMES_RECORD_SNAPSHOTS=1 AND bump
    that smell's PROMPT_VERSION (cache correctness).
    """
    prompt = build_system_prompt(SMELLS[smell_id])
    snapshot = SNAPSHOTS / f"{PROMPT_VERSIONS[smell_id]}.txt"
    if os.environ.get("HERMES_RECORD_SNAPSHOTS") == "1":
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(prompt, encoding="utf-8")
        pytest.skip("snapshot recorded (HERMES_RECORD_SNAPSHOTS=1)")
    assert snapshot.exists(), f"missing prompt snapshot for {smell_id}; record + bump version"
    assert snapshot.read_text(encoding="utf-8") == prompt


def test_snapshot_dir_has_no_stale_versions():
    if not SNAPSHOTS.exists():
        pytest.skip("snapshots not recorded yet")
    on_disk = {p.name for p in SNAPSHOTS.glob("*.txt")}
    assert on_disk == {f"{v}.txt" for v in PROMPT_VERSIONS.values()}, (
        "snapshot dir out of sync with PROMPT_VERSIONS — delete orphans from old bumps"
    )
