"""M0 scaffold tests: package imports, CLI wiring, entry point, exit codes."""

import subprocess
import sys
from importlib.metadata import entry_points

import pytest

import hermes
from hermes.cli import EXIT_DECLINED, EXIT_INTERRUPTED, EXIT_SPEC_ERROR, build_parser, main

SUBCOMMANDS = ("scan", "report", "inspect", "eval", "estimate")


def test_version_string():
    assert hermes.__version__


def test_installed_metadata_version_matches_package():
    from importlib.metadata import version

    assert version("hermes") == hermes.__version__


def test_help_lists_all_subcommands():
    help_text = build_parser().format_help()
    for name in SUBCOMMANDS:
        assert name in help_text


@pytest.mark.parametrize("name", SUBCOMMANDS)
def test_every_subcommand_parses_and_dispatches(name):
    argv = {
        "scan": ["scan", "--spec", "x.json"],
        "estimate": ["estimate", "--spec", "x.json"],
        "report": ["report", "--run", "runs/r1"],
        "inspect": ["inspect", "--spec", "x.json", "--endpoint", "GET /x"],
        "eval": ["eval"],
    }[name]
    # Unimplemented commands fail loudly with exit 1 (not argparse's 2, not a crash)
    assert main(argv) == 1


def test_bare_tags_flag_is_a_usage_error():
    # nargs="+": `--tags` with no values must be rejected, not become []
    with pytest.raises(SystemExit) as exc:
        main(["scan", "--spec", "x.json", "--tags", "--yes"])
    assert exc.value.code == 2


def test_exit_codes_match_spec():
    # docs/00-SPEC.md §6: 2 invalid invocation/spec parse failure, 3 interrupted, 4 declined
    assert (EXIT_SPEC_ERROR, EXIT_INTERRUPTED, EXIT_DECLINED) == (2, 3, 4)


def test_no_args_prints_help_and_exits_zero(capsys):
    assert main([]) == 0
    assert "scan" in capsys.readouterr().out


def test_console_script_entry_point_resolves():
    (ep,) = [e for e in entry_points(group="console_scripts") if e.name == "hermes"]
    assert ep.value == "hermes.cli:main"
    assert ep.load() is main


def test_module_execution_runs():
    out = subprocess.run(
        [sys.executable, "-m", "hermes.cli", "--version"], capture_output=True, text=True
    )
    assert out.returncode == 0
    assert "hermes" in out.stdout
