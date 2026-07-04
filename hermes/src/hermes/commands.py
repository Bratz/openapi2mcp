"""Subcommand implementations, filled in milestone by milestone.

Kept separate from cli.py so argparse wiring stays import-light and
unimplemented commands fail loudly instead of silently.
"""

from __future__ import annotations

import argparse


def _not_implemented(name: str, milestone: str) -> int:
    print(f"hermes {name}: not implemented yet (lands in {milestone} — see docs/04-BUILD-PLAN.md)")
    return 1


def cmd_inspect(args: argparse.Namespace) -> int:
    return _not_implemented("inspect", "M2")


def cmd_estimate(args: argparse.Namespace) -> int:
    return _not_implemented("estimate", "M5")


def cmd_scan(args: argparse.Namespace) -> int:
    return _not_implemented("scan", "M5")


def cmd_report(args: argparse.Namespace) -> int:
    return _not_implemented("report", "M6")


def cmd_eval(args: argparse.Namespace) -> int:
    return _not_implemented("eval", "M6")
