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
    import sys

    from hermes.reducer import reduce_operation
    from hermes.spec_loader import SpecError, find_operation, load_spec

    try:
        spec = load_spec(args.spec)
    except SpecError as exc:
        print(f"hermes inspect: {exc}", file=sys.stderr)
        return 2
    try:
        op = find_operation(spec, args.endpoint)
    except KeyError as exc:
        # Unknown endpoint key = invalid invocation (00-SPEC §6 → exit 2).
        print(f"hermes inspect: {exc.args[0]}", file=sys.stderr)
        return 2
    print(reduce_operation(spec, op).yaml, end="")
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    return _not_implemented("estimate", "M5")


def cmd_scan(args: argparse.Namespace) -> int:
    return _not_implemented("scan", "M5")


def cmd_report(args: argparse.Namespace) -> int:
    return _not_implemented("report", "M6")


def cmd_eval(args: argparse.Namespace) -> int:
    return _not_implemented("eval", "M6")
