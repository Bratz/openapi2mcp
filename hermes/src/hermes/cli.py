"""Hermes CLI entry point.

Subcommands per docs/00-SPEC.md §6: scan, report, inspect, eval, estimate.
This module stays import-light: subcommand implementations live in
hermes.commands (thin adapters that delegate to graph/report/eval modules)
and are imported lazily at dispatch time.

Flag defaults that the spec assigns real values (concurrency 8, seed 42,
out runs/) are parsed as None here and resolved by HermesConfig (M4), so
environment overrides can distinguish "flag not given" from an explicit value.
"""

from __future__ import annotations

import argparse
import sys

from hermes import __version__

EXIT_OK = 0
EXIT_SPEC_ERROR = 2  # also argparse usage errors (see DECISIONS.md)
EXIT_INTERRUPTED = 3
EXIT_DECLINED = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hermes",
        description="Detect documentation and REST smells in OpenAPI specs (arXiv:2605.14312 reproduction).",
    )
    parser.add_argument("--version", action="version", version=f"hermes {__version__}")
    sub = parser.add_subparsers(dest="command")

    scan = sub.add_parser("scan", help="Scan a spec for smells (calls the Anthropic API)")
    _add_spec_filters(scan)
    scan.add_argument("--out", default=None, help="Output directory (default: <hermes project>/runs/)")
    scan.add_argument("--run-id", default=None)
    scan.add_argument("--resume", action="store_true")
    scan.add_argument("--detect-model", default=None)
    scan.add_argument("--consolidate-model", default=None)
    scan.add_argument("--no-consolidate", action="store_true")
    scan.add_argument("--no-cache", action="store_true")
    scan.add_argument("--concurrency", type=int, default=None, help="Max concurrent LLM calls (default: 8)")
    scan.add_argument("--yes", action="store_true", help="Skip cost confirmation")

    report = sub.add_parser("report", help="Render reports from a stored run")
    report.add_argument("--run", required=True, help="Run directory (runs/<run_id>)")
    report.add_argument("--out", default=None, help="Output HTML path")
    report.add_argument("--endpoint", default=None, help='Endpoint key, e.g. "GET /users"')
    report.add_argument("--md", action="store_true", help="Emit Appendix-C markdown diagnostic report")

    inspect = sub.add_parser("inspect", help="Print the reduced endpoint representation (ERD)")
    inspect.add_argument("--spec", required=True)
    inspect.add_argument("--endpoint", required=True, help='Endpoint key, e.g. "GET /users"')

    ev = sub.add_parser("eval", help="Golden-fixture eval with multi-label metric gates")
    ev.add_argument("--live", action="store_true", help="Call the live Anthropic API (~$0.50)")
    ev.add_argument("--report", action="store_true", help="Write eval_report.json")

    estimate = sub.add_parser("estimate", help="Print endpoint/call counts and cost estimate (no API calls)")
    _add_spec_filters(estimate)

    return parser


def _add_spec_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spec", required=True, help="Path to OpenAPI/Swagger spec (JSON or YAML)")
    parser.add_argument("--tags", nargs="+", default=None)
    parser.add_argument("--paths", nargs="+", default=None, help="Path glob filters")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed (default: 42)")
    parser.add_argument("--max-endpoints", type=int, default=None)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return EXIT_OK

    import hermes.commands as commands

    return getattr(commands, f"cmd_{args.command}")(args)


if __name__ == "__main__":
    sys.exit(main())
