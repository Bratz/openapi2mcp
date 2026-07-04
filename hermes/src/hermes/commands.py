"""Subcommand implementations — thin arg→module adapters (DECISIONS M0).

Real logic lives in graph/report/eval modules. Errors print to stderr; stdout
carries pipeable content only. Exit codes per docs/00-SPEC §6.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone


def _err(name: str, message: str) -> None:
    print(f"hermes {name}: {message}", file=sys.stderr)


def _not_implemented(name: str, milestone: str) -> int:
    print(f"hermes {name}: not implemented yet (lands in {milestone} — see docs/04-BUILD-PLAN.md)")
    return 1


def _load_spec_or_exit(name: str, path: str):
    from hermes.spec_loader import SpecError, load_spec

    try:
        return load_spec(path)
    except SpecError as exc:
        _err(name, str(exc))
        return None


def _estimate(spec, args, config) -> dict:
    from hermes.config import (
        EST_CONSOLIDATION_RATE,
        EST_INPUT_TOKENS_PER_CALL,
        EST_OUTPUT_TOKENS_PER_CALL,
        call_cost_usd,
    )
    from hermes.smells.catalog import SMELLS
    from hermes.spec_loader import filter_operations

    ops = filter_operations(
        spec.operations,
        tags=args.tags,
        path_globs=args.paths,
        sample=args.sample,
        seed=config.seed,
        max_endpoints=args.max_endpoints,
    )
    detect_calls = len(ops) * len(SMELLS)
    consolidations = int(len(ops) * EST_CONSOLIDATION_RATE)
    cost = detect_calls * call_cost_usd(
        config.detect_model,
        input_tokens=EST_INPUT_TOKENS_PER_CALL,
        output_tokens=EST_OUTPUT_TOKENS_PER_CALL,
    ) + consolidations * call_cost_usd(
        config.consolidate_model,
        input_tokens=EST_INPUT_TOKENS_PER_CALL,
        output_tokens=EST_OUTPUT_TOKENS_PER_CALL,
    )
    return {
        "operations": len(ops),
        "detect_calls": detect_calls,
        "consolidations_est": consolidations,
        "cost_usd_est": round(cost, 2),
    }


def cmd_estimate(args: argparse.Namespace) -> int:
    from hermes.config import ConfigError, HermesConfig

    try:
        config = HermesConfig.resolve(args)
    except ConfigError as exc:
        _err("estimate", str(exc))
        return 2
    spec = _load_spec_or_exit("estimate", args.spec)
    if spec is None:
        return 2
    est = _estimate(spec, args, config)
    print(f"spec: {spec.title}")
    print(f"operations (after filters): {est['operations']}")
    from hermes.smells.catalog import SMELLS

    print(f"detection calls: {est['detect_calls']} ({est['operations']} ops x {len(SMELLS)} smells)")
    print(f"estimated consolidations: {est['consolidations_est']}")
    print(f"estimated cost: ~${est['cost_usd_est']} "
          f"(models: {config.detect_model} + {config.consolidate_model}; no cache savings assumed)")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    from hermes.cache import ResponseCache
    from hermes.config import ConfigError, HermesConfig
    from hermes.graph import BudgetExceeded, run_scan
    from hermes.llm import AnthropicLLM
    from hermes.store import RunStore

    try:
        config = HermesConfig.resolve(args)
    except ConfigError as exc:
        _err("scan", str(exc))
        return 2
    spec = _load_spec_or_exit("scan", args.spec)
    if spec is None:
        return 2

    est = _estimate(spec, args, config)
    print(
        f"About to scan {est['operations']} operations of '{spec.title}': "
        f"{est['detect_calls']} detection calls + ~{est['consolidations_est']} consolidations, "
        f"estimated ~${est['cost_usd_est']} (cache hits reduce this).",
        file=sys.stderr,
    )
    if not args.yes:
        if not sys.stdin.isatty():
            _err("scan", "refusing to spend without --yes (non-interactive)")
            return 4
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            _err("scan", "declined")
            return 4

    if args.resume and not args.run_id:
        _err("scan", "--resume requires --run-id (resume = same run-id; the response cache replays completed work)")
        return 2
    run_id = args.run_id or "r_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    store = RunStore(config.out_dir, run_id)
    cache = ResponseCache(config.cache_db, read_enabled=not args.no_cache)
    llm = AnthropicLLM(config)
    try:
        meta = run_scan(
            spec=spec,
            llm=llm,
            cache=cache,
            config=config,
            store=store,
            tags=args.tags,
            path_globs=args.paths,
            sample=args.sample,
            max_endpoints=args.max_endpoints,
            consolidate_enabled=not args.no_consolidate,
        )
    except BudgetExceeded as exc:
        _err("scan", f"{exc} — run again with --run-id {run_id} --resume to continue (cache makes it cheap)")
        return 3
    except KeyboardInterrupt:
        _err("scan", f"interrupted — run again with --run-id {run_id} --resume to continue")
        return 3
    finally:
        cache.close()

    _render_report(store)
    counts, usage = meta["counts"], meta["usage"]
    print(f"run: {store.dir}")
    print(f"operations: {counts['operations_scanned']}  detections: {counts['detections']}  "
          f"avg smells/endpoint: {counts['avg_smells_per_endpoint']}  errors: {counts['detector_errors']}")
    print(f"calls: {usage['calls']} ({usage['cached_replays']} cached)  cost: ${usage['cost_usd']}")
    print(f"report: {store.dir / 'report.html'}")
    return 0


def _render_report(store) -> None:
    from hermes.report.render import render_run

    (store.dir / "report.html").write_text(render_run(store), encoding="utf-8")


def cmd_inspect(args: argparse.Namespace) -> int:
    from hermes.reducer import reduce_operation
    from hermes.spec_loader import find_operation

    spec = _load_spec_or_exit("inspect", args.spec)
    if spec is None:
        return 2
    try:
        op = find_operation(spec, args.endpoint)
    except KeyError as exc:
        # Unknown endpoint key = invalid invocation (00-SPEC §6 → exit 2).
        _err("inspect", exc.args[0])
        return 2
    print(reduce_operation(spec, op).yaml, end="")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from pathlib import Path

    from hermes.report.render import render_endpoint_md, render_run
    from hermes.store import RunStore

    run_dir = Path(args.run)
    if not (run_dir / "run.json").exists():
        _err("report", f"no run.json in {run_dir} — not a hermes run directory")
        return 2
    store = RunStore(run_dir.parent, run_dir.name)
    status = store.load_run_meta().get("status", "?")
    if status != "completed":
        # Interrupted re-runs rewrite only run.json; findings/report on disk may
        # belong to an earlier completed run under the same run-id.
        _err("report", f"warning: run status is {status!r} — artifacts may be stale "
                       "(from an earlier completed run with this run-id)")
    if args.md and not args.endpoint:
        _err("report", "--md requires --endpoint (Appendix-C markdown is per-endpoint)")
        return 2
    if args.endpoint:
        if not args.md:
            _err("report", "--endpoint requires --md (Appendix-C markdown report)")
            return 2
        try:
            print(render_endpoint_md(store, args.endpoint), end="")
        except KeyError as exc:
            _err("report", exc.args[0])
            return 2
        return 0
    out = Path(args.out) if args.out else run_dir / "report.html"
    out.write_text(render_run(store), encoding="utf-8")
    print(f"wrote {out}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    import json

    from hermes.eval.harness import StaleRecordingsError, format_report, run_live_eval, run_offline_eval

    import os

    from hermes.config import ConfigError, HermesConfig

    try:
        config = HermesConfig.resolve(args)
    except ConfigError as exc:
        _err("eval", str(exc))
        return 2
    try:
        if args.live:
            from hermes.llm import AnthropicLLM

            if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
                _err("eval", "--live needs credentials: set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN); "
                             "without them every call would 401 and burn the flake-shield retries")
                return 2
            llm = AnthropicLLM(config)
            result = run_live_eval(llm, detect_model=config.detect_model)
        else:
            result = run_offline_eval(expected_detect_model=config.detect_model)
    except (FileNotFoundError, StaleRecordingsError) as exc:
        _err("eval", str(exc))
        return 2
    print(format_report(result))
    if args.report:
        out = config.out_dir / "eval_report.json"  # resolved config: HERMES_OUT honored (CLAUDE.md rule 4)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    return 0 if not result["gate_failures"] else 1
