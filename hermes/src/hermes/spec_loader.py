"""Load OpenAPI/Swagger specs and enumerate operations (docs/00-SPEC.md §2, §6).

Supports Swagger 2.0 and OpenAPI 3.x, JSON or YAML. Does NOT resolve $refs —
that is the reducer's job (M2), and unresolvable refs are FRAGMENTED-smell
evidence there, never load errors here.
"""

from __future__ import annotations

import fnmatch
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# 'trace' is OAS3-only; Swagger 2.0 path items define only these seven.
SWAGGER2_METHODS = ("get", "put", "post", "delete", "options", "head", "patch")
OAS3_METHODS = SWAGGER2_METHODS + ("trace",)


class SpecError(Exception):
    """Unreadable, unparseable, or non-OpenAPI input; maps to CLI exit code 2."""


@dataclass(frozen=True)
class OperationRef:
    path: str
    method: str  # upper-case
    operation_id: str | None
    tags: tuple[str, ...]

    @property
    def endpoint_key(self) -> str:
        return f"{self.method} {self.path}"


@dataclass
class LoadedSpec:
    version: str  # "swagger2" | "oas3"
    title: str
    raw: dict
    operations: list[OperationRef] = field(default_factory=list)


def load_spec(path: str | Path) -> LoadedSpec:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:  # missing, unreadable, is-a-directory, decode via OSError
        raise SpecError(f"cannot read spec {path}: {exc}") from exc
    try:
        if path.suffix.lower() == ".json":
            raw = json.loads(text)
        else:
            raw = yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError, UnicodeDecodeError) as exc:
        raise SpecError(f"could not parse {path.name}: {exc}") from exc

    if not isinstance(raw, dict):
        raise SpecError(f"{path.name} is not an OpenAPI document (top level is not an object)")
    if "swagger" in raw:
        if str(raw["swagger"]) != "2.0":
            raise SpecError(f"unsupported swagger version {raw['swagger']!r} (only 2.0)")
        version = "swagger2"
    elif "openapi" in raw:
        if not str(raw["openapi"]).startswith("3."):
            raise SpecError(f"unsupported openapi version {raw['openapi']!r} (only 3.x)")
        version = "oas3"
    else:
        raise SpecError(f"{path.name} has neither 'swagger' nor 'openapi' field")

    paths = raw.get("paths")
    if not isinstance(paths, dict) or not paths:
        raise SpecError(f"{path.name} has no paths")

    methods = SWAGGER2_METHODS if version == "swagger2" else OAS3_METHODS
    operations: list[OperationRef] = []
    for p, item in paths.items():
        if not isinstance(item, dict):
            continue
        for method in methods:
            op = item.get(method)
            if not isinstance(op, dict):
                continue
            raw_tags = op.get("tags")
            if not isinstance(raw_tags, list):  # absent, null, or scalar → untagged
                raw_tags = []
            tags = tuple(str(t) for t in raw_tags if isinstance(t, str))
            operations.append(
                OperationRef(
                    path=str(p),
                    method=method.upper(),
                    operation_id=op.get("operationId"),
                    tags=tags,
                )
            )

    info = raw.get("info")
    if not isinstance(info, dict):
        info = {}
    title = str(info.get("title", path.stem))
    return LoadedSpec(version=version, title=title, raw=raw, operations=operations)


def filter_operations(
    operations: list[OperationRef],
    *,
    tags: list[str] | None = None,
    path_globs: list[str] | None = None,
    sample: int | None = None,
    seed: int | None = None,
    max_endpoints: int | None = None,
) -> list[OperationRef]:
    """Apply --tags / --paths / --sample / --max-endpoints (docs/00-SPEC.md §6).

    Filters preserve spec order. Path globs are case-sensitive and match the
    whole path string (`*` crosses `/`, so "/accounts/*" includes nested
    subresources). Sampling requires an explicit seed — defaults are owned by
    HermesConfig (see DECISIONS.md M0), not this module.
    """
    result = operations
    if tags:
        wanted = set(tags)
        result = [op for op in result if wanted.intersection(op.tags)]
    if path_globs:
        result = [op for op in result if any(fnmatch.fnmatchcase(op.path, g) for g in path_globs)]
    if sample is not None and sample < len(result):
        if seed is None:
            raise ValueError("sampling requires an explicit seed (resolve the default via HermesConfig)")
        result = _stratified_sample(result, sample, seed)
    if max_endpoints is not None:
        result = result[:max_endpoints]
    return result


def _stratified_sample(operations: list[OperationRef], n: int, seed: int) -> list[OperationRef]:
    """Seeded sample of n ops, round-robin across primary tags.

    Equal allocation, not proportional: every tag with at least one operation
    is represented once before any tag gets a second pick (for n >= number of
    tags). Deterministic for a given (operations, n, seed).
    """
    rng = random.Random(seed)
    buckets: dict[str, list[OperationRef]] = defaultdict(list)
    for op in operations:
        buckets[op.tags[0] if op.tags else ""].append(op)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    ordered_tags = sorted(buckets)  # deterministic bucket order
    picked: list[OperationRef] = []
    while len(picked) < n and any(buckets.values()):
        for tag in ordered_tags:
            if buckets[tag]:
                picked.append(buckets[tag].pop())
                if len(picked) == n:
                    break
    index = {op: i for i, op in enumerate(operations)}
    picked.sort(key=index.__getitem__)
    return picked
