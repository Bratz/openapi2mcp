"""Reduced endpoint representation — ERD (docs/00-SPEC.md §5, paper Appendix B).

Every smell agent for an operation receives the same ERD: method, path,
summary, description, parameters, request body, responses, referenced
schemas (inlined), and security context — nothing else from the document.

Engineering rules:
- Local ``$ref``s are inlined, depth-capped; deeper chains become
  ``{"$truncated": <name>}``. Unresolvable refs become
  ``{"$unresolved": <ref>}`` — FRAGMENTED-smell evidence, never an error.
- Deterministic YAML (stable key order) so cache keys and snapshots hold.
- Hard token cap (~chars/3.5); over-cap ERDs are re-rendered with smaller
  inline depth, then capped descriptions, and flagged ``truncation_applied``.
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from hermes.spec_loader import LoadedSpec, OperationRef

MAX_INLINE_DEPTH = 4
# Hostile-input guard: total structural recursion depth (plain inline nesting,
# no $refs needed) — far above anything a real spec produces, far below
# Python's recursion limit. Deeper nodes collapse to $truncated markers.
MAX_STRUCTURAL_DEPTH = 60
MAX_ERD_TOKENS = 6000
CHARS_PER_TOKEN = 3.5
MAX_API_DESCRIPTION_CHARS = 1500
MAX_TAG_DESCRIPTION_CHARS = 500
MAX_SIBLING_PATHS = 10
_DESCRIPTION_CAP_WHEN_TRUNCATING = 500

# Operation keys copied into the ERD (both flavors; absent keys are skipped).
_OPERATION_KEYS = (
    "operationId",
    "summary",
    "description",
    "deprecated",
    "parameters",
    "requestBody",  # OAS3
    "consumes",     # Swagger 2.0
    "produces",     # Swagger 2.0
    "responses",
    "security",
)


@dataclass
class ERD:
    yaml: str
    truncation_applied: bool


class _Resolver:
    """Inline local JSON-pointer $refs with depth cap and cycle guard."""

    def __init__(self, root: dict, max_depth: int):
        self.root = root
        self.max_depth = max_depth

    def resolve(self, node, depth: int = 0, active: tuple[str, ...] = (), spine: int = 0):
        if isinstance(node, dict):
            if spine >= MAX_STRUCTURAL_DEPTH:  # unbounded inline nesting, no $ref needed
                return {"$truncated": "max nesting depth"}
            ref = node.get("$ref")
            if isinstance(ref, str):
                return self._resolve_ref(ref, depth, active, spine)
            return {k: self.resolve(v, depth, active, spine + 1) for k, v in node.items()}
        if isinstance(node, list):
            if spine >= MAX_STRUCTURAL_DEPTH:
                return ["$truncated: max nesting depth"]
            return [self.resolve(v, depth, active, spine + 1) for v in node]
        return node

    def _resolve_ref(self, ref: str, depth: int, active: tuple[str, ...], spine: int = 0):
        name = ref.rsplit("/", 1)[-1]
        if not ref.startswith("#/"):
            return {"$unresolved": ref}
        if ref in active:  # cycle
            return {"$truncated": name}
        if depth >= self.max_depth:
            return {"$truncated": name}
        target = self.root
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(target, dict) and part in target:
                target = target[part]
            else:
                return {"$unresolved": ref}
        return self.resolve(target, depth + 1, active + (ref,), spine + 1)


def reduce_operation(spec: LoadedSpec, op: OperationRef) -> ERD:
    """Build the ERD for one operation. Pure and deterministic.

    The 6,000-token cap is enforced on the FULL rendered document (endpoint +
    operation + context + flag line), not just the operation subtree.
    """
    for max_depth in (MAX_INLINE_DEPTH, 3, 2, 1):
        doc = _build(spec, op, max_depth=max_depth, cap_descriptions=False)
        truncated = max_depth != MAX_INLINE_DEPTH
        if truncated:
            doc["truncation_applied"] = True  # in place BEFORE measuring
        text = _dump(doc)
        if _tokens(text) <= MAX_ERD_TOKENS:
            return ERD(yaml=text, truncation_applied=truncated)
    doc = _build(spec, op, max_depth=1, cap_descriptions=True)
    doc["truncation_applied"] = True
    _truncate_until_fits(doc)
    return ERD(yaml=_dump(doc), truncation_applied=True)


def _truncate_until_fits(doc: dict) -> None:
    """Last-resort cap: collapse the largest `properties` maps and long `enum`
    lists, deepest first, until the FULL rendered ERD fits (docs/00-SPEC.md §5:
    schema bodies truncated deepest first). If every candidate is exhausted and
    the document still exceeds the cap (bulk outside schemas), it is returned
    over-cap with truncation_applied=True — see DECISIONS.md M2."""
    while _tokens(_dump(doc)) > MAX_ERD_TOKENS:
        candidates: list[tuple[int, int, dict, str]] = []  # (depth, size, parent, kind)
        _collect_truncatable(doc["operation"], 0, candidates)
        if not candidates:
            break
        candidates.sort(key=lambda c: (-c[0], -c[1]))
        _, _, node, kind = candidates[0]
        if kind == "properties":
            node["properties"] = {"$truncated": f"{len(node['properties'])} properties omitted"}
        else:  # enum
            kept = node["enum"][:10]
            node["enum"] = kept + [f"$truncated: {len(node['enum']) - 10} more values"]


def _collect_truncatable(node, depth: int, out: list) -> None:
    if isinstance(node, dict):
        props = node.get("properties")
        if isinstance(props, dict) and props and "$truncated" not in props:
            out.append((depth, len(_dump(props)), node, "properties"))
        enum = node.get("enum")
        if isinstance(enum, list) and len(enum) > 25:
            out.append((depth, len(_dump({"enum": enum})), node, "enum"))
        for v in node.values():
            _collect_truncatable(v, depth + 1, out)
    elif isinstance(node, list):
        for v in node:
            _collect_truncatable(v, depth + 1, out)


def _build(spec: LoadedSpec, op: OperationRef, *, max_depth: int, cap_descriptions: bool) -> dict:
    raw = spec.raw
    path_item = raw.get("paths", {}).get(op.path, {})
    operation = path_item.get(op.method.lower(), {})
    resolver = _Resolver(raw, max_depth)

    reduced_op: dict = {}
    for key in _OPERATION_KEYS:
        if key in operation:
            reduced_op[key] = resolver.resolve(operation[key])
    # Path-level parameters apply to every operation on the path (both flavors).
    if isinstance(path_item.get("parameters"), list):
        reduced_op["path_level_parameters"] = resolver.resolve(path_item["parameters"])
    if cap_descriptions:
        _cap_strings(reduced_op, "description", _DESCRIPTION_CAP_WHEN_TRUNCATING)

    return {
        "endpoint": {
            "method": op.method,
            "path": op.path,
            "operation_id": op.operation_id,
            "tags": list(op.tags),
        },
        "operation": reduced_op,
        "context": _context(spec, op, path_item),
    }


def _context(spec: LoadedSpec, op: OperationRef, path_item: dict) -> dict:
    raw = spec.raw
    info = raw.get("info") if isinstance(raw.get("info"), dict) else {}
    api_description = info.get("description")
    if isinstance(api_description, str) and len(api_description) > MAX_API_DESCRIPTION_CHARS:
        api_description = api_description[:MAX_API_DESCRIPTION_CHARS]

    raw_tags = raw.get("tags")
    if not isinstance(raw_tags, list):  # truthy non-list tags must not crash the scan
        raw_tags = []
    tag_descriptions = {
        t["name"]: t["description"][:MAX_TAG_DESCRIPTION_CHARS]
        for t in raw_tags
        if isinstance(t, dict) and t.get("name") in op.tags and isinstance(t.get("description"), str)
    }

    if spec.version == "swagger2":
        schemes = raw.get("securityDefinitions")
    else:
        components = raw.get("components") if isinstance(raw.get("components"), dict) else {}
        schemes = components.get("securitySchemes")
    if not isinstance(schemes, dict):  # truthy non-dict must not crash the scan
        schemes = {}
    security_schemes = {
        name: str(defn.get("type", "?")) for name, defn in schemes.items() if isinstance(defn, dict)
    }

    other_methods = sorted(
        m.upper() for m in path_item if m != op.method.lower() and isinstance(path_item.get(m), dict)
        and m in ("get", "put", "post", "delete", "options", "head", "patch", "trace")
    )
    first_segment = op.path.lstrip("/").split("/")[0]
    related = sorted(
        p for p in raw.get("paths", {})
        if p != op.path and str(p).lstrip("/").split("/")[0] == first_segment
    )[:MAX_SIBLING_PATHS]

    return {
        "api_title": spec.title,
        "api_description": api_description,
        "tag_descriptions": tag_descriptions,
        "security_schemes": security_schemes,
        "path_siblings": {"other_methods": other_methods, "related_paths": related},
    }


def _cap_strings(node, key: str, limit: int) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key and isinstance(v, str) and len(v) > limit:
                node[k] = v[:limit]
            else:
                _cap_strings(v, key, limit)
    elif isinstance(node, list):
        for v in node:
            _cap_strings(v, key, limit)


def _tokens(text: str) -> float:
    return len(text) / CHARS_PER_TOKEN


def _dump(doc: dict) -> str:
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100)
