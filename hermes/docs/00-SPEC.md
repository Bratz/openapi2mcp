# Hermes — Functional Specification

Reproduction of the system described in **"Making OpenAPI Documentation Agent-Ready: Detecting Documentation and REST Smells with a Multi-Agent LLM System"** (arXiv:2605.14312, EASE 2026), adapted to this repo's context (TCS BaNCS Swagger 2.0 spec → MCP tools).

**Scope decision (locked):** detection only, paper-faithful. Hermes detects smells and produces explainable diagnostic reports. It does NOT modify specs, generate fixes beyond textual suggestions, or regenerate MCP tools.

## 1. Purpose

Given an OpenAPI/Swagger spec, analyze **every operation (endpoint × HTTP method) individually** with specialized LLM agents and report documentation and REST design smells that impede AI-agent consumption (task planning, tool selection, payload construction). Output: machine-readable findings + a self-contained HTML dashboard.

## 2. Inputs

- Swagger 2.0 **and** OpenAPI 3.x JSON/YAML specs (Swagger 2.0 is the primary target — `api-docs.json`).
- Local `$ref`s must be resolved into each endpoint's reduced representation. External refs: resolve if trivially reachable on disk, otherwise record as a `ref_unresolved` note in the representation (do not crash).
- Specs up to at least 15 MB / 1,000 operations must be handled without loading the full document into any prompt.

## 3. Smell taxonomy (locked: paper's 9, hardcoded)

Nine smell categories, one specialized agent prompt each. Definitions below are the operational criteria each agent's prompt must encode (definition + classification criteria + 2–3 few-shot examples each, per the paper's few-shot prompting strategy).

### Documentation smells (category: `documentation`)

| ID | Name | Detect when (operationalization) |
|---|---|---|
| `LAZY` | Lazy documentation | Superficial/generic text: missing or <10-word summaries; descriptions that restate the operation name; undocumented parameters or request-body fields; generic response descriptions ("OK", "Success", "Error"); missing examples where the schema is non-trivial. |
| `BLOATED` | Bloated documentation | Excessively verbose text with low information density: descriptions >~150 words whose content could be stated in a fraction of the length; boilerplate repeated verbatim across fields; marketing/filler language that adds no operational guidance. |
| `TANGLED` | Tangled documentation | Unrelated concerns mixed in one textual fragment: a description that combines auth setup, business rules, error semantics, and changelog notes without structure; parameter docs that describe other parameters or global behavior. |
| `FRAGMENTED` | Fragmented documentation | Essential information dispersed across disconnected sections without explicit linkage: constraints stated only in a top-level/tag description but needed to call this endpoint; header requirements documented in an unrelated field; enum meanings defined elsewhere with no reference. NOTE: the agent only sees the reduced endpoint representation plus the spec-level context block (§5), so FRAGMENTED is judged against that context block. |

### REST smells (category: `rest`)

| ID | Name | Detect when (operationalization) |
|---|---|---|
| `PATH` | Path design smells | Non-resource-oriented or inconsistent paths: verbs in paths (`/getAccountDetails`), inconsistent casing/pluralization vs sibling paths, redundant path segments, CRUD-through-POST-only patterns encoded in the URL, ambiguous path parameter names (`/{id1}/{id2}`). |
| `METHOD` | HTTP method smells | Method semantics violated: GET with request body; POST used for pure reads; PUT/PATCH/DELETE semantics mismatched with the documented behavior; state-changing GET. |
| `INPUT` | Input modeling smells | Parameter/body design that breaks payload construction: required fields not marked required; stringly-typed fields that are really enums/dates/numbers; undocumented magic values; duplicated info between header/query/body; missing formats/patterns on constrained fields; giant flat objects with unclear optionality. |
| `RESPONSE` | Response modeling smells | Response contract unusable for planning: only a 200 defined with no error responses; responses with no schema; same schema for success and failure; status codes contradicting the description; untyped `object` payloads. |
| `SECURITY` | Security description smells | Auth requirements absent, contradictory, or undocumented at the operation level: no security scheme referenced although headers imply auth (e.g. `userId`, `entity` headers); documented auth headers not present in parameters; scheme referenced but never defined. |

An agent may report **0..n findings** for its (endpoint, smell) pair. Each finding must cite concrete evidence locations.

## 4. Finding schema (canonical JSON)

Everything downstream (store, dashboard, eval) consumes this shape. JSON Schema lives at `src/hermes/schemas/finding.schema.json` and is enforced on every agent response.

```json
{
  "id": "f_<sha1 of endpoint_key+smell+evidence[0].location+summary>",
  "run_id": "r_20260704T120000Z",
  "api_title": "TCS BaNCS RestFul API Documentation",
  "endpoint": {
    "path": "/accountManagement/account/balanceDetails",
    "method": "GET",
    "operation_id": "getBalanceDetails",
    "tags": ["Account Management"]
  },
  "smell": "LAZY",
  "category": "documentation",
  "severity": "medium",
  "confidence": 0.85,
  "summary": "Response descriptions are generic placeholders",
  "evidence": [
    {"location": "responses.200.description", "excerpt": "OK"}
  ],
  "justification": "1-3 sentences: why this impedes agent consumption",
  "suggestion": "1-3 sentences: concrete improvement",
  "detector": {"model": "claude-haiku-4-5", "prompt_version": "lazy-v1"}
}
```

Rules:
- `severity` ∈ {`low`, `medium`, `high`}: high = an agent will likely fail to call this endpoint correctly; medium = degraded tool selection/payload quality; low = cosmetic/consistency issue.
- `confidence` ∈ [0,1], produced by the agent.
- `evidence[].location` is a dot-path relative to the operation object (or `path`, `spec` for path-level/spec-level evidence). At least one evidence item required.
- `id` is deterministic so re-runs dedup naturally.

## 5. Reduced endpoint representation (ERD)

All agents for a given endpoint receive the **same** reduced representation (paper-faithful):

- Operation object with all local `$ref`s inlined, depth-capped at 4 levels; deeper schemas replaced by `{"$truncated": "<schema name>"}`.
- Sibling context: the endpoint's path item (other methods on the same path, names only), and up to 10 sibling path strings sharing the first path segment (for PATH consistency judgments).
- Spec-level context block: `info.title`, `info.description` (truncated to 1,500 chars), matching `tags[].description`, and `securityDefinitions`/`components.securitySchemes` names + types.
- Serialized as YAML (more token-efficient than JSON), deterministic key order.
- Hard cap: 6,000 tokens per ERD (measured with `count_tokens` once per calibration, approximated as chars/3.5 at runtime). Over-cap ERDs get schema bodies progressively truncated (deepest first) and a `truncation_applied: true` marker that is also copied into any finding's `detector` block.

## 6. CLI contract

Package name `hermes`, console script `hermes`.

```
hermes scan --spec PATH [--out DIR=runs/] [--run-id ID] [--resume]
            [--tags TAG ...] [--paths GLOB ...] [--sample N] [--seed 42]
            [--detect-model ID] [--consolidate-model ID]
            [--concurrency 8] [--max-endpoints N] [--yes]
hermes report --run DIR [--out report.html]     # re-render HTML from stored findings
hermes inspect --spec PATH --endpoint "GET /x"  # print the ERD that agents would see
hermes eval [--live] [--report]                 # golden-fixture metrics (see 02-TEST-PLAN)
hermes estimate --spec PATH [filters]           # endpoint count + call count + $ estimate, no API calls
```

Behavior requirements:
- `scan` prints an upfront estimate (endpoints, LLM calls, approximate cost) and requires `--yes` or interactive confirmation before spending money.
- `scan` writes `runs/<run_id>/findings.jsonl`, `runs/<run_id>/run.json` (config, counts, token usage, cost), `runs/<run_id>/report.html`.
- `--resume` continues an interrupted run: cached (endpoint, smell) results are not re-queried (see 01-ARCHITECTURE §caching).
- Exit codes: 0 success; 2 spec parse failure; 3 interrupted (resumable); 4 budget/confirmation declined.

## 7. HTML dashboard (locked: static, self-contained)

Single `report.html`, no external requests (inline CSS/JS, no CDN). Must render from `file://`.

Required elements:
1. Header: API title, spec version, run timestamp, model(s) used, totals (endpoints scanned, findings, avg findings/endpoint — the paper's headline metric).
2. Summary cards: findings per smell (9 bars/cards), findings per severity, top-10 worst endpoints.
3. Findings table: columns method, path, smell, severity, confidence, summary. Client-side filters: smell, category, severity, tag, free-text path search. Sortable.
4. Row expansion: evidence excerpts with locations, justification, suggestion.
5. Per-tag (business domain) rollup table.
6. Embedded raw data: findings JSON in a `<script type="application/json">` block so the file doubles as the data export.

Constraint: must stay usable at 5,000 findings (virtualized or paginated table — pagination is acceptable and simpler).

## 8. Consolidation stage

After per-(endpoint, smell) detection, a consolidator pass per endpoint (single Sonnet call, only when the endpoint has ≥2 findings):
- Dedups overlapping findings across smell categories (e.g. same missing description flagged by LAZY and INPUT — keep the more specific, note the merge).
- Normalizes severity across agents.
- Produces a 1-sentence endpoint verdict ("agent-ready" / "needs adaptation" / "not agent-consumable") recorded in `runs/<id>/endpoints.jsonl`.
Consolidation must never invent findings; it may only merge, drop-as-duplicate, or adjust severity by one step (with reason recorded).

## 9. Non-goals

- No remediation/spec rewriting.
- No live HTTP calls to the described APIs.
- No web service; CLI + static HTML only.
- No support for GraphQL/AsyncAPI.

## 10. Fidelity notes vs the paper

The paper's full text was not network-accessible when this spec was written; taxonomy names, the orchestrator/specialist architecture, shared reduced endpoint representation, few-shot structured prompting, and explainable per-finding output (justification + suggestion) are sourced from the abstract and indexed excerpts. The per-smell operational criteria in §3 are this project's own operationalization and MUST be treated as tunable (prompt files carry `prompt_version` for this reason). Record any deliberate deviation in `hermes/docs/DECISIONS.md`.
