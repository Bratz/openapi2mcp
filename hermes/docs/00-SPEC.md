# Hermes — Functional Specification

Reproduction of the system described in **"Making OpenAPI Documentation Agent-Ready: Detecting Documentation and REST Smells with a Multi-Agent LLM System"** (arXiv:2605.14312, EASE 2026), adapted to this repo's context (TCS BaNCS Swagger 2.0 spec → MCP tools). Paper ground truth is condensed in `05-PAPER-FACTS.md`; deliberate deviations are logged in `DECISIONS.md`.

**Scope decision (locked):** detection only, paper-faithful. Hermes detects smells and produces explainable diagnostic reports. It does NOT modify specs, generate fixes beyond textual suggestions, or regenerate MCP tools.

## 1. Purpose

Given an OpenAPI/Swagger spec, analyze **every operation (endpoint × HTTP method) individually** with specialized LLM agents and report documentation and REST design smells that impede AI-agent consumption (task planning, tool selection, payload construction). Output: machine-readable results + Appendix-C-style per-endpoint diagnostic reports + a self-contained HTML dashboard.

## 2. Inputs

- Swagger 2.0 **and** OpenAPI 3.x JSON/YAML specs (Swagger 2.0 is the primary target — `api-docs.json`).
- Local `$ref`s must be resolved into each endpoint's reduced representation. **Unresolvable references are not an error — they are FRAGMENTED-smell evidence** (paper Table 3): record them in the ERD as `{"$unresolved": "<ref>"}` so the FRAGMENTED agent can see them.
- Specs up to at least 15 MB / 1,000 operations must be handled without loading the full document into any prompt.

## 3. Smell taxonomy (locked: paper-actual, 9 agents)

Nine specialized agents, matching the system as it actually ran in the paper (Table 3 + Appendix C — see `05-PAPER-FACTS.md` §2–3): **5 documentation smells** (Khan et al.'s taxonomy) + **4 REST smells** (PATH and METHOD merged, as in the paper's reporting and diagnostic reports).

Each agent's prompt follows the paper's Appendix A template (definition + "typically occurs when" examples + classification criteria + per-smell scoping rules). Criteria below combine the paper's definitions with its Table 3 evidence patterns.

### Documentation smells (category: `documentation`)

| ID | Name | Detect when | Scoping rule |
|---|---|---|---|
| `LAZY` | Lazy documentation | Incomplete, vague, or generic documentation: very short/generic `summary` (e.g. "Get data"); absent or redundant `description`; semantic mismatch between summary and description; no examples or constraints; intent requires external knowledge. | Analyze ONLY the operation's `summary` and `description` (per Appendix A). |
| `BLOATED` | Bloated documentation | Excessively verbose descriptions with limited informational value; filler/boilerplate that adds no operational guidance. | Textual fields only. |
| `TANGLED` | Tangled documentation | Unrelated concerns (business logic, security, error handling, changelog) mixed within the same textual fragment. | Textual fields only. |
| `FRAGMENTED` | Fragmented documentation | **Structural**: the operation references schemas/components not present in the specification (broken/missing `$ref`s → incomplete documentation). Largely mechanically checkable — the reducer surfaces `$unresolved` markers; the agent confirms and explains. | Whole ERD; keys on `$unresolved` markers. |
| `EXCESS_STRUCTURED` | Excess structured information | Class-like definitions, nested structures, or formal specifications written **inside `summary`/`description` natural-language fields**. Normative schema structure under `definitions`/`components` is NOT the smell. | Textual fields only. |

### REST smells (category: `rest`)

| ID | Name | Detect when | Scoping rule |
|---|---|---|---|
| `PATH_AND_METHOD` | Path & Method design | Action-oriented URIs with verbs (`/getUsers`, `/updateStatus`); inconsistent naming/casing vs sibling paths; long or opaque paths with internal acronyms; non-idiomatic HTTP method use (POST for updates that should be PUT/PATCH, GET for creation, state-changing GET, GET with request body). | Path string, method, sibling-path context. |
| `INPUT` | Input modeling | Parameters/bodies specified only by type with no semantic description; ambiguous names (abbreviations, internal acronyms) unexplained; missing ranges/formats/patterns; required fields without rationale or not marked required; magic values undocumented. | Parameters + request body. |
| `RESPONSE` | Response modeling | Schemas present but minimal explanatory text; generic descriptions ("Successful Response", "OK"); no semantic clarification of payload meaning across success/error; missing error responses; unconstrained `object` payloads (the paper's `{status, data:object}` anti-pattern); status codes contradicting descriptions. | Responses section. |
| `SECURITY` | Security description | Auth missing or unclear at operation level: no scheme referenced although parameters imply auth (e.g. `userId`, `entity` headers); scheme defined but no operational guidance (how to obtain credentials, scopes, constraints); scheme referenced but never defined. | Security refs + parameters + spec-level scheme context. |

Detection is **binary per (operation, smell)** — multi-label classification per endpoint, exactly as the paper formulates it. A detected smell carries justification bullets and suggested actions (schema below).

## 4. Result schema (canonical JSON)

One record per (operation, smell) **detection** (non-detections are recorded in run stats, not as records). JSON Schema at `src/hermes/schemas/finding.schema.json`, enforced on every agent response.

```json
{
  "id": "f_<first 16 hex chars of sha1('<endpoint_key>|<smell>')>",
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
  "justification": [
    "The summary is vague and the endpoint has no description, providing insufficient guidance about purpose, behavior, or usage."
  ],
  "suggestions": [
    {
      "action_title": "[LAZY] - Improve documentation",
      "description": "Provide a complete description including endpoint purpose, expected inputs, and outputs, with usage examples."
    }
  ],
  "extensions": {
    "severity": "medium",
    "confidence": 0.85,
    "evidence": [{"location": "summary", "excerpt": "Get data"}]
  },
  "detector": {"model": "claude-haiku-4-5", "prompt_version": "lazy-v1"}
}
```

Rules:
- **Paper core** (always present, used by eval): `smell`, `justification` (≥1 bullet, complete sentences, section total ≥120 chars per the paper's explanation rule), `suggestions` (≥1, `action_title` matching `[<SMELL>] - <title>`).
- **`extensions`** (locked decision): severity ∈ {low, medium, high}, confidence ∈ [0,1], evidence dot-path locations. Used for dashboard triage and ranking; **never used in eval scoring** (the eval oracle is multi-label per endpoint).
- `id` is deterministic (`endpoint_key + smell`), so re-runs dedup naturally; multi-label detection means at most one record per (operation, smell).
- Post-validation in code: clamp confidence, enforce action-title format (rewrite prefix if the model got it wrong), stamp `detector`.

## 5. Reduced endpoint representation (ERD)

All agents for a given operation receive the **same** reduced representation (paper §3.5 / Appendix B): method, path, summary, description, parameters, request body, responses, referenced schemas, and security definitions — nothing else from the document, except the engineering additions below.

Engineering additions (ours, for scale and context):
- Document-level context: `api_title`, `info.description` (capped at 1,500 chars), and the descriptions of the operation's tags; operation metadata (`operationId`, `deprecated`, `consumes`/`produces`) retained. Smells like SECURITY/RESPONSE need document-level conventions to judge an operation fairly. (Deviation from a strict Appendix-B reading; DECISIONS M2.)
- Local `$ref`s inlined, depth-capped at 4 levels; deeper schemas replaced by `{"$truncated": "<schema name>"}`; **unresolvable refs replaced by `{"$unresolved": "<ref>"}`** (FRAGMENTED evidence, never a crash).
- Sibling context for PATH_AND_METHOD: other methods on the same path (names only) + up to 10 sibling path strings sharing the first path segment.
- Spec-level security context: `securityDefinitions`/`components.securitySchemes` names + types (needed by SECURITY).
- Serialized as YAML, deterministic key order. Cap 6,000 tokens (chars/3.5 heuristic); over-cap ERDs get schema bodies progressively truncated (deepest first) with `truncation_applied: true` copied into any resulting record's `detector` block. The cap is best-effort: when no `properties`/`enum` collapse candidates remain, an ERD may exceed it (still flagged; accepted residual risk, DECISIONS M2).

## 6. CLI contract

Package name `hermes`, console script `hermes`.

```
hermes scan --spec PATH [--out DIR=runs/] [--run-id ID] [--resume]
            [--tags TAG ...] [--paths GLOB ...] [--sample N] [--seed 42]
            [--detect-model ID] [--consolidate-model ID]
            [--concurrency 8] [--max-endpoints N] [--yes]
hermes report --run DIR [--out report.html]            # re-render dashboard
hermes report --run DIR --endpoint "GET /x" --md       # Appendix-C-style markdown diagnostic report
hermes inspect --spec PATH --endpoint "GET /x"         # print the ERD agents would see
hermes eval [--live] [--report]                        # golden-fixture multi-label metrics (02-TEST-PLAN)
hermes estimate --spec PATH [filters]                  # endpoint/call counts + $ estimate, no API calls
```

Behavior requirements:
- `scan` prints an upfront estimate (endpoints, LLM calls = ops × 9 + consolidations, approximate cost) and requires `--yes` or interactive confirmation before spending money.
- `scan` writes `runs/<run_id>/findings.jsonl`, `endpoints.jsonl` (per-endpoint smell-set + verdict), `run.json` (config, counts, token usage, cost), `report.html`.
- `--resume` continues an interrupted run via the content-hash cache (01-ARCHITECTURE §4). It requires an explicit `--run-id` (exit 2 otherwise; DECISIONS M5), and the caller must repeat the original run's filter flags — filters are echoed in `run.json` but not re-applied automatically.
- Exit codes: 0 success; 2 invalid invocation or spec parse failure (argparse usage errors share this code); 3 interrupted (resumable); 4 budget/confirmation declined.

## 7. Reports

### 7.1 Per-endpoint diagnostic report (paper Appendix C format)

For any endpoint, Hermes can emit the paper's markdown report shape:

```
## API Info          – title
## Endpoint Info     – method, path
## Model             – detect model id
## Identified Smells – comma list (display names, e.g. "Lazy, Security, Path & Method")
### Explanations     – one "### <Smell>" block per detected smell, justification bullets
## Improvement Suggestions
[SMELL] - <action title> | <description>
```

Available via `hermes report --endpoint ... --md` and rendered inside the dashboard's row expansion.

### 7.2 HTML dashboard (locked: static, self-contained)

Single `report.html`, no external requests (inline CSS/JS, no CDN). Must render from `file://`.

1. Header: API title, spec version, run timestamp, model(s), totals (endpoints scanned, detections, **avg smells/endpoint** — comparable to the paper's 4.08).
2. Summary cards: detections per smell (9), per severity (extension field), % of endpoints affected per smell (comparable to paper Table 2), top-10 worst endpoints.
3. Findings table: method, path, smell, severity, confidence, first justification bullet. Client-side filters: smell, category, severity, tag, free-text path search. Sortable.
4. Row expansion: the Appendix-C-style report for that endpoint (justifications + suggestions), plus extension evidence excerpts.
5. Per-tag (business domain) rollup table.
6. Embedded raw data: findings JSON in a `<script type="application/json">` block so the file doubles as the data export.

Constraint: usable at 5,000 findings (pagination is acceptable).

## 8. Consolidation

The paper's central **Smell Detector Agent** "orchestrates the workflow and consolidates results into a unified explainable diagnostic report". In this implementation:

- **Orchestration + aggregation is code** (the LangGraph graph collects the 9 agents' outputs into the unified per-endpoint report). This is the paper-faithful core.
- **Optional extension** (kept from our design, marked as such): for endpoints with ≥2 detections, a single Sonnet call may normalize overlapping justifications and produce a 1-sentence endpoint verdict ("agent-ready" / "needs adaptation" / "not agent-consumable") stored in `endpoints.jsonl`. It may only merge/normalize — never invent detections. Disable with `--no-consolidate`.

## 9. Non-goals

- No remediation/spec rewriting.
- No live HTTP calls to the described APIs.
- No web service; CLI + static HTML only.
- No support for GraphQL/AsyncAPI.

## 10. Paper fidelity

`05-PAPER-FACTS.md` is the authoritative extract of the paper (taxonomy, prompt template, report format, baseline results). Deviations are deliberate and logged in `DECISIONS.md`: Claude API backend (paper: local gpt-oss:120b), CLI + static dashboard (paper: interactive URL-driven tool), synthetic golden fixtures (paper: 60 expert-annotated production endpoints), and the `extensions` fields in §4.
