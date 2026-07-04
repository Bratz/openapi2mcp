# Hermes — Architecture

Locked decisions: **LangGraph** orchestration, **Claude API** (Haiku 4.5 fan-out + Sonnet 5 consolidation), content-hash **caching + resumable runs**, **static HTML** rendering. Python ≥3.11. Paper facts in `05-PAPER-FACTS.md`.

## 1. Module layout

```
hermes/
├── pyproject.toml               # name `hermes`, deps below, [project.scripts] hermes = "hermes.cli:main"
├── docs/                        # these documents
├── runs/                        # scan outputs (gitignored)
├── src/hermes/
│   ├── cli.py                   # argparse CLI (spec §6); import-light, dispatches to commands.py
│   ├── commands.py              # thin arg→module adapters; real logic lives in graph/report/eval
│   ├── config.py                # HermesConfig dataclass: models, concurrency, budgets; env overrides
│   ├── spec_loader.py           # load+validate Swagger2/OAS3, enumerate Operation records
│   ├── reducer.py               # Operation -> ERD (spec §5): ref inlining, $unresolved markers, caps, YAML
│   ├── smells/
│   │   ├── catalog.py           # the 9 hardcoded Smell objects (definition, scoping, occurs_when, guards)
│   │   └── prompts/__init__.py  # Appendix-A template assembly + PROMPT_VERSIONS dict (see DECISIONS M3)
│   ├── llm.py                   # Anthropic client wrapper: structured output, retry, usage accounting
│   ├── schemas/
│   │   ├── finding.schema.json
│   │   └── models.py            # Pydantic: Finding, Suggestion, Extensions, AgentResponse, EndpointVerdict
│   ├── graph.py                 # LangGraph StateGraph wiring (below)
│   ├── nodes.py                 # node implementations (pure functions over state)
│   ├── cache.py                 # sqlite content-hash cache
│   ├── store.py                 # findings.jsonl / endpoints.jsonl / run.json persistence
│   ├── report/
│   │   ├── render.py            # dashboard (template.html) + Appendix-C markdown per endpoint
│   │   └── template.html        # single Jinja2 template, inline CSS/JS
│   └── eval/
│       ├── harness.py           # run detection over golden fixtures, score, gate
│       └── metrics.py           # multi-label: Jaccard, F1-micro/macro, Hamming, cardinality diff; per-smell P/R
└── tests/                       # see 02-TEST-PLAN.md
```

Dependencies (keep minimal): `anthropic`, `langgraph`, `pydantic>=2`, `PyYAML`, `jinja2`, `prance` (optional — fall back to internal resolver). Dev: `pytest`, `pytest-asyncio`.

## 2. Smell catalog

`smells/catalog.py` defines exactly **9** smells (spec §3): `LAZY`, `BLOATED`, `TANGLED`, `FRAGMENTED`, `EXCESS_STRUCTURED` (documentation) and `PATH_AND_METHOD`, `INPUT`, `RESPONSE`, `SECURITY` (rest). Each `Smell` carries: `id`, `category`, `display_name` (e.g. "Path & Method" for reports), `definition`, `scoping_rule` (the "Analyze ONLY …" clause), and `occurs_when` examples (the Appendix-A `{examples}` slot / few-shots). Fan-out per scan = |operations| × 9.

## 3. LangGraph design

One `StateGraph` per scan run.

### State (TypedDict)

```python
class ScanState(TypedDict):
    config: HermesConfig
    spec_path: str
    operations: list[OperationRef]        # after load
    erds: dict[str, str]                  # endpoint_key -> ERD yaml (or on-disk paths for big runs)
    pending: list[tuple[str, str]]        # (endpoint_key, smell_id) not yet detected
    findings: Annotated[list[Finding], operator.add]
    verdicts: Annotated[list[EndpointVerdict], operator.add]
    usage: Annotated[list[UsageRecord], operator.add]   # per-call tokens+cost
```

### Nodes and edges

```
load_spec ──> reduce ──> plan ──(Send fan-out)──> detect ──> collect ──> consolidate ──> persist ──> render
```

- `load_spec`: parse spec, enumerate operations, apply `--tags/--paths/--sample` filters.
- `reduce`: build ERD per operation (pure, deterministic — unit-testable without LLM). Unresolvable refs become `$unresolved` markers (FRAGMENTED evidence), never exceptions.
- `plan`: cross-product operations × 9 smells, minus cache hits (cache hits emit their stored results directly into state).
- `detect`: **map node via LangGraph `Send`** — one task per (endpoint, smell). Executes the smell agent (§4), validates output, writes to cache. Concurrency bounded by a semaphore in `llm.py` (default 8), so rate limits are respected regardless of fan-out size.
- `collect`: group detections per endpoint (this is the code half of the paper's "central Smell Detector Agent consolidates results" — it assembles the unified per-endpoint diagnostic report).
- `consolidate` (extension, `--no-consolidate` to skip): per endpoint with ≥2 detections, one Sonnet call to normalize justifications and produce the endpoint verdict. Endpoints with 0–1 detections get a rule-based verdict.
- `persist`: append to `findings.jsonl` / `endpoints.jsonl` (idempotent by id), write `run.json` with usage/cost rollup.
- `render`: `report/render.py` → `report.html`.

### Checkpointing / resume

- `langgraph.checkpoint.sqlite.SqliteSaver` at `runs/<run_id>/checkpoint.db`, `thread_id = run_id`.
- `--resume` re-invokes with the same `thread_id`; the content-hash cache is the primary resume mechanism, the checkpointer is belt-and-suspenders. If SqliteSaver fights the Send API, drop it and rely on cache-only resume — record in DECISIONS.md.

## 4. Smell agent (detect node internals)

One LLM call per (endpoint, smell):

- **Model:** `claude-haiku-4-5` (config: `detect_model`).
- **System prompt** (stable, cacheable — `cache_control: {"type": "ephemeral"}`, byte-stable, no timestamps/run ids): assembled per the paper's **Appendix A template**, in order:
  1. Role ("You are an expert in identifying <Smell> …").
  2. Smell Definition.
  3. "This smell typically occurs when:" + the catalog's `occurs_when` examples (few-shot slot).
  4. Task statement.
  5. Classification Rules — including the smell's `scoping_rule` (e.g. LAZY: "Analyze ONLY the method summary and description") and detected/not-detected semantics.
  6. Explanation and Improvement Rules — justification bullets as complete sentences, ≥120 chars total per section; suggestions with `action_title` in exact format `[<SMELL>] - <action title>`.
- **User message:** the ERD (the Appendix-A `{openapi_json}` slot, as YAML).
- **Structured output:** `client.messages.parse()` with Pydantic (replaces the paper's "return ONLY valid JSON" prose rule — same intent, mechanically enforced):

```python
class Suggestion(BaseModel):
    action_title: str                 # "[LAZY] - Improve documentation"
    description: str

class Extensions(BaseModel):
    severity: Literal["low", "medium", "high"]
    confidence: float
    evidence: list[Evidence]          # location dot-path + excerpt

class AgentResponse(BaseModel):
    smell_detected: bool
    justification: list[str]          # empty when not detected
    suggestions: list[Suggestion]     # empty when not detected
    extensions: Extensions | None     # required when detected
```

- `max_tokens`: 2000. Omit sampling params.
- **Retry policy:** SDK default retries for 429/5xx; one application-level retry on schema-validation failure with the validation error appended. Then record `detector_error` in `run.json` and continue — never abort the scan for one endpoint.
- Post-validation in code: clamp confidence to [0,1]; enforce/repair the `[<SMELL>] - ` action-title prefix; enforce ≥120-char justification total (below threshold → one retry, then accept with `short_justification: true` flag); compute deterministic `id`; stamp `detector` metadata.

### Consolidator agent (extension)

- **Model:** `claude-sonnet-5` (config: `consolidate_model`).
- Input: endpoint path/method/tags + its detections as JSON. Output (parse): normalization edits + endpoint verdict. Applied in code; raw agent outputs retained in `findings.raw.jsonl` for audit. May not invent detections.

### Model choice (recorded deviation)

The paper selected **gpt-oss:120b** after a 7-model local bake-off because confidentiality policy barred external APIs (see 05-PAPER-FACTS §1). Our target spec is a public demo spec, so we use the Claude API for detection quality and reliable structured output; the `llm.py` seam keeps a future local/pluggable backend cheap to add. Logged in DECISIONS.md.

## 5. Caching

sqlite table `cache(key TEXT PRIMARY KEY, response_json TEXT, model TEXT, created_at TEXT)` at `runs/cache.db` (shared across runs — safe because the key includes everything that affects output):

```
key = sha256(erd_yaml + smell_id + PROMPT_VERSION[smell] + model_id)
```

- Hit → stored `AgentResponse` replayed, zero API calls, `"cached": true` in usage records.
- A prompt edit bumps that smell's `PROMPT_VERSION`, invalidating only that smell's entries.
- `hermes scan --no-cache` bypasses reads (still writes).

## 6. Cost & rate control

- `hermes estimate` and the pre-scan confirmation compute: `calls = |operations| × 9 + est. consolidations`; input ≈ ERD tokens + cache-read system prompt; output ≈ 400/call. Full 927-operation BaNCS scan ≈ 8.4k calls, order of **$15–30** with Haiku 4.5 ($1/$5 per MTok) + prompt caching; print the computed number, not this constant.
- Semaphore concurrency 8 default; on `RateLimitError` the whole pool pauses ~60s before taking new slots (simple full-pause backoff; SDK per-call retries still apply — see DECISIONS M4).
- `--max-endpoints`, `--sample N --seed S` (stratified by tag when possible) for cheap partial scans.
- Usage accounting: every call appends `UsageRecord(model, input_tokens, output_tokens, cache_read_input_tokens, cost_usd)` from `response.usage`; `run.json` carries the rollup. Price table (Haiku $1/$5, Sonnet $3/$15 per MTok) lives in `config.py` with a drift comment.

## 7. Report rendering

- Pure function: `render(run_dir) -> report.html`; `render_endpoint_md(run_dir, endpoint_key) -> str` for the Appendix-C markdown format (spec §7.1). Jinja2 template, findings embedded as JSON, vanilla JS filter/sort/paginate (page size 100). No network fetches — test greps output for `http(s)://` in src/href (none allowed except inside finding text).

## 8. Error handling philosophy

- Spec-level failures (unparseable file) fail fast with exit 2.
- Per-endpoint failures (ref bomb, oversized ERD, agent schema failure after retry) degrade to recorded warnings; the scan always completes and the report shows a "skipped/errored" section.
- All LLM calls go through `llm.py`; nothing else imports `anthropic`. This is the single mocking seam for tests.
