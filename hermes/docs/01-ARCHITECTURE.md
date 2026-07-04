# Hermes — Architecture

Locked decisions: **LangGraph** orchestration, **Claude API** (Haiku 4.5 fan-out + Sonnet consolidation), content-hash **caching + resumable runs**, **static HTML** rendering. Python ≥3.11.

## 1. Module layout

```
hermes/
├── pyproject.toml               # project + deps + [project.scripts] hermes = "hermes.cli:main"
├── docs/                        # these documents
├── runs/                        # scan outputs (gitignored)
├── src/hermes/
│   ├── cli.py                   # argparse CLI (spec §6); thin — delegates to graph/report/eval
│   ├── config.py                # HermesConfig dataclass: models, concurrency, budgets; env overrides
│   ├── spec_loader.py           # load+validate Swagger2/OAS3, enumerate Operation records
│   ├── reducer.py               # Operation -> ERD (spec §5): ref inlining, depth cap, token cap, YAML
│   ├── smells/
│   │   ├── catalog.py           # the 9 hardcoded Smell objects (id, category, definition, criteria)
│   │   └── prompts/             # one file per smell: system prompt + few-shots; PROMPT_VERSION per file
│   ├── llm.py                   # Anthropic client wrapper: structured output, retry, usage accounting
│   ├── schemas/
│   │   ├── finding.schema.json
│   │   └── models.py            # Pydantic: Finding, Evidence, AgentResponse, EndpointVerdict
│   ├── graph.py                 # LangGraph StateGraph wiring (below)
│   ├── nodes.py                 # node implementations (pure functions over state)
│   ├── cache.py                 # sqlite content-hash cache
│   ├── store.py                 # findings.jsonl / run.json persistence + loading
│   ├── report/
│   │   ├── render.py            # findings -> report.html
│   │   └── template.html        # single Jinja2 template, inline CSS/JS
│   └── eval/
│       ├── harness.py           # run detection over golden fixtures, compute metrics
│       └── metrics.py           # per-smell precision/recall/F1
└── tests/                       # see 02-TEST-PLAN.md
```

Dependencies (keep minimal): `anthropic`, `langgraph`, `pydantic>=2`, `PyYAML`, `jinja2`, `prance` (optional, reuse repo convention for ref resolution — fall back to internal resolver if absent). Dev: `pytest`, `pytest-asyncio`.

## 2. LangGraph design

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
- `reduce`: build ERD per operation (pure, deterministic — unit-testable without LLM).
- `plan`: cross-product operations × 9 smells, minus cache hits (cache hits emit their stored findings directly into state).
- `detect`: **map node via LangGraph `Send`** — one task per (endpoint, smell). Executes the smell agent (§3), validates output, writes to cache. Concurrency bounded by a semaphore in `llm.py` (default 8) rather than by the graph, so rate limits are respected regardless of fan-out size.
- `collect`: group findings per endpoint.
- `consolidate`: per endpoint with ≥2 findings, one Sonnet call (spec §8). Endpoints with 0–1 findings pass through with a rule-based verdict.
- `persist`: append to `findings.jsonl` (idempotent by finding id), write `run.json` with usage/cost rollup.
- `render`: `report/render.py` → `report.html`.

### Checkpointing / resume

- `langgraph.checkpoint.sqlite.SqliteSaver` at `runs/<run_id>/checkpoint.db`, `thread_id = run_id`.
- `--resume` re-invokes the graph with the same `thread_id`; combined with the content-hash cache (below) this makes interruption cheap even if the checkpoint is coarse. **The cache is the primary resume mechanism; the checkpointer is belt-and-suspenders.** If SqliteSaver integration fights the Send API, it is acceptable to drop the checkpointer and rely on cache-only resume — record in DECISIONS.md.

## 3. Smell agent (detect node internals)

One LLM call per (endpoint, smell):

- **Model:** `claude-haiku-4-5` (config: `HermesConfig.detect_model`).
- **System prompt** (stable, cacheable): agent role + the smell's definition, classification criteria, severity rubric, 2–3 few-shot examples (input ERD fragment → expected JSON), and the output schema. Marked with `cache_control: {"type": "ephemeral"}` so the per-smell system prompt caches across the ~927 endpoint calls for that smell. **Keep it byte-stable — no timestamps/run ids in the system prompt.**
- **User message:** the ERD.
- **Structured output:** `client.messages.parse()` with a Pydantic `AgentResponse`:

```python
class AgentFinding(BaseModel):
    summary: str
    severity: Literal["low", "medium", "high"]
    confidence: float
    evidence: list[Evidence]          # location + excerpt
    justification: str
    suggestion: str

class AgentResponse(BaseModel):
    smell_detected: bool
    findings: list[AgentFinding]      # empty when smell_detected is false
```

- `max_tokens`: 2000. No `temperature` (omit sampling params).
- **Retry policy:** SDK default retries for 429/5xx; one additional application-level retry on schema-validation failure with the validation error appended to the user message. After that, record a `detector_error` entry in `run.json` and continue (never abort the scan for one endpoint).
- Post-validation in code (not trusted from the model): clamp confidence to [0,1]; drop findings with empty evidence; compute deterministic `id`; stamp `detector` metadata.

### Consolidator agent

- **Model:** `claude-sonnet-5` (config: `consolidate_model`).
- Input: the endpoint's ERD summary (path/method/tags only) + its findings as JSON.
- Output (parse): list of finding ids to keep/merge/drop with reasons + optional severity adjustments + endpoint verdict. Applied in code; original findings retained in `findings.raw.jsonl` for audit.

## 4. Caching

sqlite table `cache(key TEXT PRIMARY KEY, response_json TEXT, model TEXT, created_at TEXT)` at `runs/cache.db` (shared across runs, keyed by content — safe because key includes everything that affects output):

```
key = sha256(erd_yaml + smell_id + PROMPT_VERSION[smell] + model_id)
```

- Hit → stored `AgentResponse` JSON replayed, zero API calls, marked `"cached": true` in usage records.
- A prompt edit bumps that smell's `PROMPT_VERSION`, naturally invalidating only that smell's entries.
- `hermes scan --no-cache` bypasses reads (still writes).

## 5. Cost & rate control

- `hermes estimate` and the pre-scan confirmation compute: `calls = |operations| × 9 + |operations with ≥2 findings|(est. 60%)`; input tokens ≈ ERD tokens + cached system prompt (cache-read priced); output ≈ 400/call. With Haiku 4.5 at $1/$5 per MTok and prompt caching, a full 927-operation BaNCS scan is ~8.4k calls, order of **$15–30**; print the computed number, not this constant.
- Semaphore concurrency 8 default; on `RateLimitError` the SDK backs off — additionally halve the semaphore for 60s (simple adaptive throttle).
- `--max-endpoints` and `--sample N --seed S` (stratified by tag when possible) for cheap partial scans.
- Usage accounting: every call appends `UsageRecord(model, input_tokens, output_tokens, cache_read_input_tokens, cost_usd)` from `response.usage`; `run.json` carries the rollup. Cost table (Haiku $1/$5, Sonnet $3/$15 per MTok) lives in `config.py` with a comment that prices drift.

## 6. Report rendering

- Pure function: `render(run_dir) -> report.html`. Jinja2 template, findings embedded as JSON, vanilla JS for filter/sort/paginate (page size 100). No network fetches; must pass a test that greps the output for `http://`/`https://` in src/href attributes (none allowed except inside finding text).
- Keep JS small and dependency-free; correctness over polish.

## 7. Error handling philosophy

- Spec-level failures (unparseable file) fail fast with exit 2.
- Per-endpoint failures (ref bomb, oversized ERD, agent schema failure after retry) degrade to recorded warnings; the scan always completes and the report shows a "skipped/errored" section.
- All LLM calls go through `llm.py`; nothing else imports `anthropic`. This is the single mocking seam for tests.
