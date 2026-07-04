# Hermes — Decision Log

Judgment calls and deliberate deviations from the paper (arXiv:2605.14312 — ground truth in `05-PAPER-FACTS.md`). Build loops append here per repo-root `CLAUDE.md` rule 6.

## 2026-07-04 — Spec-phase decisions (user-confirmed)

1. **Scope: detection only (paper-faithful).** No remediation/spec rewriting/MCP regeneration.
2. **Agent roster: paper-actual 9 agents.** 5 documentation smells (LAZY, BLOATED, TANGLED, FRAGMENTED, EXCESS_STRUCTURED) + 4 REST smells (PATH_AND_METHOD merged, INPUT, RESPONSE, SECURITY). Chosen over the taxonomy-literal reading (4+5 separate) because Table 3 and the Appendix C report show the system actually ran with merged Path&Method and an Excess-Structured check.
3. **Result schema: paper core + extensions.** Binary per (endpoint, smell) with justification bullets + `[SMELL] - [action title]` suggestions is the core; `extensions.{severity, confidence, evidence}` added for dashboard triage, **excluded from eval scoring**.
4. **Eval metrics: paper's multi-label suite as gates.** Jaccard ≥ 0.75, F1-micro ≥ 0.85, Hamming ≤ 0.12 (F1-macro + cardinality difference reported). Per-smell precision/recall demoted to a non-gating diagnostic table. Gates set slightly below the paper's best-model numbers (0.85 / 0.92 / 0.07) because our oracle is a synthetic fixture, not expert-annotated production endpoints.

## Deviations from the paper (deliberate)

| Paper | This reproduction | Why |
|---|---|---|
| Local gpt-oss:120b (7-model local bake-off; confidentiality policy barred external APIs) | Claude API: Haiku 4.5 detection fan-out, Sonnet 5 consolidation | Scan target (`api-docs.json`) is a public demo spec; Claude gives stronger detection + reliable structured output. `llm.py` is the seam for a future local/pluggable backend if confidential specs are ever in scope. |
| Interactive tool taking an OpenAPI URL | CLI + static self-contained HTML dashboard (+ Appendix-C-style markdown per-endpoint reports) | User decision (interface question); zero-infrastructure distribution. |
| Gold standard: 60 production endpoints, 2 domain experts | Synthetic seeded golden fixture (40 ops, 8 clean controls) scored with the same metric suite | No expert-annotation budget; enables fully autonomous self-verification. |
| No severity/confidence in outputs | `extensions` fields (severity, confidence, evidence locations) | Dashboard triage value; excluded from eval so fidelity of metrics is preserved. |
| Prompt demands "return ONLY valid JSON" as prose rule | Structured outputs via `client.messages.parse()` + Pydantic | Same intent, mechanically enforced; removes a whole failure class. |

## Build-phase decisions

### 2026-07-04 — M6

- **f1_macro averages over ACTIVE labels only** (labels with any tp/fp/fn); a label absent from both gold and predictions everywhere would score a vacuous 1.0 and inflate macro.
- **Flake shield re-runs only the suspect smells** (those with any FP/FN) when a gate fails, and keeps the better per-smell F1 run — cheaper than a full re-run, matches 02-TEST-PLAN §3.
- **Live eval bypasses the response cache** (fresh samples are the point; the flake shield would otherwise replay identical results). Recordings are written wholesale per live eval with a `_meta` line carrying model + prompt versions; offline replay fails loudly on any version mismatch.
- **`hermes eval --report` writes eval_report.json under hermes/runs/** (CLAUDE.md rule 4).
- **Dashboard escapes embedded JSON with `</`→`<\\/` wrapped in Markup** (autoescape must not entity-encode the JSON, but script-breakout must stay impossible).
- **Recordings layout deviation**: 02-TEST-PLAN §3 sketched `tests/fixtures/recorded/<prompt_version>/`; implemented as a single flat `responses.jsonl` with per-record prompt_version fields + a `_meta` line (same staleness guarantees, simpler diffing). Failing live runs write to `responses.failed.jsonl` and never clobber the green baseline; incomplete recordings (detector errors) are a gate failure.
- **BLOCKED: `hermes eval --live` and the M7 smoke require ANTHROPIC_API_KEY**, which this build environment does not have. Offline machinery (harness, metrics, gates, recordings plumbing, replay) is complete and unit-tested; the live gate run + committed recordings are pending credentials.

### 2026-07-04 — M5

- **Cache-only resume; no LangGraph checkpointer.** SqliteSaver adds serialization constraints on Send fan-out for no benefit — the content-hash cache already makes an interrupted run's completed pairs free on re-run (this was pre-authorized in the M2/architecture notes). `--resume` is therefore just "same run-id, cache does the work"; findings.jsonl appends stay idempotent by id.
- **Rule-based verdict** (0-1 findings, --no-consolidate, consolidator failure): 0 smells → agent-ready; any high severity or ≥3 distinct smells → not agent-consumable; else needs adaptation.
- **Node closures live in graph.py** (they bind llm/cache/config/store); `nodes.py` holds the pure helpers (verdict rule, consolidation-edit application with the ±1 clamp).
- **M5 ships a placeholder report.html** (valid, self-contained, findings table + embedded JSON) so `hermes scan` always writes a usable artifact; the full dashboard is M6.
- **Budget ceiling enforcement**: a thread-safe cost meter accumulates real + estimated spend across detect/consolidate; exceeding HERMES_MAX_COST_USD raises BudgetExceeded → exit 3 with a resume hint. Responses are cached BEFORE the budget check so a paid call is never bought twice.
- **From the M5 review**: consolidations are cached too (key = findings payload + consolidator version + model) — without this a budget-capped scan could loop paying for consolidations forever; interrupted runs (budget/Ctrl-C) still write run.json with `status: interrupted*` so run dirs are never empty; run artifacts are rewritten wholesale per completed run (append-by-id would let stale finding bodies coexist with fresh verdicts); severity re-adjustment edits for the same finding are ignored (no compounding past the ±1 clamp); embedded report JSON escapes `</` (script-breakout XSS from hostile spec text); `--resume` requires `--run-id`; `max_concurrency` gets no floor (concurrency=1 is strictly serial — verified against langgraph 1.2 internals).

### 2026-07-04 — M4 (from milestone review)

- **Threading, not asyncio**: `llm.py` uses a synchronous client + `threading.BoundedSemaphore`; LangGraph's sync executor runs Send branches in a thread pool. Simpler to test deterministically than an async stack.
- **429 backoff = full-pool pause (~60s), not halved concurrency** — docs updated to match. Pause happens BEFORE semaphore acquisition, re-checked in a loop, so throttled workers don't hold slots.
- **Failure contract**: every per-call failure (schema validation after retry, refusal/empty `parsed_output` — which the SDK returns as `None` without raising — API errors after SDK retries) surfaces as `DetectorFailure`; the graph records detection failures as `detector_error` and falls back to rule-based verdicts on consolidation failures. Scan never aborts for one endpoint.
- **Schema-retry is a clean resend with doubled max_tokens**, not a "repair" — the SDK enforces the schema server-side and discards failed responses client-side, so truncation is the dominant residual cause and the failed output is unrecoverable. Failed attempts' spend is **estimated** (worst-case output) and merged into the returned/raised UsageRecord (`estimated: true`) so HERMES_MAX_COST_USD stays honest.
- **cache_control on system blocks is currently a no-op**: per-smell system prompts (~700 tokens) are far below Haiku 4.5's 4096-token minimum cacheable prefix. Kept (harmless, future-proof); cost estimates assume NO cache savings.
- **finding.schema.json is generated**: regenerate with `python -m hermes.schemas` after any Finding model change (sync-guarded by a unit test). It describes the stored Finding record; AgentResponse enforcement happens at parse() time.
- **ConsolidationEdit requires `new_severity` iff `action=adjust_severity`**; the M5 edit-applier clamps adjustments to ±1 step of the stored severity (the model can't see the original).

### 2026-07-04 — M3 (from milestone review)

- **Prompt layout: single `smells/prompts/__init__.py`** with a `PROMPT_VERSIONS` dict instead of nine per-smell modules — the per-smell content already lives in `catalog.py` dataclasses, so per-smell modules would be nine copies of one template. 01-ARCHITECTURE layout updated.
- **Appendix-A deviations (beyond the already-logged structured-outputs swap):** the paper's "endpoint key format GET:/users" and "empty JSON object when clean" classification rules are dropped — endpoint identity is stamped by code and `smell_detected=false` replaces the empty object; the `{openapi_json}` slot moves from mid-system-prompt to the user message so the per-smell system prompt stays byte-stable and prompt-cacheable across all ~927 endpoint calls.
- **Eval-integrity rule: prompts must never quote golden-fixture-invented text.** The milestone review caught fixture-verbatim examples in occurs_when/guards (param names, class idioms, marketing strings, the clean-control's literal path); all were neutralized to paper-sourced or fixture-disjoint phrasings. Pattern-level overlap remains by design (fixture and prompts both instantiate the taxonomy) — the golden eval therefore measures "implements the taxonomy", and M7's real-corpus smoke is the generalization check.
- **Paper's ≥120-chars-per-section rule applied to BOTH sections** (justification and suggestions).
- **`-v1` prompt versions are finalized as of the M3 commit**; any later prompt-affecting edit (catalog text included) bumps the affected smell's version and re-records its snapshot.

### 2026-07-04 — M2 (from milestone review)

- **Token cap enforced on the full rendered ERD** (endpoint + operation + context + flag), not just the operation subtree. Truncation ladder, in order: inline depth 4→3→2→1, then per-field description cap (500 chars), then collapse largest `properties` maps and `enum` lists >25 values (deepest first). If bulk remains outside those shapes the ERD can still exceed the cap (flagged `truncation_applied`) — accepted residual risk, revisit if M7 hits it.
- **Implementation constants** beyond the spec-fixed ones (api_description 1500 chars, tag_description 500 chars, in-truncation description cap 500 chars, enum threshold 25/keep 10) live in reducer.py as module constants — they shape a deterministic artifact and cache keys, so they are code, not runtime config.
- **Unknown `--endpoint` on inspect exits 2** (invalid invocation per 00-SPEC §6); CLI errors go to stderr so stdout stays pipeable ERD/report content.
- **`find_operation` lives in spec_loader** next to OperationRef (M6 report needs the same lookup).
- **Snapshot recording is opt-in** (`HERMES_RECORD_SNAPSHOTS=1`); a missing baseline fails the test instead of self-recording, so a regressed reducer can never bless its own output. Baselines are committed.
- **Pooled OAS3 `#/components/requestBodies` refs consume one inline-depth level** like any other ref link — a schema reached through a pooled requestBody truncates one level earlier than the same schema inlined directly. Accepted asymmetry.

### 2026-07-04 — M1 (from milestone review)

- **Test-plan fixture numbers corrected to the fixture as built**: ~55 → ~50 seeded instances (actual 52), "1–3 smells per op" → "most 1–3, Appendix-B clone carries 5". The composition floor (every smell ≥4 in the swagger2 fixture alone, 8 clean controls) is unchanged and now test-enforced.
- **Oracle-poisoning hazards fixed after fixture audit**: the shared `PaymentRequest` schema was accidentally INPUT-smelly (fed two ops not labeled INPUT) — now fully documented; 8 unlabeled ops documented only 2xx responses (a spec-literal RESPONSE agent would flag them) — all now carry 4xx error responses. RESPONSE seeds remain distinguished by no-schema/generic-envelope evidence.
- **Sampling requires an explicit seed** (`filter_operations` raises ValueError on None) — default 42 is owned by HermesConfig per the M0 decision; equal round-robin per tag (not proportional), documented in the docstring.
- **Path globs use `fnmatch.fnmatchcase`** (case-sensitive, platform-independent); `*` crosses `/` deliberately.
- **`trace` is enumerated for OAS3 only** (not a Swagger 2.0 method).
- **Freeze notices added** to seeded_spec_oas3.yaml and expected.json (seeded_spec.yaml already had one).

### 2026-07-04 — M0 (from milestone review)

- **Exit code 2 covers argparse usage errors as well as spec parse failures.** argparse exits 2 on bad flags by design; remapping it buys nothing. 00-SPEC §6 updated.
- **CLI flag defaults that the spec assigns values (concurrency 8, seed 42, out runs/) parse as `None`** and are resolved by `HermesConfig` (M4), so env overrides can distinguish "not given" from an explicit value.
- **`commands.py` added to the architecture layout** as a thin arg→module adapter layer; real logic stays in graph/report/eval per 01-ARCHITECTURE.
- **Version single-sourced** in `hermes.__version__` via `[tool.setuptools.dynamic]`.
- **Test dirs are packages** (`__init__.py` in tests/, tests/unit/, tests/integration/) to prevent pytest import-file-mismatch collisions; integration has a placeholder smoke test until M5 so the must-pass command collects on fresh clones.
