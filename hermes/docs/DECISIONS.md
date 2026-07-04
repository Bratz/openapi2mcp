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
