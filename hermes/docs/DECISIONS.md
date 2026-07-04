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

(append below)
