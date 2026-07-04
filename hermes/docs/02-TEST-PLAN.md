# Hermes — Test Plan

Locked decisions: **seeded golden fixtures** as the quality backbone (autonomous loops self-verify without human review), scored with the **paper's multi-label metrics** (Jaccard, F1-micro/macro, Hamming loss, cardinality difference — see 05-PAPER-FACTS §1). Two tiers: deterministic (no network, always run) and live-LLM eval (explicit, costed).

## 1. Test tiers

| Tier | Command | Network | When |
|---|---|---|---|
| Unit | `pytest tests/unit -q` | none | every change; must pass before any commit |
| Integration (mocked LLM) | `pytest tests/integration -q` | none | every change |
| Offline eval | `hermes eval` | none (recorded responses) | every milestone ≥ M6 |
| Live eval | `hermes eval --live` | Anthropic API (~$0.50) | before tuning claims, before full scans, after any prompt change |
| Smoke scan | `hermes scan --spec ../api-docs.json --sample 5 --yes` | Anthropic API (~$0.15) | M7 acceptance |

The mocking seam is `hermes.llm` (01-ARCHITECTURE §8). Tests inject a `FakeLLM` returning canned `AgentResponse` objects; no test outside `--live` paths may construct a real `anthropic.Anthropic`. Enforce with a unit test that monkeypatches `anthropic.Anthropic.__init__` to raise during the unit/integration suites.

## 2. Golden fixture suite

Location: `tests/fixtures/golden/`.

### 2.1 `seeded_spec.yaml`

A hand-authored Swagger 2.0 spec, fictional "Atlas Retail Banking API", **40 operations** across 4 tags. Composition:

- **8 clean operations** — deliberately well-documented, correct REST design, full security definitions. Expected label set: empty. These are the false-positive controls. Include acceptable-lookalikes here (e.g. a genuinely appropriate POST `/accounts/search` that must NOT be flagged PATH_AND_METHOD).
- **32 seeded operations** — most seeded with 1–3 smells (the Appendix-B clone carries 5; ~50 seeded instances total), covering **every one of the 9 smells at least 4 times**, including at least one *subtle* instance per smell. Mandatory seeds for the two structural smells:
  - `FRAGMENTED`: at least one operation whose response/body `$ref` points to a definition that does not exist in the document (paper Table 3's operationalization).
  - `EXCESS_STRUCTURED`: at least one `description` containing an embedded class-like/JSON-schema-like definition in the prose.
  - Also include the paper's Appendix B anti-pattern at least once: `GET /orders/createNewOrder`-style verb path + wrong method + `GenericResponse {status, data: object}` (should light up PATH_AND_METHOD, RESPONSE, LAZY, INPUT, SECURITY like the paper's sample report).

Also: `seeded_spec_oas3.yaml` — 6 operations, OpenAPI 3.0 flavor of a subset, to lock in OAS3 support.

### 2.2 `expected.json` — multi-label oracle

Ground truth is a **smell-label set per operation**, matching the paper's multi-label formulation:

```json
{
  "getAccountBalance": {
    "smells": ["LAZY", "SECURITY"],
    "notes": {"LAZY": "generic summary, no description", "SECURITY": "auth headers in params, no scheme"}
  },
  "listBranches": {"smells": []}
}
```

`notes` are for human debugging only — scoring uses the `smells` sets exclusively (extensions like severity/confidence are never scored).

### 2.3 Authoring rules

- Every seeded smell gets a YAML comment `# SEEDED: <SMELL> — <why>` in the spec.
- Fixture frozen after M6 baseline; changes require regenerating recorded responses and re-baselining, noted in DECISIONS.md.

## 3. Metrics and gates (`hermes eval`)

`hermes.eval.harness` scans the golden fixtures (live or recorded), builds the predicted smell-set per operation, and scores against `expected.json` with the paper's metric suite:

- **Jaccard similarity** (mean per-endpoint |P∩G|/|P∪G|; define both-empty as 1.0)
- **F1-micro** and **F1-macro** over the 9 labels
- **Hamming loss** (fraction of wrong label decisions over 46 ops × 9 labels)
- **Cardinality difference** (mean predicted-labels − actual-labels; sign shows over/under-labeling)

Plus a **diagnostic (non-gating) per-smell precision/recall table** for prompt tuning.

```
metric        value   gate     paper (gpt-oss:120b)
Jaccard       0.81    ≥0.75    0.85
F1-micro      0.88    ≥0.85    0.92
F1-macro      0.70    report   0.73
Hamming       0.09    ≤0.12    0.07
Cardin.diff  -0.31    report   -0.53
```

**Gates (live eval; `--report` writes `eval_report.json`):**

- Jaccard ≥ **0.75**
- F1-micro ≥ **0.85**
- Hamming loss ≤ **0.12**
- False-positive control: ≤ 2 of the 8 clean operations may have any predicted label

`hermes eval --live` exits non-zero when a gate fails, so a build loop can iterate on prompts until green. Gates are floors for M6 acceptance (chosen slightly below the paper's best-model numbers since our oracle is synthetic); raising them later is fine — record in DECISIONS.md.

**Stability note:** live evals are stochastic. On a failed gate the harness re-runs once and takes the better result (cheap flake shield). Do not chase single-run fluctuations < 0.03.

### Recorded responses (offline eval)

After the first green live eval, record all raw `AgentResponse` JSONs to `tests/fixtures/recorded/responses.jsonl` (flat file with per-record `prompt_version` fields + a `_meta` line — DECISIONS M6). `hermes eval` (no `--live`) replays them — keeping the metrics pipeline regression-tested in CI for free. A prompt-version bump orphans recordings; the harness must fail loudly ("recordings stale, run --live") rather than silently reusing old ones.

## 4. Unit test inventory (minimum)

`tests/unit/`:

- **spec_loader**: parses `seeded_spec.yaml` (40 ops), `seeded_spec_oas3.yaml` (6 ops), and the real `../api-docs.json` (exactly 927 operations, title matches); rejects garbage with exit-2 error; tag/path/sample filters; seeded sampling determinism.
- **reducer**: local $refs inlined; depth cap with `$truncated`; **unresolvable ref → `$unresolved` marker, no exception** (FRAGMENTED evidence path); deterministic output (two runs byte-identical); token cap + `truncation_applied`; sibling-path context; security-scheme context present; ERD for a known BaNCS endpoint snapshot-tested.
- **smells/catalog**: exactly 9 smells with ids `LAZY, BLOATED, TANGLED, FRAGMENTED, EXCESS_STRUCTURED, PATH_AND_METHOD, INPUT, RESPONSE, SECURITY`; categories 5 documentation / 4 rest; every smell has `scoping_rule`, `occurs_when` (≥2 examples), non-empty `PROMPT_VERSION`; prompt assembly follows the Appendix-A section order (snapshot test).
- **llm**: schema-validation retry path; usage record math (incl. cache reads); semaphore honored under a slow FakeLLM.
- **cache**: hit avoids second call; prompt-version bump misses; `--no-cache` bypasses reads, still writes.
- **schemas**: finding id determinism; confidence clamping; `[SMELL] - ` action-title repair; short-justification flag path; detected=false with non-empty findings rejected.
- **metrics**: Jaccard/F1-micro/F1-macro/Hamming/cardinality computed on small hand-calculated fixtures (e.g. 3 endpoints × 3 labels with known confusion) — exact expected values asserted; both-empty Jaccard = 1.0 edge case.
- **store**: idempotent append; run.json rollup equals sum of usage records.
- **report**: dashboard renders 0-finding and 5,000-finding inputs; embedded JSON equals input; no external URLs in src/href; **Appendix-C markdown report matches the paper's section order** (`## API Info`, `## Endpoint Info`, `## Model`, `## Identified Smells`, `### Explanations`, `## Improvement Suggestions`) — snapshot test against a hand-written expected file.

`tests/integration/`:

- **graph end-to-end with FakeLLM**: scan the golden spec with a FakeLLM returning the expected labels → findings.jsonl matches, endpoints.jsonl has per-endpoint smell sets, report renders, run.json counts correct.
- **resume**: FakeLLM raises after N calls → re-run with `--resume` completes; total FakeLLM calls ≤ (total tasks + N), proving cache reuse.
- **consolidation**: endpoint with ≥2 detections → consolidator (Fake) normalization applied, raw outputs preserved in findings.raw.jsonl; `--no-consolidate` skips the node.
- **cli**: `hermes estimate` on golden spec prints call count = ops×9; `scan` without `--yes` on a non-TTY aborts with exit 4; `report --endpoint --md` emits the markdown report.

## 5. Live smoke criteria (M7)

- `hermes scan --spec ../api-docs.json --sample 5 --seed 1 --yes` completes; ≥1 detection (the paper found ≥1 smell in **every** endpoint of a comparable corpus — 5 BaNCS endpoints with zero detections almost certainly means a broken pipeline); report.html opens with correct counts; run.json cost < $1.
- Re-running the same command performs zero fresh detect calls (cache proof on real data).
- **Calibration sanity (directional, non-gating):** results should rhyme with paper Table 2 — RESPONSE/LAZY/INPUT common, BLOATED/TANGLED/EXCESS_STRUCTURED rare. Wild divergence (e.g. BLOATED on every endpoint, RESPONSE on none) → investigate prompts before believing the results.
