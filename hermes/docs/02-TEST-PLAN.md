# Hermes — Test Plan

Locked decision: **seeded golden fixtures** are the backbone of quality validation, so autonomous build loops can prove detection quality without a human reviewing runs. Two test tiers: deterministic (no network, always run) and live-LLM eval (explicit, costed).

## 1. Test tiers

| Tier | Command | Network | When |
|---|---|---|---|
| Unit | `pytest tests/unit -q` | none | every change; must pass before any commit |
| Integration (mocked LLM) | `pytest tests/integration -q` | none | every change |
| Offline eval | `hermes eval` | none (recorded responses) | every milestone ≥ M6 |
| Live eval | `hermes eval --live` | Anthropic API (~$0.50) | before tuning claims, before full scans, after any prompt change |
| Smoke scan | `hermes scan --spec ../api-docs.json --sample 5 --yes` | Anthropic API (~$0.15) | M7 acceptance |

The mocking seam is `hermes.llm` (see 01-ARCHITECTURE §7). Tests inject a `FakeLLM` returning canned `AgentResponse` objects; no test outside `--live` paths may construct a real `anthropic.Anthropic`. Enforce with a unit test that monkeypatches `anthropic.Anthropic.__init__` to raise during the unit/integration suites.

## 2. Golden fixture suite

Location: `tests/fixtures/golden/`.

### 2.1 `seeded_spec.yaml`

A hand-authored Swagger 2.0 spec, fictional "Atlas Retail Banking API", **40 operations** across 4 tags. Composition:

- **8 clean operations** — deliberately well-documented, correct REST design, full security definitions. Expected: zero findings. These are the false-positive controls.
- **32 seeded operations** — each seeded with 1–3 specific smells (~55 seeded smell instances total), covering **every one of the 9 smells at least 4 times**, including at least one *subtle* instance per smell (e.g. LAZY: a summary that is long but content-free; METHOD: a POST `/search` that is genuinely acceptable and must NOT be flagged — put acceptable-lookalikes in the clean set).

Also: `seeded_spec_oas3.yaml` — 6 operations, OpenAPI 3.0 flavor of a subset, to lock in OAS3 support.

### 2.2 `expected.json`

Ground-truth labels, the eval oracle:

```json
{
  "getAccountBalance": {
    "seeded": [
      {"smell": "LAZY", "location_hint": "responses.200.description", "note": "generic 'OK'"},
      {"smell": "SECURITY", "location_hint": "security", "note": "auth headers in params, no scheme"}
    ]
  },
  "listBranches": {"seeded": []}
}
```

Matching rule for scoring: a detected finding matches a seeded label iff same `operation_id` and same `smell`. (`location_hint` is for human debugging, not automated matching — smell-level matching keeps the oracle robust to phrasing.)

### 2.3 Authoring rules

- Every seeded smell instance gets a YAML comment `# SEEDED: <SMELL> — <why>` in the spec for maintainability.
- The fixture is frozen after M6 baseline; changes to it require regenerating the recorded responses and re-baselining, noted in DECISIONS.md.

## 3. Metrics and gates (`hermes eval`)

`hermes.eval.harness` scans the golden fixtures (live or from recorded responses), scores against `expected.json`, and prints a per-smell table:

```
smell       seeded  detected  TP  FP  FN  precision  recall
LAZY            8        9    8   1   0      0.89     1.00
...
OVERALL        55       58   49   9   6      0.84     0.89
```

FP counting: any finding on a clean operation, or a finding whose smell isn't seeded on that operation.

**Gates (live eval, `--report` writes `eval_report.json`):**

- Per-smell recall ≥ **0.75** and precision ≥ **0.65**
- Overall recall ≥ **0.80**, overall precision ≥ **0.70**
- Zero-findings rate on the 8 clean operations: ≤ 2 clean operations may have any finding (false-positive control)

`hermes eval --live` exits non-zero when a gate fails, so a build loop can iterate on prompts until green. Gates are floors for M6 acceptance; tuning may raise them later (record in DECISIONS.md).

**Stability note:** live evals are stochastic. The harness runs each failed gate's smell a second time and takes the better result before declaring failure (cheap flake shield). Do not chase single-run fluctuations of <0.05.

### Recorded responses (offline eval)

After the first green live eval, record all raw `AgentResponse` JSONs to `tests/fixtures/recorded/<prompt_version>/`. `hermes eval` (no `--live`) replays them — this keeps the metrics pipeline itself regression-tested for free in CI. A prompt-version bump orphans recordings; the harness must fail loudly ("recordings stale, run --live") rather than silently reusing old ones.

## 4. Unit test inventory (minimum)

`tests/unit/`:

- **spec_loader**: parses `seeded_spec.yaml` (40 ops), `seeded_spec_oas3.yaml` (6 ops), and the real `../api-docs.json` (exactly 927 operations, title matches); rejects garbage with exit-2 error; tag/path/sample filters produce expected subsets; sampling with fixed seed is deterministic.
- **reducer**: local $refs inlined; depth cap at 4 with `$truncated` markers; deterministic output (two runs byte-identical); token cap enforced with `truncation_applied` flag; sibling-path context present; spec-level context block truncated to 1,500 chars; ERD for a known BaNCS endpoint snapshot-tested.
- **smells/catalog**: exactly 9 smells, ids/categories match spec §3; every smell has a prompt file with non-empty PROMPT_VERSION, ≥2 few-shot examples.
- **llm**: schema-validation retry path (first response invalid → retried with error appended → second accepted); usage record math (cost computed from token counts incl. cache reads); semaphore honored (no more than N concurrent with a slow FakeLLM).
- **cache**: same key hit avoids second call; prompt-version bump misses; `--no-cache` bypasses reads but writes.
- **schemas**: finding id determinism; confidence clamping; empty-evidence findings dropped.
- **store**: idempotent append (re-persisting same findings doesn't duplicate); run.json rollup equals sum of usage records.
- **report**: renders 0-finding and 5,000-finding inputs; output contains embedded JSON equal to input findings; no external URLs in src/href; filters present (static grep-level assertions are fine, no browser automation).

`tests/integration/`:

- **graph end-to-end with FakeLLM**: scan the golden spec with a FakeLLM that returns the expected findings → findings.jsonl matches, report.html renders, run.json counts correct.
- **resume**: kill the graph after N detect tasks (FakeLLM raises after N calls), re-run with `--resume` → completes; total FakeLLM call count ≤ (total tasks + N) proving cache reuse.
- **consolidation**: endpoint with duplicate LAZY+INPUT findings → consolidator (Fake) merge applied, raw findings preserved in findings.raw.jsonl.
- **cli**: `hermes estimate` on golden spec prints call count = ops×9; `scan` without `--yes` on a non-TTY aborts with exit 4.

## 5. Live smoke criteria (M7)

- `hermes scan --spec ../api-docs.json --sample 5 --seed 1 --yes` completes, ≥1 finding (the paper found smells in *every* endpoint of a comparable corpus; 5 BaNCS endpoints with zero findings almost certainly means a broken pipeline, not a clean spec), report.html opens with correct counts, run.json cost < $1.
- Re-running the same command touches zero non-cached LLM calls (cache proof on real data).
