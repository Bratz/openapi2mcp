# Hermes — Build Plan for Autonomous Claude Code Loops

Ordered milestones. Rules of engagement are in the repo-root `CLAUDE.md` (work in order, acceptance commands must exit 0, commit per milestone as `hermes: M<N> <name>`, log judgment calls in `docs/DECISIONS.md`).

Every milestone lists: **Build** (what to create), **Accept** (commands that must pass), **Guard** (what must NOT happen).

---

## M0 — Scaffolding

**Build:** `hermes/pyproject.toml` (name `hermes`, Python ≥3.11, deps per 01-ARCHITECTURE §1, console script), `src/hermes/` package skeleton with empty modules, `tests/unit/test_scaffold.py`, `.gitignore` entries (runs/, .venv). (`docs/DECISIONS.md` already exists — append to it, don't recreate.)

**Accept:**
```bash
cd hermes && pip install -e ".[dev]" && hermes --help && pytest tests/unit -q
```
**Guard:** no files outside `hermes/` and root `.gitignore`/`CLAUDE.md` touched.

## M1 — Spec loading & filtering

**Build:** `spec_loader.py` (Swagger 2.0 + OAS 3.x; Operation records with path/method/operation_id/tags; tag/path-glob/sample/seed filters), the golden fixtures (`tests/fixtures/golden/seeded_spec.yaml` with SEEDED comments covering all 9 smells incl. a broken-`$ref` FRAGMENTED seed and an embedded-schema-in-description EXCESS_STRUCTURED seed, `seeded_spec_oas3.yaml`, and the **multi-label** `expected.json` — per 02-TEST-PLAN §2), unit tests per 02-TEST-PLAN §4 spec_loader bullet.

**Accept:**
```bash
pytest tests/unit/test_spec_loader.py -q
python -c "from hermes.spec_loader import load_spec; s=load_spec('../api-docs.json'); assert len(s.operations)==927, len(s.operations)"
```
**Guard:** loading api-docs.json completes < 30 s and < 2 GB RSS.

## M2 — Endpoint reduction (ERD)

**Build:** `reducer.py` per spec §5 (ref inlining with depth cap, sibling context, spec-level context, YAML serialization, token cap + truncation flag), `hermes inspect` CLI path, snapshot tests.

**Accept:**
```bash
pytest tests/unit/test_reducer.py -q
hermes inspect --spec ../api-docs.json --endpoint "GET /accountManagement/account/balanceDetails" | head -50
```
(pick any real endpoint if that one doesn't exist — verify against spec_loader output first)
**Guard:** reducer is deterministic (test runs it twice, asserts byte equality); no LLM imports.

## M3 — Smell catalog & prompts

**Build:** `smells/catalog.py` (the 9 paper-actual smells: LAZY, BLOATED, TANGLED, FRAGMENTED, EXCESS_STRUCTURED, PATH_AND_METHOD, INPUT, RESPONSE, SECURITY — spec §3), `smells/prompts/__init__.py` — a single generic assembler over the catalog (per-smell modules dropped; DECISIONS M3) following the **Appendix-A template structure** (role → definition → "typically occurs when" examples → task → classification rules incl. the smell's scoping rule → explanation/improvement rules with the `[SMELL] - [action title]` contract; source: 05-PAPER-FACTS §5), `occurs_when` examples (≥2), `PROMPT_VERSION="<smell>-v1"`; prompt-assembly function returning the exact messages payload; snapshot tests of assembled prompts (so accidental prompt drift is visible in diffs).

**Accept:**
```bash
pytest tests/unit/test_smells.py -q
```
**Guard:** system prompts contain no dynamic content (grep test: no `{run_id}`, no timestamps).

## M4 — LLM layer

**Build:** `llm.py` (Anthropic wrapper: `messages.parse()` with Pydantic `AgentResponse`, prompt caching via `cache_control` on the system block, semaphore, schema-retry, usage/cost records), `schemas/models.py` + `finding.schema.json`, `config.py`, FakeLLM test double in `tests/conftest.py`, unit tests per 02-TEST-PLAN (llm, schemas bullets) including the no-real-client enforcement test.

**Accept:**
```bash
pytest tests/unit/test_llm.py -q   # schemas tests live here too (no separate test_schemas.py)
```
**Guard:** `anthropic` imported only inside `llm.py`; unit suite passes with `ANTHROPIC_API_KEY` unset.

## M5 — Graph, cache, store

**Build:** `cache.py`, `store.py`, `nodes.py`, `graph.py` (LangGraph per 01-ARCHITECTURE §2 incl. Send fan-out, consolidator node, optional SqliteSaver), `hermes scan/estimate` CLI paths with confirmation + cost estimate, integration tests (graph e2e, resume, consolidation, cli — 02-TEST-PLAN §4 integration bullets).

**Accept:**
```bash
pytest tests/unit -q && pytest tests/integration -q
hermes estimate --spec tests/fixtures/golden/seeded_spec.yaml
```
**Guard:** resume test proves cached tasks aren't re-executed; scan without `--yes` non-interactively exits 4.

## M6 — Report + eval harness

**Build:** `report/render.py` + `template.html` (spec §7, incl. the Appendix-C markdown per-endpoint report), `hermes report [--endpoint --md]`, `eval/harness.py` + `eval/metrics.py` (multi-label: Jaccard, F1-micro/macro, Hamming, cardinality difference) + `hermes eval [--live]` with gates Jaccard ≥ 0.75 / F1-micro ≥ 0.85 / Hamming ≤ 0.12 (02-TEST-PLAN §3); the eval report must print the paper's gpt-oss:120b reference row (Jaccard 0.85, F1μ 0.92, Hamming 0.07) next to ours for comparison; report unit tests, offline-eval plumbing (recordings dir, staleness check).

**Accept:**
```bash
pytest tests/unit -q && pytest tests/integration -q
hermes eval --live          # requires ANTHROPIC_API_KEY; iterate on prompts until gates green
```
Then record responses and:
```bash
hermes eval                 # offline replay green
```
**Guard:** this is the milestone where prompt iteration happens. Budget: if gates aren't green after ~5 prompt iterations (~$3), stop and write up the failing smells + confusion patterns in DECISIONS.md instead of thrashing. Never edit `expected.json` to make a gate pass without a written justification.

## M7 — Real-corpus smoke + hardening

**Build:** nothing new by default — run and fix.

**Accept:**
```bash
hermes scan --spec ../api-docs.json --sample 5 --seed 1 --yes
# re-run same command: assert cached (run.json shows 0 fresh detect calls)
hermes scan --spec ../api-docs.json --sample 5 --seed 1 --yes
pytest tests/unit tests/integration -q
```
Plus: open criteria per 02-TEST-PLAN §5 (≥1 finding, report renders, cost < $1).
**Guard:** any crash on real BaNCS endpoints becomes a regression test (add the offending endpoint's ERD as a fixture).

## M8 (human-gated) — Full scan

Not autonomous: the full ~$15–30 scan is run by the user (`hermes scan --spec ../api-docs.json --yes`), who reviews `report.html`. Prepare a short `docs/RESULTS.md` template to fill from run.json: totals, per-smell % of endpoints side-by-side with the paper's Table 2 baseline (Response 100%, Lazy 90%, Input 88%, Security 68%, Path_and_Method 53%, Tangled 5%, Bloated 2%, Fragmented 2% — 05-PAPER-FACTS §7), avg smells/endpoint vs the paper's 4.08, and top-10 worst endpoints.

---

## Definition of done (whole project)

1. All unit + integration suites green with no network.
2. `hermes eval --live` gates green; recordings checked in; `hermes eval` (offline) green.
3. M7 smoke on real BaNCS spec green, including the cache-proof re-run.
4. Docs updated where reality diverged (DECISIONS.md), CLAUDE.md commands still accurate.
