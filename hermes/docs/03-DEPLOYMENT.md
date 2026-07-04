# Hermes — Deployment & Operations

Locked decisions: local CLI tool producing static self-contained HTML; no server component. Claude API backend.

## 1. Installation

```bash
cd hermes
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"        # dev
pip install .                  # use only
hermes --help
```

Python ≥3.11. No system dependencies beyond Python.

## 2. Configuration

Precedence: CLI flag > environment variable > default.

| Env var | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — (required for scan/eval --live) | API auth. Alternatively an active `ant auth login` profile is picked up by the SDK. |
| `HERMES_DETECT_MODEL` | `claude-haiku-4-5` | per-smell detection agents |
| `HERMES_CONSOLIDATE_MODEL` | `claude-sonnet-5` | per-endpoint consolidator |
| `HERMES_CONCURRENCY` | `8` | max concurrent LLM calls |
| `HERMES_CACHE_DB` | `runs/cache.db` | content-hash cache location |
| `HERMES_MAX_COST_USD` | unset | hard abort threshold; scan stops (resumable, exit 3) when rolled-up cost exceeds it |

Secrets policy: the API key is read from env only; never written to run.json, logs, or reports. Add a unit test greping run artifacts for `sk-ant`.

## 3. Standard workflows

**Full BaNCS scan (~927 operations, ~8.4k Haiku calls):**

```bash
hermes estimate --spec ../api-docs.json          # prints call count + $ estimate
hermes scan --spec ../api-docs.json --yes        # ~$15-30, 30-90 min at concurrency 8
# interrupted? (Ctrl-C, crash, rate-limit death):
hermes scan --spec ../api-docs.json --run-id <same id> --resume --yes
open runs/<run_id>/report.html
```

**Cheap iteration loop (prompt tuning):**

```bash
hermes scan --spec ../api-docs.json --tags "Account Management" --sample 20 --seed 7 --yes
hermes eval --live      # golden-fixture gates after any prompt change
```

**Distribute results:** `report.html` is the artifact — mail it, attach it, drop it in a share. It embeds the findings JSON, so recipients can extract machine-readable data from the same file.

## 4. Cost model

| Item | Basis | Ballpark (full BaNCS scan) |
|---|---|---|
| Detection | 927 ops × 9 smell agents (5 documentation + 4 REST, per 00-SPEC §3), Haiku 4.5 ($1/$5 per MTok), ~1.5k in / 400 out per call, system prompts cache-read after first hit per smell | ~$12–25 |
| Consolidation | ~60% of ops, Sonnet 5 ($3/$15 per MTok), ~1.2k in / 300 out | ~$3–6 |
| Re-scan after prompt change to ONE smell | only that smell's 927 calls re-run (cache) | ~$2–3 |
| Live eval | 46 ops × 9 + consolidation, Haiku | ~$0.50 |

Prices as of 2026-07; `config.py` owns the numbers. If full-scan costs become routine, consider the Batch API (50% discount, ≤24 h turnaround) — noted as a future option, not in scope now.

## 5. Operational guidance

- **Rate limits:** SDK auto-retries 429/5xx; hermes additionally halves concurrency for 60 s on a 429. If scans crawl, lower `HERMES_CONCURRENCY`; if your org tier is generous, 16 is fine.
- **Interruptions are cheap:** the content-hash cache means a re-run only pays for what hasn't completed. Never delete `runs/cache.db` casually.
- **Prompt changes invalidate deliberately:** bump the edited smell's `PROMPT_VERSION`. Never edit a prompt without bumping — silent cache poisoning.
- **Data sensitivity:** endpoint documentation text is sent to the Anthropic API. The BaNCS spec here is a public demo spec, so that is fine. The paper itself faced the confidential case and solved it by restricting model selection to locally-deployable LLMs (their winner: gpt-oss:120b — see 05-PAPER-FACTS §1); if confidential specs ever need scanning, the precedent is a local backend behind the `llm.py` seam, not sending the spec out.
- **Determinism caveat:** LLM detection is stochastic run-to-run even with caching (cache makes *repeats* deterministic, not fresh runs). Findings carry `confidence`; treat single-run diffs on unchanged specs as noise unless eval gates moved.

## 6. CI (optional, later)

A GitHub Actions workflow is intentionally out of scope for v1 (interface decision: dashboard, not CI gate). When wanted, the shape is: unit+integration on every PR (no secrets needed); `hermes eval` offline replay on every PR; `hermes eval --live` manually dispatched with `ANTHROPIC_API_KEY` from repo secrets.

## 7. Repo hygiene

- `hermes/runs/`, `hermes/.venv/`, `**/__pycache__/` in `.gitignore`.
- Never commit findings from confidential specs.
- Existing repo files (`openapi2mcp_enhanced.py`, generated tool files, `api-docs.json`) are untouched by hermes; it only reads `api-docs.json`.
