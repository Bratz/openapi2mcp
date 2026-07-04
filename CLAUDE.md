# openapi2mcp — repo guide for Claude Code

## What this repo contains

- `openapi2mcp_enhanced.py` — existing, working OpenAPI/Swagger → MCP tool generator. **Do not modify** unless a task explicitly asks for it.
- `api-docs.json` — TCS BaNCS Core Banking Swagger 2.0 spec (~14.7 MB, 907 paths / 927 operations). This is the primary real-world scan target. Treat it as read-only reference data. Never load the whole file into an LLM prompt.
- `banking_tools_new*.py`, `tcs_bancs_real_tools.py` — generated MCP tool files (build artifacts, read-only).
- `llm_system_prompts.yaml` — MCP prompt config for the banking APIs.
- `hermes/` — **the active project**: a multi-agent OpenAPI smell detector (reproduction of arXiv:2605.14312). All new work happens here.

## Hermes: where to look

All design documents live in `hermes/docs/` and are authoritative. Read them before writing code:

| Doc | Contents |
|---|---|
| `hermes/docs/00-SPEC.md` | Functional spec: smell taxonomy, finding schema, CLI contract, report requirements |
| `hermes/docs/01-ARCHITECTURE.md` | LangGraph design, module layout, LLM usage, caching, resume |
| `hermes/docs/02-TEST-PLAN.md` | Unit tests, golden fixture suite, metrics gates, how to self-verify |
| `hermes/docs/03-DEPLOYMENT.md` | Install, env vars, run commands, cost model, operational guidance |
| `hermes/docs/04-BUILD-PLAN.md` | Ordered milestones M0–M7 with acceptance criteria — follow this when building |
| `hermes/docs/05-PAPER-FACTS.md` | Ground-truth extract of the paper (taxonomy, prompt template, report format, baseline results) — paper fidelity source |
| `hermes/docs/DECISIONS.md` | Decision log + deliberate deviations from the paper |

## Ground rules for autonomous build loops

1. Work milestone-by-milestone per `hermes/docs/04-BUILD-PLAN.md`. Do not start milestone N+1 until milestone N's acceptance criteria pass.
2. Every acceptance criterion is a shell command. Run it; paste output honestly. A milestone is done only when its commands exit 0.
3. Never call the live Anthropic API from unit tests. Live calls happen only in `hermes eval --live` and `hermes scan`, both explicitly invoked.
4. All temp/output files go under `hermes/runs/` (gitignored) — never into the repo root.
5. Commit at the end of every completed milestone with message `hermes: M<N> <milestone name>`.
6. If a spec document is ambiguous, prefer the simplest interpretation, implement it, and record the decision in `hermes/docs/DECISIONS.md` rather than stopping to ask.

## Commands

```bash
cd hermes
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest tests/unit -q            # fast, no network — must always pass
pytest tests/integration -q     # mocked-LLM graph tests — must always pass
hermes eval                     # offline eval against recorded responses (exits 2 until a green --live run has recorded them)
hermes eval --live              # live-LLM eval vs golden fixtures (costs ~$0.50; needs ANTHROPIC_API_KEY)
```
