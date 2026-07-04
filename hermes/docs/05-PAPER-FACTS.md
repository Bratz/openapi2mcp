# Hermes — Ground-Truth Facts from the Paper (arXiv:2605.14312v1)

Condensed extract of the full paper text ("Making OpenAPI Documentation Agent-Ready: Detecting Documentation and REST Smells with a Multi-Agent LLM System", Lima, Pinheiro, Menezes — Sidia Institute of Technology; EASE 2026). This file is the authoritative source for paper fidelity; build loops should consult it instead of the PDF (which is not committed, for copyright reasons).

## 1. Study design

- Industrial case study at an R&D institute: **16 production APIs, ~600 endpoints**, microservice architecture, APIs stable but never designed for agent consumption.
- **Proof-of-concept probe**: two MCP servers over the same 10-endpoint FastAPI app — one with original docs, one with enriched docstrings (objectives, inputs, security, outputs, semantics). 10 automation tasks (8 multi-tool). Original docs: ~70% failed at planning, 10% of expected endpoints identified, 0 executed end-to-end. Enriched docs: 90% correct plans, 80% endpoints identified, all identified endpoints executed. Single run per task, no statistical treatment — indicative only.
- **Gold standard**: 60 endpoints (10% random), **2 domain experts** annotating multi-label smells, disagreements resolved by discussion.
- **Model selection**: smell detection formulated as **multi-label classification**; 7 locally-deployable LLMs compared under identical prompting (local-only due to confidentiality policy). Winner: **gpt-oss:120b**.

### Table 1 — model bake-off (vs expert gold standard)

| Model | Jaccard | F1-micro | F1-macro | Hamming | Cardin. diff |
|---|---|---|---|---|---|
| **gpt-oss:120b** | **0.85** | **0.92** | **0.73** | **0.07** | −0.53 |
| deepseek-r1:70b | 0.69 | 0.81 | 0.48 | 0.16 | −1.22 |
| qwen2.5vl:72b | 0.67 | 0.80 | 0.49 | 0.20 | −0.47 |
| llama3.2-vision:90b | 0.62 | 0.77 | 0.46 | 0.21 | −1.17 |
| gemma3:27b | 0.53 | 0.70 | 0.50 | 0.37 | +1.83 |
| qwen3-coder:30b | 0.53 | 0.70 | 0.38 | 0.25 | −1.72 |
| llama4:16x17b | 0.51 | 0.68 | 0.44 | 0.29 | −1.18 |

Cardinality difference = mean(predicted labels − actual labels) per endpoint; negative = under-labeling.

## 2. Taxonomy as actually implemented

Section 3.5 names 4 documentation + 5 REST categories, but the running system (Table 3 evidence patterns + the Appendix C report) shows:

- **Documentation smells (5)**: LAZY, BLOATED, TANGLED, FRAGMENTED, **EXCESS_STRUCTURED** (from Khan et al. 2021's five API documentation smells).
- **REST smells (4)**: **PATH_AND_METHOD** (reported and explained as a single merged category), INPUT, RESPONSE, SECURITY.

Definitions (Section 3.5, quoted/close-paraphrase):
- **LAZY** — superficial or generic documentation: short summaries, vague descriptions, undocumented parameters, generic response messages.
- **BLOATED** — excessively verbose descriptions with limited informational value.
- **TANGLED** — documentation mixing unrelated concerns (business logic, security, error handling) within the same textual fragment.
- **FRAGMENTED** — essential information dispersed across disconnected sections without explicit linkage.
- **PATH** — action-oriented or inconsistently named URIs that do not represent resources. **METHOD** — inappropriate or semantically inconsistent HTTP method use. (Merged in practice.)
- **INPUT** — weakly specified parameters or request bodies lacking semantic clarification.
- **RESPONSE** — inconsistent or insufficiently described response schemas, status codes, or error handling.
- **SECURITY** — missing or unclear authentication/authorization definitions.

## 3. Table 3 — recurring evidence patterns (prompt-authoring source)

| Smell | Evidence patterns observed |
|---|---|
| Lazy | Very short/generic `summary` ("Get data"); absent or redundant `description`; semantic mismatch between summary and description; no examples/constraints; intent requires external knowledge. |
| Input | Parameters typed only (`string`, `integer`) with no semantic description; ambiguous names (abbreviations, internal acronyms); missing valid ranges/formats/behavioral impact; required params without rationale. |
| Response | Schemas present (often `$ref`) but minimal explanatory text; generic descriptions ("Successful Response"); no semantic clarification of payload meaning across success/error; no interpretation guidance. |
| Security | Schemes technically defined (e.g. Bearer) but no operational guidance (how to obtain credentials, scopes, constraints); security as implicit organizational knowledge. |
| Path_and_Method | POST for updates that should be PUT/PATCH; verbs in paths (`/getUsers`, `/updateStatus`); long or opaque paths with internal acronyms. |
| Bloated/Tangled | Rare in their corpus (short, low-density text gave little room for verbosity or mixed concerns). |
| Excess_Structured | Applies when class-like definitions, nested structures, or formal specifications are written **inside `summary`/`description`**; normative schema structure under `components` is NOT the smell. None found in their corpus. |
| Fragmented | Applies when endpoints reference schemas/components **not present in the specification** (broken/missing references → incomplete documentation). None found in their corpus (all refs resolved). |

Note: detection is more reliable for smells grounded in explicit structure (Input, Response, Security) and more contextual for natural-language smells (Lazy).

## 4. Architecture facts

- **Endpoint-centric**: each operation (method + path) isolated and evaluated independently.
- **Reduced OpenAPI representation** per endpoint containing only: method, path, summaries, descriptions, parameters, request body, responses, schemas, security definitions. Purpose: token reduction + analytical focus.
- **Central Smell Detector Agent** orchestrates the workflow and **consolidates results into a unified explainable diagnostic report**; specialized agents (one per smell category) all receive the same reduced representation and analyze from their assigned perspective.
- Structured prompting: each agent prompt encodes the smell definition, classification criteria, and illustrative examples (**few-shot**).

## 5. Appendix A — specialist prompt template structure (LAZY agent shown)

Sections, in order:
1. **Role**: "You are an expert in identifying <Smell> documentation smells in API documentation."
2. **Smell Definition**: prose definition.
3. **"This smell typically occurs when:"** `{examples}` — the few-shot/occurrence examples slot.
4. **Task**: analyze the complete OpenAPI method definition provided below.
5. **OpenAPI Specification**: `{openapi_json}` (the reduced representation).
6. **Classification Rules**:
   - Per-smell scoping — for LAZY: "Analyze ONLY the method summary and description."
   - Return ONLY a valid JSON object with affected endpoints; endpoint key format `"GET:/users"`.
   - Empty JSON object when no smell found; no text outside the JSON.
7. **Explanation and Improvement Rules**:
   - Explanations ≥ 120 characters per section.
   - Exactly two sections: (1) **"Justification and evidence of the smell:"** — bullet points, complete sentences; (2) **"Suggested actions to address the smell:"** — table with an Action column, exact format **`[SMELL] - [action title]`**.

## 6. Appendix B/C — report format

Per-endpoint diagnostic report (markdown):

```
## API Info            – Title
## Endpoint Info       – Method, Path
## Model               – model id used
## Identified Smells   – comma list (e.g. Lazy, Security, Input, Response, Path & Method)
### Explanations       – one "### <Smell>" block each, bullet justifications
## Improvement Suggestions
[SMELL] - <action title> | <one-sentence description>
```

The Appendix B example (Order Management API, `GET /orders/createNewOrder`, `GenericResponse{status, data:object}`) is a canonical multi-smell fixture shape worth imitating in our golden fixtures.

## 7. Results (calibration baselines for our scans)

### Table 2 — smell distribution over 600 endpoints

| Smell category | Frequency | % of endpoints |
|---|---|---|
| Response | 600 | 100% |
| Lazy | 540 | 90% |
| Input | 530 | 88% |
| Security | 410 | 68% |
| Path_and_Method | 320 | 53% |
| Tangled | 30 | 5% |
| Bloated | 10 | 2% |
| Fragmented | 10 | 2% |

Total **2,450 smells**, ≥1 per endpoint, **avg 4.08 smells/endpoint**. Recurring anti-pattern: generic DTO responses `{status, data}` where `data` is an unconstrained object (semantic opacity — even original authors couldn't describe contents without reading source).

### Practitioner validation (24 developers, 60-endpoint reports)

- High agreement with structurally-grounded smells (80% strongly agree + 7% agree) — incomplete params, insufficient response docs, missing security.
- LAZY contested (~30% disagree) — brevity defended as fine internally until reviewers considered external/agent consumers.
- REST smells sometimes justified by legacy constraints (20% disagree, 3% strong; 32% "tried to justify").
- Reported learning effect: reviewing reports raised documentation awareness.

### Strategic outcome

- Full-ecosystem remediation ≈ 385 engineering hours vs 42 hours for the 42 endpoints needed by 18 target automation scenarios (−89%). Organization pivoted to selective adaptation + documentation standards + Hermes in governance review.

## 8. Deviations in this reproduction (see DECISIONS.md)

1. **Model backend**: paper used local gpt-oss:120b (confidentiality constraint, explicitly noted in their Limitations). We use Claude API (Haiku 4.5 detect / Sonnet 5 consolidate) — our scan target is a public demo spec.
2. **Interface**: paper describes an interactive tool taking an OpenAPI URL; we build a CLI + static HTML dashboard (plus Appendix-C-style per-endpoint markdown reports).
3. **Ground truth**: paper used 60 expert-annotated production endpoints; we use a synthetic seeded golden fixture (40 ops) with the same multi-label metric suite.
4. **Output schema**: paper core (binary smell + justification + suggestions) plus our extension fields (severity, confidence, evidence locations) which are excluded from eval scoring.
