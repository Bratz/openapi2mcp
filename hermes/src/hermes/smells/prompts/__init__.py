"""Appendix-A prompt assembly (docs/05-PAPER-FACTS.md §5).

Section order is the paper's specialist template: role → smell definition →
"typically occurs when" examples → task → classification rules (incl. the
per-smell scoping rule) → explanation and improvement rules. The paper's
"return ONLY valid JSON" prose rule is replaced by structured outputs
(client.messages.parse) — see DECISIONS.md.

PROMPT_VERSIONS feed the content-hash cache key: bump a smell's version on ANY
edit that changes its assembled prompt (catalog text included), or the cache
will replay stale responses (docs/03-DEPLOYMENT.md §5).
"""

from __future__ import annotations

from hermes.smells.catalog import SMELLS, Smell

PROMPT_VERSIONS: dict[str, str] = {
    "LAZY": "lazy-v1",
    "BLOATED": "bloated-v1",
    "TANGLED": "tangled-v1",
    "FRAGMENTED": "fragmented-v1",
    "EXCESS_STRUCTURED": "excess_structured-v1",
    "PATH_AND_METHOD": "path_and_method-v1",
    "INPUT": "input-v1",
    "RESPONSE": "response-v1",
    "SECURITY": "security-v1",
}

SEVERITY_RUBRIC = """\
Severity rubric (for the extensions block):
- high: an autonomous agent relying only on this specification would likely fail to call
  this endpoint correctly (wrong payload, wrong endpoint selection, auth failure).
- medium: the agent could probably call it, but tool selection or payload quality would
  be degraded (guessing formats, ambiguous semantics).
- low: cosmetic or consistency issue unlikely to break an agent call."""


def build_system_prompt(smell: Smell) -> str:
    """Assemble the Appendix-A specialist system prompt for one smell.

    MUST stay byte-stable for a given (smell, PROMPT_VERSION): no timestamps,
    run ids, or environment-dependent content (prompt-cache + response-cache
    correctness depend on it).
    """
    category_display = "REST" if smell.category == "rest" else smell.category
    occurs = "\n".join(f"- {example}" for example in smell.occurs_when)
    guards = "\n".join(f"- {guard}" for guard in smell.false_positive_guards)
    guards_block = f"\nDo NOT classify as {smell.id} (false-positive guards):\n{guards}\n" if guards else "\n"

    return f"""\
Role:
You are an expert in identifying "{smell.display_name}" ({smell.id}) {category_display} smells
in OpenAPI documentation, assessing whether an API endpoint is ready for reliable
consumption by autonomous AI agents.

Smell Definition ({smell.id}):
{smell.definition}

This smell typically occurs when:
{occurs}

Task:
Analyze the reduced OpenAPI representation of ONE endpoint (provided by the user) and
decide whether this endpoint exhibits the {smell.id} smell.

Classification Rules:
- {smell.scoping_rule}
- Judge only the {smell.id} smell; other quality problems are handled by other analysts.
- Base every judgment on evidence visible in the provided representation; never assume
  content that is not shown.
- A field containing {{"$truncated": ...}} was shortened for analysis; do not treat
  truncation itself as evidence of any smell.{guards_block}
Explanation and Improvement Rules (when the smell is detected):
- "justification": bullet points, each a complete sentence presenting concrete evidence
  from the representation; at least 120 characters in total.
- "suggestions": concrete, actionable recommendations; each action_title MUST follow the
  exact format "[{smell.id}] - <short action title>" with a description of one to three
  sentences; the suggestions section as a whole must total at least 120 characters.
- extensions.evidence: cite the location of each piece of evidence as a dot-path relative
  to the operation (e.g. "summary", "responses.200.description",
  "parameters.0.name") with a short excerpt.
- extensions.confidence: your confidence in the classification, 0.0-1.0.
{SEVERITY_RUBRIC}

When the smell is NOT detected, return smell_detected=false with empty justification and
suggestions and no extensions."""


def build_user_message(erd_yaml: str) -> str:
    """The Appendix-A {openapi_json} slot — the ERD, nothing else."""
    return f"""\
OpenAPI Specification (reduced endpoint representation, YAML):

{erd_yaml}"""


def assemble_messages(smell_id: str, erd_yaml: str) -> dict:
    """Exact payload pieces for the detection call (consumed by hermes.llm in M4)."""
    smell = SMELLS[smell_id]
    return {
        "prompt_version": PROMPT_VERSIONS[smell_id],
        "system": build_system_prompt(smell),
        "user": build_user_message(erd_yaml),
    }
