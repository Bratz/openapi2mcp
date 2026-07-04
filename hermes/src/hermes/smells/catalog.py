"""The nine paper-actual smells (docs/00-SPEC.md §3, docs/05-PAPER-FACTS.md §2-3).

5 documentation smells (Khan et al. taxonomy) + 4 REST smells (PATH and METHOD
merged, as the paper's system actually ran). Each Smell carries everything the
Appendix-A prompt template needs; per-smell prompt versions live in
hermes.smells.prompts (bump on any prompt-affecting edit — cache keys depend on it).
"""

from __future__ import annotations

from dataclasses import dataclass, field

DOCUMENTATION = "documentation"
REST = "rest"


@dataclass(frozen=True)
class Smell:
    id: str
    category: str  # documentation | rest
    display_name: str  # as shown in reports, e.g. "Path & Method"
    definition: str
    scoping_rule: str  # the Appendix-A "Analyze ONLY ..." clause
    occurs_when: tuple[str, ...]  # the "typically occurs when" few-shot slot
    false_positive_guards: tuple[str, ...] = field(default=())


SMELLS: dict[str, Smell] = {
    s.id: s
    for s in [
        Smell(
            id="LAZY",
            category=DOCUMENTATION,
            display_name="Lazy",
            definition=(
                "The documentation is incomplete, vague, or generic, failing to provide "
                "essential information about the operation's purpose, behavior, inputs, "
                "outputs, or usage. This includes missing examples or unclear explanations "
                "that hinder correct understanding and usage."
            ),
            scoping_rule="Analyze ONLY the operation's summary and description fields.",
            occurs_when=(
                'The summary is a generic stub such as "Get data", "Create order", or a '
                "one-word label, and there is no description at all.",
                "The description merely restates the summary or the operationId "
                '(e.g. summary "List invoices", description "Lists the invoices.") '
                "without adding purpose, behavior, or constraints.",
                "The text is long but content-free: circular wording that restates the "
                "endpoint name without explaining semantics, preconditions, or effects "
                '(e.g. "This endpoint allows updating the profile settings of the profile").',
                "The summary and description are semantically inconsistent — they describe "
                "different behaviors or contradict each other.",
                "Understanding the endpoint's intent requires external or tribal knowledge "
                "not present in the text (no examples, no constraints, no context).",
            ),
            false_positive_guards=(
                "A concise summary paired with a description that states purpose, behavior, "
                "and constraints is NOT lazy — brevity alone is not the smell.",
            ),
        ),
        Smell(
            id="BLOATED",
            category=DOCUMENTATION,
            display_name="Bloated",
            definition=(
                "The documentation is excessively verbose with limited informational value: "
                "filler, marketing language, or repetition that adds no operational guidance."
            ),
            scoping_rule="Analyze ONLY the operation's summary and description fields.",
            occurs_when=(
                "The description is dominated by promotional adjectives and superlatives "
                "(marketing filler) with little or no concrete API guidance.",
                "The same fact is restated in different words several times, so the text "
                "could shrink to a fraction of its length with no information loss.",
                "Boilerplate paragraphs are duplicated verbatim from other operations and "
                "do not describe this endpoint specifically.",
            ),
            false_positive_guards=(
                "A long description that is information-dense (constraints, examples, error "
                "semantics, paging rules) is NOT bloated — length alone is not the smell.",
            ),
        ),
        Smell(
            id="TANGLED",
            category=DOCUMENTATION,
            display_name="Tangled",
            definition=(
                "The documentation mixes unrelated concerns — business rules, authentication "
                "setup, error-handling procedures, changelog notes, support/escalation "
                "instructions — within the same textual fragment, without structure."
            ),
            scoping_rule="Analyze ONLY the operation's summary and description fields.",
            occurs_when=(
                "A single description paragraph mixes several unrelated concerns — credential "
                "acquisition, business approval thresholds, handling of specific internal error "
                "codes, deprecation timelines, regulatory citations — with no structure.",
                "Operational or organizational instructions (ticketing, escalation contacts, "
                "internal routing, data-retention policy) are interleaved with the endpoint's "
                "functional description.",
            ),
            false_positive_guards=(
                "A description that briefly notes auth scope requirements alongside behavior "
                "is NOT tangled if the concerns are relevant and coherently presented.",
            ),
        ),
        Smell(
            id="FRAGMENTED",
            category=DOCUMENTATION,
            display_name="Fragmented",
            definition=(
                "Essential information is dispersed or missing because the operation "
                "references schemas or components that are not present in the specification "
                "document — broken or unresolvable references make the documentation incomplete."
            ),
            scoping_rule=(
                "Analyze the whole reduced representation; the decisive evidence is any "
                '"$unresolved" marker, which means a $ref target does not exist in the document.'
            ),
            occurs_when=(
                'A response or request-body schema appears as {"$unresolved": '
                '"#/definitions/SomeType"} — the referenced definition is missing from the document.',
                "A parameter or security scheme reference points at a component that is "
                "never defined, so a consumer cannot know the shape of the data.",
            ),
            false_positive_guards=(
                '"$truncated" markers are NOT fragmentation — they mean the reducer shortened '
                "a deep or large schema for analysis, not that the document is broken.",
                "If every reference resolves (no $unresolved markers anywhere), do NOT report this smell.",
            ),
        ),
        Smell(
            id="EXCESS_STRUCTURED",
            category=DOCUMENTATION,
            display_name="Excess Structured",
            definition=(
                "Class-like definitions, nested structures, or formal specifications are "
                "written inside natural-language fields (summary/description) instead of the "
                "specification's schema sections."
            ),
            scoping_rule=(
                "Analyze ONLY natural-language fields (summary and description). Normative "
                "schema structure under definitions/components is the CORRECT place for "
                "structure and is never this smell."
            ),
            occurs_when=(
                "A description spells out a type definition in prose — a JSON-like object "
                "layout, an interface declaration, or a field-by-field pseudo-schema table — "
                "duplicating what belongs in the schema section.",
                "Any formal notation embedded in prose (XML/XSD fragments, protobuf messages, "
                "SQL DDL, class diagrams in text) that defines structure rather than showing "
                "an example value.",
            ),
            false_positive_guards=(
                "Small inline examples of a request/response VALUE (sample payload) are "
                "illustrative, not structural definitions — judge whether the text defines "
                "types/structure versus showing an example value.",
                "Do NOT flag the schemas in the parameters/responses sections — only "
                "structure embedded in prose.",
            ),
        ),
        Smell(
            id="PATH_AND_METHOD",
            category=REST,
            display_name="Path & Method",
            definition=(
                "The URI is action-oriented or inconsistently named rather than "
                "resource-oriented, and/or the HTTP method is used inappropriately or "
                "inconsistently with its semantics."
            ),
            scoping_rule=(
                "Analyze the path string, the HTTP method, and the sibling-path context "
                "(context.path_siblings) for consistency."
            ),
            occurs_when=(
                "Verbs are embedded in the path (/getUsers, /updateStatus, "
                "/orders/createNewOrder) instead of resource nouns.",
                "The method contradicts the documented behavior: GET performs resource "
                "creation or state changes, POST is used for updates or deletes that PUT/PATCH/DELETE "
                "should model, or GET carries a request body.",
                "Paths are long or semantically opaque, contain internal acronyms, or are "
                "inconsistent in naming/casing with sibling paths.",
                "A resource identifier is passed as a query parameter while sibling "
                "endpoints model it as a path segment.",
            ),
            false_positive_guards=(
                "POST used for a complex search/query operation with a structured filter body "
                "is an accepted pattern when the documentation justifies it (filters too large "
                "or too structured for a URL) — do NOT flag such operations.",
                "Judge method semantics from the documented behavior, not from the path name alone.",
            ),
        ),
        Smell(
            id="INPUT",
            category=REST,
            display_name="Input",
            definition=(
                "Parameters or request bodies are weakly specified and lack semantic "
                "clarification, preventing a consumer from constructing a valid request."
            ),
            scoping_rule="Analyze ONLY the parameters and request body (including inlined schemas).",
            occurs_when=(
                "Fields are specified only by type (string, integer) with no description of "
                "meaning, format, or constraints.",
                "Parameter names are ambiguous abbreviations or internal acronyms (e.g. amt, "
                "dt, prcCd) with no explanation.",
                "Valid ranges, accepted formats, enumerated values, or behavioral impact are "
                "missing (magic values undocumented).",
                "Required fields are not marked required, or requiredness has no functional rationale.",
            ),
            false_positive_guards=(
                "A body parameter wrapper named 'body' whose referenced schema documents "
                "every field (descriptions, formats, required list) is well specified — "
                "judge the resolved schema, not the wrapper name.",
            ),
        ),
        Smell(
            id="RESPONSE",
            category=REST,
            display_name="Response",
            definition=(
                "Response schemas, status codes, or error handling are inconsistent or "
                "insufficiently described, so a consumer cannot interpret results or plan "
                "around failures."
            ),
            scoping_rule="Analyze ONLY the responses section (including inlined schemas).",
            occurs_when=(
                "A response schema is present (often via $ref) but carries minimal "
                "explanatory text: no clarification of what the payload means, how success "
                "and error cases differ, or how a consumer should interpret the returned "
                "structure.",
                'A success response has no schema at all, or a generic envelope such as '
                '{"status": string, "data": object} where "data" is an unconstrained object.',
                'Response descriptions are placeholders ("OK", "Success", "Successful '
                'Response") with no semantic clarification of the payload.',
                "Status codes contradict the documented semantics (e.g. 200 where 201/204 is "
                "meant), or a single schema serves contradictory outcomes.",
                "Failure modes mentioned in the documentation have no corresponding error "
                "responses, or a state-changing operation documents no error contract at all.",
            ),
            false_positive_guards=(
                "A response set with typed success schema, described payload semantics, and "
                "relevant 4xx entries is NOT smelly merely because some theoretical status "
                "code is absent.",
            ),
        ),
        Smell(
            id="SECURITY",
            category=REST,
            display_name="Security",
            definition=(
                "Authentication or authorization requirements are missing, unclear, "
                "contradictory, or undocumented at the operation level."
            ),
            scoping_rule=(
                "Analyze the operation's security references, its parameters (for ad-hoc "
                "auth-bearing headers), its description, and the declared schemes in "
                "context.security_schemes."
            ),
            occurs_when=(
                "No security scheme is referenced although the operation clearly needs auth "
                "(money movement, personal data) or its parameters carry auth-like values "
                "(user ids, API keys, session or operator identifiers) as ordinary "
                "parameters outside any scheme.",
                "The description states an auth requirement in prose (a token, a login, a "
                "role or entitlement, admin-only access) while the operation references no "
                "security scheme.",
                "The operation references a security scheme that is never defined in the "
                "document's security definitions.",
                "A scheme is technically defined but gives no operational guidance (how to "
                "obtain credentials, required scopes, constraints).",
            ),
            false_positive_guards=(
                "An operation referencing a defined scheme whose definition documents how to "
                "obtain and use credentials is NOT smelly.",
            ),
        ),
    ]
}

DOCUMENTATION_SMELLS = tuple(s for s in SMELLS.values() if s.category == DOCUMENTATION)
REST_SMELLS = tuple(s for s in SMELLS.values() if s.category == REST)
