"""M2 tests: ERD construction (docs/02-TEST-PLAN.md §4 reducer bullet)."""

from pathlib import Path

import pytest
import yaml

from hermes.reducer import ERD, reduce_operation
from hermes.spec_loader import find_operation, load_spec

GOLDEN = Path(__file__).parent.parent / "fixtures" / "golden"
SNAPSHOTS = Path(__file__).parent.parent / "fixtures" / "snapshots"
BANCS = Path(__file__).parent.parent.parent.parent / "api-docs.json"


@pytest.fixture(scope="module")
def golden():
    return load_spec(GOLDEN / "seeded_spec.yaml")


def erd_for(spec, operation_id) -> ERD:
    op = next(o for o in spec.operations if o.operation_id == operation_id)
    return reduce_operation(spec, op)


def test_refs_are_inlined(golden):
    erd = erd_for(golden, "getAccount")
    doc = yaml.safe_load(erd.yaml)
    schema = doc["operation"]["responses"]["200"]["schema"]
    assert "$ref" not in schema
    assert schema["properties"]["iban"]["description"] == "IBAN of the account."
    assert not erd.truncation_applied


def test_unresolvable_ref_becomes_unresolved_marker(golden):
    erd = erd_for(golden, "getCustomerPreferences")
    doc = yaml.safe_load(erd.yaml)
    schema = doc["operation"]["responses"]["200"]["schema"]
    assert schema == {"$unresolved": "#/definitions/CustomerPreferences"}


def test_depth_cap_truncates_deep_ref_chains(tmp_path):
    # A -> B -> C -> D -> E chain: with cap 4, the 5th-level ref is $truncated.
    defs = {
        chr(65 + i): {"type": "object", "properties": {"next": {"$ref": f"#/definitions/{chr(66 + i)}"}}}
        for i in range(4)
    }
    defs["E"] = {"type": "string"}
    spec_file = tmp_path / "deep.yaml"
    spec_file.write_text(yaml.safe_dump({
        "swagger": "2.0",
        "info": {"title": "Deep", "version": "1"},
        "paths": {"/x": {"get": {"operationId": "x", "responses": {
            "200": {"description": "ok", "schema": {"$ref": "#/definitions/A"}}}}}},
        "definitions": defs,
    }))
    spec = load_spec(spec_file)
    doc = yaml.safe_load(erd_for(spec, "x").yaml)
    # A,B,C,D inline (depths 1-4); the 5th link (E) exceeds the cap.
    node = doc["operation"]["responses"]["200"]["schema"]
    for _ in range(4):
        node = node["properties"]["next"]
    assert node == {"$truncated": "E"}


def test_cyclic_refs_do_not_hang(tmp_path):
    spec_file = tmp_path / "cycle.yaml"
    spec_file.write_text(yaml.safe_dump({
        "swagger": "2.0",
        "info": {"title": "Cycle", "version": "1"},
        "paths": {"/x": {"get": {"operationId": "x", "responses": {
            "200": {"description": "ok", "schema": {"$ref": "#/definitions/Node"}}}}}},
        "definitions": {"Node": {"type": "object", "properties": {"child": {"$ref": "#/definitions/Node"}}}},
    }))
    spec = load_spec(spec_file)
    text = erd_for(spec, "x").yaml
    assert "$truncated" in text


def test_deterministic_output(golden):
    a = erd_for(golden, "createRefund").yaml
    b = erd_for(golden, "createRefund").yaml
    assert a == b


def test_token_cap_applies_progressive_truncation(tmp_path):
    big_schema = {
        "type": "object",
        "properties": {f"field_{i:03d}": {"type": "string", "description": "x" * 120} for i in range(300)},
    }
    spec_file = tmp_path / "big.yaml"
    spec_file.write_text(yaml.safe_dump({
        "swagger": "2.0",
        "info": {"title": "Big", "version": "1"},
        "paths": {"/x": {"post": {"operationId": "x",
            "parameters": [{"name": "body", "in": "body", "schema": {"$ref": "#/definitions/L0"}}],
            "responses": {"200": {"description": "ok", "schema": {"$ref": "#/definitions/L0"}}}}}},
        "definitions": {
            "L0": {"type": "object", "properties": {"a": {"$ref": "#/definitions/L1"}, "big": big_schema}},
            "L1": {"type": "object", "properties": {"b": {"$ref": "#/definitions/L2"}, "big": big_schema}},
            "L2": {"type": "object", "properties": {"c": {"$ref": "#/definitions/L3"}, "big": big_schema}},
            "L3": {"type": "object", "properties": {"big": big_schema}},
        },
    }))
    spec = load_spec(spec_file)
    erd = erd_for(spec, "x")
    assert erd.truncation_applied
    assert yaml.safe_load(erd.yaml)["truncation_applied"] is True
    # Cap is enforced on the FULL rendered document, no slack.
    assert len(erd.yaml) / 3.5 <= 6000


def test_token_cap_collapses_giant_enums(tmp_path):
    spec_file = tmp_path / "enums.yaml"
    spec_file.write_text(yaml.safe_dump({
        "swagger": "2.0",
        "info": {"title": "Enums", "version": "1"},
        "paths": {"/x": {"get": {"operationId": "x", "parameters": [
            {"name": "code", "in": "query", "type": "string", "enum": [f"CODE_{i:05d}" for i in range(3000)]}
        ], "responses": {"200": {"description": "ok"}}}}},
    }))
    spec = load_spec(spec_file)
    erd = erd_for(spec, "x")
    assert erd.truncation_applied
    assert len(erd.yaml) / 3.5 <= 6000
    doc = yaml.safe_load(erd.yaml)
    enum = doc["operation"]["parameters"][0]["enum"]
    assert len(enum) == 11 and "$truncated" in enum[-1]


def test_sibling_context(golden):
    doc = yaml.safe_load(erd_for(golden, "getAccount").yaml)
    siblings = doc["context"]["path_siblings"]
    assert siblings["other_methods"] == []
    assert "/accounts" in siblings["related_paths"]
    assert "/accounts/{accountId}/balance" in siblings["related_paths"]
    assert len(siblings["related_paths"]) <= 10
    # listAccounts and createAccount share /accounts: other_methods populated
    doc2 = yaml.safe_load(erd_for(golden, "listAccounts").yaml)
    assert doc2["context"]["path_siblings"]["other_methods"] == ["POST"]


def test_security_scheme_context_present(golden):
    doc = yaml.safe_load(erd_for(golden, "initiatePayment").yaml)
    assert doc["context"]["security_schemes"] == {"bearerAuth": "apiKey"}


def test_oas3_security_scheme_context():
    spec = load_spec(GOLDEN / "seeded_spec_oas3.yaml")
    doc = yaml.safe_load(erd_for(spec, "getMerchant").yaml)
    assert doc["context"]["security_schemes"] == {"bearerAuth": "http"}


def test_api_description_capped_at_1500(tmp_path):
    spec_file = tmp_path / "longinfo.yaml"
    spec_file.write_text(yaml.safe_dump({
        "swagger": "2.0",
        "info": {"title": "T", "version": "1", "description": "d" * 5000},
        "paths": {"/x": {"get": {"operationId": "x", "responses": {"200": {"description": "ok"}}}}},
    }))
    spec = load_spec(spec_file)
    doc = yaml.safe_load(erd_for(spec, "x").yaml)
    assert len(doc["context"]["api_description"]) == 1500


def test_tag_descriptions_only_for_operation_tags(golden):
    doc = yaml.safe_load(erd_for(golden, "getAccount").yaml)
    assert list(doc["context"]["tag_descriptions"]) == ["Accounts"]


def test_find_operation(golden):
    op = find_operation(golden, "get /accounts/{accountId}")
    assert op.operation_id == "getAccount"
    with pytest.raises(KeyError):
        find_operation(golden, "PATCH /accounts/{accountId}")
    with pytest.raises(KeyError):
        find_operation(golden, "nonsense")


def test_no_llm_imports():
    import hermes.reducer as mod

    source = Path(mod.__file__).read_text()
    assert "anthropic" not in source


@pytest.mark.skipif(not BANCS.exists(), reason="api-docs.json not present")
def test_bancs_erd_snapshot():
    import os

    spec = load_spec(BANCS)
    op = spec.operations[0]
    erd = reduce_operation(spec, op)
    snapshot = SNAPSHOTS / "bancs_first_endpoint.erd.yaml"
    if os.environ.get("HERMES_RECORD_SNAPSHOTS") == "1":
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(f"# {op.endpoint_key}\n{erd.yaml}")
        pytest.skip("snapshot recorded (HERMES_RECORD_SNAPSHOTS=1)")
    # A missing baseline is a FAILURE, not a record trigger — recording is
    # opt-in so a regressed reducer can never bless its own output.
    assert snapshot.exists(), "baseline missing; run with HERMES_RECORD_SNAPSHOTS=1 to record"
    assert snapshot.read_text() == f"# {op.endpoint_key}\n{erd.yaml}"
