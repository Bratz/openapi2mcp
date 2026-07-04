"""M1 tests: spec loading, enumeration, filters (docs/02-TEST-PLAN.md §4)."""

import json
from pathlib import Path

import pytest

from hermes.spec_loader import SpecError, filter_operations, load_spec

GOLDEN = Path(__file__).parent.parent / "fixtures" / "golden"
BANCS = Path(__file__).parent.parent.parent.parent / "api-docs.json"

ALL_SMELLS = {
    "LAZY", "BLOATED", "TANGLED", "FRAGMENTED", "EXCESS_STRUCTURED",
    "PATH_AND_METHOD", "INPUT", "RESPONSE", "SECURITY",
}


@pytest.fixture(scope="module")
def golden():
    return load_spec(GOLDEN / "seeded_spec.yaml")


@pytest.fixture(scope="module")
def expected():
    raw = json.loads((GOLDEN / "expected.json").read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def test_golden_swagger2_fixture_loads(golden):
    assert golden.version == "swagger2"
    assert golden.title == "Atlas Retail Banking API"
    assert len(golden.operations) == 40


def test_golden_oas3_fixture_loads():
    spec = load_spec(GOLDEN / "seeded_spec_oas3.yaml")
    assert spec.version == "oas3"
    assert len(spec.operations) == 6


def test_oracle_matches_fixture_operation_ids(golden, expected):
    oas3 = load_spec(GOLDEN / "seeded_spec_oas3.yaml")
    ids = [op.operation_id for op in golden.operations + oas3.operations]
    assert None not in ids
    assert len(set(ids)) == len(ids), "duplicate operationIds"
    assert set(expected) == set(ids)


def test_swagger2_fixture_covers_all_nine_smells_at_least_four_times(golden, expected):
    # 02-TEST-PLAN §2.1: the ≥4-per-smell floor applies to the PRIMARY fixture alone;
    # OAS3 labels must not pad the counts.
    swagger2_ids = {op.operation_id for op in golden.operations}
    counts: dict[str, int] = {}
    for op_id in swagger2_ids:
        for smell in expected[op_id]["smells"]:
            counts[smell] = counts.get(smell, 0) + 1
    assert set(counts) == ALL_SMELLS
    assert all(n >= 4 for n in counts.values()), counts


def test_oracle_uses_only_catalog_smell_ids(expected):
    for entry in expected.values():
        assert set(entry["smells"]).issubset(ALL_SMELLS), entry


def test_clean_control_count(golden, expected):
    swagger2_ids = {op.operation_id for op in golden.operations}
    clean_swagger2 = [i for i in swagger2_ids if not expected[i]["smells"]]
    assert len(clean_swagger2) == 8  # 02-TEST-PLAN §2.1
    clean_all = [i for i, v in expected.items() if not v["smells"]]
    assert len(clean_all) == 10  # + getMerchant, onboardMerchant (OAS3)


@pytest.mark.skipif(not BANCS.exists(), reason="api-docs.json not present")
def test_bancs_spec_loads_with_927_operations():
    spec = load_spec(BANCS)
    assert spec.version == "swagger2"
    assert spec.title == "TCS BaNCS RestFul API Documentation"
    assert len(spec.operations) == 927


def test_garbage_input_raises_spec_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("just: a\nrandom: doc\n")
    with pytest.raises(SpecError):
        load_spec(bad)
    notjson = tmp_path / "bad.json"
    notjson.write_text("{broken")
    with pytest.raises(SpecError):
        load_spec(notjson)
    with pytest.raises(SpecError):
        load_spec(tmp_path / "missing.yaml")
    with pytest.raises(SpecError):
        load_spec(tmp_path)  # a directory


def test_malformed_but_parseable_specs_do_not_crash(tmp_path):
    # tags: null, tags as scalar string, info: null — real-world sloppiness
    # must not escape as TypeError/AttributeError (SpecError contract or graceful skip).
    spec = tmp_path / "sloppy.yaml"
    spec.write_text(
        "swagger: '2.0'\n"
        "info:\n"
        "paths:\n"
        "  /a:\n"
        "    get:\n"
        "      tags:\n"
        "      operationId: opA\n"
        "      responses: {'200': {description: ok}}\n"
        "  /b:\n"
        "    get:\n"
        "      tags: Accounts\n"
        "      operationId: opB\n"
        "      responses: {'200': {description: ok}}\n"
    )
    loaded = load_spec(spec)
    by_id = {op.operation_id: op for op in loaded.operations}
    assert by_id["opA"].tags == ()
    assert by_id["opB"].tags == ()  # scalar string is NOT split into characters
    assert loaded.title == "sloppy"  # falls back to file stem


def test_trace_method_is_oas3_only(tmp_path):
    body = (
        "paths:\n"
        "  /x:\n"
        "    trace:\n"
        "      operationId: traceX\n"
        "      responses: {'200': {description: ok}}\n"
    )
    s2 = tmp_path / "s2.yaml"
    s2.write_text("swagger: '2.0'\n" + body)
    o3 = tmp_path / "o3.yaml"
    o3.write_text("openapi: 3.0.0\ninfo: {title: t, version: '1'}\n" + body)
    assert len(load_spec(s2).operations) == 0
    assert len(load_spec(o3).operations) == 1


def test_tag_filter(golden):
    loans = filter_operations(golden.operations, tags=["Loans"])
    assert loans
    assert all("Loans" in op.tags for op in loans)


def test_path_glob_filter_is_case_sensitive(golden):
    accounts = filter_operations(golden.operations, path_globs=["/accounts/*"])
    assert accounts
    assert all(op.path.startswith("/accounts/") for op in accounts)
    assert filter_operations(golden.operations, path_globs=["/ACCOUNTS/*"]) == []


def test_sample_is_deterministic_and_stratified(golden):
    a = filter_operations(golden.operations, sample=8, seed=7)
    b = filter_operations(golden.operations, sample=8, seed=7)
    assert a == b
    assert len(a) == 8
    # 4 tags, 8 picks -> every tag represented (round-robin stratification)
    assert {op.tags[0] for op in a} == {"Accounts", "Payments", "Customers", "Loans"}


def test_sample_without_seed_is_an_error(golden):
    with pytest.raises(ValueError):
        filter_operations(golden.operations, sample=8)


def test_max_endpoints_truncates(golden):
    assert len(filter_operations(golden.operations, max_endpoints=5)) == 5


def test_endpoint_key_format(golden):
    op = next(o for o in golden.operations if o.operation_id == "getAccount")
    assert op.endpoint_key == "GET /accounts/{accountId}"
