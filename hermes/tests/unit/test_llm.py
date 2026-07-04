"""M4 tests: LLM wrapper, schemas, config, cost math (docs/02-TEST-PLAN.md §4)."""

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from hermes.config import HermesConfig, call_cost_usd
from hermes.llm import AnthropicLLM, DetectorFailure
from hermes.schemas.models import (
    AgentResponse,
    EndpointInfo,
    Evidence,
    Extensions,
    Finding,
    Suggestion,
    build_finding,
    finding_id,
)

SRC = Path(__file__).parent.parent.parent / "src" / "hermes"


# ---------- fake anthropic client ----------


class FakeParseClient:
    """Mimics anthropic client .messages.parse for wrapper-level tests."""

    def __init__(self, outputs, usages=None):
        self.outputs = list(outputs)  # dicts (parsed_output payloads) or Exceptions
        self.usages = usages or []
        self.requests = []
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **kwargs):
        self.requests.append(kwargs)
        result = self.outputs.pop(0)
        if isinstance(result, Exception):
            raise result
        usage = self.usages.pop(0) if self.usages else SimpleNamespace(
            input_tokens=1200, output_tokens=300, cache_read_input_tokens=800, cache_creation_input_tokens=0
        )
        return SimpleNamespace(parsed_output=result, usage=usage)


GOOD = {
    "smell_detected": True,
    "justification": ["The summary is a generic stub providing no guidance about behavior or usage at all."],
    "suggestions": [{"action_title": "[LAZY] - Improve documentation", "description": "Add a real description."}],
    "extensions": {"severity": "high", "confidence": 0.9, "evidence": [{"location": "summary", "excerpt": "Get data"}]},
}


def _make_validation_error():
    try:
        AgentResponse.model_validate({"smell_detected": True, "justification": []})
    except ValidationError as exc:
        return exc
    raise AssertionError("expected ValidationError")


# ---------- wrapper behavior ----------


def test_detect_happy_path_builds_usage_and_response():
    client = FakeParseClient([GOOD])
    llm = AnthropicLLM(HermesConfig(), client=client)
    response, usage = llm.detect("LAZY", "endpoint: {}")
    assert response.smell_detected
    assert usage.model == "claude-haiku-4-5"
    assert usage.input_tokens == 1200 and usage.cache_read_input_tokens == 800
    # cost: (1200*1 + 800*0.1 + 300*5)/1e6
    assert usage.cost_usd == pytest.approx((1200 * 1 + 800 * 0.1 + 300 * 5) / 1e6)
    # prompt caching on the system block
    system = client.requests[0]["system"]
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_schema_validation_retry_resends_with_doubled_budget():
    client = FakeParseClient([_make_validation_error(), GOOD])
    llm = AnthropicLLM(HermesConfig(), client=client)
    response, usage = llm.detect("LAZY", "endpoint: {}")
    assert response.smell_detected
    assert len(client.requests) == 2
    # Clean resend (truncation is the dominant residual failure), doubled budget
    assert client.requests[1]["messages"] == client.requests[0]["messages"]
    assert client.requests[1]["max_tokens"] == client.requests[0]["max_tokens"] * 2
    # Failed attempt's estimated spend is merged into the returned record
    assert usage.estimated
    assert usage.cost_usd > 0
    assert usage.output_tokens > 300  # real 300 + estimated failed-attempt output


def test_schema_validation_failing_twice_raises_detector_failure():
    client = FakeParseClient([_make_validation_error(), _make_validation_error()])
    llm = AnthropicLLM(HermesConfig(), client=client)
    with pytest.raises(DetectorFailure):
        llm.detect("LAZY", "endpoint: {}")


def test_semaphore_bounds_concurrency():
    lock = threading.Lock()
    state = {"in_flight": 0, "max": 0}

    def slow_parse(**kwargs):
        with lock:
            state["in_flight"] += 1
            state["max"] = max(state["max"], state["in_flight"])
        time.sleep(0.02)
        with lock:
            state["in_flight"] -= 1
        return SimpleNamespace(parsed_output=dict(GOOD), usage=None)

    client = SimpleNamespace(messages=SimpleNamespace(parse=slow_parse))
    llm = AnthropicLLM(HermesConfig(concurrency=3), client=client)
    threads = [threading.Thread(target=llm.detect, args=("LAZY", "e: {}")) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert state["max"] <= 3


def test_rate_limit_trips_throttle_and_surfaces_as_detector_failure(monkeypatch):
    class RateLimitError(Exception):
        status_code = 429

    client = FakeParseClient([RateLimitError(), GOOD])
    llm = AnthropicLLM(HermesConfig(), client=client)
    with pytest.raises(DetectorFailure, match="RateLimitError"):
        llm.detect("LAZY", "e: {}")
    assert llm._throttle_until > 0
    slept = {}

    def fake_sleep(s):
        slept.setdefault("s", s)
        llm._throttle_until = 0.0  # simulate the window elapsing

    monkeypatch.setattr("hermes.llm.time.sleep", fake_sleep)
    response, _ = llm.detect("LAZY", "e: {}")
    assert response.smell_detected
    assert 0 < slept["s"] <= 60.0


def test_real_client_construction_is_forbidden_in_tests():
    with pytest.raises(RuntimeError, match="must not construct a real Anthropic client"):
        AnthropicLLM(HermesConfig())  # no injected client -> tries anthropic.Anthropic()


# ---------- schemas ----------


def test_agent_response_consistency_rules():
    with pytest.raises(ValidationError):
        AgentResponse.model_validate({"smell_detected": True, "justification": []})
    with pytest.raises(ValidationError):
        AgentResponse.model_validate(
            {"smell_detected": False, "justification": ["something"], "suggestions": [], "extensions": None}
        )
    assert not AgentResponse.model_validate({"smell_detected": False}).smell_detected


def _endpoint():
    return EndpointInfo(path="/x", method="GET", operation_id="x", tags=["T"])


def _response(**overrides):
    payload = json.loads(json.dumps(GOOD))
    payload.update(overrides)
    return AgentResponse.model_validate(payload)


def test_build_finding_id_deterministic_and_clamped():
    resp = _response(extensions={"severity": "high", "confidence": 3.7, "evidence": [{"location": "summary", "excerpt": "x"}]})
    kwargs = dict(
        response=resp, run_id="r1", api_title="T", endpoint=_endpoint(), smell_id="LAZY",
        category="documentation", model="m", prompt_version="lazy-v1",
    )
    a, b = build_finding(**kwargs), build_finding(**kwargs)
    assert a.id == b.id == finding_id("GET /x", "LAZY")
    assert a.extensions.confidence == 1.0


def test_build_finding_repairs_action_title_prefix():
    resp = _response(suggestions=[{"action_title": "Improve documentation", "description": "d"}])
    f = build_finding(
        response=resp, run_id="r", api_title="T", endpoint=_endpoint(), smell_id="LAZY",
        category="documentation", model="m", prompt_version="v",
    )
    assert f.suggestions[0].action_title == "[LAZY] - Improve documentation"


def test_build_finding_flags_short_justification_and_none_when_clean():
    short = _response(justification=["Too short."])
    f = build_finding(
        response=short, run_id="r", api_title="T", endpoint=_endpoint(), smell_id="LAZY",
        category="documentation", model="m", prompt_version="v",
    )
    assert f.detector.short_justification
    assert build_finding(
        response=AgentResponse(smell_detected=False), run_id="r", api_title="T",
        endpoint=_endpoint(), smell_id="LAZY", category="documentation", model="m", prompt_version="v",
    ) is None


def test_finding_schema_json_in_sync():
    on_disk = json.loads((SRC / "schemas" / "finding.schema.json").read_text())
    assert on_disk == Finding.model_json_schema(), (
        "finding.schema.json out of sync — regenerate from Finding.model_json_schema()"
    )


# ---------- config ----------


def test_config_env_and_args_precedence(monkeypatch):
    monkeypatch.setenv("HERMES_CONCURRENCY", "4")
    monkeypatch.setenv("HERMES_DETECT_MODEL", "env-model")
    args = SimpleNamespace(detect_model="cli-model", consolidate_model=None, concurrency=None, seed=None, out=None)
    cfg = HermesConfig.resolve(args)
    assert cfg.detect_model == "cli-model"  # CLI beats env
    assert cfg.concurrency == 4  # env beats default
    assert cfg.seed == 42  # default
    assert cfg.cache_db == cfg.out_dir / "cache.db"


def test_default_out_dir_is_project_runs():
    cfg = HermesConfig()
    assert cfg.out_dir.name == "runs"
    assert (cfg.out_dir.parent / "pyproject.toml").exists()


def test_unknown_model_costs_zero():
    assert call_cost_usd("fake-model", input_tokens=1000, output_tokens=1000) == 0.0
