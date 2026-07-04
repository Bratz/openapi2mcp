"""The single LLM seam (docs/01-ARCHITECTURE §8): nothing else imports anthropic.

AnthropicLLM wraps the Claude API: structured outputs via messages.parse(),
a concurrency semaphore, a pool-wide 429 backoff, one application-level retry
on schema-validation failure (with doubled max_tokens — truncation is the
dominant residual cause since the schema itself is enforced server-side), and
usage/cost records.

Failure contract: every per-call failure — schema validation after retry,
empty/refused output, API errors after SDK retries — surfaces as
DetectorFailure. The graph layer records it and continues; one endpoint never
aborts a scan (docs/01-ARCHITECTURE §4).

Note on cache_control: the per-smell system blocks are far below the models'
minimum cacheable prefix (4096 tokens on Haiku 4.5), so the breakpoint is
currently a harmless no-op kept for larger future prompts — cost estimates
must NOT assume cache savings (DECISIONS.md M4).

Tests inject fakes at one of two seams: an object with the LLM interface
(detect/consolidate), or a fake `client` passed to AnthropicLLM.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Protocol

from pydantic import ValidationError

from hermes.config import HermesConfig, call_cost_usd
from hermes.schemas.models import AgentResponse, ConsolidationResponse, UsageRecord
from hermes.smells.prompts import assemble_messages

CONSOLIDATOR_SYSTEM = """\
You are the central Smell Detector coordinator consolidating per-smell findings for ONE
API endpoint into a coherent diagnostic report.

Rules:
- You may only keep findings, drop exact duplicates (the same defect reported by two
  analysts), or adjust a finding's severity by at most one step (with a reason).
- You must NEVER invent new findings or new evidence.
- Then judge the endpoint overall: "agent-ready" (no material obstacles for an autonomous
  agent), "needs adaptation" (usable but degraded), or "not agent-consumable" (an agent
  would likely fail against it).
"""


class DetectorFailure(Exception):
    """LLM structured-output failure after retries (detection OR consolidation).

    Contract (docs/01-ARCHITECTURE §4, §8): the graph layer records it and the
    scan always completes — detection failures become detector_error entries in
    run.json; consolidation failures fall back to the rule-based verdict.
    Carries `usage` (UsageRecord) with the estimated spend of the failed attempts
    so cost rollups and HERMES_MAX_COST_USD stay honest.
    """

    def __init__(self, message: str, usage: UsageRecord | None = None):
        super().__init__(message)
        self.usage = usage


class LLM(Protocol):
    def detect(self, smell_id: str, erd_yaml: str) -> tuple[AgentResponse, UsageRecord]: ...

    def consolidate(self, endpoint_summary: str, findings_json: str) -> tuple[ConsolidationResponse, UsageRecord]: ...


class AnthropicLLM:
    def __init__(self, config: HermesConfig, client=None):
        if client is None:
            import anthropic  # the only real-client construction site in hermes

            client = anthropic.Anthropic()
        self._client = client
        self._config = config
        self._semaphore = threading.BoundedSemaphore(config.concurrency)
        self._throttle_lock = threading.Lock()
        self._throttle_until = 0.0  # monotonic deadline; workers pause until then after a 429

    # -- public interface ----------------------------------------------------

    def detect(self, smell_id: str, erd_yaml: str) -> tuple[AgentResponse, UsageRecord]:
        parts = assemble_messages(smell_id, erd_yaml)
        return self._parse_with_retry(
            model=self._config.detect_model,
            max_tokens=self._config.max_tokens_detect,
            system=parts["system"],
            user=parts["user"],
            output_format=AgentResponse,
        )

    def consolidate(self, endpoint_summary: str, findings_json: str) -> tuple[ConsolidationResponse, UsageRecord]:
        user = f"Endpoint:\n{endpoint_summary}\n\nFindings (JSON):\n{findings_json}"
        return self._parse_with_retry(
            model=self._config.consolidate_model,
            max_tokens=self._config.max_tokens_consolidate,
            system=CONSOLIDATOR_SYSTEM,
            user=user,
            output_format=ConsolidationResponse,
        )

    # -- internals ------------------------------------------------------------

    def _parse_with_retry(self, *, model, max_tokens, system, user, output_format):
        messages = [{"role": "user", "content": user}]
        # The SDK raises ValidationError from inside parse() and discards the
        # response, so failed attempts' real usage is unrecoverable — estimate
        # it (worst-case output = max_tokens) so cost rollups stay honest.
        failed = _estimate_usage(model, system, user, max_tokens)
        try:
            return self._call(model, max_tokens, system, messages, output_format)
        except ValidationError:
            # One application-level retry. The schema is enforced server-side by
            # parse(); residual validation failures are dominated by max_tokens
            # truncation, so double the budget. The failed output is not
            # available client-side, so this is a clean resend, not a repair.
            try:
                parsed, record = self._call(model, max_tokens * 2, system, messages, output_format)
            except ValidationError as exc2:
                raise DetectorFailure(
                    f"schema validation failed twice: {exc2}", usage=_double(failed)
                ) from exc2
            except DetectorFailure as exc2:
                exc2.usage = _merge(failed, exc2.usage)
                raise
            except Exception as exc2:  # API errors after SDK retries
                raise DetectorFailure(f"API failure on retry: {exc2!r}", usage=failed) from exc2
            return parsed, _merge(failed, record)
        except DetectorFailure:
            raise
        except Exception as exc:  # API errors after SDK retries, malformed payloads
            raise DetectorFailure(f"API failure: {exc!r}") from exc

    def _call(self, model, max_tokens, system, messages, output_format):
        self._wait_if_throttled()
        with self._semaphore:
            try:
                response = self._client.messages.parse(
                    model=model,
                    max_tokens=max_tokens,
                    system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                    messages=messages,
                    output_format=output_format,
                )
            except Exception as exc:
                if _is_rate_limit(exc):
                    self._trip_throttle()
                raise
        usage = getattr(response, "usage", None)
        record = UsageRecord(
            model=model,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )
        record.cost_usd = call_cost_usd(
            model,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            cache_read_input_tokens=record.cache_read_input_tokens,
            cache_creation_input_tokens=record.cache_creation_input_tokens,
        )
        parsed = response.parsed_output
        if parsed is None:
            # Refusal or text-block-free response: parse() returns None without
            # raising (verified against anthropic 0.116).
            raise DetectorFailure(
                f"model returned no parseable output (stop_reason={getattr(response, 'stop_reason', '?')})"
            )
        if isinstance(parsed, dict):  # tolerate fakes returning plain dicts
            parsed = output_format.model_validate(parsed)
        if isinstance(parsed, str):
            try:
                parsed = output_format.model_validate(json.loads(parsed))
            except json.JSONDecodeError as exc:
                raise ValidationError.from_exception_data("malformed-json", []) from exc
        return parsed, record

    def _wait_if_throttled(self) -> None:
        # Pool-wide backoff (docs/01-ARCHITECTURE §6): after a 429 every worker
        # pauses until the deadline BEFORE taking a semaphore slot. Loop because
        # another 429 may extend the deadline mid-sleep.
        while True:
            with self._throttle_lock:
                remaining = self._throttle_until - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(remaining)

    def _trip_throttle(self, seconds: float = 60.0) -> None:
        with self._throttle_lock:
            self._throttle_until = max(self._throttle_until, time.monotonic() + seconds)


def _is_rate_limit(exc: Exception) -> bool:
    return exc.__class__.__name__ == "RateLimitError" or getattr(exc, "status_code", None) == 429


def _estimate_usage(model: str, system: str, user: str, max_tokens: int) -> UsageRecord:
    input_est = int((len(system) + len(user)) / 3.5)
    record = UsageRecord(model=model, input_tokens=input_est, output_tokens=max_tokens, estimated=True)
    record.cost_usd = call_cost_usd(model, input_tokens=input_est, output_tokens=max_tokens)
    return record


def _double(record: UsageRecord) -> UsageRecord:
    doubled = record.model_copy()
    doubled.input_tokens *= 2
    doubled.output_tokens *= 2
    doubled.cost_usd *= 2
    return doubled


def _merge(a: UsageRecord, b: UsageRecord | None) -> UsageRecord:
    if b is None:
        return a
    merged = b.model_copy()
    merged.input_tokens += a.input_tokens
    merged.output_tokens += a.output_tokens
    merged.cache_read_input_tokens += a.cache_read_input_tokens
    merged.cache_creation_input_tokens += a.cache_creation_input_tokens
    merged.cost_usd += a.cost_usd
    merged.estimated = True
    return merged
