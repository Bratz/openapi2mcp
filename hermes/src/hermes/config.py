"""HermesConfig — single owner of runtime defaults (DECISIONS.md M0).

Precedence: CLI flag > environment variable > default. CLI flags parse as None
so this module can tell "not given" from an explicit value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

class ConfigError(Exception):
    """Invalid configuration (bad env value or flag); maps to CLI exit code 2."""


DEFAULT_DETECT_MODEL = "claude-haiku-4-5"
DEFAULT_CONSOLIDATE_MODEL = "claude-sonnet-5"
DEFAULT_CONCURRENCY = 8
DEFAULT_SEED = 42

# $/MTok (input, output) as of 2026-07 — prices drift; update alongside 03-DEPLOYMENT §4.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
}
CACHE_READ_FACTOR = 0.1
CACHE_WRITE_FACTOR = 1.25

# Estimation constants for `hermes estimate` (docs/01-ARCHITECTURE §6).
EST_INPUT_TOKENS_PER_CALL = 1500
EST_OUTPUT_TOKENS_PER_CALL = 400
EST_CONSOLIDATION_RATE = 0.6


def _default_out_dir() -> Path:
    """<hermes project>/runs when running from a checkout, else ./runs."""
    project = Path(__file__).resolve().parents[2]
    if (project / "pyproject.toml").exists():
        return project / "runs"
    return Path("runs")


@dataclass
class HermesConfig:
    detect_model: str = DEFAULT_DETECT_MODEL
    consolidate_model: str = DEFAULT_CONSOLIDATE_MODEL
    concurrency: int = DEFAULT_CONCURRENCY
    seed: int = DEFAULT_SEED
    out_dir: Path = field(default_factory=_default_out_dir)
    cache_db: Path | None = None  # resolved to out_dir/cache.db when None
    max_cost_usd: float | None = None
    max_tokens_detect: int = 2000
    max_tokens_consolidate: int = 1500

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        if self.cache_db is None:
            self.cache_db = self.out_dir / "cache.db"
        if self.concurrency < 1:
            raise ConfigError(f"concurrency must be >= 1, got {self.concurrency}")
        if self.max_tokens_detect < 1 or self.max_tokens_consolidate < 1:
            raise ConfigError("max_tokens settings must be positive")

    @classmethod
    def resolve(cls, args=None) -> "HermesConfig":
        """Env overrides, then CLI args (only when the flag was actually given)."""

        def env(name: str, cast=str):
            raw = os.environ.get(name)
            if raw in (None, ""):
                return None
            try:
                return cast(raw)
            except (ValueError, TypeError) as exc:
                raise ConfigError(f"invalid value for {name}: {raw!r}") from exc

        values = {
            "detect_model": env("HERMES_DETECT_MODEL"),
            "consolidate_model": env("HERMES_CONSOLIDATE_MODEL"),
            "concurrency": env("HERMES_CONCURRENCY", int),
            "out_dir": env("HERMES_OUT", Path),
            "cache_db": env("HERMES_CACHE_DB", Path),
            "max_cost_usd": env("HERMES_MAX_COST_USD", float),
        }
        if args is not None:
            arg_map = {
                "detect_model": getattr(args, "detect_model", None),
                "consolidate_model": getattr(args, "consolidate_model", None),
                "concurrency": getattr(args, "concurrency", None),
                "seed": getattr(args, "seed", None),
                "out_dir": getattr(args, "out", None),
            }
            for key, val in arg_map.items():
                if val is not None:
                    values[key] = Path(val) if key == "out_dir" else val
        return cls(**{k: v for k, v in values.items() if v is not None})


def call_cost_usd(
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> float:
    """Cost of one call. Unknown models cost 0.0 (recorded, flagged in run.json)."""
    if model not in PRICES_PER_MTOK:
        return 0.0
    price_in, price_out = PRICES_PER_MTOK[model]
    return (
        input_tokens * price_in
        + cache_read_input_tokens * price_in * CACHE_READ_FACTOR
        + cache_creation_input_tokens * price_in * CACHE_WRITE_FACTOR
        + output_tokens * price_out
    ) / 1_000_000
