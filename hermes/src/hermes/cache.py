"""Content-hash response cache (docs/01-ARCHITECTURE §5).

key = sha256(erd_yaml + smell_id + prompt_version + model). Shared across runs
(runs/cache.db) — safe because the key covers everything that affects output.
This cache is the PRIMARY resume mechanism: an interrupted scan re-runs
cheaply because completed (endpoint, smell) pairs replay from here.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from hermes.schemas.models import AgentResponse


def cache_key(erd_yaml: str, smell_id: str, prompt_version: str, model: str) -> str:
    payload = f"{erd_yaml}\x00{smell_id}\x00{prompt_version}\x00{model}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    """Thread-safe sqlite-backed cache. `read_enabled=False` implements --no-cache
    (bypass reads, still write)."""

    def __init__(self, db_path: Path | str, *, read_enabled: bool = True):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.read_enabled = read_enabled
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS cache ("
                "key TEXT PRIMARY KEY, response_json TEXT NOT NULL, "
                "model TEXT NOT NULL, created_at TEXT NOT NULL)"
            )
            self._conn.commit()

    def get(self, key: str) -> AgentResponse | None:
        return self.get_as(key, AgentResponse)

    def get_as(self, key: str, model_cls):
        """Typed fetch — consolidations cache ConsolidationResponse in the same table."""
        if not self.read_enabled:
            return None
        with self._lock:
            row = self._conn.execute("SELECT response_json FROM cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return model_cls.model_validate_json(row[0])

    def put(self, key: str, response, model: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, response_json, model, created_at) VALUES (?, ?, ?, ?)",
                (key, response.model_dump_json(), model, datetime.now(timezone.utc).isoformat()),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
