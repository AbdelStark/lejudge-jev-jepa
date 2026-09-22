"""Response cache and tracing helpers (RFC-0008).

Every external call goes through ``CachedCaller.call``. Key = sha256(model_id, bank_version,
canonical_state_json, sorted_question_json). Modes: ``live`` (call on miss), ``offline``
(raise on miss), ``refresh`` (call and store under a new ``uid`` — consistency study).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CACHE_PATH = Path(os.environ.get("LEJUDGE_CACHE", "artifacts/cache/jev.sqlite"))
MODES = ("live", "offline", "refresh")


class CacheMiss(RuntimeError):
    pass


def mode() -> str:
    m = os.environ.get("LEJUDGE_MODE", "offline")
    if m not in MODES:
        raise ValueError(f"LEJUDGE_MODE must be one of {MODES}, got {m!r}")
    return m


@dataclass
class CachedResponse:
    payload: dict[str, Any]
    response_model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cache_hit: bool
    key: str


def cache_key(model_id: str, bank_version: str, canonical_state: str, questions_json: str, uid: str = "") -> str:
    h = hashlib.sha256()
    for part in (model_id, bank_version, canonical_state, questions_json, uid):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class Cache:
    def __init__(self, path: Path | str = DEFAULT_CACHE_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), timeout=60)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS responses (
                key TEXT PRIMARY KEY, model_id TEXT, bank_version TEXT, uid TEXT,
                response TEXT, response_model TEXT, latency_ms REAL,
                input_tokens INTEGER, output_tokens INTEGER, created REAL)"""
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON responses(model_id, bank_version)")
        # step-level memo for hard-family questions: (model, bank, constraint text, facts json) -> P(true)
        self._conn.execute("CREATE TABLE IF NOT EXISTS stepfacts (key TEXT PRIMARY KEY, model_id TEXT, bank_version TEXT, p REAL, created REAL)")
        self._conn.commit()

    @staticmethod
    def step_key(model_id: str, bank_version: str, constraint_text: str, family: str, facts_json: str) -> str:
        return hashlib.sha256("\x00".join((model_id, bank_version, family, constraint_text, facts_json)).encode()).hexdigest()

    def get_steps(self, keys: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i : i + 500]
            q = f"SELECT key, p FROM stepfacts WHERE key IN ({','.join('?' * len(chunk))})"
            out.update({k: float(p) for k, p in self._conn.execute(q, chunk).fetchall()})
        return out

    def put_steps(self, rows: list[tuple[str, str, str, float]]) -> None:
        now = time.time()
        self._conn.executemany("INSERT OR REPLACE INTO stepfacts VALUES (?,?,?,?,?)", [(k, m, b, p, now) for k, m, b, p in rows])
        self._conn.commit()

    def count_steps(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM stepfacts").fetchone()[0])

    def get(self, key: str) -> CachedResponse | None:
        row = self._conn.execute(
            "SELECT response, response_model, latency_ms, input_tokens, output_tokens FROM responses WHERE key=?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return CachedResponse(
            payload=json.loads(row[0]),
            response_model=row[1],
            latency_ms=float(row[2]),
            input_tokens=int(row[3] or 0),
            output_tokens=int(row[4] or 0),
            cache_hit=True,
            key=key,
        )

    def put(self, key: str, model_id: str, bank_version: str, uid: str, resp: CachedResponse) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                key,
                model_id,
                bank_version,
                uid,
                json.dumps(resp.payload, sort_keys=True),
                resp.response_model,
                resp.latency_ms,
                resp.input_tokens,
                resp.output_tokens,
                time.time(),
            ),
        )
        self._conn.commit()

    def count(self, model_id: str | None = None) -> int:
        if model_id is None:
            return int(self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0])
        return int(self._conn.execute("SELECT COUNT(*) FROM responses WHERE model_id=?", (model_id,)).fetchone()[0])

    def close(self) -> None:
        self._conn.close()


_CACHES: dict[str, Cache] = {}


def get_cache(path: Path | str | None = None) -> Cache:
    p = str(Path(path) if path is not None else DEFAULT_CACHE_PATH)
    if p not in _CACHES:
        _CACHES[p] = Cache(p)
    return _CACHES[p]


class CachedCaller:
    """Wraps a callable ``fn() -> (payload, response_model, input_tokens, output_tokens)``."""

    def __init__(self, cache: Cache | None = None, retries: int = 1, retry_sleep_s: float = 0.2):
        self.cache = cache or get_cache()
        self.retries = retries
        self.retry_sleep_s = retry_sleep_s

    def call(
        self,
        model_id: str,
        bank_version: str,
        canonical_state: str,
        questions_json: str,
        fn: Callable[[], tuple[dict[str, Any], str, int, int]],
        uid: str = "",
        force_mode: str | None = None,
    ) -> CachedResponse:
        m = force_mode or mode()
        key = cache_key(model_id, bank_version, canonical_state, questions_json, uid)
        if m != "refresh":
            hit = self.cache.get(key)
            if hit is not None:
                return hit
        if m == "offline":
            raise CacheMiss(f"offline mode and no cached response for {model_id} key={key[:12]}")
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                t0 = time.perf_counter()
                payload, response_model, in_tok, out_tok = fn()
                dt = (time.perf_counter() - t0) * 1000.0
                resp = CachedResponse(payload, response_model, dt, in_tok, out_tok, False, key)
                self.cache.put(key, model_id, bank_version, uid, resp)
                return resp
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < self.retries:
                    time.sleep(self.retry_sleep_s)
        assert last_err is not None
        raise last_err


# ------------------------------------------------------------------------------ traces
class TraceWriter:
    """JSONL per episode: ``artifacts/traces/<run_id>/ep_<n>.jsonl``, one line per CEM iteration."""

    def __init__(self, run_id: str, root: Path | str = "artifacts/traces"):
        self.run_id = run_id
        self.dir = Path(root) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._fh = None
        self._episode: int | None = None
        self.context: dict[str, Any] = {}

    def open_episode(self, episode: int) -> None:
        self.close()
        self._episode = episode
        self._fh = open(self.dir / f"ep_{episode}.jsonl", "a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        if self._fh is None:
            raise RuntimeError("open_episode() first")
        rec = {"run": self.run_id, "episode": self._episode, **self.context, **record}
        self._fh.write(json.dumps(rec, sort_keys=True, default=_json_default) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    @staticmethod
    def read(path: Path | str) -> list[dict[str, Any]]:
        out = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out


def _json_default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "to_json"):
        return o.to_json()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")
