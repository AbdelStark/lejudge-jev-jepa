"""JevJudge: one ``system_one`` call per CEM iteration (split above the question cap)."""

from __future__ import annotations

import os
from typing import Any

from lejudge.judge.bank import BANK_VERSION, MAX_QUESTIONS_PER_CALL, spec_json, to_sdk
from lejudge.judge.base import assemble, build_state, empty_result, plan_questions
from lejudge.judge.cache import CachedCaller, CacheMiss, get_cache
from lejudge.judge.state import canonical_json
from lejudge.types import Constraint, GroundTruthState, JudgeResult, StepFacts

JEV_REQUEST_MODEL = os.environ.get("LEJUDGE_JEV_MODEL", "jev-latest")
JEV_PIN_PREFIX = "jev-1.13"


class JevJudge:
    name = "jev"

    def __init__(
        self,
        model: str = JEV_REQUEST_MODEL,
        pin_prefix: str = JEV_PIN_PREFIX,
        cache_path: str | None = None,
        max_questions: int = MAX_QUESTIONS_PER_CALL,
        uid: str = "",
        client: Any | None = None,
        timeout_s: float = 30.0,
    ):
        self.model = model
        self.pin_prefix = pin_prefix
        self.caller = CachedCaller(get_cache(cache_path))
        self.max_questions = max_questions
        self.uid = uid
        self._client = client
        self.timeout_s = timeout_s
        self.bank_version = BANK_VERSION

    @property
    def model_id(self) -> str:
        # Cache key uses the pinned family, not the moving alias.
        return self.pin_prefix if self.model in ("jev-latest", "jev-preview") else self.model

    def _get_client(self) -> Any:
        if self._client is None:
            from typesafe_sdk import TypeSafeClient

            self._client = TypeSafeClient(timeout=self.timeout_s)
        return self._client

    def judge(
        self,
        facts: dict[str, list[StepFacts]],
        constraints: list[Constraint],
        *,
        states: dict[str, list[GroundTruthState]] | None = None,
    ) -> JudgeResult:
        built = build_state(facts, constraints)
        specs = plan_questions(built, constraints)
        state_json = canonical_json(built.state)
        answers: dict[str, dict[str, Any]] = {}
        latency = 0.0
        in_tok = out_tok = 0
        hits = 0
        n_calls = 0
        response_model = ""
        for start in range(0, len(specs), self.max_questions):
            chunk = specs[start : start + self.max_questions]
            qjson = canonical_json([spec_json(s) for s in chunk])

            def fn(chunk=chunk) -> tuple[dict[str, Any], str, int, int]:
                client = self._get_client()
                qs = {s.key: to_sdk(s) for s in chunk}
                resp = client.system_one(state=built.state, questions=qs, model=self.model)
                if not resp.model.startswith(self.pin_prefix):
                    raise RuntimeError(f"response model {resp.model!r} is not pinned to {self.pin_prefix}")
                payload = {k: _answer_json(a) for k, a in resp.answers.items()}
                usage = resp.usage
                return payload, resp.model, int(usage.input_tokens or 0), int(usage.output_tokens or 0)

            try:
                r = self.caller.call(self.model_id, self.bank_version, state_json, qjson, fn, uid=self.uid)
            except CacheMiss:
                raise
            except Exception:  # noqa: BLE001 — judge failure never blocks planning
                res = empty_result(facts, constraints, self.name, failed=True)
                res.keys = [s.key for s in specs]
                return res
            n_calls += 1
            hits += int(r.cache_hit)
            latency += r.latency_ms
            in_tok += r.input_tokens
            out_tok += r.output_tokens
            response_model = r.response_model
            answers.update(r.payload)
        return assemble(
            built,
            constraints,
            specs,
            answers,
            latency_ms=latency,
            input_tokens=in_tok,
            output_tokens=out_tok,
            response_model=response_model,
            cache_hit=(hits == n_calls and n_calls > 0),
            n_calls=n_calls,
            name=self.name,
        )


def _answer_json(a: Any) -> dict[str, Any]:
    t = getattr(a, "type", None)
    if t == "noul":
        return {"type": "noul", "noul": float(a.noul)}
    if t == "score":
        return {
            "type": "score",
            "score": float(a.score),
            "confidence": float(a.confidence),
            "probabilities": {str(k): float(v) for k, v in a.probabilities.items()},
        }
    if t == "choice":
        return {"type": "choice", "choice": a.choice, "confidence": float(a.confidence), "probabilities": dict(a.probabilities)}
    raise TypeError(f"unknown answer type {t!r}")
