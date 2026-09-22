"""OracleJudge: the constraint's checker on ground-truth states, or on probe outputs
(``oracle-on-probes``). Produces the same key set as JevJudge."""

from __future__ import annotations

from lejudge.constraints.library import check
from lejudge.judge.base import build_state, plan_questions
from lejudge.types import Constraint, GroundTruthState, JudgeResult, StepFacts
from lejudge.vocab.pusht import Vocab


class OracleJudge:
    def __init__(self, vocab: Vocab, on_probes: bool = False):
        self.vocab = vocab
        self.on_probes = on_probes
        self.name = "oracle-on-probes" if on_probes else "oracle"

    def judge(
        self,
        facts: dict[str, list[StepFacts]],
        constraints: list[Constraint],
        *,
        states: dict[str, list[GroundTruthState]] | None = None,
    ) -> JudgeResult:
        if states is None:
            raise ValueError("OracleJudge needs `states` (index 0 = current state, then one per fact)")
        built = build_state(facts, constraints)
        specs = plan_questions(built, constraints)
        p: dict[str, dict[str, list[float]]] = {}
        conf: dict[str, dict[str, float | None]] = {}
        for k, seq in facts.items():
            p[k] = {}
            conf[k] = {}
            st = states[k]
            assert len(st) == len(seq) + 1, "states must have one more entry than facts"
            for c in constraints:
                tr = check(c, st, self.vocab)
                if c.family == "never":
                    p[k][c.id] = [1.0 if v else 0.0 for v in tr.steps]
                    conf[k][c.id] = None
                elif c.family == "always":
                    p[k][c.id] = [0.0 if v else 1.0 for v in tr.steps]
                    conf[k][c.id] = None
                elif c.family == "soft":
                    frac_ok = 1.0 - (sum(tr.steps) / len(tr.steps) if tr.steps else 0.0)
                    level = int(round(frac_ok * 4))
                    vec = [0.0] * 5
                    vec[level] = 1.0
                    p[k][c.id] = vec
                    conf[k][c.id] = 1.0
                else:
                    p[k][c.id] = [0.0] * (2 * len(seq))
                    conf[k][c.id] = None
        return JudgeResult(p=p, confidence=conf, latency_ms=0.0, input_tokens=0, output_tokens=0, response_model=self.name, cache_hit=True, n_calls=0, judge_name=self.name, keys=[s.key for s in specs])
