# Jev 1.13 jaggedness — condensed (source: https://docs.typesafe.ai/model-jaggedness/jev-1.13.md, reviewed 2026-09-17)

| # | Failure mode | Do instead (LeJudge rule) |
|---|---|---|
| 1 | Literal reading of scoping words, negations, implied conditions | Write the exact condition; boundary cases in criteria; split interpretation into two literal questions |
| 2 | Math and numbers: no counting, weak on numeric representations, Score levels not for interpolation | Count in code; bucket numbers into words; use Score expectation only for thresholding |
| 3 | Date/time comparison | Extract parts as Choice, compare in code (not used in LeJudge) |
| 4 | Indirection, double negatives | Direct instructions; name the relevant state fields |
| 5 | Large state with irrelevant detail | Filter in code; send only needed fields; cap candidates per call |
| 6 | Adversarial content steers answers | Separate trusted and untrusted fields; literal criteria; test injections |
| 7 | Contradictory instruction vs criteria | Align polarity and vocabulary |
| 8 | No guaranteed structural invariants (Noul vs Choice, P vs 1−P) | Ask each decision one way; enforce identities in code; never carry thresholds across primitives |
| 9 | Generation | Use templates or a generative model; Jev only picks |

Consistency: std-dev 0.0102 across 15 repeats in TypeSafe's own test; expect similar and verify with `lejudge consistency`.
