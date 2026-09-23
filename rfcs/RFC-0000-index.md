# RFC index and dependency order

| RFC | Title | Depends on | MVP | Status |
|---|---|---|---|---|
| 0001 | Probes and vocabulary | — | yes | Implemented |
| 0002 | Constraint library and oracle checkers | 0001 (vocab names) | yes | Implemented |
| 0003 | Judge: state, question bank, aggregation, baselines | 0001, 0002 | yes | Implemented |
| 0004 | JevCost and planner integration | 0003 | yes | Implemented, amended (population-wide judging) |
| 0005 | Language goals by retrieval | 0001, 0003 | stretch | Draft (not built) |
| 0006 | Evaluation harness, statistics and report | 0003, 0004 | yes | Implemented |
| 0007 | Demo Space | 0004 | yes | Implemented locally; not deployed |
| 0008 | Reproducibility: caching, tracing, pinning | 0003 | yes | Implemented |

Status vocabulary: `Draft` → `Accepted` → `Implemented`. An RFC is `Implemented` when its acceptance tests pass in CI and its decision-log entry is written.

Read order for a new agent: 0008 (how we record), 0001, 0002, 0003, 0004, 0006, 0007, then 0005.
