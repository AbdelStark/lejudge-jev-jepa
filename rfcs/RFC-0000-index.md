# RFC index and dependency order

| RFC | Title | Depends on | MVP |
|---|---|---|---|
| 0001 | Probes and vocabulary | — | yes |
| 0002 | Constraint library and oracle checkers | 0001 (vocab names) | yes |
| 0003 | Judge: state, question bank, aggregation, baselines | 0001, 0002 | yes |
| 0004 | JevCost and planner integration | 0003 | yes |
| 0005 | Language goals by retrieval | 0001, 0003 | stretch |
| 0006 | Evaluation harness, statistics and report | 0003, 0004 | yes |
| 0007 | Demo Space | 0004 | yes |
| 0008 | Reproducibility: caching, tracing, pinning | 0003 | yes |

Status vocabulary: `Draft` → `Accepted` → `Implemented`. An RFC is `Implemented` when its acceptance tests pass in CI and its decision-log entry is written.

Read order for a new agent: 0008 (how we record), 0001, 0002, 0003, 0004, 0006, 0007, then 0005.
