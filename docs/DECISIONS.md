# Decision log

Format: date · decision · options considered · reason · RFC. Append only.

| Date | Decision | Options | Reason | RFC |
|---|---|---|---|---|
| 2026-09-22 | MVP scope is PushT only | PushT; PushT+TwoRoom | One environment fully instrumented beats two half-done; adapters keyed by env keep the door open | SPEC |
| 2026-09-22 | Vocabulary default 3×3 grid + edge flags | 3×3; 4×4 | Smaller state; edge flags recover the boundary precision the edge constraints need; 4×4 kept for ablation | 0001 |
| 2026-09-22 | Soft constraints use Score | Noul; Score | Graded penalty without a threshold; confidence available | 0003 |
| 2026-09-22 | Judge the top-K elites per CEM iteration | all N; final elites only | A already ranks; K = 16 keeps questions ≤ 512; final_only ablated | 0004 |
| 2026-09-22 | Language goals are stretch, not MVP | in MVP; stretch | Demo works with goal images; retrieval adds a second judge path to validate | 0005 |
| | swm API snapshot (fill at M0) | | | skill |
| | LeWM PushT reproduced number (fill at M0) | | | |
| | Constraint library freeze hash (fill at M3) | | | 0002 |
| | PREREG hash (fill before M7) | | | 0006 |
