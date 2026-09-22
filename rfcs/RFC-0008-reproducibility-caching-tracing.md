# RFC-0008 — Reproducibility: caching, tracing, pinning

**Status** Draft · **Scope** MVP · **Depends on** — (read first)

## Summary

Every Jev and LLM response is cached; every planning step is traced; every model and artefact is pinned. The paper regenerates offline.

## Cache

- SQLite `artifacts/cache/jev.sqlite`: key `sha256(model_id, bank_version, canonical_state_json, sorted_question_json)` → response JSON, `response_model`, latency, tokens, timestamp.
- Canonicalisation: sorted keys; candidate ids renumbered in cost order; floats never present.
- Modes: `live` (call on miss), `offline` (raise on miss), `refresh` (call and overwrite; used only for the consistency study, with `uid` field).
- LLM baselines share the cache with `model_id` in the key.
- Cache shipped with the repo (compressed) so `make paper` needs no keys.

## Traces

JSONL per episode `artifacts/traces/<run_id>/ep_<n>.jsonl`, one line per CEM iteration:

```json
{"run":"...","episode":12,"step":3,"iter":1,"cond":"jev","set":"edges+upright","seed":0,
 "elites":["k1",...],"facts":{...},"judge":{"p":{...},"held":[...],"latency_ms":118,"input_tokens":2410,"response_model":"jev-1.13.0","cache_hit":false},
 "costA":[...],"penalty":[...],"chosen_action":[...],"executed_state":{...},"oracle":{"edges_never":false,"upright_always":true}}
```

`run_id` = git commit + config hash. Traces feed both studies and the demo.

## Pinning

- Jev: request `jev-1.13`; assert `response_model` startswith `jev-1.13`; record in every trace.
- LeWM checkpoint: HF revision hash in `configs/pusht.yaml`.
- stable-worldmodel, typesafe-sdk, torch: exact versions in `uv.lock`.
- Probes, vocab, constraint library: versioned ids in every result row.

## Determinism

- Seeds for env reset, CEM sampling, probe training, bootstrap.
- CEM on GPU may be nondeterministic across hardware; record device and CUDA version; the paper claims reproducibility of *tables* from cached traces, not bitwise replay of planning.

## Acceptance

- `lejudge report --offline` succeeds on a clean checkout with the shipped cache.
- A trace round-trips into the demo's probability panel.
- CI job runs the judge-only study on a 50-item subset offline.
