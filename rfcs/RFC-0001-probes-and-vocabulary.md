# RFC-0001 — Probes and vocabulary

**Status** Implemented · **Scope** MVP · **Depends on** —

## Summary

Turn a LeWM latent into a handful of words Jev can judge. Two steps: probes (latent → symbols) and vocabulary (symbols → words). Both are versioned artefacts with measured accuracy.

## Motivation

Jev is text-only and weak at numbers; LeWM's paper shows position and angle are linearly recoverable from its latents (r 0.97–0.99 on PushT). The bridge therefore exists; this RFC makes it a measured, versioned component.

## Probes

### Targets (PushT)

| Symbol | Type | Ground truth source |
|---|---|---|
| `agent_xy` | 2 floats, normalised to [0,1]² | env state |
| `block_xy` | 2 floats | env state |
| `block_angle` | angle; regress `(sin θ, cos θ)` | env state |
| `contact` | binary | distance(agent, block polygon) < ε |

### Training

- Input: `z` after the projector (192-d), from encoding dataset frames with the released checkpoint.
- Also fit on **imagined** latents: roll the predictor open-loop for h = 1..H from dataset frames with dataset actions; targets are the true future states. Report probe error as a function of h. This is the number the paper needs for the error decomposition.
- Heads: linear (ridge, λ by CV) and MLP (2 × 256, dropout 0.1). Linear is the default; MLP reported.
- Splits: by episode; 80/10/10.
- Metrics: R², MSE per target; for `contact`, AUROC and accuracy; **bucket accuracy** (fraction of steps whose vocabulary word matches the ground-truth word) per vocabulary field.

### Artefact

`artifacts/probes/pusht/linear@1/{weights.pt, meta.json}` with `meta.json` = checkpoint hash, dataset hash, R², bucket accuracies, imagined-latent error curve.

## Vocabulary `pusht@1`

| Field | Words | Rule |
|---|---|---|
| `block`, `agent` | `top-left, top-centre, top-right, centre-left, centre, centre-right, bottom-left, bottom-centre, bottom-right` | 3×3 grid over the arena |
| `block_edge` | `none, left edge, right edge, top edge, bottom edge` | block polygon within margin m of a wall; nearest wall wins |
| `block_angle` | `upright, tilted left, tilted right, on its side left, on its side right, upside down` | 6 bins around the canonical T orientation |
| `contact` | `true / false` | probe logit > 0 |
| `block_speed` | `still, slow, fast` | ‖Δblock_xy‖ thresholds |

Thresholds are in `vocab/pusht@1.yaml`. A `pusht@2` (4×4 grid) exists for the ablation.

## Acceptance tests

- Probe R² ≥ 0.95 on positions, ≥ 0.85 on angle (linear) on encoded frames.
- Bucket accuracy ≥ 0.90 (`block`), ≥ 0.80 (`block_angle`), ≥ 0.90 (`block_edge`) on encoded frames; curve reported on imagined latents.
- `words()` is pure and deterministic; property test: same symbols → same words.
- No numeric value appears in any `StepFacts` field.

## Alternatives considered

- Decoder → VLM caption → Jev: slow, noisy, not needed.
- Sending coordinates: rejected per jaggedness guidance.
- Object-centric latents (C-JEPA): future work; would give per-object facts directly.

## Open questions

- Margin m for `block_edge`: pick from the oracle's own definition so the words match the checker.
