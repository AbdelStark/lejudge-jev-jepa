# RFC-0005 — Language goals by retrieval (stretch)

**Status** Draft · **Scope** stretch, not MVP · **Depends on** RFC-0001, RFC-0003

## Summary

Replace the goal image with a sentence. Encode a bank of candidate goal frames from the dataset, describe each with probes and vocabulary, ask Jev one Choice over the described candidates for the user's sentence, and use the chosen frame as `o_g`. The demo becomes fully language-driven.

## Design

1. Bank: 200 frames sampled from the PushT dataset stratified over the 3×3 grid × 6 angle bins; each frame has `z`, symbolic state, words.
2. Query: user sentence (≤ 200 chars) placed under `goal_text`; candidates under `candidates` as `{g1: {...words}, ...}`.
3. Shortlist in code: filter candidates whose words share at least one grid or angle term with the sentence? No — that is the keyword trap. Instead run one Choice over all 200 (options `g1..g200 + none`); Jev's Choice handles hundreds of options as in the re-ranking and semantic-find cookbooks.
4. Gate: if `confidence < 0.6` or choice is `none`, ask the user for a clearer sentence (demo) or fall back to the goal image (study).
5. Validation: oracle description of each candidate versus the sentence's intended cell/angle; report top-1 and top-5 retrieval accuracy on 200 sentences; planning success with retrieved goals versus true goal images.

## Acceptance

- Top-1 ≥ 0.8 on unambiguous sentences; abstention ≥ 0.8 on ambiguous ones (a held-out set).
- Planning success with retrieved goals within 5 points of goal-image planning.

## Open

- Whether to retrieve from imagined reachable latents instead of dataset frames (avoids unreachable goals). Post-stretch.
