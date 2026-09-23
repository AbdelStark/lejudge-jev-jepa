"""Start-state relevance filters for the follow-up planning study (Study 2).

A (start, goal) window is *relevant* for a constraint set when the set is **satisfiable** from the
start and goal states and the expert trajectory between them is **in tension** with it, i.e. the
demonstrated way of reaching the goal violates the constraint. Both tests use only dataset states,
never a planner, so the filter is pre-registrable and identical for every condition.
"""

from __future__ import annotations

import numpy as np

from lejudge.constraints import Library, check
from lejudge.types import GroundTruthState
from lejudge.vocab import Vocab

FILTERS = ("none", "relevant")


def _cell(vocab: Vocab, g: GroundTruthState) -> str:
    return str(vocab.cell_name(vocab.block_centroid(np.array(g.block_xy), np.array(g.block_angle))))


def satisfiable(
    set_name: str, s0: GroundTruthState, goal: GroundTruthState, vocab: Vocab, lib: Library
) -> bool:
    ok = True
    for cid in lib.sets[set_name]:
        if cid == "centre_avoid":
            ok &= _cell(vocab, s0) != "centre" and _cell(vocab, goal) != "centre"
        elif cid in ("no_contact_first3", "approach_below", "no_push_tilted"):
            ok &= not s0.contact
        elif cid in (
            "edges_never",
            "top_never",
            "corner_avoid",
            "agent_bottom_only",
            "stay_left_half",
            "no_upside_down",
            "upright_always",
        ):
            c = lib.get(cid)
            ok &= (
                not check(c, [s0, s0], vocab).episode and not check(c, [goal, goal], vocab).episode
            )
    return bool(ok)


def in_tension(
    set_name: str, window: list[GroundTruthState], vocab: Vocab, lib: Library, action_block: int = 5
) -> bool:
    """The expert trajectory (env-step states from start to goal) violates at least one
    constraint of the set at the planner's cadence, excluding the start and goal frames."""
    cadence = window[::action_block]
    if cadence[-1] is not window[-1]:
        cadence = cadence + [window[-1]]
    inner = window[1:-1]
    for cid in lib.sets[set_name]:
        c = lib.get(cid)
        if cid == "centre_avoid":
            if any(_cell(vocab, x) == "centre" for x in inner):
                return True
        elif cid == "no_contact_first3":
            if any(x.contact for x in window[1 : 3 * action_block + 1]):
                return True
        else:
            if check(c, cadence, vocab).episode:
                return True
    return False


def relevant(
    set_name: str, window: list[GroundTruthState], vocab: Vocab, lib: Library, action_block: int = 5
) -> bool:
    return satisfiable(set_name, window[0], window[-1], vocab, lib) and in_tension(
        set_name, window, vocab, lib, action_block
    )
