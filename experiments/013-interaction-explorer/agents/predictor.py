"""Predictive world model — tests action rules against actual transitions.

Given an action rule and the current symbolic state, predicts what SHOULD
change. Then compares against what ACTUALLY changed. Updates rule confidence.
"""

from .dsl import ActionRule, WorldModel


def identify_controllable(
    transitions: list[dict],
) -> tuple[int | None, str]:
    """Operationally identify the controllable object.

    The controllable object is the one whose position changes are
    most consistently attributable to directional inputs.

    Returns:
        (color_id, evidence_string) or (None, "insufficient data")
    """
    if len(transitions) < 3:
        return None, "Need at least 3 transitions"

    color_move_counts: dict[int, int] = {}
    total_moves = 0

    for t in transitions:
        if t["blocked"]:
            continue
        total_moves += 1
        for change in t["sym_changes"]:
            if change.get("type") == "changed" and "center" in change:
                cid = change.get("color_id", -1)
                color_move_counts[cid] = color_move_counts.get(cid, 0) + 1

    if total_moves == 0:
        return None, "No successful moves recorded"

    best_color = None
    best_ratio = 0.0
    for color, count in color_move_counts.items():
        ratio = count / total_moves
        if ratio > best_ratio:
            best_ratio = ratio
            best_color = color

    if best_color is not None and best_ratio > 0.5:
        return best_color, f"color {best_color} changed position in {best_ratio:.0%} of successful actions"

    return None, f"No color consistently moved (best was {best_ratio:.0%})"
