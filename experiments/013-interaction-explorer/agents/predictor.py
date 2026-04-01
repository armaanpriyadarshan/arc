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


def predict_and_score(
    model: WorldModel,
    action: str,
    sym_changes: list[dict],
    blocked: bool,
    changes: int,
) -> str:
    """Score the current action rule against observed changes.

    Returns a text summary of prediction accuracy for the LLM to see.
    """
    rule = model.action_rules.get(action)
    if not rule:
        return f"No rule for {action} yet."

    rule.test_count += 1
    report = []

    if rule.effect == "move" and rule.precondition == "path_clear":
        if blocked:
            report.append(f"BLOCKED as expected (path not clear in {rule.direction})")
            rule.success_count += 1
        elif changes > 5:
            ctrl_color = model.controllable_color()
            if ctrl_color is not None:
                for change in sym_changes:
                    if change.get("color_id") == ctrl_color and change.get("type") == "changed":
                        center_change = change.get("center", {})
                        if center_change:
                            was = center_change.get("was", [0, 0])
                            now = center_change.get("now", [0, 0])
                            dr = now[0] - was[0]
                            dc = now[1] - was[1]

                            expected_dr = {"up": -1, "down": 1}.get(rule.direction, 0) * rule.distance
                            expected_dc = {"left": -1, "right": 1}.get(rule.direction, 0) * rule.distance

                            dir_match = (
                                (expected_dr != 0 and (dr * expected_dr > 0)) or
                                (expected_dc != 0 and (dc * expected_dc > 0)) or
                                (expected_dr == 0 and expected_dc == 0)
                            )

                            if dir_match:
                                report.append(f"CORRECT: controllable moved {rule.direction}")
                                rule.success_count += 1
                            else:
                                report.append(f"WRONG: expected {rule.direction} but got delta=({dr},{dc})")
                            break
                else:
                    report.append(f"Controllable color not found in changes")
            else:
                report.append("No controllable object identified yet")
        else:
            report.append(f"Expected move but only {changes} cells changed")

    elif rule.effect == "no_op":
        if changes == 0:
            report.append("CORRECT: no-op as predicted")
            rule.success_count += 1
        else:
            report.append(f"WRONG: expected no-op but {changes} cells changed")

    else:
        report.append(f"Rule type '{rule.effect}' — {changes} cells changed")

    if rule.test_count > 0:
        rule.confidence = rule.accuracy

    return "; ".join(report) if report else "No prediction made."


def classify_interaction(
    sym_changes: list[dict],
    blocked: bool,
    changes: int,
    died: bool,
    score_delta: int,
    ctrl_color: int | None,
) -> str:
    """Classify what happened when the controllable contacted an object.

    Returns one of: blocked, collected, died, score_up, transformed, pass_through, unknown
    """
    if died:
        return "died"
    if score_delta > 0:
        return "score_up"
    if blocked and changes < 10:
        return "blocked"

    # Check if the target object disappeared (collected)
    for change in sym_changes:
        if change.get("type") == "disappeared" and change.get("color_id") != ctrl_color:
            return "collected"

    # Check if an object changed color/shape (transformed)
    for change in sym_changes:
        if change.get("type") == "changed" and change.get("color_id") != ctrl_color:
            if "shape" in change or "size" in change:
                return "transformed"

    if changes < 5:
        return "pass_through"

    return "unknown"
