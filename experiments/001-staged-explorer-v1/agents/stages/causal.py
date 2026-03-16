"""Stage 3: Causal Testing — test object interactions.

Deliberately move toward interesting objects and record what happens.
Each interaction becomes a test case for the world model.
"""

import json
import logging
from dataclasses import dataclass, field

from arcengine import FrameData, GameAction, GameState

from ..perception.differ import diff_frames
from ..perception.objects import ObjectCatalog, detect_objects, track_objects

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    """A single observed state transition — becomes a world model test case."""
    action: str
    grid_before: list[list[int]]
    grid_after: list[list[int]]
    score_before: int
    score_after: int
    state_before: str
    state_after: str
    num_changes: int
    description: str  # human-readable summary for LLM consumption


@dataclass
class CausalResult:
    transitions: list[Transition]
    catalog: ObjectCatalog
    actions_used: int

    def summary(self) -> str:
        lines = [f"Causal Testing ({self.actions_used} actions, {len(self.transitions)} transitions):"]
        for t in self.transitions:
            lines.append(
                f"  {t.action}: changes={t.num_changes} "
                f"score={t.score_before}->{t.score_after} "
                f"state={t.state_before}->{t.state_after} "
                f"| {t.description}"
            )
        return "\n".join(lines)

    def to_test_cases(self) -> list[dict]:
        """Export transitions as test cases for world model validation."""
        return [
            {
                "action": t.action,
                "grid_before": t.grid_before,
                "grid_after": t.grid_after,
                "score_before": t.score_before,
                "score_after": t.score_after,
            }
            for t in self.transitions
        ]


class CausalStage:
    """Deliberately explore to test object interactions."""

    def __init__(self, budget: int = 20) -> None:
        self.budget = budget

    def run(
        self,
        step_fn,
        current_frame: FrameData,
        catalog: ObjectCatalog,
    ) -> tuple[CausalResult, FrameData]:
        transitions: list[Transition] = []
        actions_used = 0
        frame = current_frame

        # Strategy: cycle through movement actions, recording every transition.
        # Focus on actions that produce change.
        movement_cycle = [
            GameAction.ACTION1, GameAction.ACTION3,
            GameAction.ACTION2, GameAction.ACTION4,
            GameAction.ACTION1, GameAction.ACTION1,
            GameAction.ACTION4, GameAction.ACTION4,
            GameAction.ACTION2, GameAction.ACTION2,
            GameAction.ACTION3, GameAction.ACTION3,
            GameAction.ACTION1, GameAction.ACTION4,
            GameAction.ACTION2, GameAction.ACTION3,
            GameAction.ACTION5,  # test non-movement actions too
            GameAction.ACTION1, GameAction.ACTION1,
            GameAction.ACTION4, GameAction.ACTION4,
        ]

        prev_objects = detect_objects(frame.frame[-1]) if frame.frame else []

        action_idx = 0
        stale_count = 0  # track consecutive no-change actions

        while actions_used < self.budget and action_idx < len(movement_cycle):
            action = movement_cycle[action_idx]
            action_idx += 1

            grid_before = frame.frame[-1] if frame.frame else []
            score_before = frame.levels_completed
            state_before = frame.state

            frame = step_fn(action)
            actions_used += 1

            grid_after = frame.frame[-1] if frame.frame else []

            # Handle game over
            if frame.state == GameState.GAME_OVER:
                transitions.append(Transition(
                    action=action.name,
                    grid_before=grid_before,
                    grid_after=grid_after,
                    score_before=score_before,
                    score_after=frame.levels_completed,
                    state_before=state_before.name,
                    state_after=frame.state.name,
                    num_changes=0,
                    description="GAME OVER — agent died",
                ))
                frame = step_fn(GameAction.RESET)
                actions_used += 1
                prev_objects = detect_objects(frame.frame[-1]) if frame.frame else []
                stale_count = 0
                continue

            if grid_before and grid_after:
                diff = diff_frames(grid_before, grid_after)

                # Classify the transition
                score_changed = frame.levels_completed != score_before
                desc = _describe_transition(action, diff.num_changed, score_changed, frame.state)

                transitions.append(Transition(
                    action=action.name,
                    grid_before=grid_before,
                    grid_after=grid_after,
                    score_before=score_before,
                    score_after=frame.levels_completed,
                    state_before=state_before.name,
                    state_after=frame.state.name,
                    num_changes=diff.num_changed,
                    description=desc,
                ))

                # Track objects
                curr_objects = detect_objects(grid_after)
                catalog = track_objects(catalog, prev_objects, curr_objects)
                prev_objects = curr_objects

                if diff.num_changed == 0:
                    stale_count += 1
                else:
                    stale_count = 0

                # If stuck (3 consecutive no-change), try a different direction
                if stale_count >= 3:
                    stale_count = 0
                    # Skip ahead in the cycle
                    action_idx = min(action_idx + 2, len(movement_cycle))

                logger.info(
                    f"[causal] {action.name}: changes={diff.num_changed} "
                    f"score={score_before}->{frame.levels_completed} | {desc}"
                )

        result = CausalResult(
            transitions=transitions,
            catalog=catalog,
            actions_used=actions_used,
        )
        logger.info(result.summary())
        return result, frame


def _describe_transition(
    action: GameAction, num_changes: int, score_changed: bool, state: GameState
) -> str:
    parts = []
    if num_changes == 0:
        parts.append("no visible change (blocked?)")
    elif num_changes < 20:
        parts.append(f"small change ({num_changes} cells)")
    elif num_changes < 100:
        parts.append(f"moderate change ({num_changes} cells)")
    else:
        parts.append(f"large change ({num_changes} cells)")

    if score_changed:
        parts.append("SCORE CHANGED")
    if state == GameState.WIN:
        parts.append("WIN")

    return "; ".join(parts)
