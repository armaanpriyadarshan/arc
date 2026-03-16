"""Stage 1: Sensorimotor — learn what each action does.

Try each action systematically, diff the result, record the effect.
Output: action→effect map, inferred background colors, collected grids for downstream stages.
"""

import logging
from dataclasses import dataclass, field

from arcengine import FrameData, GameAction, GameState

from ..perception.differ import DiffResult, diff_frames
from ..perception.objects import infer_background_from_diffs

logger = logging.getLogger(__name__)

Grid = list[list[int]]


@dataclass
class ActionEffect:
    """What happened when an action was taken."""
    action: GameAction
    diff: DiffResult
    movement_direction: tuple[int, int] | None
    state_before: GameState
    state_after: GameState
    score_before: int
    score_after: int


@dataclass
class SensorimotorResult:
    """Output of the sensorimotor stage."""
    action_effects: dict[str, list[ActionEffect]]
    actions_used: int
    background_colors: set[int]  # inferred from frame diffs
    collected_grids: list[Grid]  # all grids observed, for downstream use

    def get_movement_actions(self) -> list[GameAction]:
        """Return actions that caused visible change."""
        movers = []
        for name, effects in self.action_effects.items():
            if any(e.diff.num_changed > 0 for e in effects):
                movers.append(effects[0].action)
        return movers

    def get_no_effect_actions(self) -> list[GameAction]:
        """Return actions that had no visible effect."""
        no_effect = []
        for name, effects in self.action_effects.items():
            if all(e.diff.num_changed == 0 for e in effects):
                no_effect.append(effects[0].action)
        return no_effect

    def summary(self) -> str:
        lines = ["Sensorimotor Results:"]
        for name, effects in self.action_effects.items():
            changes = [e.diff.num_changed for e in effects]
            avg = sum(changes) / len(changes) if changes else 0
            dirs = [e.movement_direction for e in effects if e.movement_direction]
            dir_str = str(dirs[0]) if dirs else "none"
            lines.append(f"  {name}: avg_changes={avg:.0f}, direction={dir_str}")
        lines.append(f"  Inferred background colors: {sorted(self.background_colors)}")
        lines.append(f"  Total actions used: {self.actions_used}")
        return "\n".join(lines)


class SensorimotorStage:
    """Try each action and record its effect."""

    def __init__(self, budget: int = 12) -> None:
        self.budget = budget

    def run(
        self,
        step_fn,
        current_frame: FrameData,
    ) -> tuple[SensorimotorResult, FrameData]:
        effects: dict[str, list[ActionEffect]] = {}
        actions_used = 0
        frame = current_frame
        collected_grids: list[Grid] = []

        # Collect initial grid
        if frame.frame:
            collected_grids.append(frame.frame[-1])

        # Only test simple actions + ACTION6 with default coords
        test_actions = [a for a in GameAction if a is not GameAction.RESET]

        for action in test_actions:
            if actions_used >= self.budget:
                break

            for trial in range(2):
                if actions_used >= self.budget:
                    break

                grid_before = frame.frame[-1] if frame.frame else []
                state_before = frame.state
                score_before = frame.levels_completed

                if action.is_complex():
                    action.set_data({"x": 32, "y": 32})

                frame = step_fn(action)
                actions_used += 1

                grid_after = frame.frame[-1] if frame.frame else []
                if grid_after:
                    collected_grids.append(grid_after)

                if grid_before and grid_after:
                    diff = diff_frames(grid_before, grid_after)
                    movement = _estimate_movement(diff, grid_before, grid_after)
                else:
                    diff = DiffResult([], [[]], 0, [])
                    movement = None

                effect = ActionEffect(
                    action=action,
                    diff=diff,
                    movement_direction=movement,
                    state_before=state_before,
                    state_after=frame.state,
                    score_before=score_before,
                    score_after=frame.levels_completed,
                )

                name = action.name
                if name not in effects:
                    effects[name] = []
                effects[name].append(effect)

                logger.info(
                    f"[sensorimotor] {name} trial={trial}: "
                    f"changed={diff.num_changed} movement={movement}"
                )

                if frame.state in (GameState.GAME_OVER,):
                    frame = step_fn(GameAction.RESET)
                    actions_used += 1
                    if frame.frame:
                        collected_grids.append(frame.frame[-1])

        # Infer background from all collected grids
        bg_colors = infer_background_from_diffs(collected_grids)
        logger.info(f"[sensorimotor] inferred background colors: {sorted(bg_colors)}")

        result = SensorimotorResult(
            action_effects=effects,
            actions_used=actions_used,
            background_colors=bg_colors,
            collected_grids=collected_grids,
        )
        logger.info(result.summary())
        return result, frame


def _estimate_movement(
    diff: DiffResult, grid_before: Grid, grid_after: Grid
) -> tuple[int, int] | None:
    """Estimate movement direction from a diff.

    Strategy: find cells where a non-background value appeared vs disappeared.
    Uses the actual grid values instead of hardcoded background assumptions.
    """
    if diff.is_empty or not diff.regions:
        return None

    # Find the largest changed region
    largest = max(diff.regions, key=len)
    if len(largest) < 4:
        return None

    changes = diff.changed_values()

    # For each changed cell, check if the old value appeared somewhere nearby
    # in the new frame (indicating movement rather than state change)
    # Simpler approach: centroid of cells that gained rare colors vs lost them
    from collections import Counter
    before_colors: Counter[int] = Counter()
    after_colors: Counter[int] = Counter()
    for r, c in largest:
        if (r, c) in changes:
            old, new = changes[(r, c)]
            before_colors[old] += 1
            after_colors[new] += 1

    # The "moving" color is one that appears in both before and after
    # but at different positions
    moving_colors = set(before_colors.keys()) & set(after_colors.keys())
    # Remove colors that dominate (likely background)
    total = len(largest)
    moving_colors = {c for c in moving_colors if before_colors[c] < total * 0.8}

    if not moving_colors:
        # Fall back: just compute centroid shift of the changed region split by before/after
        gained: list[tuple[int, int]] = []
        lost: list[tuple[int, int]] = []
        for r, c in largest:
            if (r, c) in changes:
                old, new = changes[(r, c)]
                # "gained" = cell now has a less common color
                # "lost" = cell now has a more common color
                gained.append((r, c))
                lost.append((r, c))
        return None

    # Cells where moving color disappeared vs appeared
    lost: list[tuple[int, int]] = []
    gained: list[tuple[int, int]] = []
    for r, c in largest:
        if (r, c) in changes:
            old, new = changes[(r, c)]
            if old in moving_colors and new not in moving_colors:
                lost.append((r, c))
            elif new in moving_colors and old not in moving_colors:
                gained.append((r, c))

    if not gained or not lost:
        return None

    g_r = sum(r for r, _ in gained) / len(gained)
    g_c = sum(c for _, c in gained) / len(gained)
    l_r = sum(r for r, _ in lost) / len(lost)
    l_c = sum(c for _, c in lost) / len(lost)

    dr = round(g_r - l_r)
    dc = round(g_c - l_c)

    if dr == 0 and dc == 0:
        return None

    return (dr, dc)
