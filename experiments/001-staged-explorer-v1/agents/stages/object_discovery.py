"""Stage 2: Object Discovery — catalog persistent entities.

Uses the perception layer to detect objects (per-color connected components)
and track them across movements. Uses background colors inferred by stage 1.
"""

import logging
from dataclasses import dataclass

from arcengine import FrameData, GameAction

from ..perception.objects import (
    ObjectCatalog,
    detect_objects,
    track_objects,
)
from .sensorimotor import SensorimotorResult

logger = logging.getLogger(__name__)


@dataclass
class ObjectDiscoveryResult:
    catalog: ObjectCatalog
    actions_used: int

    def summary(self) -> str:
        return (
            f"Object Discovery ({self.actions_used} actions):\n"
            f"{self.catalog.summary()}"
        )


class ObjectDiscoveryStage:
    """Detect and classify objects by taking deliberate movement actions."""

    def __init__(self, budget: int = 8) -> None:
        self.budget = budget

    def run(
        self,
        step_fn,
        current_frame: FrameData,
        sensorimotor: SensorimotorResult,
    ) -> tuple[ObjectDiscoveryResult, FrameData]:
        bg = sensorimotor.background_colors
        catalog = ObjectCatalog(background_colors=bg)
        actions_used = 0
        frame = current_frame

        # Use actions that caused change in sensorimotor stage
        movement_actions = sensorimotor.get_movement_actions()
        if not movement_actions:
            movement_actions = [
                GameAction.ACTION1, GameAction.ACTION2,
                GameAction.ACTION3, GameAction.ACTION4,
            ]

        # Initial detection with inferred background
        grid = frame.frame[-1] if frame.frame else []
        prev_objects = detect_objects(grid, background=bg) if grid else []
        logger.info(
            f"[object_discovery] initial detection: {len(prev_objects)} objects "
            f"(bg={sorted(bg)})"
        )

        # Take movement actions and track
        action_sequence = movement_actions * 2  # repeat to get more tracking data
        for action in action_sequence:
            if actions_used >= self.budget:
                break

            frame = step_fn(action)
            actions_used += 1

            grid = frame.frame[-1] if frame.frame else []
            curr_objects = detect_objects(grid, background=bg) if grid else []

            catalog = track_objects(catalog, prev_objects, curr_objects, agent_acted=True)
            prev_objects = curr_objects

            logger.info(
                f"[object_discovery] after {action.name}: "
                f"{len(curr_objects)} objects, "
                f"{len(catalog.controllable)} controllable, "
                f"{len(catalog.static)} static"
            )

        result = ObjectDiscoveryResult(catalog=catalog, actions_used=actions_used)
        logger.info(result.summary())
        return result, frame
