"""Base agent class — thin wrapper around arc-agi SDK."""

import logging
import time
from abc import ABC, abstractmethod

from arc_agi import Arcade, EnvironmentWrapper
from arcengine import FrameData, GameAction, GameState

logger = logging.getLogger(__name__)


class Agent(ABC):
    MAX_ACTIONS: int = 80

    def __init__(self, game_id: str) -> None:
        self.game_id = game_id
        self.arcade = Arcade()
        self.scorecard_id = self.arcade.open_scorecard()
        self.env: EnvironmentWrapper = self.arcade.make(game_id, scorecard_id=self.scorecard_id)
        self.frames: list[FrameData] = []
        self.action_counter = 0
        self.timer = 0.0

    def run(self) -> None:
        """Main game loop."""
        self.timer = time.time()

        frame = self._step(GameAction.RESET)
        self.frames.append(frame)

        while not self.is_done(frame) and self.action_counter < self.MAX_ACTIONS:
            action = self.choose_action(self.frames, frame)
            frame = self._step(action)
            self.frames.append(frame)
            self.action_counter += 1
            logger.info(
                f"[{self.game_id}] action={action.name} "
                f"levels={frame.levels_completed} "
                f"state={frame.state.name} "
                f"step={self.action_counter}"
            )

        elapsed = round(time.time() - self.timer, 2)
        logger.info(
            f"[{self.game_id}] done — {self.action_counter} actions, "
            f"{elapsed}s, state={frame.state.name}"
        )
        self.close()

    def close(self) -> None:
        """Close the scorecard and print results."""
        if not self.scorecard_id:
            return
        scorecard = self.arcade.close_scorecard(self.scorecard_id)
        if scorecard:
            logger.info("--- SCORECARD ---")
            import json
            logger.info(json.dumps(scorecard.model_dump(), indent=2))
        self.scorecard_id = None

    def _step(self, action: GameAction) -> FrameData:
        raw = self.env.step(action)
        if raw is None:
            logger.warning(f"env.step({action.name}) returned None, reusing last frame")
            if self.frames:
                return self.frames[-1]
            return FrameData(levels_completed=0)
        return FrameData(
            game_id=raw.game_id,
            frame=[arr.tolist() for arr in raw.frame],
            state=raw.state,
            levels_completed=raw.levels_completed,
            win_levels=raw.win_levels,
            guid=raw.guid,
            full_reset=raw.full_reset,
            available_actions=raw.available_actions,
        )

    def is_done(self, frame: FrameData) -> bool:
        return frame.state is GameState.WIN

    @abstractmethod
    def choose_action(
        self, frames: list[FrameData], latest: FrameData
    ) -> GameAction:
        ...
