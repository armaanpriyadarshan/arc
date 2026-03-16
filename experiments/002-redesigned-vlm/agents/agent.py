"""Compositional Discovery Agent — orchestrator.

Single loop: the model observes, reasons, and decides what to do next.
No hardcoded phases. The model controls its own learning process.
"""

import json
import logging
import time

from arcengine import FrameData, GameAction, GameState

from .memory import GameKnowledge
from .models import ModelRouter
from .perception import DiffAnalyzer, SceneDescriber
from .planner import Planner
from .skills import SkillLibrary

logger = logging.getLogger(__name__)


class CompositionalAgent:
    MAX_ACTIONS = 100

    def __init__(self, game_id: str) -> None:
        from arc_agi import Arcade
        self.game_id = game_id
        self.arcade = Arcade()
        self.scorecard_id = self.arcade.open_scorecard()
        self.env = self.arcade.make(game_id, scorecard_id=self.scorecard_id)
        self.frames: list[FrameData] = []

        # Components
        self.knowledge = GameKnowledge()
        self.router = ModelRouter()
        self.differ = DiffAnalyzer()
        self.scene = SceneDescriber(model_call_fn=self.router.call)
        self.skills = SkillLibrary()
        self.planner = Planner(
            self.knowledge, self.router, self.differ, self.scene, self.skills
        )

    def run(self) -> None:
        timer = time.time()
        logger.info("=" * 60)
        logger.info(f"Compositional Agent on {self.game_id}")
        logger.info("=" * 60)

        frame = self._step(GameAction.RESET, "Initial reset")

        if frame.state == GameState.WIN:
            self._close()
            return

        # Set up known inputs and available actions
        self.planner.setup_primitives(frame)

        # Single unified loop — the model decides what to do
        frame = self.planner.run(frame, self._step, self.MAX_ACTIONS)

        elapsed = round(time.time() - timer, 2)
        logger.info("=" * 60)
        logger.info(
            f"FINISHED: actions={self.planner.action_counter} "
            f"state={frame.state.name} score={frame.levels_completed} "
            f"deaths={self.planner.total_deaths} "
            f"model_calls={self.router.stats()} time={elapsed}s"
        )
        logger.info(f"Knowledge:\n{self.knowledge.compact_text()}")
        logger.info(f"Skills:\n{self.skills.summary()}")
        logger.info("=" * 60)
        self._close()

    def _step(self, action: GameAction, reasoning: str = "") -> FrameData:
        try:
            raw = self.env.step(action, reasoning=reasoning)
        except Exception as e:
            logger.warning(f"env.step({action.name}) exception: {e}")
            raw = None

        if raw is None:
            logger.warning(f"env.step({action.name}) returned None")
            if self.frames:
                return self.frames[-1]
            return FrameData(levels_completed=0)

        frame = FrameData(
            game_id=raw.game_id,
            frame=[arr.tolist() for arr in raw.frame],
            state=raw.state,
            levels_completed=raw.levels_completed,
            win_levels=raw.win_levels,
            guid=raw.guid,
            full_reset=raw.full_reset,
            available_actions=raw.available_actions,
        )
        self.frames.append(frame)
        return frame

    def _close(self) -> None:
        if not self.scorecard_id:
            return
        scorecard = self.arcade.close_scorecard(self.scorecard_id)
        if scorecard:
            logger.info("--- SCORECARD ---")
            logger.info(json.dumps(scorecard.model_dump(), indent=2))
        self.scorecard_id = None
