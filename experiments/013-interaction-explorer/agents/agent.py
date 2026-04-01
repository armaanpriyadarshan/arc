"""Interaction-focused explorer agent.

Three phases:
  Phase 1: Auto-probe — discover actions and controllable object
  Phase 2: Interaction testing — approach and test each object type
  Phase 3: Goal-directed play — use discovered mechanics

Key difference from experiment 004: the agent doesn't just navigate
toward goals. It systematically tests what each object type does
when contacted, building an interaction map before committing
to a strategy.
"""

import json
import logging
import os
import time

from arcengine import FrameData, GameAction, GameState
from openai import OpenAI

from .dsl import ActionRule, GoalHypothesis, InteractionRule, ObjectRole, WorldModel
from .interaction import InteractionPlanner
from .predictor import classify_interaction, identify_controllable, predict_and_score
from .symbolic import grid_to_symbolic, diff_symbolic
from .vision import diff_b64, grid_b64, input_text, input_image_b64

logger = logging.getLogger(__name__)


class InteractionExplorer:
    MAX_ACTIONS = 100
    PROBE_BUDGET = 12
    INTERACTION_BUDGET = 40

    def __init__(self, game_id: str) -> None:
        from arc_agi import Arcade
        self.game_id = game_id
        self.arcade = Arcade()
        self.scorecard_id = self.arcade.open_scorecard()
        self.env = self.arcade.make(game_id, scorecard_id=self.scorecard_id)
        self.frames: list[FrameData] = []
        self.action_counter = 0
        self.total_deaths = 0

        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        self.current_grid: list[list[int]] = []
        self.prev_grid: list[list[int]] = []
        self.model = WorldModel()
        self.planner = InteractionPlanner()
        self.transitions: list[dict] = []
        self.llm_calls = 0

    def _step(self, action: GameAction, reasoning: str = "") -> FrameData:
        try:
            raw = self.env.step(action, reasoning=reasoning)
        except Exception as e:
            logger.warning(f"env.step({action.name}) exception: {e}")
            raw = None
        if raw is None:
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

    def _parse_json(self, raw: str) -> dict:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            pass
        cleaned = raw
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1]
            if "```" in cleaned:
                cleaned = cleaned.split("```", 1)[0]
        try:
            return json.loads(cleaned.strip())
        except (json.JSONDecodeError, ValueError):
            pass
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except (json.JSONDecodeError, ValueError):
                pass
        return {}

    def run(self) -> None:
        timer = time.time()
        logger.info("=" * 60)
        logger.info(f"Interaction Explorer on {self.game_id}")
        logger.info("=" * 60)

        frame = self._step(GameAction.RESET)
        self.current_grid = frame.frame[-1] if frame.frame else []
        available = frame.available_actions or [1, 2, 3, 4]

        if frame.state == GameState.WIN:
            self._close()
            return

        # === PHASE 1 ===
        logger.info("=== PHASE 1: ACTION PROBING ===")
        frame = self._probe_actions(frame, available)
        if frame.state == GameState.WIN:
            self._close()
            return

        ctrl_color, ctrl_evidence = identify_controllable(self.transitions)
        if ctrl_color is not None:
            self.model.object_roles.append(ObjectRole(
                color=ctrl_color, role="controllable",
                evidence=ctrl_evidence, confidence=0.9,
            ))
            logger.info(f"[controllable] color={ctrl_color}: {ctrl_evidence}")

        self._induce_model(frame)

        # === PHASE 2 ===
        logger.info("=== PHASE 2: INTERACTION TESTING ===")
        frame = self._test_interactions(frame)
        if frame.state == GameState.WIN:
            self._close()
            return

        # === PHASE 3 ===
        logger.info("=== PHASE 3: GOAL-DIRECTED PLAY ===")
        frame = self._play(frame)

        elapsed = round(time.time() - timer, 2)
        logger.info("=" * 60)
        logger.info(
            f"FINISHED: actions={self.action_counter} state={frame.state.name} "
            f"score={frame.levels_completed} deaths={self.total_deaths} "
            f"llm_calls={self.llm_calls} time={elapsed}s"
        )
        logger.info(f"World model:\n{self.model.to_dsl_text()}")
        logger.info(f"Interactions:\n{self.planner.summary()}")
        logger.info("=" * 60)
        self._close()
