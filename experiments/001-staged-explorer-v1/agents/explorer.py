"""Staged exploration agent — the orchestrator.

Runs through stages sequentially:
1. Sensorimotor (learn action effects)
2. Object Discovery (catalog entities)
3. Causal Testing (test interactions)
4. Goal Inference (hypothesize win condition)
5. Planning (execute toward goal)
"""

import logging

from arcengine import FrameData, GameAction, GameState

from .base import Agent
from .perception.objects import ObjectCatalog
from .stages.sensorimotor import SensorimotorStage
from .stages.object_discovery import ObjectDiscoveryStage
from .stages.causal import CausalStage
from .stages.goal_inference import GoalInferenceStage
from .stages.planning import PlanningStage
from .world_model import WorldModel

logger = logging.getLogger(__name__)


class ExplorerAgent(Agent):
    MAX_ACTIONS = 200

    # Stage budgets
    SENSORIMOTOR_BUDGET = 12
    OBJECT_DISCOVERY_BUDGET = 8
    CAUSAL_BUDGET = 20
    PLANNING_BUDGET = 160
    MODEL = "gpt-4o-mini"

    def __init__(self, game_id: str) -> None:
        super().__init__(game_id)
        self.world_model = WorldModel()
        self.total_actions = 0

    def run(self) -> None:
        """Override the base run loop — we drive stages directly instead of choose_action."""
        import time
        self.timer = time.time()

        # Initial reset
        logger.info("=" * 60)
        logger.info(f"EXPERIMENT: Staged Explorer v1 on {self.game_id}")
        logger.info("=" * 60)

        frame = self._step(GameAction.RESET)
        self.frames.append(frame)
        self.total_actions += 1

        if frame.state == GameState.WIN:
            logger.info("Won immediately after reset (?).")
            return

        # --- STAGE 1: Sensorimotor ---
        logger.info("\n" + "=" * 40)
        logger.info("STAGE 1: SENSORIMOTOR")
        logger.info("=" * 40)

        sm_stage = SensorimotorStage(budget=self.SENSORIMOTOR_BUDGET)
        sm_result, frame = sm_stage.run(self._step_and_track, frame)
        self.frames.append(frame)

        if frame.state == GameState.WIN:
            self._finish("Won during sensorimotor stage")
            return

        # --- STAGE 2: Object Discovery ---
        logger.info("\n" + "=" * 40)
        logger.info("STAGE 2: OBJECT DISCOVERY")
        logger.info("=" * 40)

        od_stage = ObjectDiscoveryStage(budget=self.OBJECT_DISCOVERY_BUDGET)
        od_result, frame = od_stage.run(self._step_and_track, frame, sm_result)
        self.frames.append(frame)

        if frame.state == GameState.WIN:
            self._finish("Won during object discovery stage")
            return

        # --- STAGE 3: Causal Testing ---
        logger.info("\n" + "=" * 40)
        logger.info("STAGE 3: CAUSAL TESTING")
        logger.info("=" * 40)

        causal_stage = CausalStage(budget=self.CAUSAL_BUDGET)
        causal_result, frame = causal_stage.run(
            self._step_and_track, frame, od_result.catalog
        )
        self.frames.append(frame)

        # Build world model from causal observations
        self.world_model.add_tests_from_causal(causal_result.to_test_cases())

        if frame.state == GameState.WIN:
            self._finish("Won during causal testing stage")
            return

        # --- STAGE 4: Goal Inference ---
        logger.info("\n" + "=" * 40)
        logger.info("STAGE 4: GOAL INFERENCE")
        logger.info("=" * 40)

        goal_stage = GoalInferenceStage(model=self.MODEL)
        goal_result = goal_stage.run(sm_result, causal_result, od_result.catalog)
        # Goal inference uses no game actions, only LLM

        # --- STAGE 5: Planning ---
        logger.info("\n" + "=" * 40)
        logger.info("STAGE 5: PLANNING")
        logger.info("=" * 40)

        remaining_budget = self.MAX_ACTIONS - self.total_actions
        plan_budget = min(self.PLANNING_BUDGET, remaining_budget)

        planning_stage = PlanningStage(budget=plan_budget, model=self.MODEL)
        plan_actions, frame = planning_stage.run(
            self._step_and_track, frame, goal_result, causal_result, od_result.catalog
        )
        self.frames.append(frame)

        self._finish(f"Planning complete — {frame.state.name}")

    def _step_and_track(self, action: GameAction) -> FrameData:
        """Step wrapper that tracks total action count."""
        frame = self._step(action)
        self.total_actions += 1
        self.action_counter = self.total_actions
        return frame

    def _finish(self, msg: str) -> None:
        import time
        elapsed = round(time.time() - self.timer, 2)
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"FINISHED: {msg}")
        logger.info(
            f"Total actions: {self.total_actions} | "
            f"Final state: {self.frames[-1].state.name} | "
            f"Score: {self.frames[-1].levels_completed} | "
            f"Time: {elapsed}s"
        )
        logger.info(f"World model: {self.world_model.summary()}")
        logger.info("=" * 60)
        self.close()

    def choose_action(
        self, frames: list[FrameData], latest: FrameData
    ) -> GameAction:
        """Not used — stages drive actions directly. Required by base class."""
        return GameAction.ACTION1
