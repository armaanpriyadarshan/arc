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

    def _probe_actions(self, frame: FrameData, available: list[int]) -> FrameData:
        """Phase 1: Take each action 2-3 times systematically."""
        actions = [GameAction.from_id(a) for a in available if a <= 7]

        for action in actions:
            for trial in range(3):
                if self.action_counter >= self.PROBE_BUDGET:
                    break

                grid_before = self.current_grid
                sym_before = grid_to_symbolic(grid_before)

                frame = self._step(action)
                self.action_counter += 1
                self.current_grid = frame.frame[-1] if frame.frame else grid_before

                sym_after = grid_to_symbolic(self.current_grid)
                sym_changes = diff_symbolic(sym_before, sym_after)
                changes = sum(1 for r in range(64) for c in range(64)
                              if grid_before[r][c] != self.current_grid[r][c])
                blocked = 0 < changes < 10

                self.transitions.append({
                    "action": action.name,
                    "changes": changes,
                    "blocked": blocked,
                    "sym_changes": sym_changes,
                })

                logger.info(
                    f"[probe] #{self.action_counter} {action.name}: "
                    f"{'BLOCKED' if blocked else f'{changes}ch'}"
                )

                if frame.state == GameState.GAME_OVER:
                    self.total_deaths += 1
                    frame = self._step(GameAction.RESET)
                    self.action_counter += 1
                    self.current_grid = frame.frame[-1] if frame.frame else []
                    break

                if frame.state == GameState.WIN:
                    return frame

        return frame

    def _test_interactions(self, frame: FrameData) -> FrameData:
        """Phase 2: Approach and test interactions with each object type."""
        ctrl_color = self.model.controllable_color()
        if ctrl_color is None:
            logger.info("[interact] No controllable object — skipping phase 2")
            return frame

        interaction_end = self.PROBE_BUDGET + self.INTERACTION_BUDGET
        prev_score = frame.levels_completed
        stuck_count = 0

        while self.action_counter < min(interaction_end, self.MAX_ACTIONS):
            if frame.state == GameState.WIN:
                return frame

            if frame.state == GameState.GAME_OVER:
                self.total_deaths += 1
                logger.info(f"[interact] DIED #{self.total_deaths}")
                frame = self._step(GameAction.RESET)
                self.action_counter += 1
                self.current_grid = frame.frame[-1] if frame.frame else []
                continue

            sym = grid_to_symbolic(self.current_grid)
            target_color = self.planner.next_target(sym, ctrl_color)

            if target_color is None:
                logger.info("[interact] All reachable objects tested")
                break

            # Navigate toward target
            action_name = self.planner.navigation_direction(
                sym, ctrl_color, target_color, self.model.action_rules
            )

            if action_name is None:
                # No clear direction — try cycling through actions
                available = frame.available_actions or [1, 2, 3, 4]
                idx = self.action_counter % len(available)
                action_name = f"ACTION{available[idx]}"

            try:
                action = GameAction.from_name(action_name)
            except (ValueError, KeyError):
                action = GameAction.ACTION1

            self.planner.record_approach(target_color)

            grid_before = self.current_grid
            sym_before = sym

            frame = self._step(action, reasoning=f"approaching color {target_color}")
            self.action_counter += 1
            self.current_grid = frame.frame[-1] if frame.frame else grid_before

            sym_after = grid_to_symbolic(self.current_grid)
            sym_changes = diff_symbolic(sym_before, sym_after)
            changes = sum(1 for r in range(64) for c in range(64)
                          if grid_before[r][c] != self.current_grid[r][c])
            blocked = 0 < changes < 10

            self.transitions.append({
                "action": action.name, "changes": changes,
                "blocked": blocked, "sym_changes": sym_changes,
            })

            # Check if we made contact with the target
            score_delta = frame.levels_completed - prev_score
            died = frame.state == GameState.GAME_OVER

            dist = None
            from .symbolic import proximity_to
            dist = proximity_to(sym_after, ctrl_color, target_color)

            if dist is not None and dist <= 2 or died or score_delta > 0:
                # Contact or near-contact — classify the interaction
                effect = classify_interaction(
                    sym_changes, blocked, changes, died, score_delta, ctrl_color
                )
                self.planner.record_test(target_color, effect)
                self.model.interaction_rules.append(InteractionRule(
                    target_color=target_color,
                    target_role="unknown",
                    effect=effect,
                    evidence=f"contact at action #{self.action_counter}",
                    test_count=1,
                    confidence=0.7,
                ))
                logger.info(
                    f"[interact] #{self.action_counter} CONTACT color={target_color}: "
                    f"effect={effect} score_delta={score_delta}"
                )
                prev_score = frame.levels_completed
                stuck_count = 0
            else:
                logger.info(
                    f"[interact] #{self.action_counter} {action.name} -> "
                    f"{'BLOCKED' if blocked else f'{changes}ch'} "
                    f"dist_to_{target_color}={dist}"
                )
                if blocked:
                    stuck_count += 1
                else:
                    stuck_count = 0

            # If stuck, try a different direction
            if stuck_count >= 3:
                stuck_count = 0
                self.planner.record_approach(target_color)
                self.planner.record_approach(target_color)
                self.planner.record_approach(target_color)
                logger.info(f"[interact] Stuck approaching color={target_color}, moving on")

        logger.info(f"[interact] Phase 2 done: {self.planner.summary()}")
        return frame
