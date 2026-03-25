"""World-model induction agent.

Phase 1 (first ~12 actions): Systematic action probing
- Take each available action 2-3 times from the initial state
- Record structured transitions
- Code identifies controllable object operationally
- GPT proposes action rules and goal hypotheses in DSL format

Phase 2 (remaining actions): Goal-directed play with model refinement
- Use induced action model to pursue the best goal hypothesis
- When predictions fail, GPT refines rules
- When blocked, GPT proposes alternative strategies
"""

import json
import logging
import os
import time

from arcengine import FrameData, GameAction, GameState
from openai import OpenAI

from .dsl import ActionRule, GoalHypothesis, ObjectRole, WorldModel
from .predictor import identify_controllable, predict_and_score
from .symbolic import grid_to_symbolic, diff_symbolic
from .vision import grid_b64, diff_b64, input_text, input_image_b64

logger = logging.getLogger(__name__)


class WorldModelAgent:
    MAX_ACTIONS = 100
    PROBE_BUDGET = 12  # actions for phase 1

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
        self.prev_symbolic: dict | None = None
        self.model = WorldModel()
        self.transitions: list[dict] = []  # recorded transitions for analysis
        self.llm_calls = 0

    def run(self) -> None:
        timer = time.time()
        logger.info("=" * 60)
        logger.info(f"World-Model Induction Agent on {self.game_id}")
        logger.info("=" * 60)

        frame = self._step(GameAction.RESET)
        self.current_grid = frame.frame[-1] if frame.frame else []
        available = frame.available_actions or [1, 2, 3, 4]

        if frame.state == GameState.WIN:
            self._close()
            return

        # === PHASE 1: Systematic probing ===
        logger.info("=== PHASE 1: ACTION PROBING ===")
        frame = self._probe_actions(frame, available)

        if frame.state == GameState.WIN:
            self._close()
            return

        # Identify controllable object from transitions
        ctrl_color, ctrl_evidence = identify_controllable(self.transitions)
        if ctrl_color is not None:
            self.model.object_roles.append(ObjectRole(
                color=ctrl_color, role="controllable",
                evidence=ctrl_evidence, confidence=0.9,
            ))
            logger.info(f"[controllable] color={ctrl_color}: {ctrl_evidence}")

        # Ask GPT to induce action rules and goal hypotheses from transitions
        self._induce_model(frame)

        # === PHASE 2: Goal-directed play ===
        logger.info("=== PHASE 2: GOAL-DIRECTED PLAY ===")
        frame = self._play(frame)

        elapsed = round(time.time() - timer, 2)
        logger.info("=" * 60)
        logger.info(
            f"FINISHED: actions={self.action_counter} state={frame.state.name} "
            f"score={frame.levels_completed} deaths={self.total_deaths} "
            f"llm_calls={self.llm_calls} time={elapsed}s"
        )
        logger.info(f"World model:\n{self.model.to_dsl_text()}")
        logger.info("=" * 60)
        self._close()

    def _probe_actions(self, frame: FrameData, available: list[int]) -> FrameData:
        """Phase 1: Take each action 2-3 times systematically. Record transitions."""
        actions = [GameAction.from_id(a) for a in available if a <= 7]
        initial_symbolic = grid_to_symbolic(self.current_grid)
        self.prev_symbolic = initial_symbolic

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

    def _induce_model(self, frame: FrameData) -> None:
        """Ask GPT to propose action rules and goal hypotheses from probe data."""
        self.llm_calls += 1

        # Summarize transitions
        trans_text = []
        for t in self.transitions:
            changes_summary = []
            for c in t["sym_changes"][:5]:
                changes_summary.append(json.dumps(c))
            trans_text.append(
                "{}: {}".format(t['action'], 'BLOCKED' if t['blocked'] else f"{t['changes']}ch")
                + (f"\n  changes: {'; '.join(changes_summary)}" if changes_summary else "")
            )

        ctrl = self.model.controllable_color()
        ctrl_text = f"Operationally identified controllable object: color={ctrl}" if ctrl else "Controllable object not yet identified"

        content = [
            input_text(
                "You are analyzing an unknown game. Below are the results of systematically "
                "testing each available action 2-3 times from the initial state.\n\n"
                f"PROBE RESULTS:\n" + "\n".join(trans_text) + "\n\n"
                f"{ctrl_text}\n\n"
                "Based on these transitions, propose:\n"
                "1. ACTION RULES: what does each action do? (move, push, toggle, etc.)\n"
                "2. OBJECT ROLES: which objects are controllable, blocking, goal-like?\n"
                "3. GOAL HYPOTHESES: what might the objective be?\n\n"
                "Respond with JSON:\n"
                '{"action_rules": {"ACTION1": {"effect": "move", "direction": "up", "distance": 5, '
                '"target": "controllable", "precondition": "path_clear"}, ...},\n'
                ' "object_roles": [{"color": 9, "role": "controllable", "evidence": "..."}],\n'
                ' "goal_hypotheses": [{"type": "contact", "description": "reach the white cross", '
                '"target": "color_0", "confidence": 0.5}]}'
            ),
            input_text("Current game frame:"),
            input_image_b64(grid_b64(self.current_grid)),
        ]

        try:
            response = self.client.responses.create(
                model="gpt-5.4",
                input=[{"role": "user", "content": content}],
                max_output_tokens=2000,
                temperature=0.2,
            )
            raw = response.output_text or "{}"
        except Exception as e:
            logger.warning(f"Induction API error: {e}")
            return

        data = self._parse_json(raw)

        # Parse action rules
        for name, rule_data in data.get("action_rules", {}).items():
            self.model.action_rules[name] = ActionRule(
                action=name,
                effect=rule_data.get("effect", "unknown"),
                direction=rule_data.get("direction", ""),
                distance=rule_data.get("distance", 0),
                target=rule_data.get("target", ""),
                precondition=rule_data.get("precondition", ""),
                confidence=0.5,
            )

        # Parse object roles
        for role_data in data.get("object_roles", []):
            self.model.object_roles.append(ObjectRole(
                color=role_data.get("color", -1),
                role=role_data.get("role", "unknown"),
                evidence=role_data.get("evidence", ""),
                confidence=role_data.get("confidence", 0.5),
            ))

        # Parse goal hypotheses
        for goal_data in data.get("goal_hypotheses", []):
            self.model.goal_hypotheses.append(GoalHypothesis(
                description=goal_data.get("description", ""),
                type=goal_data.get("type", "unknown"),
                target=goal_data.get("target", ""),
                confidence=goal_data.get("confidence", 0.3),
            ))

        logger.info(f"[induce] Model:\n{self.model.to_dsl_text()}")

    def _play(self, frame: FrameData) -> FrameData:
        """Phase 2: Goal-directed play using the induced model."""
        actions_since_llm = 0
        prev_grid = self.current_grid
        prev_score = frame.levels_completed

        while self.action_counter < self.MAX_ACTIONS:
            if frame.state == GameState.WIN:
                logger.info("WIN!")
                break

            if frame.state == GameState.GAME_OVER:
                self.total_deaths += 1
                self.model.add_observation(f"Died at action {self.action_counter}")
                logger.info(f"DIED #{self.total_deaths}")
                frame = self._step(GameAction.RESET)
                self.action_counter += 1
                self.current_grid = frame.frame[-1] if frame.frame else []
                self.prev_symbolic = None
                # Re-induce after death
                self._refine_model(frame, "Died. What went wrong? Revise the model.")
                actions_since_llm = 0
                continue

            # Choose action based on current model
            action_name = self._choose_action(frame)
            try:
                action = GameAction.from_name(action_name)
            except (ValueError, KeyError):
                action = GameAction.ACTION1

            # Execute
            grid_before = self.current_grid
            sym_before = grid_to_symbolic(grid_before)
            frame = self._step(action, reasoning=self.model.to_dsl_text()[:80])
            self.action_counter += 1
            actions_since_llm += 1
            self.current_grid = frame.frame[-1] if frame.frame else grid_before

            sym_after = grid_to_symbolic(self.current_grid)
            sym_changes = diff_symbolic(sym_before, sym_after)
            changes = sum(1 for r in range(64) for c in range(64)
                          if grid_before[r][c] != self.current_grid[r][c])
            blocked = 0 < changes < 10

            # Score prediction
            score_report = predict_and_score(self.model, action.name, sym_changes, blocked, changes)

            logger.info(
                f"#{self.action_counter} {action.name}: "
                f"{'BLOCKED' if blocked else f'{changes}ch'} "
                f"score={frame.levels_completed} | {score_report}"
            )

            # Track
            self.transitions.append({
                "action": action.name, "changes": changes,
                "blocked": blocked, "sym_changes": sym_changes,
            })
            prev_grid = grid_before

            # Trigger model refinement on significant events
            if frame.levels_completed > prev_score:
                prev_score = frame.levels_completed
                self.model.add_observation(
                    f"LEVEL COMPLETED! Score {frame.levels_completed} after {action.name}"
                )
                logger.info(f"[level-up] Level {frame.levels_completed}! Reflecting then re-probing.")
                self._reflect_on_level(frame)
                # Re-probe: grid layout has changed
                self.transitions.clear()
                self.prev_symbolic = None
                available = frame.available_actions or [1, 2, 3, 4]
                frame = self._probe_actions(frame, available)
                self.current_grid = frame.frame[-1] if frame.frame else self.current_grid
                ctrl_color, ctrl_evidence = identify_controllable(self.transitions)
                if ctrl_color is not None:
                    self.model.object_roles.append(ObjectRole(
                        color=ctrl_color, role="controllable",
                        evidence=ctrl_evidence, confidence=0.9,
                    ))
                actions_since_llm = 0

            elif changes > 200:
                self.model.add_observation(f"Large change ({changes} cells) after {action.name}")
                self._refine_model(frame, f"Large change ({changes} cells). Level transition?")
                actions_since_llm = 0

            elif actions_since_llm >= 20:
                self._refine_model(frame, "20 actions since last check. Am I making progress?")
                actions_since_llm = 0

        return frame

    def _choose_action(self, frame: FrameData) -> str:
        """Choose the best action based on current model.

        If we have a goal and know the controllable object's position,
        pick the action that moves toward the goal.
        If model is uncertain, pick the action with lowest test count (explore).
        """
        goal = self.model.best_goal()
        ctrl_color = self.model.controllable_color()

        # If we have a goal target color, find it in the current grid
        if goal and goal.target and ctrl_color is not None:
            sym = grid_to_symbolic(self.current_grid)
            ctrl_pos = None
            target_pos = None

            for obj in sym.get("objects", []):
                if obj["color_id"] == ctrl_color:
                    ctrl_pos = obj["center"]
                elif goal.target in str(obj.get("color", "")) or str(obj.get("color_id", "")) == goal.target:
                    target_pos = obj["center"]

            if ctrl_pos and target_pos:
                dr = target_pos[0] - ctrl_pos[0]
                dc = target_pos[1] - ctrl_pos[1]

                # Pick direction that reduces distance most
                candidates = []
                for name, rule in self.model.action_rules.items():
                    if rule.effect == "move" and rule.direction:
                        expected_dr = {"up": -1, "down": 1}.get(rule.direction, 0)
                        expected_dc = {"left": -1, "right": 1}.get(rule.direction, 0)
                        score = dr * expected_dr + dc * expected_dc
                        candidates.append((score, name))

                if candidates:
                    candidates.sort(key=lambda x: -x[0])
                    return candidates[0][1]

        # Fallback: pick least-tested action (explore)
        available = frame.available_actions or [1, 2, 3, 4]
        action_names = [f"ACTION{i}" for i in available]
        least_tested = min(action_names,
                          key=lambda a: self.model.action_rules.get(a, ActionRule(action=a, effect="unknown")).test_count)
        return least_tested

    def _refine_model(self, frame: FrameData, trigger: str) -> None:
        """Ask GPT to refine the world model based on new evidence."""
        self.llm_calls += 1

        recent_trans = self.transitions[-10:]
        trans_text = []
        for t in recent_trans:
            trans_text.append("{}: {}".format(t['action'], 'BLOCKED' if t['blocked'] else f"{t['changes']}ch"))

        content = [
            input_text(
                f"TRIGGER: {trigger}\n\n"
                f"CURRENT WORLD MODEL:\n{self.model.to_dsl_text()}\n\n"
                f"RECENT TRANSITIONS:\n" + "\n".join(trans_text) + "\n\n"
                f"Score: {frame.levels_completed} | Deaths: {self.total_deaths} | "
                f"Actions: {self.action_counter}/{self.MAX_ACTIONS}\n\n"
                "Refine the world model. Update action rules, object roles, or goal hypotheses. "
                "Focus on what's WRONG in the current model and fix it.\n\n"
                "Respond with JSON:\n"
                '{"action_rules": {...}, "object_roles": [...], "goal_hypotheses": [...], '
                '"observations": ["key insight"]}'
            ),
            input_text("Current frame:"),
            input_image_b64(grid_b64(self.current_grid)),
        ]

        try:
            response = self.client.responses.create(
                model="gpt-5.4",
                input=[{"role": "user", "content": content}],
                max_output_tokens=1500,
                temperature=0.2,
            )
            raw = response.output_text or "{}"
        except Exception as e:
            logger.warning(f"Refine API error: {e}")
            return

        data = self._parse_json(raw)

        # Update action rules (merge, don't replace)
        for name, rule_data in data.get("action_rules", {}).items():
            self.model.action_rules[name] = ActionRule(
                action=name,
                effect=rule_data.get("effect", "unknown"),
                direction=rule_data.get("direction", ""),
                distance=rule_data.get("distance", 0),
                target=rule_data.get("target", ""),
                precondition=rule_data.get("precondition", ""),
                confidence=rule_data.get("confidence", 0.5),
            )

        # Add new observations
        for obs in data.get("observations", []):
            self.model.add_observation(obs)

        # Update goals
        for g in data.get("goal_hypotheses", []):
            self.model.goal_hypotheses.append(GoalHypothesis(
                description=g.get("description", ""),
                type=g.get("type", "unknown"),
                target=g.get("target", ""),
                confidence=g.get("confidence", 0.3),
            ))

        logger.info(f"[refine] {trigger}")
        logger.info(f"[model] {self.model.to_dsl_text()[:300]}")

    def _reflect_on_level(self, frame: FrameData) -> None:
        """Reflect on a just-completed level. Ask GPT what worked and what to carry forward."""
        self.llm_calls += 1

        recent_trans = self.transitions[-15:]
        trans_text = []
        for t in recent_trans:
            trans_text.append("{}: {}".format(
                t['action'], 'BLOCKED' if t['blocked'] else f"{t['changes']}ch"
            ))

        content = [
            input_text(
                f"LEVEL {frame.levels_completed} COMPLETED!\n\n"
                f"You just completed a level in this unknown game. Reflect on what "
                f"succeeded so you can apply the same strategy to the next level.\n\n"
                f"CURRENT WORLD MODEL:\n{self.model.to_dsl_text()}\n\n"
                f"RECENT ACTION SEQUENCE (leading to completion):\n"
                + "\n".join(trans_text) + "\n\n"
                f"Score: {frame.levels_completed} | Deaths: {self.total_deaths} | "
                f"Actions: {self.action_counter}/{self.MAX_ACTIONS}\n\n"
                "Answer these questions:\n"
                "1. WINNING STRATEGY: What sequence of actions or approach led to completing this level?\n"
                "2. GAME MECHANICS: What universal rules about this game did you confirm? "
                "(e.g., action effects, object interactions, win conditions)\n"
                "3. CARRY FORWARD: What knowledge should be applied to the next level? "
                "The layout will change but the game mechanics stay the same.\n\n"
                "Respond with JSON:\n"
                '{"winning_strategy": "what you did to win this level",\n'
                ' "confirmed_mechanics": ["universal game rule 1", "universal game rule 2"],\n'
                ' "carry_forward": ["knowledge to apply to next level"],\n'
                ' "goal_for_next_level": {"description": "what to try first", '
                '"type": "contact/collect/clear/pattern", "confidence": 0.6}}'
            ),
            input_text("Final frame of completed level:"),
            input_image_b64(grid_b64(self.current_grid)),
        ]

        try:
            response = self.client.responses.create(
                model="gpt-5.4",
                input=[{"role": "user", "content": content}],
                max_output_tokens=1500,
                temperature=0.2,
            )
            raw = response.output_text or "{}"
        except Exception as e:
            logger.warning(f"Level reflection API error: {e}")
            return

        data = self._parse_json(raw)

        strategy = data.get("winning_strategy", "")
        if strategy:
            self.model.add_observation(f"LEVEL {frame.levels_completed} WON: {strategy}")

        for mechanic in data.get("confirmed_mechanics", []):
            self.model.add_observation(f"CONFIRMED: {mechanic}")

        for knowledge in data.get("carry_forward", []):
            self.model.add_observation(f"CARRY FORWARD: {knowledge}")

        next_goal = data.get("goal_for_next_level", {})
        if next_goal and next_goal.get("description"):
            self.model.goal_hypotheses = [GoalHypothesis(
                description=next_goal["description"],
                type=next_goal.get("type", "unknown"),
                target=next_goal.get("target", ""),
                confidence=next_goal.get("confidence", 0.5),
                evidence=f"Inferred from level {frame.levels_completed} success",
            )]

        logger.info(f"[level-reflect] Level {frame.levels_completed}: {strategy}")
        logger.info(f"[level-reflect] Carry forward: {data.get('carry_forward', [])}")

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
