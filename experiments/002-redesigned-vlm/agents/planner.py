"""Unified planner — one loop, model decides what to do.

Changes from prior iteration (based on log analysis):
1. Abort stuck plans (3 consecutive similar outcomes)
2. Show before/after/diff images so model sees what its plan DID
3. Track and surface plan failures so model doesn't repeat them
4. Record skill outcomes (success/failure tracking)
5. Smarter fallback (systematic probe, not random cycling)
6. Re-describe scene on large changes
7. Split think into analyze + plan (two calls per cycle)
"""

import json
import logging
from typing import Callable

from arcengine import FrameData, GameAction, GameState

from .memory import GameKnowledge, Hypothesis, Primitive, Rule, Transition
from .models import ModelRouter, REASONING_MODEL, VISION_MODEL
from .perception import DiffAnalyzer, SceneDescriber
from .skills import SkillLibrary
from .vision import grid_to_b64, side_by_side_b64, image_block, text_block

logger = logging.getLogger(__name__)

StepFn = Callable[[GameAction, str], FrameData]

AGENT_CONTEXT = (
    "You are an agent playing a turn-based game displayed as a 64x64 pixel grid with 16 colors. "
    "The game has hidden rules that you must figure out through observation and experimentation. "
    "The game has multiple levels. Your score (levels_completed) increases when you complete a level. "
    "You win by completing all levels. You lose (GAME_OVER) if you fail — then you must RESET and try again. "
    "Your inputs are directional controls (up/down/left/right) and possibly action/click buttons. "
    "You must discover what the game wants you to do by observing how the grid changes in response to your inputs."
)

INPUT_MAP = {
    "ACTION1": "up", "ACTION2": "down", "ACTION3": "left", "ACTION4": "right",
    "ACTION5": "action (enter/spacebar)", "ACTION6": "click (x,y)", "ACTION7": "undo",
}


class Planner:
    def __init__(self, knowledge: GameKnowledge, router: ModelRouter,
                 differ: DiffAnalyzer, scene: SceneDescriber, skills: SkillLibrary) -> None:
        self.k = knowledge
        self.router = router
        self.differ = differ
        self.scene = scene
        self.skills = skills
        self.action_counter = 0
        self.total_deaths = 0
        self.highest_score = 0
        self._scene_described = False
        self._last_plan_grid: list[list[int]] | None = None  # frame at start of last plan
        self.failed_plans: list[dict] = []  # Change 3: track failures

    def setup_primitives(self, frame: FrameData) -> None:
        available = frame.available_actions or [1, 2, 3, 4]
        self.k.available_actions = available
        for i in range(1, 8):
            name = f"ACTION{i}"
            self.k.primitives[name] = Primitive(
                action=name, is_available=(i in available),
                direction=INPUT_MAP.get(name, "unknown"),
                confidence=1.0 if i in available else 0.0,
            )
        for name, p in self.k.primitives.items():
            if p.is_available and p.direction in ("up", "down", "left", "right"):
                self.skills.add(f"move_{p.direction}", [name], description=f"Single {p.direction} step")

    def run(self, frame: FrameData, step: StepFn, budget: int) -> FrameData:
        logger.info("=== PLAYING ===")

        while self.action_counter < budget:
            if frame.state == GameState.WIN:
                logger.info("WIN!")
                break

            if frame.state == GameState.GAME_OVER:
                self.total_deaths += 1
                self.k.death_events.append({"step": self.action_counter, "deaths": self.total_deaths})
                logger.info(f"DIED #{self.total_deaths}")
                frame = step(GameAction.RESET, f"Reset after death #{self.total_deaths}")
                self.action_counter += 1
                continue

            self.k.dedup_hypotheses()
            self.k.dedup_rules()

            # Scene description: first time + on large changes
            if not self._scene_described:
                self.k.scene_description = self.scene.describe(frame.frame[-1])
                self._scene_described = True
                logger.info(f"[scene] {self.k.scene_description[:300]}...")

            # Change 8: Split into analyze (what happened) + plan (what to do)
            current_grid = frame.frame[-1]
            self._analyze(frame)
            plan, goal = self._plan(frame, budget - self.action_counter)

            logger.info(f"[plan] ({len(plan)} actions) {goal}")

            # Save grid for before/after comparison next cycle
            self._last_plan_grid = current_grid

            # Execute
            expanded = self.skills.expand(plan)
            executed: list[str] = []
            replan_reason = None
            consecutive_stuck = 0
            last_change_count: int | None = None

            for action_name in expanded:
                if self.action_counter >= budget:
                    break

                action_name = str(action_name).strip()
                try:
                    action = GameAction.from_name(action_name)
                except (ValueError, KeyError, AttributeError):
                    continue
                if action.value not in self.k.available_actions and action != GameAction.RESET:
                    continue

                grid_before = frame.frame[-1]
                score_before = frame.levels_completed

                frame = step(action, f"{goal[:80]} [{len(executed)+1}/{len(expanded)}]")
                self.action_counter += 1

                grid_after = frame.frame[-1]
                diff = self.differ.compute(action_name, grid_before, grid_after)
                blocked = self.differ.is_blocked(diff)
                score_delta = frame.levels_completed - score_before
                is_death = frame.state == GameState.GAME_OVER

                self.k.log_action(self.action_counter, action_name, diff.total_changes,
                                  frame.levels_completed, blocked=blocked)
                self.k.transitions.record(Transition(
                    action=action_name, changes=diff.total_changes,
                    blocked=blocked, score_delta=score_delta,
                    large_change=diff.total_changes > 100, death=is_death,
                ))
                executed.append(action_name)

                logger.info(f"#{self.action_counter} {action_name}: "
                            f"{diff.total_changes}ch {'BLOCKED' if blocked else ''} "
                            f"score={frame.levels_completed}")

                # Change 1: Abort stuck plans
                if last_change_count is not None and abs(diff.total_changes - last_change_count) < 5:
                    consecutive_stuck += 1
                else:
                    consecutive_stuck = 0
                last_change_count = diff.total_changes

                if consecutive_stuck >= 3:
                    replan_reason = f"stuck ({consecutive_stuck} similar outcomes in a row)"
                    break

                # Events
                if frame.state == GameState.WIN:
                    break
                if is_death:
                    replan_reason = "died"
                    break
                if frame.levels_completed > self.highest_score:
                    self.highest_score = frame.levels_completed
                    self.k.score_events.append({"step": self.action_counter, "score": frame.levels_completed,
                                                "actions": executed.copy(), "goal": goal})
                    self.skills.record_success_sequence(executed.copy(), f"Score to {frame.levels_completed}")
                    logger.info(f"*** SCORE UP: {frame.levels_completed} ***")
                    self.k.scene_description = self.scene.describe(frame.frame[-1])
                    replan_reason = f"score increased to {frame.levels_completed}"
                    break

                # Change 6: Re-describe scene on large changes
                if diff.total_changes > 100:
                    self.k.scene_description = self.scene.describe(frame.frame[-1])
                    replan_reason = f"large change ({diff.total_changes} cells)"
                    break

            # Change 3: Record plan failure
            if replan_reason and not replan_reason.startswith("score"):
                self.failed_plans.append({
                    "goal": goal, "actions": executed[:8],
                    "reason": replan_reason, "step": self.action_counter,
                })
                logger.info(f"[replan] {replan_reason}")

            # Change 4: Record skill outcomes
            plan_succeeded = (replan_reason is None) or replan_reason.startswith("score")
            for item in plan:
                if item in self.skills.skills:
                    self.skills.record_use(item, plan_succeeded)

        return frame

    # ── Change 8: Two-phase thinking ────────────────────────────

    def _analyze(self, frame: FrameData) -> None:
        """Phase 1: Analyze what happened. Update knowledge. Uses vision model."""
        if not self._last_plan_grid:
            return  # nothing to analyze on first cycle

        content = [
            text_block(
                f"{AGENT_CONTEXT}\n\n"
                f"CURRENT KNOWLEDGE:\n{self.k.compact_text()}\n\n"
                "Side-by-side: frame BEFORE your last plan (left) vs frame NOW (right):"
            ),
            image_block(side_by_side_b64(self._last_plan_grid, frame.frame[-1], "BEFORE", "AFTER")),
        ]

        # Change 3: Surface failures
        if self.failed_plans:
            failures = "\nRECENT FAILURES (do NOT repeat):\n"
            for f in self.failed_plans[-3:]:
                failures += f"  - '{f['goal']}' failed: {f['reason']} after {f['actions']}\n"
            content.append(text_block(failures))

        content.append(text_block(
            "\nAnalyze what happened since your last plan. What changed? What did you learn?\n"
            "Update your understanding of the game.\n\n"
            "Respond with JSON:\n"
            '{"analysis": "what happened and what you learned",\n'
            ' "new_rules": [{"description": "...", "confidence": 0.5}],\n'
            ' "updated_hypotheses": [{"statement": "...", "status": "confirmed/refuted/untested", "priority": 1}]}'
        ))

        response = self.router.call(VISION_MODEL, content, max_tokens=1000)
        data = self._extract_json(response)

        analysis = data.get("analysis", "")
        if analysis:
            logger.info(f"[analyze] {analysis[:200]}")

        for r in data.get("new_rules", []):
            self.k.rules.append(Rule(id=f"rule_{len(self.k.rules)}",
                                     description=r.get("description", ""), confidence=r.get("confidence", 0.5)))

        for h in data.get("updated_hypotheses", []):
            statement = h.get("statement", "")
            status = h.get("status", "untested")
            # Update existing or add new
            found = False
            for existing in self.k.hypotheses:
                if existing.statement.lower()[:40] == statement.lower()[:40]:
                    existing.status = status
                    found = True
                    break
            if not found:
                self.k.hypotheses.append(Hypothesis(
                    id=f"hyp_{len(self.k.hypotheses)}", statement=statement,
                    priority=h.get("priority", 3), status=status,
                ))

    def _plan(self, frame: FrameData, remaining: int) -> tuple[list[str], str]:
        """Phase 2: Decide what to do next. Uses reasoning model."""
        content = [
            text_block(
                f"{AGENT_CONTEXT}\n\n"
                f"{self.k.compact_text()}\n\n"
                f"SKILLS:\n{self.skills.summary()}\n\n"
                f"Actions remaining: {remaining}\n\n"
            ),
        ]

        # Change 3: Surface failures in plan prompt too
        if self.failed_plans:
            failures = "RECENT FAILURES (do NOT repeat these approaches):\n"
            for f in self.failed_plans[-3:]:
                failures += f"  - '{f['goal']}' failed: {f['reason']} after {f['actions']}\n"
            content.append(text_block(failures + "\n"))

        # Show side-by-side if we have a previous frame, otherwise just current
        if self._last_plan_grid:
            content.extend([
                text_block("Side-by-side (before last plan vs now):"),
                image_block(side_by_side_b64(self._last_plan_grid, frame.frame[-1], "BEFORE", "NOW")),
            ])
        else:
            content.extend([
                text_block("Current frame:"),
                image_block(grid_to_b64(frame.frame[-1])),
            ])

        content.append(text_block(
            "\nDecide what to do next. Think step by step.\n"
            "If previous plans failed, try something FUNDAMENTALLY different.\n"
            "You can explore, test hypotheses, execute strategies, or define new skills.\n\n"
            "Respond with JSON:\n"
            '{"plan": ["ACTION3", "ACTION1", ...],\n'
            ' "goal": "what this plan aims to achieve",\n'
            ' "new_skills": [{"name": "...", "actions": [...], "description": "..."}],\n'
            ' "new_hypotheses": [{"statement": "...", "priority": 1, "test_plan": [...]}]}'
        ))

        response = self.router.call(REASONING_MODEL, content, max_tokens=3000)
        return self._parse_plan(response)

    def _parse_plan(self, response: str) -> tuple[list[str], str]:
        data = self._extract_json(response)
        plan = data.get("plan", [])
        goal = data.get("goal", "")

        for s in data.get("new_skills", []):
            actions = s.get("actions", [])
            if actions and all(isinstance(a, str) for a in actions):
                self.skills.add(s.get("name", f"skill_{len(self.skills.skills)}"),
                                actions, description=s.get("description", ""), source="synthesis")

        for h in data.get("new_hypotheses", []):
            test_plan = [a for a in (h.get("test_plan") or []) if isinstance(a, str) and a.startswith("ACTION")]
            self.k.hypotheses.append(Hypothesis(
                id=f"hyp_{len(self.k.hypotheses)}", statement=h.get("statement", ""),
                priority=h.get("priority", 3), test_plan=test_plan,
            ))

        if not plan:
            # Change 5: Systematic probe fallback instead of random cycling
            for hyp in sorted(self.k.hypotheses, key=lambda h: h.priority):
                if hyp.status == "untested" and hyp.test_plan:
                    return hyp.test_plan, f"Testing: {hyp.statement}"

            available = [f"ACTION{i}" for i in self.k.available_actions]
            return available[:4], "Systematic probe: try each available action once"

        return plan, goal

    def _extract_json(self, response: str) -> dict:
        try:
            return json.loads(response)
        except (json.JSONDecodeError, ValueError):
            pass
        cleaned = response
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1]
            if "```" in cleaned:
                cleaned = cleaned.split("```", 1)[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1]
            if "```" in cleaned:
                cleaned = cleaned.split("```", 1)[0]
        try:
            return json.loads(cleaned.strip())
        except (json.JSONDecodeError, ValueError):
            pass
        start = response.find("{")
        end = response.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(response[start:end + 1])
            except (json.JSONDecodeError, ValueError):
                pass
        return {}
