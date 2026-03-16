"""Unified planner — one loop, model decides what to do.

No hardcoded phases. Each cycle:
1. Model sees: current frame image + accumulated knowledge + recent action results
2. Model outputs: a plan (action sequence) + reasoning + any new knowledge
3. Agent executes the plan
4. On events (score change, death, large change, plan done), cycle back to 1

The model controls its own learning — it can explore, test hypotheses,
or execute strategies. It decides when to shift between these modes.
"""

import json
import logging
from typing import Callable

from arcengine import FrameData, GameAction, GameState

from .memory import GameKnowledge, Hypothesis, Primitive, Rule, Transition
from .models import ModelRouter, REASONING_MODEL
from .perception import DiffAnalyzer, SceneDescriber
from .skills import SkillLibrary
from .vision import grid_to_b64, image_block, text_block

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

# Universal ARC-AGI-3 input mappings (API-level, not game-specific)
INPUT_MAP = {
    "ACTION1": "up",
    "ACTION2": "down",
    "ACTION3": "left",
    "ACTION4": "right",
    "ACTION5": "action (enter/spacebar)",
    "ACTION6": "click (x,y)",
    "ACTION7": "undo",
}


class Planner:
    def __init__(
        self,
        knowledge: GameKnowledge,
        router: ModelRouter,
        differ: DiffAnalyzer,
        scene: SceneDescriber,
        skills: SkillLibrary,
    ) -> None:
        self.k = knowledge
        self.router = router
        self.differ = differ
        self.scene = scene
        self.skills = skills
        self.action_counter = 0
        self.total_deaths = 0
        self.highest_score = 0
        self._scene_described = False

    def setup_primitives(self, frame: FrameData) -> None:
        """Register available actions with known API-level input mappings."""
        available = frame.available_actions or [1, 2, 3, 4]
        self.k.available_actions = available

        for i in range(1, 8):
            name = f"ACTION{i}"
            direction = INPUT_MAP.get(name, "unknown")
            self.k.primitives[name] = Primitive(
                action=name,
                is_available=(i in available),
                direction=direction,
                confidence=1.0 if i in available else 0.0,
            )

        # Create basic movement skills
        for name, p in self.k.primitives.items():
            if p.is_available and p.direction in ("up", "down", "left", "right"):
                self.skills.add(f"move_{p.direction}", [name], description=f"Single {p.direction} step")

    def run(self, frame: FrameData, step: StepFn, budget: int) -> FrameData:
        """Main loop. Model decides what to do each cycle."""
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

            # Dedup before thinking
            self.k.dedup_hypotheses()
            self.k.dedup_rules()

            # Get scene description once (first cycle) and on score changes
            if not self._scene_described:
                self.k.scene_description = self.scene.describe(frame.frame[-1])
                self._scene_described = True
                logger.info(f"[scene] {self.k.scene_description[:200]}...")

            # Ask model what to do
            plan, goal, reasoning = self._think(frame, budget - self.action_counter)
            logger.info(f"[plan] ({len(plan)} actions) {goal}")
            if reasoning:
                logger.info(f"[reasoning] {reasoning[:200]}")

            # Execute the plan
            expanded = self.skills.expand(plan)
            executed: list[str] = []
            replan_reason = None

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

                self.k.log_action(
                    self.action_counter, action_name, diff.total_changes,
                    frame.levels_completed, blocked=blocked,
                )
                self.k.transitions.record(Transition(
                    action=action_name, changes=diff.total_changes,
                    blocked=blocked, score_delta=score_delta,
                    large_change=diff.total_changes > 100, death=is_death,
                ))
                executed.append(action_name)

                logger.info(
                    f"#{self.action_counter} {action_name}: "
                    f"{diff.total_changes}ch {'BLOCKED' if blocked else ''} "
                    f"score={frame.levels_completed}"
                )

                # Events that trigger re-thinking
                if frame.state == GameState.WIN:
                    break

                if is_death:
                    replan_reason = "died"
                    break

                if frame.levels_completed > self.highest_score:
                    self.highest_score = frame.levels_completed
                    self.k.score_events.append({
                        "step": self.action_counter,
                        "score": frame.levels_completed,
                        "actions": executed.copy(),
                        "goal": goal,
                    })
                    self.skills.record_success_sequence(
                        executed.copy(), f"Score to {frame.levels_completed}"
                    )
                    logger.info(f"*** SCORE UP: {frame.levels_completed} ***")
                    # Re-describe scene on score change (level might have changed)
                    self.k.scene_description = self.scene.describe(frame.frame[-1])
                    replan_reason = f"score increased to {frame.levels_completed}"
                    break

                if diff.total_changes > 100:
                    replan_reason = f"large change ({diff.total_changes} cells)"
                    break

            if replan_reason:
                logger.info(f"[replan] {replan_reason}")

        return frame

    def _think(self, frame: FrameData, remaining: int) -> tuple[list[str], str, str]:
        """Ask the model what to do. Returns (plan, goal, reasoning)."""
        content = [
            text_block(
                f"{AGENT_CONTEXT}\n\n"
                f"{self.k.compact_text()}\n\n"
                f"SKILLS:\n{self.skills.summary()}\n\n"
                f"Actions remaining: {remaining}\n\n"
                f"Current frame:"
            ),
            image_block(grid_to_b64(frame.frame[-1])),
            text_block(
                "\nLook at the image and everything you know. Decide what to do next.\n\n"
                "You can:\n"
                "- Explore: try actions to learn how the game works\n"
                "- Test a hypothesis: execute a specific sequence to confirm/deny a theory\n"
                "- Execute a strategy: carry out a plan to complete the level\n"
                "- Define new skills: name useful action sequences for reuse\n\n"
                "Think step by step about the current situation.\n"
                "If previous plans failed, explain WHY and try something DIFFERENT.\n\n"
                "Respond with JSON:\n"
                "{\n"
                '  "reasoning": "your analysis of the current situation",\n'
                '  "plan": ["ACTION3", "ACTION1", ...],\n'
                '  "goal": "what this plan aims to achieve",\n'
                '  "new_skills": [{"name": "...", "actions": [...], "description": "..."}],\n'
                '  "new_rules": [{"description": "...", "confidence": 0.5}],\n'
                '  "new_hypotheses": [{"statement": "...", "priority": 1, "test_plan": [...]}]\n'
                "}"
            ),
        ]

        response = self.router.call(REASONING_MODEL, content, max_tokens=3000)
        return self._parse_response(response)

    def _parse_response(self, response: str) -> tuple[list[str], str, str]:
        """Parse model response. Returns (plan, goal, reasoning)."""
        data = self._extract_json(response)

        plan = data.get("plan", [])
        goal = data.get("goal", "")
        reasoning = data.get("reasoning", "")

        # Add new skills
        for s in data.get("new_skills", []):
            actions = s.get("actions", [])
            if actions and all(isinstance(a, str) for a in actions):
                self.skills.add(
                    s.get("name", f"skill_{len(self.skills.skills)}"),
                    actions,
                    description=s.get("description", ""),
                    source="synthesis",
                )

        # Add new rules
        for r in data.get("new_rules", []):
            self.k.rules.append(Rule(
                id=f"rule_{len(self.k.rules)}",
                description=r.get("description", ""),
                confidence=r.get("confidence", 0.5),
            ))

        # Add new hypotheses
        for h in data.get("new_hypotheses", []):
            test_plan = [a for a in (h.get("test_plan") or []) if isinstance(a, str) and a.startswith("ACTION")]
            self.k.hypotheses.append(Hypothesis(
                id=f"hyp_{len(self.k.hypotheses)}",
                statement=h.get("statement", ""),
                priority=h.get("priority", 3),
                test_plan=test_plan,
            ))

        if not plan:
            # Fallback: try an untested hypothesis
            for hyp in sorted(self.k.hypotheses, key=lambda h: h.priority):
                if hyp.status == "untested" and hyp.test_plan:
                    return hyp.test_plan, f"Testing: {hyp.statement}", "Fallback to hypothesis test"

            plan = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"] * 5
            goal = "Fallback exploration"
            reasoning = "Could not parse plan from model response"

        return plan, goal, reasoning

    def _extract_json(self, response: str) -> dict:
        try:
            return json.loads(response)
        except (json.JSONDecodeError, ValueError):
            pass

        # Strip markdown fences
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
