"""Unified planner with three-layer memory.

Layer 1 (GridMemory) and Layer 2 (EpisodicBuffer) feed structured data to the LLM.
Layer 3 (GameKnowledge) is updated by the LLM on significant events.

Fixes from analysis:
- Stuck detection only fires on BLOCKED moves, not successful movement
- Compressed grid text sent alongside image for precise spatial reasoning
"""

import json
import logging
from typing import Callable

from arcengine import FrameData, GameAction, GameState

from .episodic import EpisodicBuffer
from .grid_memory import GridMemory, compress_grid
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
    # Test flags — set from config
    SEND_IMAGE = True       # Hypothesis 2: set False to test without image
    MINIMAL_PROMPT = True   # Hypothesis 1: set True for stripped-down prompt

    def __init__(self, knowledge: GameKnowledge, router: ModelRouter,
                 differ: DiffAnalyzer, grid_mem: GridMemory, episodic: EpisodicBuffer,
                 scene: SceneDescriber, skills: SkillLibrary) -> None:
        self.k = knowledge
        self.router = router
        self.differ = differ
        self.grid_mem = grid_mem
        self.episodic = episodic
        self.scene = scene
        self.skills = skills
        self.action_counter = 0
        self.total_deaths = 0
        self.highest_score = 0
        self._scene_described = False
        self._last_plan_grid: list[list[int]] | None = None
        self.failed_plans: list[dict] = []

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

            # Scene description only when images are enabled
            if self.SEND_IMAGE and not self._scene_described:
                self.k.scene_description = self.scene.describe(frame.frame[-1])
                self._scene_described = True
                logger.info(f"[scene] {self.k.scene_description[:300]}...")

            # Think: single call with all three memory layers
            current_grid = frame.frame[-1]
            plan, goal = self._think(frame, budget - self.action_counter)
            logger.info(f"[plan] ({len(plan)} actions) {goal}")

            self._last_plan_grid = current_grid

            # Execute
            expanded = self.skills.expand(plan)
            executed: list[str] = []
            replan_reason = None
            consecutive_blocked = 0  # FIX: only count BLOCKED moves

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

                # Update all three memory layers
                self.grid_mem.update(action_name, grid_before, grid_after)
                self.episodic.record(
                    self.action_counter, action_name, diff.total_changes,
                    blocked, frame.levels_completed, score_delta, is_death, grid_after,
                )
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

                # FIX Priority 1: Stuck detection only on BLOCKED moves
                if blocked:
                    consecutive_blocked += 1
                else:
                    consecutive_blocked = 0

                if consecutive_blocked >= 5:
                    replan_reason = f"stuck: {consecutive_blocked} consecutive blocked moves"
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
                    if self.SEND_IMAGE:
                        self.k.scene_description = self.scene.describe(frame.frame[-1])
                    replan_reason = f"score increased to {frame.levels_completed}"
                    break
                if diff.total_changes > 100:
                    if self.SEND_IMAGE:
                        self.k.scene_description = self.scene.describe(frame.frame[-1])
                    replan_reason = f"large change ({diff.total_changes} cells)"
                    break

            # Record plan outcome
            if replan_reason and not replan_reason.startswith("score"):
                self.failed_plans.append({
                    "goal": goal, "actions": executed[:8],
                    "reason": replan_reason, "step": self.action_counter,
                })
                logger.info(f"[replan] {replan_reason}")

            plan_succeeded = (replan_reason is None) or replan_reason.startswith("score")
            for item in plan:
                if isinstance(item, str) and item in self.skills.skills:
                    self.skills.record_use(item, plan_succeeded)

        return frame

    def _think(self, frame: FrameData, remaining: int) -> tuple[list[str], str]:
        """Single LLM call. Prompt size and image controlled by flags."""
        grid = frame.frame[-1]
        grid_text = compress_grid(grid)

        if self.MINIMAL_PROMPT:
            # Hypothesis 1: Stripped-down prompt — just essentials
            # Last 5 action results from episodic buffer
            recent = ""
            if self.episodic.steps:
                for s in self.episodic.steps[-5:]:
                    status = "BLOCKED" if s.blocked else f"{s.changes}ch"
                    recent += f"  #{s.number} {s.action}: {status} score={s.score}\n"

            failures = ""
            if self.failed_plans:
                for f in self.failed_plans[-2:]:
                    failures += f"  - '{f['goal'][:40]}' failed: {f['reason']}\n"

            content = [
                text_block(
                    f"{AGENT_CONTEXT}\n\n"
                    f"Actions remaining: {remaining}\n"
                    f"Score: {frame.levels_completed} | Deaths: {self.total_deaths}\n\n"
                    f"RECENT ACTIONS:\n{recent}\n"
                    + (f"FAILURES:\n{failures}\n" if failures else "")
                    + f"{grid_text}\n\n"
                    "Output a plan of 10-25 actions. Respond with JSON:\n"
                    '{"plan": ["ACTION1", ...], "goal": "what this achieves"}'
                ),
            ]
        else:
            # Full prompt with all three layers
            content = [
                text_block(
                    f"{AGENT_CONTEXT}\n\n"
                    f"{self.k.compact_text()}\n\n"
                    f"{self.grid_mem.compact_text()}\n\n"
                    f"{self.episodic.compact_text()}\n\n"
                    f"SKILLS:\n{self.skills.summary()}\n\n"
                    f"Actions remaining: {remaining}\n\n"
                    f"{grid_text}\n\n"
                ),
            ]

            if self.failed_plans:
                failures = "RECENT FAILURES (do NOT repeat):\n"
                for f in self.failed_plans[-3:]:
                    failures += f"  - '{f['goal']}' failed: {f['reason']} after {f['actions']}\n"
                content.append(text_block(failures))

            content.append(text_block(
                "\nThink step by step. If previous plans failed, try something different.\n\n"
                "Respond with JSON:\n"
                '{"plan": ["ACTION3", "ACTION1", ...],\n'
                ' "goal": "what this plan aims to achieve",\n'
                ' "new_skills": [{"name": "...", "actions": [...], "description": "..."}],\n'
                ' "new_rules": [{"description": "...", "confidence": 0.5}],\n'
                ' "new_hypotheses": [{"statement": "...", "priority": 1, "test_plan": [...]}]}'
            ))

        # Hypothesis 2: Only add image if SEND_IMAGE is True
        if self.SEND_IMAGE:
            if self._last_plan_grid:
                content.extend([
                    text_block("Side-by-side (before vs now):"),
                    image_block(side_by_side_b64(self._last_plan_grid, grid, "BEFORE", "NOW")),
                ])
            else:
                content.extend([
                    text_block("Current frame:"),
                    image_block(grid_to_b64(grid)),
                ])

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

        for r in data.get("new_rules", []):
            self.k.rules.append(Rule(id=f"rule_{len(self.k.rules)}",
                                     description=r.get("description", ""), confidence=r.get("confidence", 0.5)))

        for h in data.get("new_hypotheses", []):
            test_plan = [a for a in (h.get("test_plan") or []) if isinstance(a, str) and a.startswith("ACTION")]
            self.k.hypotheses.append(Hypothesis(
                id=f"hyp_{len(self.k.hypotheses)}", statement=h.get("statement", ""),
                priority=int(h.get("priority", 3)), test_plan=test_plan,
            ))

        if not plan:
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
