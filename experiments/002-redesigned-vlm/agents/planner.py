"""Unified planner with curated tiered prompt.

Fixes from iteration 13 analysis:
1. Action name aliases (LEFT→ACTION3, etc.)
2. Large-change threshold raised to 200
3. Scene re-describe only on score changes
4. Tiered prompt (~700 tokens context, not raw dump)
5. Persistent current_theory field
6. Constrained output format
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
    "You are an agent playing a turn-based game on a 64x64 pixel grid with 16 colors. "
    "The game has hidden rules you must discover. It has multiple levels — your score "
    "(levels_completed) increases when you complete one. You lose (GAME_OVER) if you fail."
)

INPUT_MAP = {
    "ACTION1": "up", "ACTION2": "down", "ACTION3": "left", "ACTION4": "right",
    "ACTION5": "action (enter/spacebar)", "ACTION6": "click (x,y)", "ACTION7": "undo",
}

# Change 1: Action name aliases
ACTION_ALIASES = {
    "up": "ACTION1", "down": "ACTION2", "left": "ACTION3", "right": "ACTION4",
    "north": "ACTION1", "south": "ACTION2", "west": "ACTION3", "east": "ACTION4",
    "action": "ACTION5", "click": "ACTION6", "undo": "ACTION7",
    "move_up": "ACTION1", "move_down": "ACTION2", "move_left": "ACTION3", "move_right": "ACTION4",
}


class Planner:
    SEND_IMAGE = True
    LARGE_CHANGE_THRESHOLD = 200  # Change 2: raised from 100

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
        self._last_plan_goal: str = ""
        self._last_plan_outcome: str = ""
        self.failed_plans: list[dict] = []
        self._consecutive_parse_failures: int = 0

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
                self._last_plan_outcome = "DIED"
                logger.info(f"DIED #{self.total_deaths}")
                frame = step(GameAction.RESET, f"Reset after death #{self.total_deaths}")
                self.action_counter += 1
                continue

            self.k.dedup_hypotheses()
            self.k.dedup_rules()

            # Change 3: Scene description only on first cycle and score changes
            if self.SEND_IMAGE and not self._scene_described:
                self.k.scene_description = self.scene.describe(frame.frame[-1])
                self._scene_described = True
                logger.info(f"[scene] {self.k.scene_description[:300]}...")

            current_grid = frame.frame[-1]
            plan, goal = self._think(frame, budget - self.action_counter)
            logger.info(f"[plan] ({len(plan)} actions) {goal}")

            self._last_plan_grid = current_grid
            self._last_plan_goal = goal

            # Execute
            expanded = self.skills.expand(plan)
            executed: list[str] = []
            replan_reason = None
            consecutive_blocked = 0

            for action_name in expanded:
                if self.action_counter >= budget:
                    break

                action_name = str(action_name).strip()

                # Change 1: Normalize action names via aliases
                action = self._resolve_action(action_name)
                if action is None:
                    continue
                if action.value not in self.k.available_actions and action != GameAction.RESET:
                    continue

                grid_before = frame.frame[-1]
                score_before = frame.levels_completed

                frame = step(action, f"{goal[:80]} [{len(executed)+1}/{len(expanded)}]")
                self.action_counter += 1

                grid_after = frame.frame[-1]
                diff = self.differ.compute(action.name, grid_before, grid_after)
                blocked = self.differ.is_blocked(diff)
                score_delta = frame.levels_completed - score_before
                is_death = frame.state == GameState.GAME_OVER

                self.grid_mem.update(action.name, grid_before, grid_after)
                self.episodic.record(
                    self.action_counter, action.name, diff.total_changes,
                    blocked, frame.levels_completed, score_delta, is_death, grid_after,
                )
                self.k.log_action(self.action_counter, action.name, diff.total_changes,
                                  frame.levels_completed, blocked=blocked)
                self.k.transitions.record(Transition(
                    action=action.name, changes=diff.total_changes,
                    blocked=blocked, score_delta=score_delta,
                    large_change=diff.total_changes > self.LARGE_CHANGE_THRESHOLD, death=is_death,
                ))
                executed.append(action.name)

                logger.info(f"#{self.action_counter} {action.name}: "
                            f"{diff.total_changes}ch {'BLOCKED' if blocked else ''} "
                            f"score={frame.levels_completed}")

                if blocked:
                    consecutive_blocked += 1
                else:
                    consecutive_blocked = 0

                if consecutive_blocked >= 5:
                    replan_reason = f"stuck: {consecutive_blocked} consecutive blocked moves"
                    break

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
                    # Change 3: Only re-describe on score change
                    if self.SEND_IMAGE:
                        self.k.scene_description = self.scene.describe(frame.frame[-1])
                    replan_reason = f"score increased to {frame.levels_completed}"
                    break

                # Change 2: Higher threshold for large-change replans
                if diff.total_changes > self.LARGE_CHANGE_THRESHOLD:
                    replan_reason = f"large change ({diff.total_changes} cells)"
                    break

            # Record outcome
            if replan_reason:
                self._last_plan_outcome = replan_reason
                if not replan_reason.startswith("score"):
                    self.failed_plans.append({
                        "goal": goal, "actions": executed[:8],
                        "reason": replan_reason, "step": self.action_counter,
                    })
                logger.info(f"[replan] {replan_reason}")
            else:
                self._last_plan_outcome = f"completed ({len(executed)} actions)"

            plan_succeeded = (replan_reason is None) or replan_reason.startswith("score")
            for item in plan:
                if isinstance(item, str) and item in self.skills.skills:
                    self.skills.record_use(item, plan_succeeded)

        return frame

    # ── Change 4: Tiered prompt ──────────────────────────────────

    def _think(self, frame: FrameData, remaining: int) -> tuple[list[str], str]:
        # Skip LLM after 2 consecutive parse failures
        if self._consecutive_parse_failures >= 2:
            self._consecutive_parse_failures = 0
            available = [f"ACTION{i}" for i in self.k.available_actions]
            logger.info("[think] Skipping model — recent parse failures")
            return available[:4], "Probe (skipping model — recent parse failures)"

        grid = frame.frame[-1]

        # Tier 1: Always included (compact grid near active area)
        grid_text = compress_grid(grid, self.grid_mem, max_rows=16)
        tier1 = (
            f"{AGENT_CONTEXT}\n\n"
            f"INPUTS: ACTION1=up, ACTION2=down, ACTION3=left, ACTION4=right\n"
            f"Score: {frame.levels_completed} | Deaths: {self.total_deaths} | Actions left: {remaining}\n\n"
            f"{grid_text}"
        )

        # Tier 2: Recent context
        tier2_parts = []
        if self._last_plan_goal:
            tier2_parts.append(f"LAST PLAN: \"{self._last_plan_goal}\" → {self._last_plan_outcome}")

        if self.episodic.steps:
            tier2_parts.append("LAST 5 ACTIONS:")
            for s in self.episodic.steps[-5:]:
                status = "BLOCKED" if s.blocked else f"{s.changes}ch"
                tier2_parts.append(f"  #{s.number} {s.action}: {status}")

        if self.failed_plans:
            tier2_parts.append("FAILURES (don't repeat):")
            for f in self.failed_plans[-2:]:
                tier2_parts.append(f"  - \"{f['goal'][:50]}\" → {f['reason']}")

        tier2 = "\n".join(tier2_parts)

        # Tier 3: Accumulated knowledge (strictly bounded)
        tier3_parts = []

        # Change 5: Persistent theory
        if self.k.current_theory:
            tier3_parts.append(f"YOUR THEORY: {self.k.current_theory}")

        top_rules = sorted(self.k.rules, key=lambda r: -r.confidence)[:3]
        if top_rules:
            tier3_parts.append("CONFIRMED RULES:")
            for r in top_rules:
                tier3_parts.append(f"  - {r.description}")

        active_hyps = [h for h in self.k.hypotheses if h.status not in ("confirmed", "refuted")]
        top_hyps = sorted(active_hyps, key=lambda h: h.priority)[:3]
        if top_hyps:
            tier3_parts.append("HYPOTHESES TO TEST:")
            for h in top_hyps:
                tier3_parts.append(f"  - {h.statement}")

        tier3 = "\n".join(tier3_parts)

        # Assemble
        content = [
            text_block(f"{tier1}\n\n{tier2}\n\n{tier3}"),
        ]

        # Image
        if self.SEND_IMAGE:
            if self._last_plan_grid:
                content.extend([
                    text_block("Side-by-side (before last plan vs now):"),
                    image_block(side_by_side_b64(self._last_plan_grid, grid, "BEFORE", "NOW")),
                ])
            else:
                content.extend([
                    text_block("Current frame:"),
                    image_block(grid_to_b64(grid)),
                ])

        # Change 6: Constrained output format
        content.append(text_block(
            "\nPlan 10-25 actions to make progress. Think step by step.\n\n"
            "RULES:\n"
            "- Actions MUST be: ACTION1, ACTION2, ACTION3, ACTION4 (not LEFT/RIGHT/UP/DOWN)\n"
            "- If you keep getting BLOCKED, try a different direction\n"
            "- If a plan failed, try a fundamentally different approach\n\n"
            "Respond with JSON:\n"
            '{"plan": ["ACTION1", "ACTION4", "ACTION4", ...],\n'
            ' "goal": "what this achieves",\n'
            ' "theory": "one sentence: what you think this game is and how to win"}'
        ))

        response = self.router.call(REASONING_MODEL, content, max_tokens=4000)
        plan, goal = self._parse_plan(response)

        if not plan or goal.startswith("Probe"):
            self._consecutive_parse_failures += 1
        else:
            self._consecutive_parse_failures = 0

        return plan, goal

    # ── Parse + resolve ──────────────────────────────────────────

    def _resolve_action(self, name: str) -> GameAction | None:
        """Resolve an action name, including aliases like LEFT/RIGHT/UP/DOWN."""
        # Direct match
        try:
            return GameAction.from_name(name)
        except (ValueError, KeyError, AttributeError):
            pass

        # Alias lookup
        normalized = name.lower().strip().replace(" ", "_")
        if normalized in ACTION_ALIASES:
            alias = ACTION_ALIASES[normalized]
            if alias is None:
                return None  # WAIT, NOOP etc.
            try:
                return GameAction.from_name(alias)
            except (ValueError, KeyError, AttributeError):
                pass

        return None

    def _parse_plan(self, response: str) -> tuple[list[str], str]:
        data = self._extract_json(response)
        plan = data.get("plan", [])
        goal = data.get("goal", "")

        # Change 5: Store theory
        theory = data.get("theory", "")
        if theory:
            self.k.current_theory = theory
            logger.info(f"[theory] {theory}")

        # Normalize plan entries through aliases
        normalized_plan = []
        for item in plan:
            if not isinstance(item, str):
                continue
            action = self._resolve_action(item)
            if action:
                normalized_plan.append(action.name)

        if not normalized_plan:
            for hyp in sorted(self.k.hypotheses, key=lambda h: h.priority):
                if hyp.status == "untested" and hyp.test_plan:
                    return hyp.test_plan, f"Testing: {hyp.statement}"
            available = [f"ACTION{i}" for i in self.k.available_actions]
            return available[:4], "Probe: try each action once"

        # Store any new rules/hypotheses (lightweight, from the response)
        for r in data.get("new_rules", []):
            if isinstance(r, dict):
                self.k.rules.append(Rule(id=f"rule_{len(self.k.rules)}",
                                         description=r.get("description", ""), confidence=r.get("confidence", 0.5)))

        for h in data.get("new_hypotheses", []):
            if isinstance(h, dict):
                test_plan = [a for a in (h.get("test_plan") or []) if isinstance(a, str) and a.startswith("ACTION")]
                self.k.hypotheses.append(Hypothesis(
                    id=f"hyp_{len(self.k.hypotheses)}", statement=h.get("statement", ""),
                    priority=int(h.get("priority", 3)), test_plan=test_plan,
                ))

        return normalized_plan, goal

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
