"""Stage 5: Planning — plan-then-execute, not LLM-per-action.

LLM generates a multi-step plan (sequence of actions). Agent executes
the plan without LLM calls. Re-plans only on prediction mismatch,
score change, death, or plan exhaustion. Targets 3-8 LLM calls total.
"""

import json
import logging
import os
import time

from arcengine import FrameData, GameAction, GameState
from openai import OpenAI

from ..perception.differ import diff_frames
from ..perception.objects import ObjectCatalog
from .causal import CausalResult
from .goal_inference import GoalInferenceResult

logger = logging.getLogger(__name__)


class PlanningStage:
    """Plan-then-execute: LLM outputs action sequences, agent executes them."""

    def __init__(self, budget: int = 160, model: str = "gpt-4o-mini") -> None:
        self.budget = budget
        self.model = model

    def run(
        self,
        step_fn,
        current_frame: FrameData,
        goal: GoalInferenceResult,
        causal: CausalResult,
        catalog: ObjectCatalog,
    ) -> tuple[int, FrameData]:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        actions_used = 0
        frame = current_frame
        llm_calls = 0

        hypothesis = goal.best_hypothesis()
        goal_desc = hypothesis.description if hypothesis else "Unknown — explore and find the goal"
        strategy = hypothesis.suggested_strategy if hypothesis else "Try movement and interaction"

        execution_log: list[str] = []  # accumulates across re-plans

        while actions_used < self.budget:
            if frame.state == GameState.WIN:
                logger.info("[planning] WIN!")
                break

            if frame.state == GameState.GAME_OVER:
                logger.info("[planning] GAME_OVER — resetting")
                frame = step_fn(GameAction.RESET)
                actions_used += 1
                execution_log.append("DIED -> RESET")
                continue

            # --- Ask LLM for a plan ---
            grid = frame.frame[-1] if frame.frame else []
            prompt = _build_plan_prompt(
                goal_desc, strategy, catalog, frame, grid, execution_log
            )

            plan_data = None
            for attempt in range(5):
                try:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        response_format={"type": "json_object"},
                        max_tokens=500,
                    )
                    raw = response.choices[0].message.content or "{}"
                    plan_data = json.loads(raw)
                    llm_calls += 1
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    logger.warning(f"[planning] API error (attempt {attempt+1}): {e}, retrying in {wait}s")
                    time.sleep(wait)

            if plan_data is None:
                plan_data = {
                    "plan": ["ACTION1", "ACTION4", "ACTION2", "ACTION3"] * 5,
                    "reasoning": "API unavailable — exploring",
                }
                llm_calls += 1

            action_names = plan_data.get("plan", [])
            reasoning = plan_data.get("reasoning", "")
            logger.info(
                f"[planning] LLM call #{llm_calls}: "
                f"plan={action_names[:10]}{'...' if len(action_names) > 10 else ''} "
                f"| {reasoning[:100]}"
            )

            if not action_names:
                action_names = ["ACTION1", "ACTION4", "ACTION2", "ACTION3"]

            # --- Execute the plan ---
            replan_reason = None
            for i, action_name in enumerate(action_names):
                if actions_used >= self.budget:
                    break

                try:
                    action = GameAction.from_name(action_name)
                except (ValueError, KeyError):
                    continue

                if action.is_complex():
                    action.set_data({"x": 32, "y": 32})

                grid_before = frame.frame[-1] if frame.frame else []
                score_before = frame.levels_completed

                frame = step_fn(action)
                actions_used += 1

                grid_after = frame.frame[-1] if frame.frame else []

                # Check for events that trigger re-planning
                if frame.state == GameState.WIN:
                    execution_log.append(f"{action_name} -> WIN!")
                    break

                if frame.state == GameState.GAME_OVER:
                    execution_log.append(f"{action_name} -> DIED")
                    replan_reason = "died"
                    break

                if frame.levels_completed != score_before:
                    execution_log.append(
                        f"{action_name} -> SCORE {score_before}->{frame.levels_completed}"
                    )
                    replan_reason = "score_changed"
                    break

                # Track what happened
                if grid_before and grid_after:
                    diff = diff_frames(grid_before, grid_after)
                    execution_log.append(
                        f"{action_name}: {diff.num_changed} changes"
                    )
                else:
                    execution_log.append(f"{action_name}: no grid")

            # Keep execution log manageable
            if len(execution_log) > 40:
                execution_log = execution_log[-30:]

            if replan_reason:
                logger.info(f"[planning] re-planning because: {replan_reason}")

        logger.info(
            f"[planning] done — {actions_used} actions, {llm_calls} LLM calls, "
            f"state={frame.state.name}"
        )
        return actions_used, frame


SYSTEM_PROMPT = """You are playing an unknown game on a 64x64 grid. Each cell is an integer 0-15 (colors).

You must output a PLAN — a sequence of actions to execute. The plan will be executed WITHOUT further LLM calls until it completes or an important event happens (score change, death).

Respond with JSON:
{
  "reasoning": "your analysis of the current situation and what the plan aims to achieve",
  "plan": ["ACTION1", "ACTION4", "ACTION2", ...]
}

RULES:
- Plan should be 10-30 actions long. Be strategic, not random.
- Available actions: ACTION1 (up), ACTION2 (down), ACTION3 (left), ACTION4 (right), ACTION5, RESET
- If previous plans failed or got stuck, try a fundamentally different approach.
- Use the execution log to avoid repeating failed strategies.
- Think about spatial navigation: if you need to reach something, plan a path.
"""


def _build_plan_prompt(
    goal: str,
    strategy: str,
    catalog: ObjectCatalog,
    frame: FrameData,
    grid: list[list[int]],
    execution_log: list[str],
) -> str:
    parts = [
        f"GOAL: {goal}",
        f"STRATEGY: {strategy}",
        f"Score: {frame.levels_completed}",
        f"State: {frame.state.name}",
        "",
        "KNOWN OBJECTS:",
        catalog.summary(),
    ]

    if execution_log:
        parts.append("")
        parts.append("EXECUTION LOG (most recent):")
        for entry in execution_log[-20:]:
            parts.append(f"  {entry}")

    # Compact grid summary
    if grid:
        parts.append("")
        parts.append(f"Grid: {len(grid)}x{len(grid[0])}")

        # Show rows with non-uniform content
        interesting = []
        for r_idx, row in enumerate(grid):
            unique = set(row)
            if len(unique) > 1:  # skip uniform rows
                interesting.append(f"r{r_idx}: {row}")
        if len(interesting) <= 25:
            parts.append("Active rows:\n" + "\n".join(interesting))
        else:
            parts.append("Active rows (first 25):\n" + "\n".join(interesting[:25]))

    parts.append("\nOutput your plan.")
    return "\n".join(parts)
