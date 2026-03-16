"""VLM Explorer v2 — fixed based on iteration 1 failure analysis.

Iteration 1 problems:
- VLM didn't understand actions are MOVEMENT (thought they were "interactions")
- Obsessively repeated ACTION2 near "white cross" for 100+ actions
- 2ch = wall hit wasn't recognized
- 1 VLM call per action too expensive, responses were garbage

Fixes:
- Discovery phase: take each action TWICE, show side-by-side before/after images
  AND explicitly compute movement direction from diff centroids
- Tell the VLM upfront: "52ch = moved, 2ch = hit wall, 0ch = no effect"
- Play phase: VLM outputs PLANS (10-20 actions), not single actions
- Show action counts in reflection to prevent repetition loops
- Include diff image with EVERY VLM call so it sees spatial changes
"""

import json
import logging
import os
import time
from collections import Counter

from arcengine import FrameData, GameAction, GameState
from openai import OpenAI

from .vision import diff_to_b64, grid_to_b64, image_block, text_block

logger = logging.getLogger(__name__)


class VLMExplorerAgent:
    MAX_ACTIONS = 200
    MODEL = "gpt-4o"

    def __init__(self, game_id: str) -> None:
        from arc_agi import Arcade
        self.game_id = game_id
        self.arcade = Arcade()
        self.scorecard_id = self.arcade.open_scorecard()
        self.env = self.arcade.make(game_id, scorecard_id=self.scorecard_id)
        self.frames: list[FrameData] = []
        self.action_counter = 0
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        self.game_knowledge = ""
        self.action_history: list[str] = []
        self.action_counts: Counter = Counter()
        self.total_deaths = 0
        self.highest_score = 0

    def run(self) -> None:
        timer = time.time()
        logger.info("=" * 60)
        logger.info(f"EXPERIMENT 002 (iter 2): VLM Explorer on {self.game_id}")
        logger.info("=" * 60)

        frame = self._step(GameAction.RESET)

        if frame.state == GameState.WIN:
            self._close()
            return

        # === PHASE 1: DISCOVERY ===
        logger.info("=== PHASE 1: DISCOVERY ===")
        frame = self._discovery_phase(frame)

        if frame.state == GameState.WIN:
            self._close()
            return

        # === PHASE 2: PLAY ===
        logger.info("=== PHASE 2: PLAY ===")
        frame = self._play_phase(frame)

        elapsed = round(time.time() - timer, 2)
        logger.info("=" * 60)
        logger.info(
            f"FINISHED: actions={self.action_counter} state={frame.state.name} "
            f"score={frame.levels_completed} deaths={self.total_deaths} time={elapsed}s"
        )
        logger.info(f"Final game knowledge:\n{self.game_knowledge}")
        logger.info("=" * 60)
        self._close()

    def _discovery_phase(self, frame: FrameData) -> FrameData:
        """Take each directional action twice. Compute movement from diffs. Show VLM everything."""
        initial_grid = frame.frame[-1]

        test_actions = [
            (GameAction.ACTION1, "ACTION1"),
            (GameAction.ACTION2, "ACTION2"),
            (GameAction.ACTION3, "ACTION3"),
            (GameAction.ACTION4, "ACTION4"),
        ]

        # For each action: take it twice, record before/after grids and compute movement
        discovery_data = []
        for action, name in test_actions:
            for trial in range(2):
                grid_before = frame.frame[-1]
                frame = self._step(action)
                grid_after = frame.frame[-1]

                changes = self._count_changes(grid_before, grid_after)
                direction = self._compute_movement_direction(grid_before, grid_after)

                discovery_data.append({
                    "action": name,
                    "trial": trial,
                    "changes": changes,
                    "direction": direction,
                    "before_b64": grid_to_b64(grid_before),
                    "after_b64": grid_to_b64(grid_after),
                    "diff_b64": diff_to_b64(grid_before, grid_after),
                })

                self.action_history.append(f"{name}: {changes}ch dir={direction}")

                if frame.state == GameState.GAME_OVER:
                    frame = self._step(GameAction.RESET)
                    self.total_deaths += 1
                if frame.state == GameState.WIN:
                    return frame

        # Build movement summary from data
        movement_summary = self._summarize_movements(discovery_data)
        logger.info(f"[discovery] Movement summary: {movement_summary}")

        # Now ask VLM to analyze with all the computed info
        content = [
            text_block(
                "You are analyzing an unknown video game on a 64x64 pixel grid with 16 colors.\n\n"
                "CRITICAL FACTS I've already computed:\n"
                f"{movement_summary}\n\n"
                "KEY PATTERNS:\n"
                "- When ~52 cells change: the player MOVED successfully\n"
                "- When ~2 cells change: the player HIT A WALL (couldn't move)\n"
                "- When 0 cells change: the action has no effect\n\n"
                "I'll show you the initial frame, then each action's result with a diff overlay.\n"
                "Based on all this, describe:\n"
                "1. What the game looks like (player, walls, objects, UI elements)\n"
                "2. What you think the GOAL might be\n"
                "3. What strategy you'd recommend\n\n"
                "End with a GAME RULES section.\n"
            ),
            text_block("INITIAL FRAME:"),
            image_block(grid_to_b64(initial_grid)),
        ]

        # Show first trial of each action
        for d in discovery_data:
            if d["trial"] == 0:
                content.extend([
                    text_block(f"\n{d['action']}: {d['changes']} cells changed, movement direction: {d['direction']}"),
                    text_block("Result:"),
                    image_block(d["after_b64"]),
                    text_block("Diff (red = changes):"),
                    image_block(d["diff_b64"]),
                ])

        response = self._vlm_call(content, "discovery")
        self.game_knowledge = f"MOVEMENT MAP:\n{movement_summary}\n\nVLM ANALYSIS:\n{response}"

        logger.info(f"[discovery] VLM analysis:\n{response[:500]}...")
        return frame

    def _play_phase(self, frame: FrameData) -> FrameData:
        """VLM outputs action PLANS (10-20 steps), executed without VLM calls until re-plan needed."""
        actions_since_replan = 0

        while self.action_counter < self.MAX_ACTIONS:
            if frame.state == GameState.WIN:
                logger.info("[play] WIN!")
                break

            if frame.state == GameState.GAME_OVER:
                self.total_deaths += 1
                logger.info(f"[play] DIED #{self.total_deaths}")

                # Quick death analysis
                death_note = f"DEATH #{self.total_deaths} at action {self.action_counter}. " \
                             f"Recent: {', '.join(self.action_history[-10:])}. " \
                             f"Likely cause: ran out of energy (too many actions without progress)."
                self.game_knowledge += f"\n\n{death_note}"
                logger.info(f"[play] {death_note}")

                frame = self._step(GameAction.RESET)
                actions_since_replan = 0
                continue

            # Get a plan from VLM
            grid = frame.frame[-1]
            plan, reasoning = self._get_plan(frame, grid)
            logger.info(f"[play] plan ({len(plan)} steps): {plan[:15]}{'...' if len(plan)>15 else ''} | {reasoning[:100]}")

            # Execute the plan
            for action_name in plan:
                if self.action_counter >= self.MAX_ACTIONS:
                    break

                try:
                    action = GameAction.from_name(action_name)
                except (ValueError, KeyError):
                    continue

                grid_before = frame.frame[-1]
                score_before = frame.levels_completed

                frame = self._step(action)
                grid_after = frame.frame[-1]

                changes = self._count_changes(grid_before, grid_after)
                self.action_counts[action_name] += 1
                actions_since_replan += 1

                score_changed = frame.levels_completed > self.highest_score
                if score_changed:
                    self.highest_score = frame.levels_completed
                    logger.info(f"[play] *** SCORE UP: {frame.levels_completed} ***")
                    self.game_knowledge += f"\n\nSCORE to {frame.levels_completed} after {action_name}!"

                wall_hit = changes <= 5 and changes > 0

                self.action_history.append(
                    f"{action_name}: {changes}ch" +
                    (" WALL" if wall_hit else "") +
                    (" SCORE!" if score_changed else "")
                )

                logger.info(
                    f"[play] #{self.action_counter} {action_name}: "
                    f"{changes}ch score={frame.levels_completed}"
                    + (" WALL" if wall_hit else "")
                    + (" SCORE!" if score_changed else "")
                )

                # Re-plan triggers
                if frame.state in (GameState.WIN, GameState.GAME_OVER):
                    break
                if score_changed:
                    logger.info("[play] re-planning: score changed!")
                    break

            # If plan exhausted normally, will loop back and get new plan

        return frame

    def _get_plan(self, frame: FrameData, grid: list[list[int]]) -> tuple[list[str], str]:
        """Ask VLM for a multi-step plan."""
        # Build action stats
        stats = ", ".join(f"{k}:{v}" for k, v in sorted(self.action_counts.items()))

        content = [
            text_block(
                f"GAME KNOWLEDGE:\n{self.game_knowledge}\n\n"
                f"Score: {frame.levels_completed} | Deaths: {self.total_deaths} | "
                f"Actions: {self.action_counter}/{self.MAX_ACTIONS}\n"
                f"Action counts so far: {stats or 'none'}\n"
                f"Recent history: {', '.join(self.action_history[-15:])}\n\n"
                f"REMINDERS:\n"
                f"- ~52 cell changes = successful move\n"
                f"- ~2 cell changes = hit a wall, try different direction\n"
                f"- If you keep hitting walls, you need to navigate AROUND them\n"
                f"- You have limited energy — don't waste moves hitting walls\n"
            ),
            text_block("Current frame:"),
            image_block(grid_to_b64(grid)),
            text_block(
                "\nOutput a PLAN of 10-20 actions as JSON:\n"
                '{"plan": ["ACTION1", "ACTION4", ...], "reasoning": "what you\'re trying to do"}\n\n'
                "Think about WHERE the player is and WHERE it needs to go. "
                "Plan a PATH, not random moves. If you keep hitting walls in one direction, "
                "try going around them."
            ),
        ]

        response = self._vlm_call(content, "plan")
        return self._parse_plan(response)

    def _parse_plan(self, response: str) -> tuple[list[str], str]:
        """Extract plan from VLM response."""
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                plan = data.get("plan", [])
                reasoning = data.get("reasoning", "")
                if plan:
                    return plan, reasoning
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback: extract ACTION keywords
        actions = []
        for word in response.split():
            word = word.strip('",[]')
            if word in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"):
                actions.append(word)
        if actions:
            return actions[:20], response[:100]

        return ["ACTION1", "ACTION4", "ACTION2", "ACTION3"] * 3, "fallback plan"

    def _compute_movement_direction(self, before: list[list[int]], after: list[list[int]]) -> str:
        """Compute movement direction from frame diff using centroid shift of changed cells."""
        changed = []
        for r in range(len(before)):
            for c in range(len(before[0])):
                if before[r][c] != after[r][c]:
                    changed.append((r, c))

        if len(changed) < 4:
            return "none (wall hit or no effect)"

        # Split changed cells into "gained non-floor" and "lost non-floor"
        # The player moved FROM cells that became floor-like TO cells that became player-like
        gained = []  # cells where new value is rarer (likely player arrived)
        lost = []    # cells where old value was rarer (likely player left)

        for r, c in changed:
            old, new = before[r][c], after[r][c]
            # Simple heuristic: the less common value in the grid is the "interesting" one
            gained.append((r, c, new))
            lost.append((r, c, old))

        # Compute centroids of all changed cells — the centroid shift IS the movement
        avg_r = sum(r for r, c in changed) / len(changed)
        avg_c = sum(c for r, c in changed) / len(changed)

        # Compare with centroid of changed cells weighted by whether they gained or lost rare colors
        # Simpler: just look at the bounding box shift
        min_r = min(r for r, c in changed)
        max_r = max(r for r, c in changed)
        min_c = min(c for r, c in changed)
        max_c = max(c for r, c in changed)

        # The changed region is typically a band in the direction of movement
        height = max_r - min_r + 1
        width = max_c - min_c + 1

        if height > width * 2:
            # Tall thin band = vertical movement
            # Check if top or bottom cells gained player colors
            top_cells = [(r, c) for r, c in changed if r < avg_r]
            bot_cells = [(r, c) for r, c in changed if r > avg_r]
            # The side where player color appeared is where player moved TO
            top_new = [after[r][c] for r, c in top_cells]
            bot_new = [after[r][c] for r, c in bot_cells]
            # More diverse colors = player area (player has multiple colors)
            if len(set(top_new)) > len(set(bot_new)):
                return "UP (row decreasing)"
            else:
                return "DOWN (row increasing)"
        elif width > height * 2:
            left_cells = [(r, c) for r, c in changed if c < avg_c]
            right_cells = [(r, c) for r, c in changed if c > avg_c]
            left_new = [after[r][c] for r, c in left_cells]
            right_new = [after[r][c] for r, c in right_cells]
            if len(set(left_new)) > len(set(right_new)):
                return "LEFT (col decreasing)"
            else:
                return "RIGHT (col increasing)"
        else:
            return f"unclear (changed region {width}x{height})"

    def _summarize_movements(self, data: list[dict]) -> str:
        """Build a human-readable movement summary from discovery data."""
        lines = []
        for d in data:
            lines.append(f"  {d['action']} trial {d['trial']}: {d['changes']}ch, direction={d['direction']}")

        # Deduce action-to-direction mapping
        action_dirs: dict[str, list[str]] = {}
        for d in data:
            if d["action"] not in action_dirs:
                action_dirs[d["action"]] = []
            action_dirs[d["action"]].append(d["direction"])

        lines.append("\nInferred action mapping:")
        for action, dirs in action_dirs.items():
            # Most common direction
            common = Counter(dirs).most_common(1)[0][0] if dirs else "unknown"
            lines.append(f"  {action} = {common}")

        return "\n".join(lines)

    def _count_changes(self, before: list[list[int]], after: list[list[int]]) -> int:
        return sum(1 for r in range(len(before)) for c in range(len(before[0])) if before[r][c] != after[r][c])

    def _vlm_call(self, content: list[dict], label: str) -> str:
        for attempt in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=self.MODEL,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=2000,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"[{label}] API error (attempt {attempt+1}): {e}, retry in {wait}s")
                time.sleep(wait)
        return "API unavailable."

    def _step(self, action: GameAction) -> FrameData:
        raw = self.env.step(action)
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
        self.action_counter += 1
        return frame

    def _close(self) -> None:
        if not self.scorecard_id:
            return
        scorecard = self.arcade.close_scorecard(self.scorecard_id)
        if scorecard:
            logger.info("--- SCORECARD ---")
            logger.info(json.dumps(scorecard.model_dump(), indent=2))
        self.scorecard_id = None
