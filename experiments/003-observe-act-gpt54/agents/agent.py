"""Observe-then-act agent using GPT-5.4 Responses API.

Each turn the model gets:
1. Symbolic state (objects, relations, positions from grid analysis)
2. Current frame image (only when requested or on first turn/death)
3. Diff-highlighted side-by-side when previous frame exists
4. Its own persistent notes and hypothesis

It outputs: observation, hypothesis, notes, and one action.
Single API call per turn.
"""

import json
import logging
import os
import time

from arcengine import FrameData, GameAction, GameState
from openai import OpenAI

from .symbolic import grid_to_symbolic, diff_symbolic
from .vision import grid_b64, diff_b64, input_text, input_image_b64

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are playing an unknown turn-based game on a 64x64 grid. Each cell is an integer 0-15 (mapped to colors).
Your score (levels_completed) increases when you complete a level. GAME_OVER means you failed.

You have up to 7 possible actions (ACTION1-ACTION7). Each maps to a keyboard/mouse input:
- Simple actions might be: up, down, left, right, spacebar, undo
- Complex actions (like clicking) require x,y coordinates
- Not all actions may be available in every game

Each turn you receive:
- A symbolic description of objects on the grid (colors, shapes, positions, relations)
- Images showing what changed since last turn (red outlines = changed cells)
- Your own persistent notes from previous turns

You must output JSON with:
- observation: what you notice about the current state and what changed
- hypothesis: your current theory about game rules and how to win
- notes: anything you want to remember next turn (color meanings, layout, strategy)
- action: which action to take (e.g. "ACTION1")
- action_args: for complex actions, {"x": int, "y": int} — omit for simple actions

Build hypotheses. Test them systematically. Update your understanding.

Once you have a hypothesis about the goal, act on it. Don't keep exploring indefinitely.
You have limited actions and the game state may be changing against you."""


class ToolUseAgent:
    MAX_ACTIONS = 100

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
        self.prev_image_grid: list[list[int]] | None = None
        self.prev_symbolic: dict | None = None
        self.hypothesis = ""
        self.notes = ""
        self.recent_actions: list[str] = []
        self.llm_calls = 0

    def run(self) -> None:
        timer = time.time()
        logger.info("=" * 60)
        logger.info(f"GPT-5.4 Agent on {self.game_id}")
        logger.info("=" * 60)

        frame = self._step(GameAction.RESET)
        self.current_grid = frame.frame[-1] if frame.frame else []

        if frame.state == GameState.WIN:
            self._close()
            return

        while self.action_counter < self.MAX_ACTIONS:
            if frame.state == GameState.WIN:
                logger.info("WIN!")
                break

            if frame.state == GameState.GAME_OVER:
                self.total_deaths += 1
                logger.info(f"DIED #{self.total_deaths}")
                self.recent_actions.append("DIED")
                frame = self._step(GameAction.RESET)
                self.action_counter += 1
                self.current_grid = frame.frame[-1] if frame.frame else []
                self.prev_image_grid = None
                self.prev_symbolic = None
                continue

            # Build input and call model
            action = self._think_and_act(frame)

            # Execute
            grid_before = self.current_grid
            frame = self._step(action, reasoning=self.hypothesis[:80])
            self.action_counter += 1
            self.current_grid = frame.frame[-1] if frame.frame else grid_before

            changes = sum(1 for r in range(64) for c in range(64)
                          if grid_before[r][c] != self.current_grid[r][c])
            blocked = 0 < changes < 10

            result = f"{action.name}: {'BLOCKED' if blocked else f'{changes} cells changed'}"
            self.recent_actions.append(result)
            if len(self.recent_actions) > 15:
                self.recent_actions = self.recent_actions[-15:]

            logger.info(f"#{self.action_counter} {result} score={frame.levels_completed}")

            # Force fresh image on score change
            if frame.levels_completed > 0 and "SCORE" not in " ".join(self.recent_actions[-3:]):
                self.recent_actions.append(f"*** SCORE: {frame.levels_completed} ***")
                self.prev_image_grid = None

        elapsed = round(time.time() - timer, 2)
        logger.info("=" * 60)
        logger.info(
            f"FINISHED: actions={self.action_counter} state={frame.state.name} "
            f"score={frame.levels_completed} deaths={self.total_deaths} "
            f"llm_calls={self.llm_calls} time={elapsed}s"
        )
        logger.info(f"Hypothesis: {self.hypothesis}")
        logger.info(f"Notes: {self.notes}")
        logger.info("=" * 60)
        self._close()

    def _think_and_act(self, frame: FrameData) -> GameAction:
        """Single GPT-5.4 call: sees symbolic state + images, outputs observation + action."""
        self.llm_calls += 1

        available = frame.available_actions or [1, 2, 3, 4]
        avail_str = ", ".join(f"ACTION{i}" for i in available)
        recent = "\n".join(self.recent_actions[-10:]) if self.recent_actions else "(first turn)"

        # Symbolic state
        symbolic = grid_to_symbolic(self.current_grid)

        # Diff against previous symbolic state
        sym_changes = diff_symbolic(self.prev_symbolic, symbolic) if self.prev_symbolic else []
        self.prev_symbolic = symbolic

        # Build input content
        changes_text = ""
        if sym_changes:
            changes_text = f"CHANGES SINCE LAST ACTION:\n{json.dumps(sym_changes, indent=1)}\n\n"

        content = [
            input_text(SYSTEM_PROMPT),
            input_text(
                f"\nScore: {frame.levels_completed} | Deaths: {self.total_deaths} | "
                f"Actions: {self.action_counter}/{self.MAX_ACTIONS}\n"
                f"Available actions: {avail_str}\n\n"
                + (f"YOUR NOTES:\n{self.notes}\n\n" if self.notes else "")
                + (f"HYPOTHESIS: {self.hypothesis}\n\n" if self.hypothesis else "")
                + f"RECENT ACTIONS:\n{recent}\n\n"
                + changes_text
                + f"SYMBOLIC STATE:\n{json.dumps(symbolic, indent=1)}\n"
            ),
        ]

        # Images: side-by-side diff if we have a previous, otherwise just current
        if self.prev_image_grid:
            content.append(input_text("Side-by-side: PREVIOUS frame vs CURRENT frame (red outlines = changed cells):"))
            content.append(input_image_b64(diff_b64(self.prev_image_grid, self.current_grid)))
        else:
            content.append(input_text("Current game frame:"))
            content.append(input_image_b64(grid_b64(self.current_grid)))

        self.prev_image_grid = self.current_grid

        content.append(input_text(
            "\nRespond with JSON:\n"
            '{"observation": "what you see and what changed",\n'
            ' "hypothesis": "your theory about game rules and goal",\n'
            ' "notes": "persistent notes for next turn",\n'
            ' "action": "ACTION1",\n'
            ' "action_args": {} }\n'
            "(action_args only needed for complex actions like clicking)"
        ))

        # Call GPT-5.4 Responses API
        try:
            response = self.client.responses.create(
                model="gpt-5.4",
                input=[{"role": "user", "content": content}],
                max_output_tokens=1500,
                temperature=0.2,
            )
            raw = response.output_text or "{}"
        except Exception as e:
            logger.warning(f"API error: {e}")
            time.sleep(2)
            return GameAction.ACTION1

        # Parse response
        data = self._parse_json(raw)

        obs = data.get("observation", "")
        hyp = data.get("hypothesis", "")
        notes = data.get("notes", "")
        action_name = data.get("action", "ACTION1")
        action_args = data.get("action_args", {})

        if hyp:
            self.hypothesis = hyp
        if notes:
            self.notes = notes

        logger.info(f"[observe] {obs[:200]}")
        if hyp:
            logger.info(f"[hypothesis] {hyp[:150]}")

        # Resolve action
        try:
            action = GameAction.from_name(action_name)
            if action.is_complex() and action_args:
                action.set_data({
                    "x": int(action_args.get("x", 32)),
                    "y": int(action_args.get("y", 32)),
                })
        except (ValueError, KeyError):
            action = GameAction.ACTION1

        return action

    def _parse_json(self, raw: str) -> dict:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            pass
        # Try stripping markdown fences
        cleaned = raw
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
