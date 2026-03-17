"""Program-synthesis agent — the model writes code to analyze and navigate the grid.

1. Model sees the image + knows it has a 64x64 grid of ints 0-15
2. Model writes Python code to analyze the grid (find objects, walls, paths)
3. Code executes, model reads output
4. Model writes more code or outputs actions based on what it learned
5. Actions execute, model gets results, loop

The model does spatial reasoning by WRITING PROGRAMS, not by thinking about coordinates.
"""

import json
import logging
import os
import time

from arcengine import FrameData, GameAction, GameState
from openai import OpenAI

from .sandbox import run_program
from .vision import grid_to_b64, image_block, text_block

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are playing a turn-based game on a 64x64 grid. Each cell is an integer 0-15 (colors).
INPUTS: ACTION1=up(row-4), ACTION2=down(row+4), ACTION3=left(col-4), ACTION4=right(col+4).

You have two tools:
1. run_code: Execute Python code with `grid` (64x64 list of ints) available. Use print() to see results.
2. take_actions: Execute a list of game actions.

APPROACH:
- Write code to analyze the grid. Find objects, walls, the player, paths.
- The grid is raw data — you can compute anything: connected components, flood fill, BFS pathfinding.
- Write code first, understand the layout, THEN decide what actions to take.
- After taking actions, write more code to see what changed.

IMPORTANT: Each move shifts the player by ~4 cells. BLOCKED means you hit a wall."""

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "Execute Python code with `grid` (64x64 int list) and `ROWS`/`COLS` (both 64) available. Use print() to output results. No imports needed — basic builtins are available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "take_actions",
            "description": "Execute a sequence of game actions. Each costs one action from your budget. Returns results for each action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]},
                        "description": "List of actions to execute in order",
                    },
                },
                "required": ["actions"],
            },
        },
    },
]


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
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.llm_calls = 0

    def run(self) -> None:
        timer = time.time()
        logger.info("=" * 60)
        logger.info(f"Program-Synthesis Agent on {self.game_id}")
        logger.info("=" * 60)

        frame = self._step(GameAction.RESET)
        self.current_grid = frame.frame[-1] if frame.frame else []

        if frame.state == GameState.WIN:
            self._close()
            return

        # Initial prompt with image
        self.messages.append({
            "role": "user",
            "content": [
                text_block(
                    f"Game started. Score: {frame.levels_completed}. "
                    f"You have {self.MAX_ACTIONS} actions.\n\n"
                    "Here is the game frame. The grid variable contains the raw 64x64 data.\n"
                    "Write code to analyze the grid — find distinct regions, objects, the player, walls. "
                    "Then decide what actions to take."
                ),
                image_block(grid_to_b64(frame.frame[-1])),
            ],
        })

        # Main loop
        while self.action_counter < self.MAX_ACTIONS:
            if frame.state == GameState.WIN:
                logger.info("WIN!")
                break

            if frame.state == GameState.GAME_OVER:
                self.total_deaths += 1
                logger.info(f"DIED #{self.total_deaths}")
                frame = self._step(GameAction.RESET)
                self.action_counter += 1
                self.current_grid = frame.frame[-1] if frame.frame else []
                self.messages.append({
                    "role": "user",
                    "content": f"GAME_OVER! Death #{self.total_deaths}. Reset. "
                               f"Score: {frame.levels_completed}. Actions left: {self.MAX_ACTIONS - self.action_counter}. "
                               "Analyze the new grid and try again.",
                })
                continue

            # Call model
            self.llm_calls += 1
            try:
                response = self.client.chat.completions.create(
                    model="o4-mini",
                    messages=self.messages,
                    tools=TOOL_DEFINITIONS,
                    max_completion_tokens=16000,
                )
            except Exception as e:
                logger.warning(f"API error: {e}")
                time.sleep(2)
                continue

            msg = response.choices[0].message

            # Serialize assistant message
            assistant_msg: dict = {"role": "assistant"}
            if msg.content:
                assistant_msg["content"] = msg.content
                logger.info(f"[think] {msg.content[:200]}")
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
            self.messages.append(assistant_msg)

            if not msg.tool_calls:
                self.messages.append({
                    "role": "user",
                    "content": f"Actions left: {self.MAX_ACTIONS - self.action_counter}. "
                               "Use run_code to analyze or take_actions to act.",
                })
                continue

            # Process tool calls
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    self.messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": "Invalid JSON arguments",
                    })
                    continue

                if name == "run_code":
                    code = args.get("code", "")
                    logger.info(f"[code] {code[:150]}...")
                    result = run_program(code, self.current_grid)
                    logger.info(f"[result] {result[:200]}")
                    self.messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": result,
                    })

                elif name == "take_actions":
                    action_names = args.get("actions", [])
                    results = []
                    for action_name in action_names:
                        if self.action_counter >= self.MAX_ACTIONS:
                            break
                        try:
                            action = GameAction.from_name(action_name)
                        except (ValueError, KeyError):
                            results.append(f"{action_name}: INVALID")
                            continue

                        grid_before = self.current_grid
                        frame = self._step(action, reasoning="")
                        self.action_counter += 1
                        self.current_grid = frame.frame[-1] if frame.frame else grid_before

                        changes = sum(1 for r in range(64) for c in range(64)
                                      if grid_before[r][c] != self.current_grid[r][c])
                        blocked = 0 < changes < 10

                        if blocked:
                            results.append(f"{action_name}: BLOCKED")
                        else:
                            results.append(f"{action_name}: MOVED ({changes} cells changed)")

                        logger.info(f"#{self.action_counter} {action_name}: "
                                    f"{'BLOCKED' if blocked else f'{changes}ch'} "
                                    f"score={frame.levels_completed}")

                        if frame.state in (GameState.WIN, GameState.GAME_OVER):
                            results.append(f"State: {frame.state.name}")
                            break

                        if frame.levels_completed > 0 and "SCORE" not in " ".join(results):
                            results.append(f"*** SCORE: {frame.levels_completed} ***")

                    result_text = "\n".join(results)
                    result_text += f"\nActions left: {self.MAX_ACTIONS - self.action_counter}"
                    result_text += "\nThe `grid` variable is now updated with the current frame."

                    self.messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": result_text,
                    })

            # Trim conversation to prevent overflow
            if len(self.messages) > 40:
                # Keep system + first user (with image) + last 25
                trimmed = self.messages[-25:]
                while trimmed and isinstance(trimmed[0], dict) and trimmed[0].get("role") == "tool":
                    trimmed.pop(0)
                while trimmed and isinstance(trimmed[0], dict) and trimmed[0].get("tool_calls"):
                    trimmed.pop(0)
                self.messages = self.messages[:2] + trimmed

        elapsed = round(time.time() - timer, 2)
        logger.info("=" * 60)
        logger.info(
            f"FINISHED: actions={self.action_counter} state={frame.state.name} "
            f"score={frame.levels_completed} deaths={self.total_deaths} "
            f"llm_calls={self.llm_calls} time={elapsed}s"
        )
        logger.info("=" * 60)
        self._close()

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
