"""Code-driven agent with rare LLM strategy calls.

Code handles: position tracking, wall mapping, pathfinding, exploration.
LLM handles: looking at the game, hypothesizing the goal, setting strategy.
LLM is called ~3-5 times total. Everything else is code.
"""

import json
import logging
import os
import time

from arcengine import FrameData, GameAction, GameState
from openai import OpenAI

from .navigator import Navigator
from .tools import GridTools
from .vision import grid_to_b64, image_block, text_block

logger = logging.getLogger(__name__)


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
        self.nav = Navigator()
        self.tools = GridTools()

        self.strategy: str = ""
        self.targets: list[tuple[int, int]] = []  # positions to visit
        self.current_target_idx = 0
        self.llm_calls = 0

    def run(self) -> None:
        timer = time.time()
        logger.info("=" * 60)
        logger.info(f"Code-Driven Agent on {self.game_id}")
        logger.info("=" * 60)

        frame = self._step(GameAction.RESET)
        if frame.frame:
            self.tools.current_grid = frame.frame[-1]

        if frame.state == GameState.WIN:
            self._close()
            return

        # === LLM CALL 1: Look at the game and set initial strategy ===
        self._strategize(frame, "Initial analysis — what is this game? What objects do you see? Where should I go first?")

        # === MAIN LOOP: code-driven execution ===
        consecutive_blocked = 0
        actions_since_strategy = 0

        while self.action_counter < self.MAX_ACTIONS:
            if frame.state == GameState.WIN:
                logger.info("WIN!")
                break

            if frame.state == GameState.GAME_OVER:
                self.total_deaths += 1
                logger.info(f"DIED #{self.total_deaths}")
                frame = self._step(GameAction.RESET)
                self.action_counter += 1
                if frame.frame:
                    self.tools.current_grid = frame.frame[-1]
                    self.nav = Navigator()  # reset nav on death
                self._strategize(frame, f"Died (death #{self.total_deaths}). What should I do differently?")
                consecutive_blocked = 0
                actions_since_strategy = 0
                continue

            # Pick action: navigate to target or explore
            if self.targets and self.current_target_idx < len(self.targets):
                target = self.targets[self.current_target_idx]
                action_name = self.nav.direction_to(target)
                if action_name is None:
                    action_name = self.nav.explore_action()
                # Check if we reached the target
                if self.nav.pos and abs(self.nav.pos[0] - target[0]) < 6 and abs(self.nav.pos[1] - target[1]) < 6:
                    logger.info(f"Reached target {self.current_target_idx}: {target}")
                    self.current_target_idx += 1
                    if self.current_target_idx >= len(self.targets):
                        self._strategize(frame, "Reached all targets. What next?")
                        actions_since_strategy = 0
            else:
                action_name = self.nav.explore_action()

            # Execute
            try:
                action = GameAction.from_name(action_name)
            except (ValueError, KeyError):
                action = GameAction.ACTION1

            grid_before = frame.frame[-1]
            score_before = frame.levels_completed
            frame = self._step(action, reasoning=self.strategy[:80])
            self.action_counter += 1
            actions_since_strategy += 1

            grid_after = frame.frame[-1]
            changes = sum(1 for r in range(64) for c in range(64)
                          if grid_before[r][c] != grid_after[r][c])
            blocked = 0 < changes < 10

            self.nav.update(action.name, grid_before, grid_after, changes, blocked)
            self.tools.update(grid_after, action.name, changes, blocked)

            if blocked:
                consecutive_blocked += 1
            else:
                consecutive_blocked = 0

            logger.info(f"#{self.action_counter} {action.name}: {changes}ch "
                        f"{'BLOCKED' if blocked else ''} "
                        f"pos={self.nav.pos} score={frame.levels_completed}")

            # Re-strategize triggers (RARE — only on significant events)
            if frame.levels_completed > score_before:
                logger.info(f"*** SCORE UP: {frame.levels_completed} ***")
                self._strategize(frame, f"Score increased to {frame.levels_completed}! What happened and what next?")
                actions_since_strategy = 0
                consecutive_blocked = 0

            elif consecutive_blocked >= 8:
                self._strategize(frame, f"Stuck — {consecutive_blocked} blocked moves in a row. Need a new approach.")
                actions_since_strategy = 0
                consecutive_blocked = 0

            elif actions_since_strategy >= 30:
                self._strategize(frame, "30 actions since last strategy check. Am I making progress?")
                actions_since_strategy = 0

        elapsed = round(time.time() - timer, 2)
        logger.info("=" * 60)
        logger.info(
            f"FINISHED: actions={self.action_counter} state={frame.state.name} "
            f"score={frame.levels_completed} deaths={self.total_deaths} "
            f"llm_calls={self.llm_calls} time={elapsed}s"
        )
        logger.info(f"Strategy: {self.strategy}")
        logger.info(f"Nav: {self.nav.status()}")
        logger.info("=" * 60)
        self._close()

    def _strategize(self, frame: FrameData, trigger: str) -> None:
        """Rare LLM call (~3-5 per game). Looks at the image + grid data, sets strategy and targets."""
        self.llm_calls += 1
        logger.info(f"[strategy #{self.llm_calls}] {trigger}")

        # Build compact grid info from tools
        color_info = self.tools.color_summary()
        nav_status = self.nav.status()

        content = [
            text_block(
                f"You are playing a turn-based game. 64x64 grid, 16 colors.\n"
                f"ACTION1=up, ACTION2=down, ACTION3=left, ACTION4=right.\n\n"
                f"TRIGGER: {trigger}\n\n"
                f"Score: {frame.levels_completed} | Deaths: {self.total_deaths} | "
                f"Actions: {self.action_counter}/{self.MAX_ACTIONS}\n\n"
                f"{nav_status}\n\n"
                f"{color_info}\n\n"
                f"Current frame:"
            ),
            image_block(grid_to_b64(frame.frame[-1])),
            text_block(
                "\nAnalyze the image. What do you see? What is the game about?\n"
                "Then give me:\n"
                "1. A one-sentence STRATEGY (what should I do to complete the level)\n"
                "2. A list of TARGET POSITIONS (row, col) to visit in order\n\n"
                "Respond with JSON:\n"
                '{"strategy": "one sentence", "targets": [[row,col], [row,col], ...]}'
            ),
        ]

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model="o4-mini",
                    messages=[{"role": "user", "content": content}],
                    max_completion_tokens=2000,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content or "{}"
                break
            except Exception as e:
                logger.warning(f"[strategy] API error: {e}")
                time.sleep(2 ** attempt)
                raw = "{}"

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}

        self.strategy = data.get("strategy", self.strategy or "Explore the grid systematically")
        raw_targets = data.get("targets", [])

        # Parse targets
        self.targets = []
        self.current_target_idx = 0
        for t in raw_targets:
            if isinstance(t, list) and len(t) == 2:
                try:
                    self.targets.append((int(t[0]), int(t[1])))
                except (ValueError, TypeError):
                    pass

        logger.info(f"[strategy] {self.strategy}")
        logger.info(f"[strategy] targets: {self.targets}")

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
