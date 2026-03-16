"""VLM Explorer — vision-first agent with zero game knowledge.

The agent knows ONLY:
- The game is a 64x64 grid with 16 colors
- It can take actions and gets back a new frame, state, and score
- It needs to maximize score / reach WIN state
- The game tells it which actions are available via available_actions

Architecture (inspired by upstream MultiModalLLM's two-phase loop):
1. DISCOVERY: Take each available action, show VLM side-by-side before/after
2. PLAY: Every 3 actions, show VLM the latest frame + what happened, get next 3 actions

All VLM reasoning is logged to reasoning.log for debugging.
Move sequence is tracked in moves.log.
"""

import json
import logging
import os
import time
from collections import Counter

from arcengine import FrameData, GameAction, GameState
from openai import OpenAI

from .vision import diff_to_b64, grid_to_b64, side_by_side_b64, image_block, text_block

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
        self.move_log: list[dict] = []  # structured move tracking
        self.total_deaths = 0
        self.highest_score = 0

        # Open reasoning log
        log_dir = os.path.dirname(os.path.dirname(__file__))
        self.reasoning_log = open(os.path.join(log_dir, "reasoning.log"), "w")
        self.moves_log = open(os.path.join(log_dir, "moves.log"), "w")

    def run(self) -> None:
        timer = time.time()
        logger.info("=" * 60)
        logger.info(f"EXPERIMENT 002: VLM Explorer on {self.game_id}")
        logger.info("=" * 60)

        frame = self._step(GameAction.RESET)

        if frame.state == GameState.WIN:
            self._close()
            return

        # Get available actions from the game itself
        available = frame.available_actions if frame.available_actions else []
        logger.info(f"Available actions from game: {available}")

        # === PHASE 1: DISCOVERY ===
        logger.info("=== PHASE 1: DISCOVERY ===")
        frame = self._discovery_phase(frame, available)

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
        logger.info("=" * 60)
        self._close()

    def _discovery_phase(self, frame: FrameData, available: list) -> FrameData:
        """Take each available action, show VLM side-by-side comparisons."""
        initial_grid = frame.frame[-1]

        # Map available action IDs to GameAction
        test_actions = []
        for action_id in available:
            try:
                test_actions.append(GameAction.from_id(action_id))
            except (ValueError, KeyError):
                pass
        if not test_actions:
            test_actions = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]

        content = [
            text_block(
                "You are observing an unknown video game displayed as a 64x64 pixel grid "
                "with 16 possible colors. You can take actions and the game produces a new frame.\n\n"
                "I will show you the initial frame, then for each available action: "
                "a SIDE-BY-SIDE comparison (before on left, after on right) so you can directly "
                "compare what changed.\n\n"
                f"The game reports these actions are available: {[a.name for a in test_actions]}\n\n"
                "For EACH action, describe precisely:\n"
                "- What visual element moved or changed?\n"
                "- In which DIRECTION did it move? (up/down/left/right)\n"
                "- How far did it move (in pixels)?\n"
                "- Did anything else change? (counters, bars, indicators)\n\n"
                "Then provide your overall analysis of the game."
            ),
            text_block("\n--- INITIAL FRAME ---"),
            image_block(grid_to_b64(initial_grid)),
        ]

        for action in test_actions:
            grid_before = frame.frame[-1]
            frame = self._step(action, reasoning={"phase": "discovery", "testing": action.name})
            grid_after = frame.frame[-1]

            changes = self._count_changes(grid_before, grid_after)
            self._log_move(action.name, changes, frame.levels_completed, frame.state.name)

            content.extend([
                text_block(f"\n--- {action.name} ({changes} pixels changed) ---"),
                text_block("Side-by-side (BEFORE left, AFTER right):"),
                image_block(side_by_side_b64(grid_before, grid_after, "BEFORE", f"AFTER {action.name}")),
                text_block("Diff (red = changed pixels):"),
                image_block(diff_to_b64(grid_before, grid_after)),
            ])

            if frame.state == GameState.GAME_OVER:
                content.append(text_block(f"⚠️ GAME_OVER after {action.name}"))
                frame = self._step(GameAction.RESET)
                self.total_deaths += 1
            if frame.state == GameState.WIN:
                return frame

        # Also test doing same action twice to see consistent movement
        if len(test_actions) >= 1:
            action = test_actions[0]
            grid_before = frame.frame[-1]
            frame = self._step(action, reasoning={"phase": "discovery", "testing": f"{action.name} (repeat)"})
            grid_after = frame.frame[-1]
            changes = self._count_changes(grid_before, grid_after)
            self._log_move(action.name, changes, frame.levels_completed, frame.state.name)

            content.extend([
                text_block(f"\n--- {action.name} AGAIN ({changes} pixels changed) ---"),
                text_block("Side-by-side:"),
                image_block(side_by_side_b64(grid_before, grid_after, "BEFORE", f"AFTER {action.name} (2nd time)")),
            ])

            if frame.state == GameState.GAME_OVER:
                frame = self._step(GameAction.RESET)
                self.total_deaths += 1

        content.append(text_block(
            "\n\nBased on ALL side-by-side comparisons, tell me:\n"
            "1. What does each action do? (which direction does it move things?)\n"
            "2. What visual elements do you see?\n"
            "3. What do you think the objective is?\n"
            "4. What strategy would you use?\n\n"
            "End with a GAME RULES section."
        ))

        response = self._vlm_call(content, "discovery")
        self.game_knowledge = response
        self._log_reasoning("DISCOVERY", response)
        logger.info(f"[discovery] VLM analysis:\n{response[:500]}...")
        return frame

    def _play_phase(self, frame: FrameData) -> FrameData:
        """VLM-guided play. Show frame every 3 actions, get next 3."""
        batch_size = 3

        while self.action_counter < self.MAX_ACTIONS:
            if frame.state == GameState.WIN:
                logger.info("[play] WIN!")
                break

            if frame.state == GameState.GAME_OVER:
                self.total_deaths += 1
                logger.info(f"[play] DIED #{self.total_deaths}")

                # Death analysis with frame
                death_content = [
                    text_block(
                        f"GAME OVER (death #{self.total_deaths}).\n\n"
                        f"Your understanding:\n{self._compact_knowledge()}\n\n"
                        f"Recent moves:\n{self._recent_moves(15)}\n\n"
                        "What happened? Update your understanding."
                    ),
                    text_block("Last frame:"),
                    image_block(grid_to_b64(frame.frame[-1])),
                ]
                death_analysis = self._vlm_call(death_content, "death")
                self.game_knowledge += f"\n\nDEATH #{self.total_deaths}: {death_analysis}"
                self._log_reasoning(f"DEATH #{self.total_deaths}", death_analysis)

                frame = self._step(GameAction.RESET)
                continue

            # Get next batch of actions
            grid_before_batch = frame.frame[-1]
            actions, reasoning = self._get_actions(frame, batch_size)
            self._log_reasoning(f"PLAN (step {self.action_counter})", f"Actions: {actions}\nReasoning: {reasoning}")
            logger.info(f"[play] batch: {actions} | {reasoning[:80]}")

            # Execute the batch
            for action_name in actions:
                if self.action_counter >= self.MAX_ACTIONS:
                    break

                try:
                    action = GameAction.from_name(action_name)
                except (ValueError, KeyError):
                    continue

                grid_before = frame.frame[-1]
                score_before = frame.levels_completed

                # Pass VLM reasoning through to the game API so it appears in replay
                frame = self._step(action, reasoning={"vlm_reasoning": reasoning, "action": action_name, "step": self.action_counter})
                grid_after = frame.frame[-1]

                changes = self._count_changes(grid_before, grid_after)
                self._log_move(action_name, changes, frame.levels_completed, frame.state.name)

                score_changed = frame.levels_completed > self.highest_score
                if score_changed:
                    self.highest_score = frame.levels_completed
                    self.game_knowledge += f"\n\nSCORE to {frame.levels_completed} after {action_name}!"
                    logger.info(f"[play] *** SCORE UP: {frame.levels_completed} ***")
                    break  # re-plan

                logger.info(
                    f"[play] #{self.action_counter} {action_name}: {changes}px score={frame.levels_completed}"
                )

                if frame.state in (GameState.WIN, GameState.GAME_OVER):
                    break

        return frame

    def _get_actions(self, frame: FrameData, count: int) -> tuple[list[str], str]:
        """Ask VLM for next N actions based on current frame."""
        content = [
            text_block(
                f"Your understanding of this game:\n{self._compact_knowledge()}\n\n"
                f"Score: {frame.levels_completed} | Deaths: {self.total_deaths} | "
                f"Actions: {self.action_counter}/{self.MAX_ACTIONS}\n\n"
                f"Recent moves:\n{self._recent_moves(10)}\n"
            ),
            text_block("Current frame:"),
            image_block(grid_to_b64(frame.frame[-1])),
            text_block(
                f"\nChoose your next {count} actions. Respond with JSON:\n"
                '{\"actions\": [\"ACTION1\", \"ACTION3\", ...], \"reasoning\": \"what you see and why\"}\n\n'
                "Look at the image carefully. Where are things? What should you do next?"
            ),
        ]

        response = self._vlm_call(content, "actions")
        self._log_reasoning(f"GET_ACTIONS (step {self.action_counter})", response)
        return self._parse_actions(response, count)

    def _parse_actions(self, response: str, count: int) -> tuple[list[str], str]:
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                actions = data.get("actions", data.get("plan", []))
                reasoning = data.get("reasoning", "")
                if actions:
                    return actions[:count], reasoning
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback
        actions = []
        for word in response.split():
            word = word.strip('",[]')
            if word.startswith("ACTION") and word in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"):
                actions.append(word)
        if actions:
            return actions[:count], response[:100]

        return ["ACTION1", "ACTION2", "ACTION3"], "fallback"

    def _compact_knowledge(self) -> str:
        """Return game knowledge truncated to avoid token bloat."""
        if len(self.game_knowledge) > 3000:
            return self.game_knowledge[:1500] + "\n...(truncated)...\n" + self.game_knowledge[-1000:]
        return self.game_knowledge

    def _recent_moves(self, n: int) -> str:
        """Format recent moves for context."""
        recent = self.move_log[-n:]
        return "\n".join(
            f"  #{m['step']} {m['action']}: {m['changes']}px score={m['score']} state={m['state']}"
            for m in recent
        )

    def _log_move(self, action: str, changes: int, score: int, state: str) -> None:
        entry = {
            "step": self.action_counter,
            "action": action,
            "changes": changes,
            "score": score,
            "state": state,
        }
        self.move_log.append(entry)
        self.moves_log.write(json.dumps(entry) + "\n")
        self.moves_log.flush()

    def _log_reasoning(self, label: str, content: str) -> None:
        self.reasoning_log.write(f"\n{'='*60}\n")
        self.reasoning_log.write(f"[{label}] step={self.action_counter}\n")
        self.reasoning_log.write(f"{'='*60}\n")
        self.reasoning_log.write(content + "\n")
        self.reasoning_log.flush()

    def _count_changes(self, before: list[list[int]], after: list[list[int]]) -> int:
        return sum(1 for r in range(len(before)) for c in range(len(before[0]))
                   if before[r][c] != after[r][c])

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

    def _step(self, action: GameAction, reasoning: dict | str = "") -> FrameData:
        raw = self.env.step(action, reasoning=reasoning)
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
        self.reasoning_log.close()
        self.moves_log.close()
        if not self.scorecard_id:
            return
        scorecard = self.arcade.close_scorecard(self.scorecard_id)
        if scorecard:
            logger.info("--- SCORECARD ---")
            logger.info(json.dumps(scorecard.model_dump(), indent=2))
        self.scorecard_id = None
