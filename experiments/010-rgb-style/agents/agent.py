"""RGB-style agent: faithful port of RGB-Agent logic with GPT-5.4.

Text-only ASCII grid (no images), batch action planning with queue draining,
state-action memory, grid diffs, score-aware queue flushing.
"""

import json
import logging
import os
import time
from pathlib import Path

from arcengine import FrameData, GameAction, GameState as ArcGameState
from openai import OpenAI

from .game_state import GameState
from .action_queue import ActionQueue, QueueExhausted
from .grid_utils import format_grid_ascii
from .prompts import INITIAL_PROMPT, RESUME_PROMPT, ACTIONS_ADDENDUM, PYTHON_ADDENDUM

logger = logging.getLogger(__name__)

_RETRY_NUDGE = (
    "CRITICAL: Your previous response was missing the [ACTIONS] section. "
    "You MUST end your response with an [ACTIONS] section containing a JSON action plan. "
    "Do NOT write actions to a file — output them directly in your response text."
)


class RGBStyleAgent:
    MAX_ACTIONS = 200
    HIGH_REASONING_CALLS = 3  # first N analyzer calls use high effort
    PLAN_SIZE = 5
    ANALYZER_RETRIES = 5
    MESSAGE_WINDOW = 10

    def __init__(self, game_id: str) -> None:
        from arc_agi import Arcade
        self.game_id = game_id
        self.arcade = Arcade()
        self.scorecard_id = self.arcade.open_scorecard()
        self.env = self.arcade.make(game_id, scorecard_id=self.scorecard_id)
        self.frames: list[FrameData] = []

        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        self._state = GameState(game_id=game_id)
        self._queue = ActionQueue()
        self.messages: list[dict] = []
        self._is_first_analysis = True
        self.llm_calls = 0
        self.total_actions = 0
        self.total_deaths = 0

        # Prompt log
        self.log_path = Path("experiment.log")

    def run(self) -> None:
        timer = time.time()
        logger.info("=" * 60)
        logger.info(f"RGB-Style Agent on {self.game_id}")
        logger.info("=" * 60)

        self._state.reset()
        self._queue.reset()

        # Initial reset
        frame = self._env_step(GameAction.RESET)
        obs = self._format_observation(frame)
        arc_state = frame.state
        arc_score = frame.levels_completed

        self._state.record_env_update(observation=obs, reward=0.0, done=False)

        # Log initial board
        grid = self._state.render_board()
        if grid:
            self._write_log(
                f"\n{'='*80}\n"
                f"Action 0 | Level 1 | INITIAL STATE\n"
                f"Score: {arc_score} | State: {arc_state.name}\n"
                f"{'='*80}\n\n"
                f"[INITIAL BOARD STATE]\n{grid}\n\n"
            )

        level_num = 1
        max_score = 0

        # Main game loop
        while self.total_actions < self.MAX_ACTIONS:
            if arc_state == ArcGameState.WIN:
                logger.info("WIN!")
                break

            try:
                action_dict = self._next_action()
            except QueueExhausted:
                logger.info("queue exhausted at action %d — firing analyzer", self.total_actions)
                loaded = False
                for attempt in range(self.ANALYZER_RETRIES):
                    nudge = _RETRY_NUDGE if attempt > 0 or self.total_actions > 0 else ""
                    logger.info("analyzer attempt %d/%d action=%d nudge=%s",
                                attempt + 1, self.ANALYZER_RETRIES, self.total_actions, bool(nudge))
                    if self._fire_analyzer(self.total_actions, arc_score, retry_nudge=nudge):
                        loaded = True
                        break
                    logger.warning("analyzer attempt %d/%d failed", attempt + 1, self.ANALYZER_RETRIES)
                if not loaded:
                    logger.error("all analyzer attempts failed — ending run")
                    break
                try:
                    action_dict = self._next_action()
                except QueueExhausted:
                    logger.error("queue still empty after analyzer — ending run")
                    break

            action_result = self._state.record_action(action_dict)
            action = action_result.pop("action")
            reasoning = action_result.pop("reasoning", "")

            frame = self._env_step(action, reasoning=reasoning[:200])
            self.total_actions += 1

            prev_score = arc_score
            arc_state = frame.state
            arc_score = frame.levels_completed
            max_score = max(max_score, arc_score)

            obs = self._format_observation(frame)
            self._state.record_env_update(observation=obs, reward=0.0, done=False)
            self._queue.check_score(arc_score)

            # Log action
            changes = "unknown"
            plan_info = f" | Plan {self._queue.plan_index}/{self._queue.plan_total}" if self._queue.plan_total > 0 else ""
            logger.info(f"#{self.total_actions} {action_dict['name']}{plan_info} score={arc_score}")

            # Write post-action board to log
            board = self._state.render_board()
            if board:
                self._write_log(
                    f"\n{'='*80}\n"
                    f"Action {self.total_actions} | Level {level_num}{plan_info}\n"
                    f"Score: {arc_score} | State: {arc_state.name}\n"
                    f"{'='*80}\n\n"
                )
                # Log the observation/action phases from game state
                last_step = self._state.trajectory.steps[-1] if self._state.trajectory.steps else None
                if last_step and last_step.chat_completions:
                    for msg in last_step.chat_completions:
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')
                        self._write_log(f"[{role.upper()}]\n{content}\n\n")

                self._write_log(f"[POST-ACTION BOARD STATE]\nScore: {arc_score}\n{board}\n\n")

            # Level completed
            if arc_score > prev_score and arc_state not in (ArcGameState.WIN, ArcGameState.GAME_OVER):
                logger.info(f"[level-up] Level {arc_score}! (was {prev_score})")
                level_num += 1
                continue

            if arc_state == ArcGameState.GAME_OVER:
                self.total_deaths += 1
                logger.info(f"GAME_OVER #{self.total_deaths}")

        elapsed = round(time.time() - timer, 2)
        logger.info("=" * 60)
        logger.info(
            f"FINISHED: actions={self.total_actions} state={arc_state.name} "
            f"score={max_score} deaths={self.total_deaths} "
            f"llm_calls={self.llm_calls} time={elapsed}s"
        )
        logger.info("=" * 60)
        self._close()

    def _next_action(self) -> dict:
        """Get next action: auto-reset, queue drain, or raise QueueExhausted."""
        obs = self._state.last_observation or {}
        state = obs.get("state", "NOT_PLAYED")

        # Auto-reset on game over
        if state in ("NOT_PLAYED", "GAME_OVER") and self._state.last_executed_action != "RESET":
            return {"name": "RESET", "data": {}, "obs_text": "Game Over, starting new game.", "action_text": ""}

        grid_raw, grid_text = self._state.process_frame(obs)
        score = obs.get("score", 0)

        use_queued = bool(self._queue and not self._queue.score_changed)
        if not use_queued:
            self._queue.score_changed = False

        self._state.build_observation_context(
            grid_text, score, grid_raw, use_queued=use_queued, queue=self._queue,
        )

        if use_queued and self._queue:
            action = self._queue.pop()
            label = f"plan step {self._queue.plan_index}/{self._queue.plan_total}"
            action["obs_text"] = ""
            action["action_text"] = f"[queued {label}]"

            self._state._last_action_prompt = f"[Queued {label} — no model call]"
            self._state._last_action_response = (
                f"Tool Call: {action['name']}({json.dumps(action['data'])})\n"
                f"Content: Executing pre-planned action ({label})"
            )
            logger.info("queue drain -> %s (%s, %d remaining)",
                         action.get("name"), label, len(self._queue))
            return action

        logger.info("queue empty — need new plan from analyzer")
        raise QueueExhausted("Queue empty, no actions from analyzer")

    def _fire_analyzer(self, action_num: int, arc_score: int, retry_nudge: str = "") -> bool:
        """Call GPT-5.4 to analyze game state and produce an action plan."""
        # Write current board to log before analysis
        board = self._state.render_board()
        if board:
            self._write_log(f"[POST-ACTION BOARD STATE]\nScore: {arc_score}\n{board}\n\n")

        # Build the context
        obs = self._state.last_observation or {}
        grid_raw, grid_text = self._state.process_frame(obs)
        score = obs.get("score", 0)
        context = self._state.build_observation_context(
            grid_text, score, grid_raw, use_queued=False, queue=self._queue,
        )

        # Build system prompt
        if self._is_first_analysis:
            # For first analysis, use INITIAL_PROMPT adapted for direct context
            system = (
                "You are a strategic advisor for an AI agent playing a grid-based puzzle game.\n"
                "The agent's current game state is provided below.\n\n"
                "Most games have some form of timer mechanism. A score increase means a level was solved.\n\n"
                "Deeply analyze the game state to understand what the agent should do.\n\n"
                "Your response MUST contain ALL sections below — the agent cannot act without [ACTIONS]:\n"
                "1. A detailed strategic briefing (explain your reasoning, be specific with coordinates)\n"
                "2. Followed by exactly this separator and a 2-3 sentence action plan:\n\n"
                "[PLAN]\n"
                "<concise action plan the agent should follow until the next analysis>\n"
            )
            self._is_first_analysis = False
        else:
            system = (
                "The game state has been updated since your last analysis.\n\n"
                "Analyze the latest state and update your strategic briefing.\n"
                "Focus on what changed: new moves, score transitions, and whether the agent followed\n"
                "your previous plan or diverged.\n\n"
                "Your response MUST contain ALL three sections below — the agent cannot act without [ACTIONS]:\n"
                "1. A detailed strategic briefing (explain your reasoning, be specific with coordinates)\n"
                "2. Followed by exactly this separator and a 2-3 sentence action plan:\n\n"
                "[PLAN]\n"
                "<concise action plan the agent should follow until the next analysis>\n"
            )

        system += ACTIONS_ADDENDUM.format(plan_size=self.PLAN_SIZE)
        system += (
            "\n\nYou can reason about the grid programmatically. The grid is provided as ASCII text "
            "where each character represents a cell value 0-15. Analyze patterns, count objects, "
            "and identify spatial relationships from the character patterns.\n"
        )

        # Build user message
        user_msg = context
        if retry_nudge:
            user_msg += f"\n\n{retry_nudge}"

        # Append to conversation
        self.messages.append({"role": "user", "content": user_msg})

        # Trim to window
        trimmed = self.messages[-self.MESSAGE_WINDOW:]
        # Ensure starts with user
        while trimmed and trimmed[0].get("role") != "user":
            trimmed = trimmed[1:]

        self.llm_calls += 1
        try:
            response = self.client.responses.create(
                model="gpt-5.4",
                instructions=system,
                input=trimmed,
                reasoning={"effort": "high" if self.llm_calls <= self.HIGH_REASONING_CALLS else "low"},
                max_output_tokens=16000,
            )
            raw = response.output_text or ""
        except Exception as e:
            logger.warning(f"API error: {e}")
            time.sleep(2)
            return False

        # Append assistant response
        self.messages.append({"role": "assistant", "content": raw})

        if not raw:
            logger.warning("analyzer returned empty response at action %d", action_num)
            return False

        # Write analyzer exchange to log
        self._write_log(
            f"\n--- ANALYZER CALL (action {action_num}, llm_call {self.llm_calls}) ---\n"
            f"[SYSTEM PROMPT]\n{system[:500]}...\n\n"
            f"[OBSERVATION_PHASE]\n{user_msg[:500]}...\n\n"
            f"[ASSISTANT]\n{raw}\n\n"
        )

        # Parse response — same logic as RGB-Agent's GameRunner._fire_analyzer
        hint = "\n".join(line.rstrip() for line in raw.split("\n"))

        actions_text = None
        if "\n[ACTIONS]\n" in hint:
            hint, actions_text = hint.split("\n[ACTIONS]\n", 1)
            actions_text = actions_text.strip()

        if "\n[PLAN]\n" in hint:
            full_hint, plan = hint.split("\n[PLAN]\n", 1)
            full_hint, plan = full_hint.strip(), plan.strip()
        else:
            full_hint = plan = hint

        self._state.set_external_hint(full_hint)
        self._state.set_persistent_hint(plan)

        if actions_text:
            if self._queue.load(actions_text):
                logger.info("analyzer at action %d: loaded action plan (%d chars)", action_num, len(actions_text))
                return True
            logger.warning("analyzer at action %d: load rejected the plan", action_num)
            return False

        logger.warning("analyzer at action %d: hint received but NO [ACTIONS] section", action_num)
        return False

    def _write_log(self, text: str) -> None:
        """Append to the prompt log file."""
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(text)

    def _env_step(self, action: GameAction, reasoning: str = "") -> FrameData:
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

    def _format_observation(self, frame: FrameData) -> dict:
        """Format frame into observation dict matching RGB-Agent's format."""
        return {
            "game_id": getattr(frame, "game_id", self.game_id),
            "state": frame.state.name if frame.state else "NOT_PLAYED",
            "score": frame.levels_completed,
            "frame": frame.frame,
            "available_actions": frame.available_actions,
            "guid": getattr(frame, "guid", None),
        }

    def _close(self) -> None:
        if not self.scorecard_id:
            return
        scorecard = self.arcade.close_scorecard(self.scorecard_id)
        if scorecard:
            logger.info("--- SCORECARD ---")
            logger.info(json.dumps(scorecard.model_dump(), indent=2))
        self.scorecard_id = None
