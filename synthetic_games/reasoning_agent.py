"""Reasoning LLM agent adapter for play.py.

Ports the ReasoningLLM logic from ARC-AGI-3-Agents into play.py's agent
interface so you can run it against local synthetic games.

Usage:
    uv run python play.py --game wiring --agent reasoning_agent.py:ReasoningAgent
    uv run python play.py --game wiring --agent reasoning_agent.py:ReasoningAgent --seed 0 -v

Requires OPENAI_API_KEY in environment or ../.env
"""

import json
import logging
import os
import textwrap
import time
from typing import Any

import openai
from openai import OpenAI as OpenAIClient

logger = logging.getLogger(__name__)

# Load .env from project root (one level up from synthetic_games/)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ENV_PATH = os.path.join(_HERE, "..", ".env")
if os.path.isfile(_ENV_PATH):
    with open(_ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip("\"'"))


# ---------------------------------------------------------------------------
# Action name <-> int mapping
# ---------------------------------------------------------------------------

ACTION_NAME_TO_ID = {
    "RESET": 0,
    "ACTION1": 1,
    "ACTION2": 2,
    "ACTION3": 3,
    "ACTION4": 4,
    "ACTION5": 5,
    "ACTION6": 6,
    "ACTION7": 7,
}


# ---------------------------------------------------------------------------
# ReasoningAgent — play.py-compatible adapter
# ---------------------------------------------------------------------------

class ReasoningAgent:
    """Reasoning LLM agent (o4-mini) adapted for play.py's agent interface.

    Ports the core loop from ARC-AGI-3-Agents/agents/templates/llm_agents.py
    (ReasoningLLM) into the simpler play.py contract:
        __init__(game, available_actions, config)
        choose_action(frame_data) -> int | (6, x, y)
        is_done(frame_data) -> bool
    """

    MODEL = "gpt-5.2"
    MESSAGE_LIMIT = 10

    def __init__(self, game, available_actions, config):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Add it to your environment or .env file."
            )
        self._client = OpenAIClient(api_key=api_key)
        self._available_actions = available_actions
        self._game_id = getattr(game, '_game_id', 'unknown')
        self._messages: list[dict[str, Any]] = []
        self._latest_tool_call_id = "call_init"
        self._first_call = True
        self._token_count = 0
        self._turn = 0

        # Suppress noisy HTTP logs from openai/httpx
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

    # ------------------------------------------------------------------
    # play.py interface
    # ------------------------------------------------------------------

    def choose_action(self, frame_data) -> int | tuple[int, int, int]:
        """Pick the next action. Returns int or (6, x, y) for clicks."""

        if self._first_call:
            self._first_call = False
            return self._initial_action(frame_data)

        return self._standard_action(frame_data)

    def is_done(self, frame_data) -> bool:
        state = frame_data.state.name if hasattr(frame_data.state, "name") else str(frame_data.state)
        return state == "WIN"

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def _initial_action(self, frame_data):
        """First turn: set up conversation context, then pick an action."""
        # System-level user prompt
        user_prompt = self._build_user_prompt()
        self._push({"role": "user", "content": user_prompt})

        # Fake assistant RESET (mirrors the original LLM agent bootstrap)
        self._push({
            "role": "assistant",
            "tool_calls": [{
                "id": self._latest_tool_call_id,
                "type": "function",
                "function": {"name": "RESET", "arguments": "{}"},
            }],
        })

        # Feed the frame we got back from the initial reset
        self._push({
            "role": "tool",
            "tool_call_id": self._latest_tool_call_id,
            "content": self._build_frame_prompt(frame_data),
        })

        # Observation step
        self._do_observation()

        # Action step
        return self._do_action(frame_data)

    def _standard_action(self, frame_data):
        """Subsequent turns: feed frame, observe, pick action."""
        # Feed the frame from the last action
        self._push({
            "role": "tool",
            "tool_call_id": self._latest_tool_call_id,
            "content": self._build_frame_prompt(frame_data),
        })

        # Observation step
        self._do_observation()

        # Action step
        return self._do_action(frame_data)

    def _call_api(self, **kwargs):
        """Call OpenAI API with rate-limit retry + exponential backoff."""
        max_retries = 5
        wait = 30  # initial wait seconds
        for attempt in range(max_retries):
            try:
                return self._client.chat.completions.create(**kwargs)
            except openai.RateLimitError:
                if attempt == max_retries - 1:
                    raise
                print(f"\n  [rate limited — waiting {wait}s before retry ({attempt+1}/{max_retries})]")
                time.sleep(wait)
                wait = min(wait * 2, 120)  # cap at 2 minutes

    def _do_observation(self):
        """Send messages to o4-mini for a strategy observation (no tool call)."""
        response = self._call_api(
            model=self.MODEL,
            messages=self._messages,
        )
        content = response.choices[0].message.content or ""
        reasoning_tokens = self._get_reasoning_tokens(response)
        self._track_tokens(response)
        self._push({"role": "assistant", "content": content})

        # Print reasoning trace to terminal
        self._turn += 1
        print(f"\n{'='*60}")
        print(f"  TURN {self._turn} — OBSERVATION")
        print(f"{'='*60}")
        if reasoning_tokens:
            print(f"  [reasoning tokens: {reasoning_tokens}]")
        print(content)

    def _do_action(self, frame_data):
        """Send messages to o4-mini to pick a tool-call action."""
        # Add a turn prompt
        self._push({"role": "user", "content": self._build_user_prompt()})

        response = self._call_api(
            model=self.MODEL,
            messages=self._messages,
            tools=self._build_tools(),
            tool_choice="required",
        )
        reasoning_tokens = self._get_reasoning_tokens(response)
        self._track_tokens(response)

        msg = response.choices[0].message
        tool_call = msg.tool_calls[0]
        self._latest_tool_call_id = tool_call.id

        # Push the assistant message (including extra tool_calls handling)
        self._push(msg)
        for tc in msg.tool_calls[1:]:
            self._push({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": "Error: only one action per turn.",
            })

        name = tool_call.function.name
        action_id = ACTION_NAME_TO_ID.get(name, 5)

        # Print action trace to terminal
        print(f"\n  ACTION: {name}", end="")
        if action_id == 6:
            try:
                data = json.loads(tool_call.function.arguments or "{}")
                print(f" (click x={data.get('x')}, y={data.get('y')})", end="")
            except json.JSONDecodeError:
                pass
        if reasoning_tokens:
            print(f"  [reasoning tokens: {reasoning_tokens}]", end="")
        print(f"  [total tokens: {self._token_count}]")
        print(f"{'─'*60}")

        if action_id == 6:
            try:
                data = json.loads(tool_call.function.arguments or "{}")
                x = int(data.get("x", 0))
                y = int(data.get("y", 0))
            except (json.JSONDecodeError, ValueError):
                x, y = 0, 0
            return (6, x, y)

        return action_id

    # ------------------------------------------------------------------
    # Prompt builders (ported from llm_agents.py)
    # ------------------------------------------------------------------

    def _build_user_prompt(self) -> str:
        action_names = [f"ACTION{a}" for a in self._available_actions]
        return textwrap.dedent(f"""\
            # CONTEXT:
            You are an agent playing a dynamic game called "{self._game_id}".
            Your objective is to WIN and avoid GAME_OVER while minimizing actions.

            One action produces one Frame. One Frame is made of one or more sequential
            Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
            INT<0,15> values.

            The grid has two regions:
            - PLAYFIELD (upper portion): the interactive game area. Only clicks here have any effect.
            - HUD (bottom rows, typically rows 60-63): read-only status display. Clicks here do nothing.

            When using ACTION6 (click), only click within the playfield area. The HUD is for
            information only — do not click on it.

            Available actions: {', '.join(action_names)}
            You must ONLY use these actions. No other actions are valid.

            # TURN:
            Call exactly one action.""")

    def _build_frame_prompt(self, frame_data) -> str:
        state = frame_data.state.name if hasattr(frame_data.state, "name") else str(frame_data.state)
        grid_str = self._pretty_print_grid(frame_data.frame)
        return textwrap.dedent(f"""\
            # State:
            {state}

            # Score:
            {frame_data.levels_completed}

            # Frame:
            {grid_str}

            # TURN:
            Reply with a few sentences of plain-text strategy observation about the frame to inform your next action.""")

    def _pretty_print_grid(self, frame) -> str:
        lines = []
        if isinstance(frame, list):
            for i, block in enumerate(frame):
                lines.append(f"Grid {i}:")
                if isinstance(block, list):
                    for row in block:
                        lines.append(f"  {row}")
                lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool definitions (ported from llm_agents.py)
    # ------------------------------------------------------------------

    def _build_tools(self) -> list[dict[str, Any]]:
        empty_params: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
        all_functions = {
            0: {
                "name": "RESET",
                "description": "Start or restart a game. Must be called first when NOT_PLAYED or after GAME_OVER to play again.",
                "parameters": empty_params,
            },
            1: {
                "name": "ACTION1",
                "description": "Send this simple input action (1, W, Up).",
                "parameters": empty_params,
            },
            2: {
                "name": "ACTION2",
                "description": "Send this simple input action (2, S, Down).",
                "parameters": empty_params,
            },
            3: {
                "name": "ACTION3",
                "description": "Send this simple input action (3, A, Left).",
                "parameters": empty_params,
            },
            4: {
                "name": "ACTION4",
                "description": "Send this simple input action (4, D, Right).",
                "parameters": empty_params,
            },
            5: {
                "name": "ACTION5",
                "description": "Send this simple input action (5, Enter, Spacebar, Delete).",
                "parameters": empty_params,
            },
            6: {
                "name": "ACTION6",
                "description": "Click a cell in the playfield. Only clicks on the interactive game area have any effect — the bottom HUD rows (60-63) are read-only status display.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "string",
                            "description": "Coordinate X which must be Int<0,63>",
                        },
                        "y": {
                            "type": "string",
                            "description": "Coordinate Y which must be Int<0,63>",
                        },
                    },
                    "required": ["x", "y"],
                    "additionalProperties": False,
                },
            },
            7: {
                "name": "ACTION7",
                "description": "Send this simple input action (7, Z, Undo).",
                "parameters": empty_params,
            },
        }
        # Only include actions the game actually supports
        functions = [all_functions[a] for a in self._available_actions if a in all_functions]
        return [
            {"type": "function", "function": {**f, "strict": True}}
            for f in functions
        ]

    # ------------------------------------------------------------------
    # Message management (ported from llm_agents.py)
    # ------------------------------------------------------------------

    def _push(self, message):
        """Append message with FIFO windowing (keeps last MESSAGE_LIMIT)."""
        self._messages.append(message)
        if len(self._messages) > self.MESSAGE_LIMIT:
            self._messages = self._messages[-self.MESSAGE_LIMIT:]
            # Don't let the window start on a tool response
            while self._messages and (
                self._messages[0].get("role") if isinstance(self._messages[0], dict)
                else getattr(self._messages[0], "role", None)
            ) == "tool":
                self._messages.pop(0)

    def _track_tokens(self, response):
        if hasattr(response, "usage") and response.usage:
            tokens = response.usage.total_tokens
            self._token_count += tokens

    def _get_reasoning_tokens(self, response) -> int | None:
        """Extract reasoning token count from o4-mini response."""
        try:
            return response.usage.completion_tokens_details.reasoning_tokens
        except (AttributeError, TypeError):
            return None
