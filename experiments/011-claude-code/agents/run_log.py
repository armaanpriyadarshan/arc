"""Structured run log for the v5 agent.

Maintains a plain-text, section-marked log of every turn that the LLM can
query via read_log / grep_log tool calls.  Also flushes to disk at end of run.
"""

import re

__all__ = ["RunLog", "RESPONSES_TOOLS", "CHAT_TOOLS"]

# ---------------------------------------------------------------------------
# Tool schemas — two formats for the two OpenAI API surfaces
# ---------------------------------------------------------------------------

RESPONSES_TOOLS = [
    {
        "type": "function",
        "name": "read_log",
        "description": (
            "Read lines from your turn-by-turn game log. "
            "Returns the specified range of lines so you can review past turns."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "offset": {
                    "type": "integer",
                    "description": "Starting line number (1-based). Default: 1",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of lines to return. Default: 100, max: 200",
                },
            },
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "grep_log",
        "description": (
            "Search your game log for lines matching a text or regex pattern. "
            "Returns matching lines with surrounding context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Text or regex pattern to search for",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Lines of context around each match. Default: 2",
                },
            },
            "required": ["pattern"],
        },
        "strict": False,
    },
]

CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_log",
            "description": (
                "Read lines from your turn-by-turn game log. "
                "Returns the specified range of lines so you can review past turns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "offset": {
                        "type": "integer",
                        "description": "Starting line number (1-based, default 1)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Lines to return (default 100, max 200)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_log",
            "description": (
                "Search your game log for lines matching a text or regex pattern. "
                "Returns matching lines with surrounding context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text or regex to search for",
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Context lines around each match (default 2)",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# RunLog
# ---------------------------------------------------------------------------

_READ_MAX_CHARS = 4000
_GREP_MAX_CHARS = 3000


class RunLog:
    """Structured, LLM-readable log of the agent's own run."""

    def __init__(self) -> None:
        self._lines: list[str] = []

    # -- writers -------------------------------------------------------------

    def write_header(self, game_id: str, max_actions: int) -> None:
        self._lines.append(f"=== GAME: {game_id} | MAX_ACTIONS={max_actions} ===")
        self._lines.append("")

    def write_probe(self, probe_facts: str) -> None:
        self._lines.append("=== PROBE RESULTS ===")
        for line in probe_facts.strip().splitlines():
            self._lines.append(f"  {line}")
        self._lines.append("=== END PROBE ===")
        self._lines.append("")

    def write_event(self, event_type: str, details: str) -> None:
        self._lines.append(f"*** {event_type}: {details} ***")
        self._lines.append("")

    def write_turn(
        self,
        turn: int,
        action_count: int,
        score: int,
        mode: str,
        goal: str,
        plan_step: str,
        observation: str,
        hypotheses: str,
        actions_taken: list[str],
        action_results: list[str],
        interactions: list,
        notes: str,
    ) -> None:
        self._lines.append(
            f"=== TURN {turn} | ACTIONS={action_count} | SCORE={score} | MODE={mode} ==="
        )
        self._lines.append(f"GOAL: {goal or '(none)'}")
        if plan_step:
            self._lines.append(f"STEP: {plan_step}")
        if observation:
            self._lines.append(f"OBSERVATION: {observation}")
        if hypotheses:
            self._lines.append("HYPOTHESES:")
            for line in hypotheses.strip().splitlines():
                self._lines.append(f"  {line}")
        if actions_taken:
            self._lines.append(f"ACTIONS: {actions_taken}")
        if action_results:
            self._lines.append("RESULTS:")
            for r in action_results:
                self._lines.append(f"  {r}")
        if interactions:
            self._lines.append("INTERACTIONS:")
            for entry in interactions:
                if isinstance(entry, dict):
                    act = entry.get("action", "?")
                    obs = entry.get("observed", "?")
                    self._lines.append(f"  {act} -> {obs}")
        if notes:
            self._lines.append(f"NOTES: {notes}")
        self._lines.append(f"=== END TURN {turn} ===")
        self._lines.append("")

    # -- readers (called by tool dispatch) -----------------------------------

    def line_count(self) -> int:
        return len(self._lines)

    def read_lines(self, offset: int = 1, limit: int = 100) -> str:
        """Return lines [offset .. offset+limit-1] (1-based). Capped output."""
        limit = max(1, min(limit, 200))
        offset = max(1, offset)
        start = offset - 1  # convert to 0-based
        end = start + limit
        selected = self._lines[start:end]
        if not selected:
            return f"(no lines at offset {offset}, log has {len(self._lines)} lines)"

        out_lines: list[str] = []
        total = 0
        for i, line in enumerate(selected):
            numbered = f"{start + i + 1}: {line}"
            total += len(numbered) + 1
            if total > _READ_MAX_CHARS:
                out_lines.append(
                    f"... (truncated at {_READ_MAX_CHARS} chars, "
                    f"use offset={start + i + 1} to continue)"
                )
                break
            out_lines.append(numbered)
        return "\n".join(out_lines)

    def grep(self, pattern: str, context_lines: int = 2) -> str:
        """Search log lines for *pattern*. Returns matches with context."""
        context_lines = max(0, min(context_lines, 5))
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return f"(invalid regex: {pattern})"

        match_indices: set[int] = set()
        for i, line in enumerate(self._lines):
            if regex.search(line):
                for j in range(
                    max(0, i - context_lines), min(len(self._lines), i + context_lines + 1)
                ):
                    match_indices.add(j)

        if not match_indices:
            return f"(no matches for '{pattern}')"

        out_lines: list[str] = []
        total = 0
        prev = -2
        for idx in sorted(match_indices):
            if idx > prev + 1:
                out_lines.append("---")
            numbered = f"{idx + 1}: {self._lines[idx]}"
            total += len(numbered) + 1
            if total > _GREP_MAX_CHARS:
                out_lines.append(
                    f"... (truncated at {_GREP_MAX_CHARS} chars, "
                    f"narrow your pattern for more specific results)"
                )
                break
            out_lines.append(numbered)
            prev = idx
        return "\n".join(out_lines)

    # -- persistence ---------------------------------------------------------

    def flush_to_disk(self, path: str) -> None:
        with open(path, "w") as f:
            f.write("\n".join(self._lines))
            f.write("\n")
