"""Structured run log + board log for the LLM-vision agent.

Maintains a plain-text, section-marked log of every turn that the LLM can
query via read_log / grep_log tool calls.  Also maintains a text-based board
log that the LLM can query via read_board / grep_board tool calls.
"""

import re

__all__ = ["RunLog", "BoardLog", "RESPONSES_TOOLS", "CHAT_TOOLS"]

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
    {
        "type": "function",
        "name": "read_board",
        "description": (
            "Read specific rows from the CURRENT game board. The board is a 64x64 grid "
            "where each cell is a hex digit (0-F) representing a color. Returns the color "
            "legend, column headers, and the requested rows. Use this to verify exact cell "
            "positions, count objects, or inspect specific regions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "row_start": {
                    "type": "integer",
                    "description": "First row to return (0-63). Default: 0",
                },
                "row_end": {
                    "type": "integer",
                    "description": "Last row to return (0-63). Default: 63",
                },
            },
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "grep_board",
        "description": (
            "Search the board state HISTORY for a pattern. The board is logged after "
            "every action, so you can compare states over time. Example: grep_board('action #5') "
            "to see the board after action 5. grep_board('E') to find rows with green cells."
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
    {
        "type": "function",
        "function": {
            "name": "read_board",
            "description": (
                "Read specific rows from the CURRENT game board. The board is a 64x64 "
                "grid where each cell is a hex digit (0-F) representing a color."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "row_start": {
                        "type": "integer",
                        "description": "First row to return (0-63, default 0)",
                    },
                    "row_end": {
                        "type": "integer",
                        "description": "Last row to return (0-63, default 63)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_board",
            "description": (
                "Search the board state HISTORY for a pattern. Board is logged after "
                "every action. Use to compare states over time or find specific colors."
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


# ---------------------------------------------------------------------------
# BoardLog — text-based board state log, queryable by LLM via tools
# ---------------------------------------------------------------------------

_BOARD_READ_MAX_CHARS = 6000  # boards are large; allow more chars
_BOARD_GREP_MAX_CHARS = 4000

COLOR_NAMES = {
    0: "white", 1: "off-white", 2: "light-gray", 3: "gray",
    4: "dark-gray", 5: "black", 6: "magenta", 7: "pink",
    8: "red", 9: "blue", 10: "light-blue", 11: "yellow",
    12: "orange", 13: "maroon", 14: "green", 15: "purple",
}


class BoardLog:
    """Text-based board state log, queryable by the LLM via tools.

    After every action, the agent calls ``update()`` with the current grid.
    The LLM can then use ``get_rows()`` to read the current board, or
    ``grep()`` to search the board history.
    """

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._current_grid: list[list[int]] | None = None
        self._current_legend: str = ""
        self._current_col_header: str = ""
        self._current_rows: list[str] = []  # cached formatted rows for get_rows

    def update(self, grid: list[list[int]], action_num: int) -> None:
        """Write the current board state to the log."""
        self._current_grid = grid
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0

        # Color legend (only colors present)
        colors_present = sorted(set(cell for row in grid for cell in row))
        self._current_legend = "Colors: " + " ".join(
            f"{c:X}={COLOR_NAMES.get(c, '?')}" for c in colors_present
        )

        # Column header — tens digit on first line, ones digit on second
        tens = "     " + "".join(f"{c // 10}" for c in range(w))
        ones = "     " + "".join(f"{c % 10}" for c in range(w))
        self._current_col_header = tens + "\n" + ones

        # Build row strings
        self._current_rows = []
        for r in range(h):
            row_str = "".join(f"{grid[r][c]:X}" for c in range(w))
            self._current_rows.append(f" {r:02d}: {row_str}")

        # Append to history
        self._lines.append(f"=== BOARD (after action #{action_num}) ===")
        self._lines.append(self._current_legend)
        self._lines.append(tens)
        self._lines.append(ones)
        for row_line in self._current_rows:
            self._lines.append(row_line)
        self._lines.append(f"=== END BOARD ===")
        self._lines.append("")

    def get_rows(self, row_start: int = 0, row_end: int = 63) -> str:
        """Return specific rows of the current board with legend and column headers."""
        if not self._current_rows:
            return "(no board state recorded yet)"

        row_start = max(0, row_start)
        row_end = min(len(self._current_rows) - 1, row_end)

        out_lines = [
            self._current_legend,
            self._current_col_header,
        ]
        for r in range(row_start, row_end + 1):
            out_lines.append(self._current_rows[r])

        result = "\n".join(out_lines)
        if len(result) > _BOARD_READ_MAX_CHARS:
            result = result[:_BOARD_READ_MAX_CHARS] + "\n... (truncated)"
        return result

    def grep(self, pattern: str, context_lines: int = 2) -> str:
        """Search the board history for *pattern*. Returns matches with context."""
        context_lines = max(0, min(context_lines, 5))
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return f"(invalid regex: {pattern})"

        match_indices: set[int] = set()
        for i, line in enumerate(self._lines):
            if regex.search(line):
                for j in range(
                    max(0, i - context_lines),
                    min(len(self._lines), i + context_lines + 1),
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
            if total > _BOARD_GREP_MAX_CHARS:
                out_lines.append(
                    f"... (truncated at {_BOARD_GREP_MAX_CHARS} chars, "
                    f"narrow your pattern for more specific results)"
                )
                break
            out_lines.append(numbered)
            prev = idx
        return "\n".join(out_lines)

    def flush_to_disk(self, path: str) -> None:
        with open(path, "w") as f:
            f.write("\n".join(self._lines))
            f.write("\n")
