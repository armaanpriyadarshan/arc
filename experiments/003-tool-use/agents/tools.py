"""Grid inspection tools — the model calls these to investigate the game.

These are FREE (no game actions consumed). The model decides what to query.
Tools return structured text the model can reason about.
"""

import hashlib
import json
from typing import Any

Grid = list[list[int]]


class GridTools:
    """Stateful toolkit that holds the current and previous grids + diff history."""

    def __init__(self) -> None:
        self.current_grid: Grid | None = None
        self.previous_grid: Grid | None = None
        self.grids_seen: list[Grid] = []  # bounded
        self.action_history: list[dict] = []  # bounded
        self._change_counts: list[list[int]] | None = None

    def update(self, grid: Grid, action: str, changes: int, blocked: bool) -> None:
        """Called after every game action."""
        self.previous_grid = self.current_grid
        self.current_grid = grid
        self.grids_seen.append(grid)
        if len(self.grids_seen) > 20:
            self.grids_seen = self.grids_seen[-20:]

        self.action_history.append({
            "action": action, "changes": changes, "blocked": blocked,
        })
        if len(self.action_history) > 30:
            self.action_history = self.action_history[-30:]

        # Update change counts
        if self._change_counts is None:
            self._change_counts = [[0] * 64 for _ in range(64)]
        if self.previous_grid:
            for r in range(64):
                for c in range(64):
                    if self.previous_grid[r][c] != grid[r][c]:
                        self._change_counts[r][c] += 1

    # ── Tools the model can call ─────────────────────────────────

    def get_cell(self, row: int, col: int) -> str:
        """Get the color value at a specific cell."""
        if not self.current_grid:
            return "No grid loaded."
        if 0 <= row < 64 and 0 <= col < 64:
            return f"Cell ({row},{col}) = {self.current_grid[row][col]}"
        return f"Out of bounds: ({row},{col})"

    def get_row(self, row: int) -> str:
        """Get all values in a row, run-length encoded."""
        if not self.current_grid or row < 0 or row >= 64:
            return f"Invalid row: {row}"
        vals = self.current_grid[row]
        runs = []
        current = vals[0]
        count = 1
        for c in range(1, 64):
            if vals[c] == current:
                count += 1
            else:
                runs.append(f"{current}x{count}")
                current = vals[c]
                count = 1
        runs.append(f"{current}x{count}")
        return f"Row {row}: {' '.join(runs)}"

    def get_region(self, r1: int, c1: int, r2: int, c2: int) -> str:
        """Get a rectangular region of the grid."""
        if not self.current_grid:
            return "No grid loaded."
        r1, r2 = max(0, r1), min(63, r2)
        c1, c2 = max(0, c1), min(63, c2)
        lines = [f"Region ({r1},{c1})-({r2},{c2}):"]
        for r in range(r1, r2 + 1):
            lines.append(f"  r{r}: {self.current_grid[r][c1:c2+1]}")
        return "\n".join(lines)

    def diff_last_action(self) -> str:
        """Show what changed in the last action."""
        if not self.current_grid or not self.previous_grid:
            return "No previous grid to diff against."
        changes = []
        for r in range(64):
            for c in range(64):
                if self.previous_grid[r][c] != self.current_grid[r][c]:
                    changes.append((r, c, self.previous_grid[r][c], self.current_grid[r][c]))
        if not changes:
            return "No changes."

        # Group by transition
        transitions: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for r, c, old, new in changes:
            transitions.setdefault((old, new), []).append((r, c))

        lines = [f"{len(changes)} cells changed:"]
        for (old, new), cells in sorted(transitions.items(), key=lambda x: -len(x[1])):
            coords = cells[:10]
            coord_str = ", ".join(f"({r},{c})" for r, c in coords)
            more = f" +{len(cells) - 10} more" if len(cells) > 10 else ""
            lines.append(f"  {old}→{new} at: {coord_str}{more}")
        return "\n".join(lines)

    def find_color(self, color: int) -> str:
        """Find all cells with a specific color value."""
        if not self.current_grid:
            return "No grid loaded."
        cells = []
        for r in range(64):
            for c in range(64):
                if self.current_grid[r][c] == color:
                    cells.append((r, c))
        if not cells:
            return f"Color {color} not found."
        # Bounding box
        rows = [p[0] for p in cells]
        cols = [p[1] for p in cells]
        return (f"Color {color}: {len(cells)} cells, "
                f"bbox=({min(rows)},{min(cols)})-({max(rows)},{max(cols)}), "
                f"center=({sum(rows)//len(cells)},{sum(cols)//len(cells)})")

    def color_summary(self) -> str:
        """Count how many cells of each color exist."""
        if not self.current_grid:
            return "No grid loaded."
        counts: dict[int, int] = {}
        for r in range(64):
            for c in range(64):
                v = self.current_grid[r][c]
                counts[v] = counts.get(v, 0) + 1
        lines = ["Color distribution:"]
        for color, count in sorted(counts.items(), key=lambda x: -x[1]):
            pct = count * 100 // 4096
            lines.append(f"  color {color}: {count} cells ({pct}%)")
        return "\n".join(lines)

    def hotspots(self, min_changes: int = 3) -> str:
        """Cells that changed many times — likely interactive or part of a moving entity."""
        if not self._change_counts:
            return "No change data yet."
        spots = []
        for r in range(64):
            for c in range(64):
                if self._change_counts[r][c] >= min_changes:
                    spots.append((r, c, self._change_counts[r][c]))
        spots.sort(key=lambda x: -x[2])
        if not spots:
            return f"No cells changed {min_changes}+ times yet."
        lines = [f"Hotspots (changed {min_changes}+ times):"]
        for r, c, count in spots[:15]:
            lines.append(f"  ({r},{c}): {count} changes, current={self.current_grid[r][c]}")
        return "\n".join(lines)

    def recent_actions(self, n: int = 10) -> str:
        """Show the last N actions and their results."""
        recent = self.action_history[-n:]
        if not recent:
            return "No actions taken yet."
        lines = ["Recent actions:"]
        for i, a in enumerate(recent):
            status = "BLOCKED" if a["blocked"] else f"{a['changes']}ch"
            lines.append(f"  {a['action']}: {status}")
        return "\n".join(lines)

    def frame_signature(self) -> str:
        """Hash of the current grid — detect repeated states."""
        if not self.current_grid:
            return "No grid loaded."
        flat = bytes(cell for row in self.current_grid for cell in row)
        sig = hashlib.md5(flat).hexdigest()[:8]
        count = sum(1 for g in self.grids_seen
                    if hashlib.md5(bytes(cell for row in g for cell in row)).hexdigest()[:8] == sig)
        return f"Frame signature: {sig} (seen {count} time(s))"

    # ── Tool dispatch ────────────────────────────────────────────

    TOOL_DEFINITIONS = [
        {
            "type": "function",
            "function": {
                "name": "get_cell",
                "description": "Get the color value at a specific grid cell",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "row": {"type": "integer", "description": "Row (0-63)"},
                        "col": {"type": "integer", "description": "Column (0-63)"},
                    },
                    "required": ["row", "col"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_row",
                "description": "Get all values in a row, run-length encoded",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "row": {"type": "integer", "description": "Row (0-63)"},
                    },
                    "required": ["row"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_region",
                "description": "Get a rectangular region of the grid",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "r1": {"type": "integer"}, "c1": {"type": "integer"},
                        "r2": {"type": "integer"}, "c2": {"type": "integer"},
                    },
                    "required": ["r1", "c1", "r2", "c2"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "diff_last_action",
                "description": "Show exactly what cells changed in the last game action",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_color",
                "description": "Find all cells with a specific color value, returns count and bounding box",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "color": {"type": "integer", "description": "Color value (0-15)"},
                    },
                    "required": ["color"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "color_summary",
                "description": "Count how many cells of each color exist in the current grid",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "hotspots",
                "description": "Find cells that changed many times (likely interactive or moving)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "min_changes": {"type": "integer", "description": "Minimum change count (default 3)"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recent_actions",
                "description": "Show the last N actions and their results",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer", "description": "Number of recent actions (default 10)"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "take_action",
                "description": "Execute a game action. This costs one action from your budget.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
                            "description": "ACTION1=up, ACTION2=down, ACTION3=left, ACTION4=right",
                        },
                    },
                    "required": ["action"],
                },
            },
        },
    ]

    def dispatch(self, name: str, args: dict) -> str:
        """Call a tool by name with arguments."""
        if name == "get_cell":
            return self.get_cell(args["row"], args["col"])
        elif name == "get_row":
            return self.get_row(args["row"])
        elif name == "get_region":
            return self.get_region(args["r1"], args["c1"], args["r2"], args["c2"])
        elif name == "diff_last_action":
            return self.diff_last_action()
        elif name == "find_color":
            return self.find_color(args["color"])
        elif name == "color_summary":
            return self.color_summary()
        elif name == "hotspots":
            return self.hotspots(args.get("min_changes", 3))
        elif name == "recent_actions":
            return self.recent_actions(args.get("n", 10))
        elif name == "take_action":
            return "__TAKE_ACTION__"  # sentinel — handled by the agent
        else:
            return f"Unknown tool: {name}"
