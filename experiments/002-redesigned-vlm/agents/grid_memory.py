"""Layer 1: Grid Memory — 64×64 array of cell metadata. Zero LLM calls.

Tracks per-cell: change count, color history, which action last changed it.
Derives: static mask, activity heatmap, movement corridors, interaction hotspots.
"""

from dataclasses import dataclass, field

Grid = list[list[int]]


@dataclass
class CellMeta:
    change_count: int = 0
    colors_seen: set[int] = field(default_factory=set)
    last_action: str = ""


class GridMemory:
    def __init__(self, rows: int = 64, cols: int = 64) -> None:
        self.rows = rows
        self.cols = cols
        self.cells = [[CellMeta() for _ in range(cols)] for _ in range(rows)]
        self.total_updates = 0

    def update(self, action: str, before: Grid, after: Grid) -> None:
        """Update cell metadata from a frame diff."""
        self.total_updates += 1
        for r in range(self.rows):
            for c in range(self.cols):
                if before[r][c] != after[r][c]:
                    cell = self.cells[r][c]
                    cell.change_count += 1
                    cell.colors_seen.add(before[r][c])
                    cell.colors_seen.add(after[r][c])
                    cell.last_action = action

    def record_initial(self, grid: Grid) -> None:
        """Record colors from the initial frame."""
        for r in range(self.rows):
            for c in range(self.cols):
                self.cells[r][c].colors_seen.add(grid[r][c])

    def static_cell_count(self) -> int:
        return sum(1 for r in range(self.rows) for c in range(self.cols)
                   if self.cells[r][c].change_count == 0)

    def hotspots(self, min_changes: int = 5) -> list[tuple[int, int, int]]:
        """Cells that changed many times — likely dynamic/interactive. Returns (row, col, count)."""
        spots = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.cells[r][c].change_count >= min_changes:
                    spots.append((r, c, self.cells[r][c].change_count))
        spots.sort(key=lambda x: -x[2])
        return spots[:20]

    def activity_by_row(self) -> list[int]:
        """Total changes per row — shows which rows are active."""
        return [sum(self.cells[r][c].change_count for c in range(self.cols))
                for r in range(self.rows)]

    def compact_text(self) -> str:
        """Summary for LLM consumption."""
        if self.total_updates == 0:
            return "GRID MEMORY: No data yet."

        static = self.static_cell_count()
        pct = static * 100 // (self.rows * self.cols)
        lines = [f"GRID MEMORY ({self.total_updates} frames observed):"]
        lines.append(f"  Static cells: {static}/{self.rows * self.cols} ({pct}%)")

        # Most active rows (likely HUD or movement corridors)
        row_activity = self.activity_by_row()
        active_rows = [(r, count) for r, count in enumerate(row_activity) if count > 0]
        active_rows.sort(key=lambda x: -x[1])
        if active_rows:
            lines.append("  Most active rows:")
            for r, count in active_rows[:8]:
                lines.append(f"    row {r}: {count} total changes")

        # Hotspots
        spots = self.hotspots()
        if spots:
            lines.append("  Hotspots (cells that changed many times):")
            for r, c, count in spots[:10]:
                colors = sorted(self.cells[r][c].colors_seen)
                lines.append(f"    ({r},{c}): {count} changes, colors={colors}")

        return "\n".join(lines)


def compress_grid(grid: Grid, grid_mem: 'GridMemory | None' = None, max_rows: int = 20) -> str:
    """Run-length encode only the most relevant rows."""

    if grid_mem and grid_mem.total_updates > 5:
        row_activity = grid_mem.activity_by_row()
        active_rows = [r for r, count in enumerate(row_activity) if count > 0]
        if active_rows:
            min_r = max(0, min(active_rows) - 3)
            max_r = min(63, max(active_rows) + 3)
            rows_to_show = range(min_r, max_r + 1)
        else:
            rows_to_show = range(64)
    else:
        rows_to_show = range(64)

    lines = []
    for r in rows_to_show:
        row = grid[r]
        runs = []
        current = row[0]
        count = 1
        for c in range(1, len(row)):
            if row[c] == current:
                count += 1
            else:
                runs.append(f"{current}x{count}")
                current = row[c]
                count = 1
        runs.append(f"{current}x{count}")
        if len(runs) > 1:
            lines.append(f"  r{r}: {' '.join(runs)}")
        if len(lines) >= max_rows:
            lines.append(f"  ... ({len(list(rows_to_show)) - max_rows} more rows)")
            break

    if not lines:
        return ""
    return "GRID (active area, run-length encoded):\n" + "\n".join(lines)
