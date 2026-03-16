"""Frame diffing — pure Python, no LLM.

Compares two 64x64 grids and produces a structured diff:
which cells changed, what regions were affected, summary stats.
"""

from dataclasses import dataclass, field

Grid = list[list[int]]


@dataclass
class DiffResult:
    """Result of diffing two frames."""
    changed_cells: list[tuple[int, int, int, int]]  # (row, col, old_val, new_val)
    change_mask: list[list[bool]]  # 64x64 boolean mask
    num_changed: int
    regions: list[set[tuple[int, int]]]  # connected components of changed cells

    @property
    def is_empty(self) -> bool:
        return self.num_changed == 0

    def changed_values(self) -> dict[tuple[int, int], tuple[int, int]]:
        """Map of (row, col) -> (old_val, new_val) for changed cells."""
        return {(r, c): (old, new) for r, c, old, new in self.changed_cells}


def diff_frames(before: Grid, after: Grid) -> DiffResult:
    """Diff two 64x64 grids. Returns structured change information."""
    rows = len(before)
    cols = len(before[0]) if rows > 0 else 0

    changed_cells: list[tuple[int, int, int, int]] = []
    change_mask = [[False] * cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            if before[r][c] != after[r][c]:
                changed_cells.append((r, c, before[r][c], after[r][c]))
                change_mask[r][c] = True

    regions = _connected_components(change_mask, rows, cols)

    return DiffResult(
        changed_cells=changed_cells,
        change_mask=change_mask,
        num_changed=len(changed_cells),
        regions=regions,
    )


def _connected_components(
    mask: list[list[bool]], rows: int, cols: int
) -> list[set[tuple[int, int]]]:
    """Find connected components in a boolean mask (4-connected)."""
    visited = [[False] * cols for _ in range(rows)]
    components: list[set[tuple[int, int]]] = []

    for r in range(rows):
        for c in range(cols):
            if mask[r][c] and not visited[r][c]:
                component: set[tuple[int, int]] = set()
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if (
                        0 <= cr < rows
                        and 0 <= cc < cols
                        and mask[cr][cc]
                        and not visited[cr][cc]
                    ):
                        visited[cr][cc] = True
                        component.add((cr, cc))
                        stack.extend([(cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)])
                if component:
                    components.append(component)

    return components
