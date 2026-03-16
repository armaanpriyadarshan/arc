"""Object detection and tracking — pure Python, no LLM.

Detects objects as connected components of same-color cells.
Tracks objects across frames by matching position and color.
Background is inferred, not hardcoded.
"""

from collections import Counter
from dataclasses import dataclass, field

Grid = list[list[int]]


def infer_background(grid: Grid) -> set[int]:
    """Infer background colors from frequency. The most common color is background."""
    counts: Counter[int] = Counter()
    for row in grid:
        for val in row:
            counts[val] += 1
    if not counts:
        return set()
    # Background is any color covering >15% of the grid
    total = sum(counts.values())
    return {color for color, count in counts.items() if count / total > 0.15}


def infer_background_from_diffs(
    grids: list[Grid],
) -> set[int]:
    """Infer background from multiple frames. Colors that dominate static cells are background."""
    if not grids:
        return set()
    # Cells that never change across any pair of frames are structural
    rows = len(grids[0])
    cols = len(grids[0][0]) if rows > 0 else 0

    # Count how often each color appears in cells that never changed
    static_counts: Counter[int] = Counter()
    changed_ever = [[False] * cols for _ in range(rows)]

    for i in range(1, len(grids)):
        for r in range(rows):
            for c in range(cols):
                if grids[i][r][c] != grids[i - 1][r][c]:
                    changed_ever[r][c] = True

    for r in range(rows):
        for c in range(cols):
            if not changed_ever[r][c]:
                static_counts[grids[0][r][c]] += 1

    if not static_counts:
        return infer_background(grids[0])

    total_static = sum(static_counts.values())
    # Background = colors making up >15% of static cells
    return {color for color, count in static_counts.items() if count / total_static > 0.15}


@dataclass
class GameObject:
    """A detected object in a single frame."""
    id: int
    color: int  # the single color this object is made of
    cells: set[tuple[int, int]]  # (row, col)
    bbox: tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def center(self) -> tuple[float, float]:
        rs = [r for r, _ in self.cells]
        cs = [c for _, c in self.cells]
        return (sum(rs) / len(rs), sum(cs) / len(cs))

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1


@dataclass
class TrackedObject:
    """An object tracked across multiple frames."""
    id: int
    history: list[GameObject]
    is_controllable: bool = False
    is_static: bool = True
    interaction_count: int = 0

    @property
    def latest(self) -> GameObject:
        return self.history[-1]

    @property
    def color(self) -> int:
        return self.latest.color

    @property
    def moved(self) -> bool:
        if len(self.history) < 2:
            return False
        return self.history[-1].center != self.history[-2].center


@dataclass
class ObjectCatalog:
    """All tracked objects across the game so far."""
    objects: dict[int, TrackedObject] = field(default_factory=dict)
    _next_id: int = 0
    background_colors: set[int] = field(default_factory=set)

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    @property
    def controllable(self) -> list[TrackedObject]:
        return [o for o in self.objects.values() if o.is_controllable]

    @property
    def static(self) -> list[TrackedObject]:
        return [o for o in self.objects.values() if o.is_static]

    @property
    def interactive(self) -> list[TrackedObject]:
        return [o for o in self.objects.values() if o.interaction_count > 0]

    def summary(self) -> str:
        lines = [
            f"ObjectCatalog: {len(self.objects)} tracked objects "
            f"(bg colors: {sorted(self.background_colors)})"
        ]
        for obj in self.objects.values():
            tag = []
            if obj.is_controllable:
                tag.append("controllable")
            if obj.is_static:
                tag.append("static")
            if obj.interaction_count > 0:
                tag.append(f"interactive({obj.interaction_count})")
            tags = ", ".join(tag) if tag else "unknown"
            latest = obj.latest
            lines.append(
                f"  obj-{obj.id}: color={latest.color} "
                f"size={latest.size} center=({latest.center[0]:.0f},{latest.center[1]:.0f}) "
                f"bbox={latest.bbox} [{tags}]"
            )
        return "\n".join(lines)


def detect_objects(grid: Grid, background: set[int] | None = None) -> list[GameObject]:
    """Detect objects as connected components of same-color, non-background cells.

    Key change: objects are per-color. Adjacent cells of different colors
    are separate objects, even if both are non-background.
    """
    bg = background if background is not None else infer_background(grid)
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    visited = [[False] * cols for _ in range(rows)]
    objects: list[GameObject] = []
    obj_id = 0

    for r in range(rows):
        for c in range(cols):
            color = grid[r][c]
            if color not in bg and not visited[r][c]:
                # BFS: only connect cells of the SAME color
                component: set[tuple[int, int]] = set()
                stack = [(r, c)]
                min_r, min_c = r, c
                max_r, max_c = r, c

                while stack:
                    cr, cc = stack.pop()
                    if (
                        0 <= cr < rows
                        and 0 <= cc < cols
                        and not visited[cr][cc]
                        and grid[cr][cc] == color
                    ):
                        visited[cr][cc] = True
                        component.add((cr, cc))
                        min_r = min(min_r, cr)
                        min_c = min(min_c, cc)
                        max_r = max(max_r, cr)
                        max_c = max(max_c, cc)
                        stack.extend([
                            (cr - 1, cc), (cr + 1, cc),
                            (cr, cc - 1), (cr, cc + 1),
                        ])

                if len(component) >= 2:  # skip single-pixel noise
                    objects.append(GameObject(
                        id=obj_id,
                        color=color,
                        cells=component,
                        bbox=(min_r, min_c, max_r, max_c),
                    ))
                    obj_id += 1

    return objects


def track_objects(
    catalog: ObjectCatalog,
    prev_objects: list[GameObject],
    curr_objects: list[GameObject],
    agent_acted: bool = True,
) -> ObjectCatalog:
    """Match objects across frames and update tracking state.

    Matching: same color, closest center.
    """
    matched_prev: set[int] = set()
    matched_curr: set[int] = set()

    # Build candidates: same color, ranked by distance
    candidates: list[tuple[float, int, int]] = []
    for pi, po in enumerate(prev_objects):
        for ci, co in enumerate(curr_objects):
            if po.color == co.color and abs(po.size - co.size) < max(po.size, co.size) * 0.5:
                pr, pc = po.center
                cr, cc = co.center
                dist = abs(pr - cr) + abs(pc - cc)
                candidates.append((dist, pi, ci))

    candidates.sort()
    matches: list[tuple[int, int]] = []
    for dist, pi, ci in candidates:
        if pi not in matched_prev and ci not in matched_curr:
            matches.append((pi, ci))
            matched_prev.add(pi)
            matched_curr.add(ci)

    for pi, ci in matches:
        po = prev_objects[pi]
        co = curr_objects[ci]

        # Find existing tracked object
        tracked = None
        for t in catalog.objects.values():
            if t.latest.color == po.color and t.latest.center == po.center:
                tracked = t
                break

        if tracked is None:
            tid = catalog._new_id()
            tracked = TrackedObject(id=tid, history=[po])
            catalog.objects[tid] = tracked

        tracked.history.append(co)

        if po.center != co.center:
            tracked.is_static = False
            if agent_acted:
                tracked.is_controllable = True

    for ci, co in enumerate(curr_objects):
        if ci not in matched_curr:
            tid = catalog._new_id()
            catalog.objects[tid] = TrackedObject(id=tid, history=[co])

    return catalog
