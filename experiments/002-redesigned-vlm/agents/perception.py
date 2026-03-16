"""Perception layer — raw diff computation (code) + scene description (gpt-4o).

The ONLY code computation is: what cells changed between frames.
Everything else (interpretation, tracking, navigation) is the model's job.
"""

import logging
from dataclasses import dataclass, field

from .vision import grid_to_b64, image_block, text_block

logger = logging.getLogger(__name__)

Grid = list[list[int]]


@dataclass
class CellChange:
    row: int
    col: int
    old_val: int
    new_val: int


@dataclass
class Diff:
    action: str
    changes: list[CellChange]
    total_changes: int

    def as_text(self) -> str:
        if self.total_changes == 0:
            return f"{self.action}: no changes"

        lines = [f"{self.action}: {self.total_changes} cells changed"]
        transitions: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for c in self.changes:
            key = (c.old_val, c.new_val)
            if key not in transitions:
                transitions[key] = []
            transitions[key].append((c.row, c.col))

        for (old, new), cells in sorted(transitions.items(), key=lambda x: -len(x[1])):
            coords = cells[:12]
            coord_str = ", ".join(f"({r},{c})" for r, c in coords)
            more = f" +{len(cells) - 12} more" if len(cells) > 12 else ""
            lines.append(f"  {old}→{new} at: {coord_str}{more}")

        return "\n".join(lines)


class DiffAnalyzer:
    """Raw diff computation. No interpretation."""

    def __init__(self) -> None:
        self.all_diffs: list[Diff] = []

    def compute(self, action: str, before: Grid, after: Grid) -> Diff:
        changes = []
        for r in range(len(before)):
            for c in range(len(before[0])):
                if before[r][c] != after[r][c]:
                    changes.append(CellChange(r, c, before[r][c], after[r][c]))

        diff = Diff(action=action, changes=changes, total_changes=len(changes))
        self.all_diffs.append(diff)
        return diff

    def is_blocked(self, diff: Diff) -> bool:
        return 0 < diff.total_changes < 10

    def recent_diffs_text(self, n: int = 8) -> str:
        return "\n\n".join(d.as_text() for d in self.all_diffs[-n:])


class SceneDescriber:
    """gpt-4o vision model. Called sparingly for scene understanding."""

    def __init__(self, model_call_fn) -> None:
        self._call = model_call_fn
        self.call_count = 0

    def describe(self, grid: Grid) -> str:
        self.call_count += 1
        content = [
            text_block(
                "You are observing a turn-based game on a 64x64 pixel grid with 16 colors. "
                "The game has hidden rules you must figure out. It has multiple levels to complete. "
                "Describe everything you see:\n"
                "- Distinct visual elements, their approximate grid positions, colors, shapes\n"
                "- Any UI-like elements (bars, counters, indicators)\n"
                "- Spatial layout and navigable paths\n"
                "- What looks interactive or important for completing the level\n\n"
                "Be precise about positions. Grid is 64x64, top-left=(0,0), bottom-right=(63,63)."
            ),
            image_block(grid_to_b64(grid)),
        ]
        from .models import VISION_MODEL
        return self._call(VISION_MODEL, content, 800, False)
