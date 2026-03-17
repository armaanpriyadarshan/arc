"""Code-driven navigation. Zero LLM calls.

Tracks position from diffs, builds a wall map, does BFS pathfinding,
executes systematic exploration. The LLM sets the destination,
code gets there.
"""

from collections import deque

Grid = list[list[int]]


class Navigator:
    def __init__(self) -> None:
        self.pos: tuple[int, int] | None = None
        self.walls: set[tuple[int, int, str]] = set()
        self.visited: set[tuple[int, int]] = set()
        self.entity_colors: set[int] = set()
        self._prev_grid: Grid | None = None
        self._recent_positions: list[tuple[int, int]] = []  # for oscillation detection
        self._direction_preference: list[str] = ["ACTION1", "ACTION4", "ACTION2", "ACTION3"]

    def update(self, action: str, before: Grid, after: Grid, changes: int, blocked: bool) -> None:
        """Update position and wall map from a game action result."""
        if blocked and self.pos:
            r, c = self.pos
            self.walls.add((r, c, action))
            return

        if changes < 10:
            return

        # Find where entity colors appeared (new position)
        appeared: dict[int, list[tuple[int, int]]] = {}
        disappeared: dict[int, list[tuple[int, int]]] = {}

        for r in range(len(before)):
            for c in range(len(before[0])):
                if before[r][c] != after[r][c]:
                    old, new = before[r][c], after[r][c]
                    appeared.setdefault(new, []).append((r, c))
                    disappeared.setdefault(old, []).append((r, c))

        # On first move, identify entity colors from what moved
        if not self.entity_colors:
            common = set(appeared.keys()) & set(disappeared.keys())
            for color in common:
                if 2 <= len(appeared[color]) <= 20:
                    self.entity_colors.add(color)

        # Find new position: centroid of cells where entity colors appeared
        if self.entity_colors:
            cells = []
            for color in self.entity_colors:
                if color in appeared:
                    cells.extend(appeared[color])
            if cells:
                new_r = sum(r for r, c in cells) // len(cells)
                new_c = sum(c for r, c in cells) // len(cells)
                self.pos = (new_r, new_c)
                self.visited.add(self.pos)
                self._recent_positions.append(self.pos)
                if len(self._recent_positions) > 10:
                    self._recent_positions = self._recent_positions[-10:]

        self._prev_grid = after

    def direction_to(self, target: tuple[int, int]) -> str | None:
        """Get the next action to move toward target, avoiding known walls and oscillation."""
        if not self.pos:
            return None

        # If oscillating, give up on this target and just explore
        if self.is_oscillating():
            return self.explore_action()

        r, c = self.pos
        tr, tc = target

        candidates = []
        if tr < r and (r, c, "ACTION1") not in self.walls:
            candidates.append(("ACTION1", abs(tr - r)))
        if tr > r and (r, c, "ACTION2") not in self.walls:
            candidates.append(("ACTION2", abs(tr - r)))
        if tc < c and (r, c, "ACTION3") not in self.walls:
            candidates.append(("ACTION3", abs(tc - c)))
        if tc > c and (r, c, "ACTION4") not in self.walls:
            candidates.append(("ACTION4", abs(tc - c)))

        if candidates:
            candidates.sort(key=lambda x: -x[1])
            return candidates[0][0]

        # Direct paths all blocked — try perpendicular directions to go around
        perpendicular = []
        if tr != r:  # trying to go vertically, blocked — try horizontal
            if (r, c, "ACTION3") not in self.walls:
                perpendicular.append("ACTION3")
            if (r, c, "ACTION4") not in self.walls:
                perpendicular.append("ACTION4")
        if tc != c:  # trying to go horizontally, blocked — try vertical
            if (r, c, "ACTION1") not in self.walls:
                perpendicular.append("ACTION1")
            if (r, c, "ACTION2") not in self.walls:
                perpendicular.append("ACTION2")

        if perpendicular:
            return perpendicular[0]

        return None

    def is_oscillating(self) -> bool:
        """Detect if we're bouncing between the same 2-3 positions."""
        if len(self._recent_positions) < 6:
            return False
        last_6 = self._recent_positions[-6:]
        unique = set(last_6)
        return len(unique) <= 2

    def explore_action(self) -> str:
        """Pick an action that explores unvisited territory."""
        if not self.pos:
            return "ACTION1"

        r, c = self.pos

        # If oscillating, rotate preferences to try a different direction
        if self.is_oscillating():
            self._direction_preference = self._direction_preference[1:] + self._direction_preference[:1]
            self._recent_positions.clear()

        for action in self._direction_preference:
            if (r, c, action) not in self.walls:
                return action

        # All directions blocked — rotate preference and try anything
        self._direction_preference = self._direction_preference[1:] + self._direction_preference[:1]
        return self._direction_preference[0]

    def plan_path(self, target: tuple[int, int], max_steps: int = 20) -> list[str]:
        """Generate a sequence of actions to reach target."""
        if not self.pos:
            return [self.explore_action()] * min(5, max_steps)

        actions = []
        r, c = self.pos

        for _ in range(max_steps):
            if abs(r - target[0]) < 4 and abs(c - target[1]) < 4:
                break  # close enough

            action = self.direction_to(target)
            if not action:
                break

            actions.append(action)
            # Estimate new position
            if action == "ACTION1":
                r -= 4
            elif action == "ACTION2":
                r += 4
            elif action == "ACTION3":
                c -= 4
            elif action == "ACTION4":
                c += 4

        return actions if actions else [self.explore_action()] * min(5, max_steps)

    def status(self) -> str:
        pos = f"({self.pos[0]},{self.pos[1]})" if self.pos else "unknown"
        return (f"Position: {pos}. Visited: {len(self.visited)} locations. "
                f"Known walls: {len(self.walls)}. Entity colors: {sorted(self.entity_colors)}.")
