"""Interaction testing planner.

Decides which objects to approach next and plans navigation sequences
to reach them. Tracks which objects have been tested.
"""

from .symbolic import COLOR_NAMES, proximity_to


class InteractionPlanner:
    """Plans and tracks interaction tests with game objects."""

    def __init__(self) -> None:
        self.tested_colors: set[int] = set()
        self.test_results: dict[int, str] = {}  # color -> effect
        self.approach_attempts: dict[int, int] = {}  # color -> times we tried to approach

    def next_target(self, sym: dict, ctrl_color: int | None) -> int | None:
        """Pick the next object color to test interaction with.

        Prioritizes by: untested > small objects > close objects.
        """
        if ctrl_color is None:
            return None

        candidates = []
        for obj in sym.get("objects", []):
            cid = obj["color_id"]
            if cid == ctrl_color:
                continue
            if cid in self.tested_colors:
                continue
            if obj["shape"] == "background":
                continue
            # Prefer objects we haven't tried to approach much
            attempts = self.approach_attempts.get(cid, 0)
            if attempts >= 5:
                continue
            dist = proximity_to(sym, ctrl_color, cid)
            if dist is None:
                continue
            candidates.append((attempts, obj["size"], dist, cid))

        if not candidates:
            return None

        # Sort: fewest attempts, then smallest, then closest
        candidates.sort()
        return candidates[0][3]

    def record_approach(self, color: int) -> None:
        self.approach_attempts[color] = self.approach_attempts.get(color, 0) + 1

    def record_test(self, color: int, effect: str) -> None:
        self.tested_colors.add(color)
        self.test_results[color] = effect

    def navigation_direction(
        self, sym: dict, ctrl_color: int, target_color: int, action_rules: dict
    ) -> str | None:
        """Return the action name that moves the controllable toward the target."""
        ctrl_pos = None
        target_pos = None

        for obj in sym.get("objects", []):
            if obj["color_id"] == ctrl_color:
                ctrl_pos = obj["center"]
            elif obj["color_id"] == target_color:
                if target_pos is None:
                    target_pos = obj["center"]
                elif ctrl_pos:
                    # Pick the closest target
                    old_d = abs(target_pos[0] - ctrl_pos[0]) + abs(target_pos[1] - ctrl_pos[1])
                    new_d = abs(obj["center"][0] - ctrl_pos[0]) + abs(obj["center"][1] - ctrl_pos[1])
                    if new_d < old_d:
                        target_pos = obj["center"]

        if not ctrl_pos or not target_pos:
            return None

        dr = target_pos[0] - ctrl_pos[0]
        dc = target_pos[1] - ctrl_pos[1]

        candidates = []
        for name, rule in action_rules.items():
            if rule.effect != "move" or not rule.direction:
                continue
            expected_dr = {"up": -1, "down": 1}.get(rule.direction, 0)
            expected_dc = {"left": -1, "right": 1}.get(rule.direction, 0)
            score = dr * expected_dr + dc * expected_dc
            candidates.append((score, name))

        if not candidates:
            return None

        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1] if candidates[0][0] > 0 else None

    def summary(self) -> str:
        lines = [f"Tested {len(self.tested_colors)} object types:"]
        for color, effect in sorted(self.test_results.items()):
            name = COLOR_NAMES.get(color, f"color_{color}")
            lines.append(f"  {name} (color={color}): {effect}")
        untested = len(self.approach_attempts) - len(self.tested_colors)
        if untested > 0:
            lines.append(f"  {untested} types still being approached")
        return "\n".join(lines)
