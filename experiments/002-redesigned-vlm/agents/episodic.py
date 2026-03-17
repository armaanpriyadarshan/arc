"""Layer 2: Episodic Buffer — rolling window of structured transitions. Zero LLM calls.

Clusters recent actions into episodes. Detects patterns:
stuck loops, exploration progress, repeated states.
"""

import hashlib
from dataclasses import dataclass, field

Grid = list[list[int]]


@dataclass
class Step:
    number: int
    action: str
    changes: int
    blocked: bool
    score: int
    score_delta: int
    death: bool
    frame_sig: str  # md5 hash of grid for state comparison


@dataclass
class Episode:
    """A cluster of related steps."""
    steps: list[Step]
    label: str  # "moved_right_4", "blocked_up_3", "explored_new_area"

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def actions(self) -> list[str]:
        return [s.action for s in self.steps]

    def summary(self) -> str:
        return f"{self.label} ({self.length} steps): {', '.join(self.actions)}"


class EpisodicBuffer:
    def __init__(self, window_size: int = 30) -> None:
        self.steps: list[Step] = []
        self.window_size = window_size
        self._seen_sigs: dict[str, int] = {}  # sig -> count
        self._blocked_positions: list[tuple[str, int]] = []  # (action, step) for blocked moves

    def record(self, number: int, action: str, changes: int, blocked: bool,
               score: int, score_delta: int, death: bool, grid: Grid) -> None:
        sig = self._sig(grid)
        self._seen_sigs[sig] = self._seen_sigs.get(sig, 0) + 1

        step = Step(number=number, action=action, changes=changes, blocked=blocked,
                    score=score, score_delta=score_delta, death=death, frame_sig=sig)
        self.steps.append(step)

        if blocked:
            self._blocked_positions.append((action, number))

        # Keep bounded
        if len(self.steps) > self.window_size:
            self.steps = self.steps[-self.window_size:]

    def cluster_episodes(self) -> list[Episode]:
        """Cluster recent steps into episodes by action consistency."""
        if not self.steps:
            return []

        episodes: list[Episode] = []
        current: list[Step] = [self.steps[0]]

        for step in self.steps[1:]:
            prev = current[-1]
            # Same episode if: same action, or same blocked status
            same_action = step.action == prev.action
            both_blocked = step.blocked and prev.blocked
            both_moving = not step.blocked and not prev.blocked and step.changes > 5 and prev.changes > 5

            if same_action or both_blocked or both_moving:
                current.append(step)
            else:
                episodes.append(self._label_episode(current))
                current = [step]

        episodes.append(self._label_episode(current))
        return episodes

    def _label_episode(self, steps: list[Step]) -> Episode:
        if all(s.blocked for s in steps):
            actions = set(s.action for s in steps)
            label = f"blocked_{'+'.join(sorted(actions))}_{len(steps)}"
        elif all(s.score_delta > 0 for s in steps):
            label = f"scoring_{len(steps)}"
        elif all(s.death for s in steps):
            label = f"death"
        elif len(set(s.action for s in steps)) == 1:
            action = steps[0].action
            label = f"{action}_x{len(steps)}"
        else:
            label = f"mixed_{len(steps)}"
        return Episode(steps=steps, label=label)

    def detect_stuck_loop(self) -> bool:
        """Detect if the last 5+ actions are a repeating blocked pattern."""
        recent = self.steps[-8:]
        if len(recent) < 5:
            return False
        blocked_count = sum(1 for s in recent if s.blocked)
        return blocked_count >= 5

    def detect_repeated_state(self) -> bool:
        """Check if current grid state has been seen 3+ times."""
        if not self.steps:
            return False
        current_sig = self.steps[-1].frame_sig
        return self._seen_sigs.get(current_sig, 0) >= 3

    def consecutive_blocked(self) -> int:
        """Count consecutive blocked moves at the end of the buffer."""
        count = 0
        for step in reversed(self.steps):
            if step.blocked:
                count += 1
            else:
                break
        return count

    def compact_text(self) -> str:
        if not self.steps:
            return "EPISODES: No data yet."

        episodes = self.cluster_episodes()
        lines = [f"RECENT EPISODES ({len(self.steps)} steps):"]
        for ep in episodes[-8:]:
            lines.append(f"  {ep.summary()}")

        # Pattern alerts
        if self.detect_stuck_loop():
            lines.append("  ⚠ STUCK: last 5+ actions were blocked")
        if self.detect_repeated_state():
            lines.append("  ⚠ LOOP: current state seen 3+ times (going in circles)")

        cb = self.consecutive_blocked()
        if cb >= 3:
            lines.append(f"  ⚠ {cb} consecutive blocked moves")

        return "\n".join(lines)

    def _sig(self, grid: Grid) -> str:
        flat = bytes(cell for row in grid for cell in row)
        return hashlib.md5(flat).hexdigest()[:8]
