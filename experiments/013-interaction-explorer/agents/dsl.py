"""DSL for world model hypotheses.

Four layers:
1. Object roles — operational definitions (controllable, blocking, goal-like)
2. Action semantics — what each action does as an executable rule
3. Interaction rules — what happens when the controllable contacts other objects
4. Goal hypotheses — candidate objectives

These are structured data that GPT proposes and code evaluates.
"""

from dataclasses import dataclass, field


@dataclass
class ObjectRole:
    """Operationally-defined object role."""
    color: int
    role: str  # "controllable", "blocking", "goal_like", "hazard", "indicator", "collectible", "unknown"
    evidence: str = ""
    confidence: float = 0.0


@dataclass
class ActionRule:
    """Hypothesis about what an action does."""
    action: str  # "ACTION1", etc.
    effect: str  # "move", "push", "toggle", "recolor", "no_op", "unknown"
    direction: str = ""  # "up", "down", "left", "right", "" for non-directional
    distance: int = 0
    target: str = ""  # "controllable", "adjacent", "all", "global"
    precondition: str = ""  # "path_clear", "adjacent_to_X", "always"
    confidence: float = 0.0
    test_count: int = 0
    success_count: int = 0

    @property
    def accuracy(self) -> float:
        return self.success_count / self.test_count if self.test_count > 0 else 0.0


@dataclass
class InteractionRule:
    """What happens when the controllable contacts a specific object type."""
    target_color: int
    target_role: str  # role from ObjectRole
    effect: str  # "pass_through", "blocked", "collected", "died", "teleported", "score_up", "unknown"
    evidence: str = ""
    test_count: int = 0
    confidence: float = 0.0
