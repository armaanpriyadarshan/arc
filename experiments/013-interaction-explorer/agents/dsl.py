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


@dataclass
class GoalHypothesis:
    """Candidate objective."""
    description: str
    type: str  # "contact", "collect", "clear", "survive", "pattern", "unknown"
    target: str = ""
    confidence: float = 0.0
    evidence: str = ""


@dataclass
class WorldModel:
    """The complete induced world model."""
    object_roles: list[ObjectRole] = field(default_factory=list)
    action_rules: dict[str, ActionRule] = field(default_factory=dict)
    interaction_rules: list[InteractionRule] = field(default_factory=list)
    goal_hypotheses: list[GoalHypothesis] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)

    def controllable_color(self) -> int | None:
        for obj in self.object_roles:
            if obj.role == "controllable" and obj.confidence > 0.5:
                return obj.color
        return None

    def best_goal(self) -> GoalHypothesis | None:
        if not self.goal_hypotheses:
            return None
        return max(self.goal_hypotheses, key=lambda g: g.confidence)

    def get_interaction(self, target_color: int) -> InteractionRule | None:
        for rule in self.interaction_rules:
            if rule.target_color == target_color:
                return rule
        return None

    def untested_colors(self, sym: dict) -> list[int]:
        """Return object colors we haven't tested interactions with."""
        tested = {r.target_color for r in self.interaction_rules}
        ctrl = self.controllable_color()
        result = []
        for obj in sym.get("objects", []):
            cid = obj["color_id"]
            if cid != ctrl and cid not in tested and obj["shape"] != "background":
                result.append(cid)
        return list(set(result))

    def add_observation(self, obs: str) -> None:
        self.observations.append(obs)
        if len(self.observations) > 15:
            self.observations = self.observations[-15:]

    def to_dsl_text(self) -> str:
        """Render the world model as compact DSL text for the LLM."""
        lines = []

        if self.object_roles:
            lines.append("OBJECTS:")
            for obj in sorted(self.object_roles, key=lambda o: -o.confidence):
                lines.append(f"  color={obj.color} role={obj.role} conf={obj.confidence:.1f} ({obj.evidence})")

        if self.action_rules:
            lines.append("\nACTION RULES:")
            for name, rule in sorted(self.action_rules.items()):
                acc = f" acc={rule.accuracy:.0%}" if rule.test_count > 0 else ""
                lines.append(
                    f"  {name}: {rule.effect}"
                    + (f" {rule.direction}" if rule.direction else "")
                    + (f" dist={rule.distance}" if rule.distance else "")
                    + (f" target={rule.target}" if rule.target else "")
                    + (f" pre={rule.precondition}" if rule.precondition else "")
                    + f" conf={rule.confidence:.1f}{acc}"
                )

        if self.interaction_rules:
            lines.append("\nINTERACTION RULES:")
            for ir in sorted(self.interaction_rules, key=lambda r: -r.confidence):
                lines.append(
                    f"  color={ir.target_color} ({ir.target_role}): {ir.effect} "
                    f"conf={ir.confidence:.1f} tests={ir.test_count} ({ir.evidence})"
                )

        if self.goal_hypotheses:
            lines.append("\nGOAL HYPOTHESES:")
            for g in sorted(self.goal_hypotheses, key=lambda g: -g.confidence):
                lines.append(f"  [{g.type}] {g.description} conf={g.confidence:.1f}")

        if self.observations:
            lines.append(f"\nKEY OBSERVATIONS ({len(self.observations)}):")
            for obs in self.observations[-5:]:
                lines.append(f"  - {obs}")

        return "\n".join(lines) if lines else "(empty world model)"
