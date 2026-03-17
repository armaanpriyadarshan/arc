"""Structured memory as a DSL. All Pydantic models, JSON-serializable.

This is the knowledge base — not a growing string, but typed data
that the planner queries and the models update.
"""

from pydantic import BaseModel, Field


class Primitive(BaseModel):
    """A single action and its observed effect."""
    action: str
    direction: str = "unknown"
    avg_changes: float = 0.0
    is_available: bool = True
    confidence: float = 0.0
    notes: str = ""


class Skill(BaseModel):
    """A reusable action sequence with measured outcomes."""
    name: str
    actions: list[str]
    description: str = ""
    precondition: str = ""
    expected_effect: str = ""
    times_used: int = 0
    times_succeeded: int = 0
    source: str = "discovery"  # "discovery", "synthesis", "success_recording"

    @property
    def success_rate(self) -> float:
        return self.times_succeeded / self.times_used if self.times_used > 0 else 0.0


class GridObject(BaseModel):
    """A detected entity on the grid."""
    id: str
    description: str = ""
    colors: list[int] = Field(default_factory=list)
    bounding_box: tuple[int, int, int, int] = (0, 0, 0, 0)
    interaction_notes: str = ""
    first_seen_step: int = 0


class Rule(BaseModel):
    """A discovered game mechanic."""
    id: str
    description: str
    evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.5


class Hypothesis(BaseModel):
    """A testable claim about the game."""
    id: str
    statement: str
    test_plan: list[str] = Field(default_factory=list)
    status: str = "untested"  # untested, testing, confirmed, refuted
    evidence_for: list[str] = Field(default_factory=list)
    evidence_against: list[str] = Field(default_factory=list)
    priority: int = 5  # 1=highest


class Transition(BaseModel):
    """A recorded state transition — action from a position produced an outcome."""
    action: str
    from_region: str = ""  # compact description of where we were
    changes: int = 0
    blocked: bool = False
    score_delta: int = 0
    large_change: bool = False  # >100 cells changed
    death: bool = False

    @property
    def outcome_key(self) -> str:
        if self.death:
            return "death"
        if self.blocked:
            return "blocked"
        if self.score_delta > 0:
            return "score"
        if self.large_change:
            return "special"
        return "normal"


class TransitionTable(BaseModel):
    """Probabilistic model of action effects. Tracks what outcomes each action produces."""
    counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    # counts[action_name][outcome_key] = count

    def record(self, transition: Transition) -> None:
        if transition.action not in self.counts:
            self.counts[transition.action] = {}
        outcome = transition.outcome_key
        self.counts[transition.action][outcome] = self.counts[transition.action].get(outcome, 0) + 1

    def compact_text(self) -> str:
        if not self.counts:
            return "No transitions recorded."
        lines = ["TRANSITION TABLE:"]
        for action, outcomes in sorted(self.counts.items()):
            total = sum(outcomes.values())
            parts = [f"{k}={v}/{total}" for k, v in sorted(outcomes.items())]
            lines.append(f"  {action}: {', '.join(parts)}")
        return "\n".join(lines)


class ActionLogEntry(BaseModel):
    """One entry in the bounded action log."""
    step: int
    action: str
    changes: int
    score: int
    direction: str = ""
    blocked: bool = False
    notes: str = ""


class GameKnowledge(BaseModel):
    """The complete knowledge base."""
    primitives: dict[str, Primitive] = Field(default_factory=dict)
    objects: list[GridObject] = Field(default_factory=list)
    rules: list[Rule] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    transitions: TransitionTable = Field(default_factory=TransitionTable)
    score_events: list[dict] = Field(default_factory=list)
    death_events: list[dict] = Field(default_factory=list)
    scene_description: str = ""
    current_theory: str = ""  # one-sentence game theory, persists across calls
    action_log: list[ActionLogEntry] = Field(default_factory=list)
    available_actions: list[int] = Field(default_factory=list)

    def dedup_hypotheses(self) -> None:
        """Remove near-duplicate hypotheses, keeping highest priority."""
        if len(self.hypotheses) <= 5:
            return
        seen: dict[str, Hypothesis] = {}
        for h in self.hypotheses:
            # Normalize: lowercase, strip, first 50 chars as key
            key = h.statement.lower().strip()[:50]
            if key not in seen or h.priority < seen[key].priority:
                seen[key] = h
        self.hypotheses = list(seen.values())

    def dedup_rules(self) -> None:
        """Remove near-duplicate rules, keeping highest confidence."""
        if len(self.rules) <= 5:
            return
        seen: dict[str, Rule] = {}
        for r in self.rules:
            key = r.description.lower().strip()[:50]
            if key not in seen or r.confidence > seen[key].confidence:
                seen[key] = r
        self.rules = list(seen.values())

    def log_action(self, step: int, action: str, changes: int, score: int,
                   direction: str = "", blocked: bool = False, notes: str = "") -> None:
        self.action_log.append(ActionLogEntry(
            step=step, action=action, changes=changes, score=score,
            direction=direction, blocked=blocked, notes=notes,
        ))
        # Bounded: keep last 30
        if len(self.action_log) > 30:
            self.action_log = self.action_log[-30:]

    def compact_text(self) -> str:
        """Render the knowledge base as compact text for LLM consumption."""
        sections = []

        # Primitives
        if self.primitives:
            sections.append("ACTIONS:")
            for name, p in sorted(self.primitives.items()):
                avail = "" if p.is_available else " [UNAVAILABLE]"
                sections.append(
                    f"  {name}: {p.direction} ({p.avg_changes:.0f} changes, "
                    f"conf={p.confidence:.1f}){avail}"
                )

        # Skills
        if self.skills:
            sections.append("\nSKILLS:")
            for s in self.skills:
                rate = f"{s.times_succeeded}/{s.times_used}" if s.times_used > 0 else "untested"
                actions_str = "→".join(s.actions[:8])
                if len(s.actions) > 8:
                    actions_str += f"...({len(s.actions)} total)"
                sections.append(f"  {s.name}: [{actions_str}] success={rate} — {s.description}")

        # Rules
        confirmed_rules = [r for r in self.rules if r.confidence >= 0.5]
        if confirmed_rules:
            sections.append("\nRULES:")
            for r in confirmed_rules:
                sections.append(f"  - {r.description} (conf={r.confidence:.1f})")

        # Objects
        if self.objects:
            sections.append("\nOBJECTS:")
            for o in self.objects:
                sections.append(f"  {o.id}: {o.description} at {o.bounding_box} colors={o.colors}")

        # Hypotheses
        active = [h for h in self.hypotheses if h.status not in ("confirmed", "refuted")]
        if active:
            sections.append("\nHYPOTHESES:")
            for h in sorted(active, key=lambda h: h.priority):
                has_plan = f" [{len(h.test_plan)} actions]" if h.test_plan else ""
                sections.append(f"  (p{h.priority}) [{h.status}] {h.statement}{has_plan}")

        # Transition probabilities
        tt = self.transitions.compact_text()
        if tt != "No transitions recorded.":
            sections.append(f"\n{tt}")

        # Scene
        if self.scene_description:
            sections.append(f"\nSCENE: {self.scene_description[:500]}")

        # Score/death events
        if self.score_events:
            sections.append(f"\nSCORE EVENTS: {len(self.score_events)}")
            for e in self.score_events[-3:]:
                sections.append(f"  {e}")
        if self.death_events:
            sections.append(f"\nDEATHS: {len(self.death_events)}")
            for e in self.death_events[-2:]:
                sections.append(f"  {e}")

        # Recent actions
        if self.action_log:
            sections.append(f"\nRECENT ({len(self.action_log)} entries):")
            for e in self.action_log[-10:]:
                blocked = " BLOCKED" if e.blocked else ""
                sections.append(f"  #{e.step} {e.action}: {e.changes}ch {e.direction}{blocked} score={e.score}")

        return "\n".join(sections)
