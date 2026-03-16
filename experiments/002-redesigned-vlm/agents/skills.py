"""Skill library — reusable action sequences with success tracking.

Skills are named, composable, and track their own effectiveness.
"""

import logging

from .memory import Skill

logger = logging.getLogger(__name__)


class SkillLibrary:
    def __init__(self) -> None:
        self.skills: dict[str, Skill] = {}

    def add(self, name: str, actions: list[str], description: str = "",
            source: str = "discovery", precondition: str = "",
            expected_effect: str = "") -> Skill:
        skill = Skill(
            name=name,
            actions=actions,
            description=description,
            source=source,
            precondition=precondition,
            expected_effect=expected_effect,
        )
        self.skills[name] = skill
        logger.info(f"[skills] added: {name} = {actions[:8]}{'...' if len(actions) > 8 else ''}")
        return skill

    def record_use(self, name: str, succeeded: bool) -> None:
        if name in self.skills:
            self.skills[name].times_used += 1
            if succeeded:
                self.skills[name].times_succeeded += 1

    def record_success_sequence(self, actions: list[str], description: str) -> Skill:
        """Record an action sequence that produced a positive outcome."""
        name = f"success_{len(self.skills)}"
        return self.add(name, actions, description=description, source="success_recording")

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)

    def expand(self, plan: list[str]) -> list[str]:
        """Expand skill references in a plan to raw actions.

        If a plan item is a skill name, replace it with the skill's actions.
        Otherwise keep it as-is (it's a raw action).
        """
        expanded = []
        for item in plan:
            if item in self.skills:
                expanded.extend(self.skills[item].actions)
            else:
                expanded.append(item)
        return expanded

    def as_list(self) -> list[Skill]:
        return list(self.skills.values())

    def summary(self) -> str:
        if not self.skills:
            return "No skills yet."
        lines = []
        for s in self.skills.values():
            rate = f"{s.times_succeeded}/{s.times_used}" if s.times_used > 0 else "untested"
            lines.append(f"  {s.name}: {s.actions[:6]}{'...' if len(s.actions) > 6 else ''} [{rate}] — {s.description}")
        return "\n".join(lines)
