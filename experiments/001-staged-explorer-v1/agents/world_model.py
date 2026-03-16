"""World Model — LLM-synthesized step function with test-driven refinement.

The world model is a Python function `step(grid, action) -> grid` that
predicts the next frame given the current frame and an action.

The test suite is a list of real (grid_before, action, grid_after) triples.
When the model is refined, all past tests must still pass.
"""

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """A single world model test case from a real observation."""
    action: str
    grid_before: list[list[int]]
    grid_after: list[list[int]]
    score_before: int
    score_after: int


@dataclass
class WorldModel:
    """Container for the world model and its test suite."""
    test_suite: list[TestCase] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)  # natural language rules extracted

    def add_test(
        self,
        action: str,
        grid_before: list[list[int]],
        grid_after: list[list[int]],
        score_before: int = 0,
        score_after: int = 0,
    ) -> None:
        self.test_suite.append(TestCase(
            action=action,
            grid_before=grid_before,
            grid_after=grid_after,
            score_before=score_before,
            score_after=score_after,
        ))

    def add_tests_from_causal(self, test_cases: list[dict]) -> None:
        """Import test cases from CausalResult.to_test_cases()."""
        for tc in test_cases:
            self.add_test(
                action=tc["action"],
                grid_before=tc["grid_before"],
                grid_after=tc["grid_after"],
                score_before=tc.get("score_before", 0),
                score_after=tc.get("score_after", 0),
            )

    def summary(self) -> str:
        lines = [f"WorldModel: {len(self.test_suite)} test cases, {len(self.rules)} rules"]
        for rule in self.rules:
            lines.append(f"  - {rule}")
        return "\n".join(lines)
