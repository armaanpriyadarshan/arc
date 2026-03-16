"""Stage 4: Goal Inference — hypothesize win conditions.

Uses LLM to analyze transitions, score changes, and object catalog
to propose what the agent needs to do to win.
"""

import json
import logging
import os
import time
from dataclasses import dataclass

from openai import OpenAI

from ..perception.objects import ObjectCatalog
from .causal import CausalResult
from .sensorimotor import SensorimotorResult

logger = logging.getLogger(__name__)


@dataclass
class GoalHypothesis:
    """A hypothesis about what the win condition is."""
    description: str
    confidence: str  # "low", "medium", "high"
    evidence: list[str]
    suggested_strategy: str


@dataclass
class GoalInferenceResult:
    hypotheses: list[GoalHypothesis]
    raw_response: str

    def best_hypothesis(self) -> GoalHypothesis | None:
        if not self.hypotheses:
            return None
        priority = {"high": 3, "medium": 2, "low": 1}
        return max(self.hypotheses, key=lambda h: priority.get(h.confidence, 0))

    def summary(self) -> str:
        lines = ["Goal Inference:"]
        for h in self.hypotheses:
            lines.append(
                f"  [{h.confidence}] {h.description}\n"
                f"    strategy: {h.suggested_strategy}\n"
                f"    evidence: {', '.join(h.evidence)}"
            )
        return "\n".join(lines)


class GoalInferenceStage:
    """Use LLM to infer what the win condition might be."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model

    def run(
        self,
        sensorimotor: SensorimotorResult,
        causal: CausalResult,
        catalog: ObjectCatalog,
    ) -> GoalInferenceResult:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

        # Build evidence summary for the LLM
        evidence = self._build_evidence(sensorimotor, causal, catalog)

        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": evidence},
                    ],
                    response_format={"type": "json_object"},
                )
                break
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"[goal_inference] API error (attempt {attempt+1}): {e}, retrying in {wait}s")
                time.sleep(wait)
        else:
            logger.error("[goal_inference] All API attempts failed")
            return GoalInferenceResult(
                hypotheses=[GoalHypothesis(
                    description="API unavailable — explore with movement",
                    confidence="low",
                    evidence=["api_error"],
                    suggested_strategy="Navigate systematically using ACTION1-4",
                )],
                raw_response="",
            )

        raw = response.choices[0].message.content or "{}"
        hypotheses = self._parse_response(raw)

        result = GoalInferenceResult(hypotheses=hypotheses, raw_response=raw)
        logger.info(result.summary())
        return result

    def _build_evidence(
        self,
        sensorimotor: SensorimotorResult,
        causal: CausalResult,
        catalog: ObjectCatalog,
    ) -> str:
        parts = [
            "# Evidence from exploration\n",
            "## Action Effects (Sensorimotor Stage)",
            sensorimotor.summary(),
            "",
            "## Object Catalog",
            catalog.summary(),
            "",
            "## Observed Transitions (Causal Stage)",
        ]

        for t in causal.transitions:
            parts.append(
                f"- {t.action}: {t.description} "
                f"(score: {t.score_before}->{t.score_after}, "
                f"state: {t.state_before}->{t.state_after})"
            )

        # Note any score changes
        score_changes = [t for t in causal.transitions if t.score_before != t.score_after]
        if score_changes:
            parts.append("\n## Score Changes Observed")
            for t in score_changes:
                parts.append(f"- {t.action}: score went from {t.score_before} to {t.score_after}")

        deaths = [t for t in causal.transitions if t.state_after == "GAME_OVER"]
        if deaths:
            parts.append("\n## Deaths Observed")
            for t in deaths:
                parts.append(f"- Died after {t.action}: {t.description}")

        return "\n".join(parts)

    def _parse_response(self, raw: str) -> list[GoalHypothesis]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return [GoalHypothesis(
                description="Failed to parse LLM response",
                confidence="low",
                evidence=["parse_error"],
                suggested_strategy="Continue exploring with movement actions",
            )]

        hypotheses = []
        for h in data.get("hypotheses", []):
            hypotheses.append(GoalHypothesis(
                description=h.get("description", "unknown"),
                confidence=h.get("confidence", "low"),
                evidence=h.get("evidence", []),
                suggested_strategy=h.get("strategy", "explore more"),
            ))

        if not hypotheses:
            hypotheses.append(GoalHypothesis(
                description=data.get("description", "Could not infer goal"),
                confidence=data.get("confidence", "low"),
                evidence=data.get("evidence", []),
                suggested_strategy=data.get("strategy", "explore more"),
            ))

        return hypotheses


SYSTEM_PROMPT = """You are analyzing observations from an agent exploring an unknown game.
The game is played on a 64x64 grid where each cell is an integer 0-15 (representing colors).
The agent can take actions (movement, interaction) and observes how the grid changes.

From the evidence provided, infer what the win condition might be.

Respond with JSON in this format:
{
  "hypotheses": [
    {
      "description": "what the agent needs to do to win",
      "confidence": "low|medium|high",
      "evidence": ["list of observations supporting this"],
      "strategy": "concrete action plan to test or execute this hypothesis"
    }
  ]
}

Be specific about game mechanics you can infer. Consider:
- What do different colors/numbers represent?
- What objects can the agent interact with?
- What causes score changes?
- What causes death/game over?
- What spatial patterns suggest a goal (doors, targets, collectibles)?
"""
