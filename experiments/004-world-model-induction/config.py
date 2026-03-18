"""Experiment 004: World Model Induction

Treat game-playing as hidden-mechanics program induction.
Phase 1: Systematic action probing + operational identification of controllable object
Phase 2: GPT induces action rules and goal hypotheses in a DSL
Phase 3: Goal-directed play with predictive model scoring and refinement

Key innovations vs experiment 003:
- Controllable object identified OPERATIONALLY (which object responds to inputs)
- Action rules as executable hypotheses, not vague text
- Predictive scoring: does the rule predict what actually happened?
- Separation of action semantics induction from goal pursuit
"""

AGENT = "explorer"
GAME = "ls20"
MAX_ACTIONS = 100
