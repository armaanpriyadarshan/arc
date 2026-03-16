"""Experiment 001: Staged Explorer v1

First full run of the staged exploration agent against LockSmith.
Tests the complete pipeline: perception → sensorimotor → object discovery
→ causal testing → goal inference → planning.

Hypothesis: A staged agent that builds understanding bottom-up
(action effects → objects → interactions → goals) can discover
LockSmith's rules and make progress without any game-specific knowledge.

Result: 0/7 levels. Perception failed — see analysis.md.
"""

AGENT = "explorer"
GAME = "ls20"
SCORECARD_ID = "832a07d8-e387-47e6-9eaf-a2f2cf1099d3"
SCORECARD_URL = "https://three.arcprize.org/scorecards/832a07d8-e387-47e6-9eaf-a2f2cf1099d3"

# Stage action budgets
SENSORIMOTOR_BUDGET = 12
OBJECT_DISCOVERY_BUDGET = 8
CAUSAL_BUDGET = 20
PLANNING_BUDGET = 160

MAX_ACTIONS = 200

MODEL = "gpt-4o-mini"
