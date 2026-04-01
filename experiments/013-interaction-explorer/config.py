"""Experiment 013: Interaction Explorer

Key insight from experiment 004: the agent navigates well but doesn't
investigate game objects deeply enough. Hypotheses are about paths
("go up to reach X") rather than mechanics ("interacting with X does Y").

This experiment adds an explicit interaction-testing phase between
auto-probe and goal-directed play. After identifying objects and
basic movement, the agent systematically approaches and tests
interactions with each discovered object type.

Three phases:
  Phase 1: Auto-probe (same as 004) — discover actions and controllable
  Phase 2: Interaction testing — approach each object type, try all actions near it
  Phase 3: Goal-directed play — use discovered mechanics to pursue goals
"""

AGENT = "explorer"
GAME = "ls20"
MAX_ACTIONS = 100
PROBE_BUDGET = 12
INTERACTION_BUDGET = 40
