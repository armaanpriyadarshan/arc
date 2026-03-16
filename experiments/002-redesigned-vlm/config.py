"""Experiment 002: Three-Model Agent (Eyes + Brain + Hands)

Complete redesign from experiment 001 (0/7 levels).

Architecture:
  Eyes (gpt-4o): per-action visual observation, multi-turn conversation
  Brain (o4-mini): periodic strategic reasoning, hypothesis journal
  Hands (gpt-4o-mini): per-action executor, text-only, reads plan
  Tracker: code-only spatial tracking from diffs, zero API calls

Reasoning annotations are readable English text in the ARC replay.
"""

AGENT = "explorer"
GAME = "ls20"
MAX_ACTIONS = 200

EYES_MODEL = "gpt-4o"
BRAIN_MODEL = "o4-mini"
HANDS_MODEL = "gpt-4o-mini"

BRAIN_INTERVAL = 15  # brain called every N actions
DISCOVERY_BUDGET = 8
