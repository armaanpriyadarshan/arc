"""Experiment 006: Persistent state agent

Builds on experiment 005 (program synthesis with sandbox). Key change: structured
game state is auto-populated by code every turn, not left for the model to build.

Auto-populated sandbox variables:
- action_log: full history of every action taken [{turn, action, blocked, cells_changed, large_change}]
- player_pos: approximate [row, col] of the controllable object (auto-updated)
- last_trigger: details of the most recent large change (>100 cells), including changed cells
- trigger_history: list of all trigger events [{turn, total_changed}]

The model also sees a compact action summary instead of raw recent action text,
and trigger diffs are auto-computed on large changes.

References experiments 004-005: core architecture (probe, symbolic, hypothesis-driven)
works well but the model wastes actions re-discovering state that code could track
automatically.
"""

AGENT = "explorer"
GAME = "ls20"
MAX_ACTIONS = 100
REASONING_MODEL = "o4-mini"
VISION_MODEL = "gpt-4o"
