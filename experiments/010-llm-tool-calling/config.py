"""Experiment 010: RGB-Agent on synthetic games (Docker + OpenCode)

Uses the actual RGB-Agent code unmodified: GameRunner, GameState, ActionQueue,
OpenCodeAgent (Docker sandbox with OpenCode CLI). The only custom code is run.py
which wires the components together and adds optional pygame visualization.

The RGB-Agent analyzer:
- Writes a growing prompt log file with full game history
- OpenCode (inside Docker) reads the log with Read/Grep/Python tools
- Returns [PLAN] and [ACTIONS] sections with batched action plans
- Maintains cell-level grid diffs, state-action memory, click target analysis

References:
- RGB-Agent: https://github.com/alexisfox7/RGB-Agent
- Experiment 004 (current best): observe-then-act with auto-probe + symbolic state
"""

AGENT = "rgb_agent"
GAME = "wiring"
MAX_ACTIONS = 500
MODEL = "openai/gpt-5.4"
