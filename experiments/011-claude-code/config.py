"""Experiment 011-claude-code: Claude Code as reasoning engine

Replaces direct OpenAI API calls with Claude Code CLI subprocess calls.
The game loop remains in Python (arcengine), but instead of calling
client.responses.create(), we call `claude -p` with a prompt that
references a run log file on disk. Claude Code reads and analyzes
the log using its built-in Read/Grep/Bash tools, then outputs an
action plan.

Architecture:
- Python game loop (arcengine) manages the environment
- RunLog writes structured turn data to disk after each action
- ClaudeCodeAnalyzer calls `claude -p` to analyze the log
- Action queue (RGB-Agent style) batches and drains planned actions
- Symbolic state, vision, auto-probe from v5
"""

AGENT = "explorer"
GAME = "ls20"
MAX_ACTIONS = 200
PLAN_SIZE = 5
