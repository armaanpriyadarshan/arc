"""Experiment 003: Tool-use agent

The model drives its own investigation. Instead of pre-computing what
information to send, the model gets tools to query the grid, compute diffs,
check cell history, and execute actions. It decides what to ask for.

References experiment 002 analysis: the core bottleneck was that we kept
guessing what the model needed and getting it wrong. The model should
formulate its own programs and queries.
"""

AGENT = "explorer"
GAME = "ls20"
MAX_ACTIONS = 100
REASONING_MODEL = "o4-mini"
VISION_MODEL = "gpt-4o"
