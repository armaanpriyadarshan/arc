"""Experiment 005: Program synthesis agent

Copy of experiment 004 (observe-act with auto-probe, symbolic state,
hypothesis-driven actions) with one key addition: the model can synthesize
and execute Python programs to build whatever representations it needs.

The model gets access to the current grid and can write programs to analyze
it however it wants -- build maps, track positions, compute paths, detect
patterns, whatever it considers important.

References experiment 004 analysis: the core architecture works well but
the model doesn't investigate game objects deeply enough or build rich
spatial representations. Program synthesis gives it the power to compute
whatever it needs.
"""

AGENT = "explorer"
GAME = "ls20"
MAX_ACTIONS = 100
REASONING_MODEL = "o4-mini"
VISION_MODEL = "gpt-4o"
