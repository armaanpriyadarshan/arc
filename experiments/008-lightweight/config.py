"""Experiment 008: Lightweight agent — no DAG, no structured outputs

Tests whether a simple GPT-5.4 agent with low reasoning effort performs
comparably to the heavy DAG architecture from experiment 007. If the
bottleneck is strategic (wrong target selection) rather than tactical
(bad routes), then all the DAG overhead is wasted.

Same infrastructure as 007 (auto-probe, enhanced symbolic state,
conversational sliding window, action history, sandbox) but with a
minimal output format and no mandatory code.
"""

AGENT = "explorer"
GAME = "ls20"
MAX_ACTIONS = 100
