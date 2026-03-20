"""Experiment 007: Conversational sliding-window agent

Same core as experiment 005 (auto-probe, symbolic state, hypothesis-driven
actions, program synthesis) with one key architectural change: the agent
maintains a CONVERSATION with the model using a sliding window of the last
10 messages, instead of building one massive prompt per turn.

The model can reference its own prior reasoning without us re-injecting
hypotheses/notes. System prompt goes via `instructions` parameter.
Uses reasoning effort "high" and no temperature.

References experiment 005 analysis: stateless single-call means the model
can't naturally build on its own prior reasoning. Conversational context
gives it memory through the conversation itself.
"""

AGENT = "explorer"
GAME = "ls20"
MAX_ACTIONS = 100
REASONING_MODEL = "o4-mini"
VISION_MODEL = "gpt-4o"
