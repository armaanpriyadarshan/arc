"""Experiment 009: OPAL — Observe, Perceive, Act, Loop

A reactive agent that minimizes LLM calls by:
1. Observe: Auto-probe all actions, classify effect types (movement/toggle/transform/parameterized/null)
2. Perceive: One LLM call to interpret the game and declare progress indicators
3. Act: Code executes the LLM's action sequence
4. Loop: LLM called again only on surprise, stagnation, or sequence completion

Key innovations over experiments 004-008:
- Progress indicators: LLM declares what progress looks like upfront; code monitors them
- Stagnation detection: N actions with no indicator movement forces LLM course correction
- No DAG, no structured outputs — LLM gets full reasoning ability back
- Game-agnostic action classification works for any action type (not just 4-directional)

References:
- Exp 003: GPT-5.4 + symbolic state is the right combo (1/7 best)
- Exp 004: Auto-probe eliminates direction confusion; simplicity wins
- Exp 005: Optional code doesn't work; hypothesis competition improves mechanic ID
- Exp 007: q6 verification dropped BLOCKED 65%; strategic target selection is bottleneck
- Exp 008: DAG overhead may be wasted if bottleneck is strategic
"""

AGENT = "opal"
GAME = "ls20"
MAX_ACTIONS = 100
