"""Experiment 002: VLM-based agent with visual reasoning

Complete redesign from experiment 001 which scored 0/7.
001 failed because code-only perception couldn't identify the player,
background, or movement directions.

New approach: Let a VLM SEE the game as images.
- Phase 1: VLM analyzes initial frame + diffs from 4 test actions to understand the game
- Phase 2: VLM-guided action loop with visual diff feedback after every action
- Self-correction: when stuck (no progress for N actions), VLM re-analyzes the full situation

No hardcoded game knowledge. The VLM discovers everything visually.
"""

AGENT = "vlm_explorer"
GAME = "ls20"
MAX_ACTIONS = 200
MODEL = "gpt-4o"  # need vision capability
