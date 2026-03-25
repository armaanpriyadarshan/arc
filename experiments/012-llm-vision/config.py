"""Experiment 012: LLM Vision — replace symbolic.py with GPT-5.4 vision summarizer.

Instead of code-based connected component analysis, a dedicated GPT-5.4 call
looks at the diff image and produces a structured JSON scene description.
That summary feeds into the reasoning agent (same v5 hypothesis-driven loop).

The reasoning agent also gets:
- The diff image directly (can verify/augment the summary)
- Text board access via read_board/grep_board tools (raw 64x64 hex grid)
"""

AGENT = "explorer"
GAME = "ls20"
MAX_ACTIONS = 150
