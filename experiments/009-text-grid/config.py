"""Experiment 009: Text grid — raw grid as compact text + two-phase observe-then-act

Builds on experiment 008's lightweight approach but adopts ideas from GuidedLLM:
send a sampled text grid (every 5th cell, 13x13) so the model sees actual cell
values at movement resolution, add an analysis field for two-phase reasoning
within one API call, and include both sampled grid and symbolic objects.

Tests whether raw spatial data at movement resolution improves navigation
and game understanding compared to symbolic-only perception.
"""

AGENT = "explorer"
GAME = "ls20"
MAX_ACTIONS = 100
