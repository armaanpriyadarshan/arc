"""Experiment 004-v5: Performance-optimized observe-act agent

Builds on v4 with 4 wall-clock optimizations:
1. Skip image generation when grid unchanged (send plain frame, not stale diff)
2. Memoize grid_to_symbolic() with LRU cache (maxsize=4)
5. Repeat clean navigation batches without LLM calls (directional-only, EXECUTE mode)
8. Parallelize image generation + symbolic analysis (ThreadPoolExecutor, cache-aware)

REVERTED: Optimization 9 (SCALE 8->4) degraded GPT-5.4 spatial reasoning,
causing shorter action batches and 50% more LLM calls.

These optimizations do NOT change the agent's decision-making logic
(except #5, which reduces LLM calls for routine navigation).
"""

AGENT = "explorer"
GAME = "ls20"
MAX_ACTIONS = 150
