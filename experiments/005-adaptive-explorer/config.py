"""Experiment 005 — Adaptive Explorer agent configuration."""

EXPERIMENT = "005-adaptive-explorer"
AGENT = "adaptive"
GAME = "ls20"
SCORECARD_ID = "00def162-09ba-411d-bf36-8a7a28d4dc5f"  # v5 run5 (best: 1/7)
RESULT = "1/7 levels via BFS exploration (v5). Combo search and random exploration failed. Multi-lock levels need visual pattern matching."

CONFIG = {
    "grid_size": 64,
    "num_colors": 16,
    "max_episode_steps": 500,
    "expected_step_size": 5,
    "proximity_threshold": 3,
}
