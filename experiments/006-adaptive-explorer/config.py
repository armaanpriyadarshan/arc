"""Experiment 005 — Adaptive Explorer agent configuration."""

EXPERIMENT = "005-adaptive-explorer"
AGENT = "adaptive"
GAME = "ls20"
SCORECARD_ID = None  # Set after first run
RESULT = ""

CONFIG = {
    # Environment
    "grid_size": 64,
    "num_colors": 16,

    # Agent
    "max_episode_steps": 500,
    "calibration_steps_per_action": 3,
    "exploration_budget_fraction": 0.6,  # 60% of actions for exploration, 40% for exploitation
    "interaction_test_budget": 30,  # actions to test each object interaction
    "min_energy_reserve": 20,  # stop exploring if fewer than this many steps remain before expected GAME_OVER

    # Movement
    "expected_step_size": 5,  # pixels per step (common in ARC games)
    "proximity_threshold": 3,  # pixels within which positions are "same"

    # Logging
    "log_interval": 10,
}
