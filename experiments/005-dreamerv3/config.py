"""Experiment 004 — DreamerV3 agent configuration.

All hyperparameters live here. Also serves as the experiment metadata
record (scorecard ID, result summary) per repo convention.
"""

EXPERIMENT = "004-dreamerv3"
AGENT = "dreamer"
GAME = "ls20"
SCORECARD_ID = "e3f7630a-91d2-414a-abd4-40a0b340ec6d"  # Perception run 4
RESULT = "DreamerV3: 0/7 across 5 runs. Perception agent: 1/7 in 500 actions. Code-based perception >> pure RL for ARC."

CONFIG = {
    # Environment
    "grid_size": 64,
    "num_colors": 16,
    "num_base_actions": 5,  # RESET(0) + ACTION1-4; games may use fewer

    # World Model - Encoder
    "encoder_channels": [32, 64, 128],

    # World Model - RSSM
    "gru_hidden_dim": 256,
    "stochastic_categories": 8,
    "stochastic_classes": 8,

    # World Model - Training
    "kl_free_bits": 0.1,   # was 1.0 — lowered to avoid stochastic state collapse
    "kl_balance_prior": 0.8,
    "kl_balance_posterior": 0.5,

    # Actor-Critic
    "imagination_horizon": 8,
    "gamma": 0.997,
    "lambda_": 0.95,
    "actor_entropy_coef": 3e-3,
    "critic_slow_ema": 0.98,
    "return_norm_percentile_low": 5,
    "return_norm_percentile_high": 95,
    "return_norm_decay": 0.99,

    # Training (tuned for CPU, single game)
    "replay_buffer_capacity": 50_000,
    "batch_size": 8,
    "sequence_length": 32,
    "train_ratio": 32,
    "world_model_lr": 3e-4,
    "actor_lr": 1e-4,
    "critic_lr": 1e-4,
    "max_grad_norm": 100.0,
    "seed": 42,

    # Collection
    "num_random_episodes": 20,
    "max_episode_steps": 500,
    "collect_interval": 50,  # world model train steps between collections

    # LLM
    "llm_model": "claude-sonnet-4-20250514",
    "llm_reward_shaping_enabled": True,
    "llm_diagnosis_patience": 100,
    "llm_max_calls_per_game": 15,

    # Logging
    "log_interval": 10,
    "eval_interval": 50,
    "save_interval": 100,
    "max_iterations": 5000,
}
