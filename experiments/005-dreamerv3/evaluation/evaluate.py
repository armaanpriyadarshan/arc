"""Evaluation utilities for the DreamerV3 agent.

Provides ``evaluate_agent`` for running the current policy in the real
environment (no training, no exploration noise) and ``log_training_stats``
for structured logging of training metrics.
"""

import logging
from typing import Callable

import numpy as np
import torch

logger = logging.getLogger(__name__)


def evaluate_agent(
    env,
    actor,
    world_model,
    preprocess_fn: Callable,
    num_episodes: int = 5,
    max_steps: int = 200,
) -> dict:
    """Run the actor in the real environment with no training or exploration.

    Each episode starts from a fresh reset. The actor selects actions
    greedily (using ``sample``, which still uses stochastic sampling — for
    a fully greedy policy, set temperature to 0 in the actor). The RSSM
    state is reset at the beginning of each episode.

    Args:
        env: ``ARCEnvWrapper`` instance. Must support ``reset()`` and
             ``step(action_type, x, y)``.
        actor: ``HierarchicalActor`` — policy network.
        world_model: ``WorldModel`` — provides encoder and RSSM for
                     observation encoding during rollout.
        preprocess_fn: Callable that converts a (64, 64) int array to a
                       (16, 64, 64) float tensor (e.g. ``preprocess_observation``).
        num_episodes: Number of evaluation episodes to run (default 5).
        max_steps: Maximum steps per episode before forced termination.

    Returns:
        Dict containing:
            ``mean_return``           — average episode return across episodes.
            ``max_return``            — best return seen in this eval batch.
            ``mean_steps``            — average steps taken per episode.
            ``max_levels_completed``  — highest ``levels_completed`` seen.
            ``win_rate``              — fraction of episodes that ended in WIN.
    """
    device = next(actor.parameters()).device
    num_actions = actor.num_actions

    episode_returns: list[float] = []
    episode_steps: list[int] = []
    levels_completed_list: list[int] = []
    wins: int = 0

    actor.eval()
    with torch.no_grad():
        for ep in range(num_episodes):
            obs = env.reset()
            h, z = world_model.rssm.initial_state(1, device)
            prev_action = torch.zeros(1, dtype=torch.long, device=device)

            ep_return = 0.0
            steps = 0
            info: dict = {"levels_completed": 0, "state": "NOT_STARTED"}

            for step in range(max_steps):
                # Encode current observation
                obs_tensor = preprocess_fn(obs).unsqueeze(0).to(device)  # (1, 16, 64, 64)
                embed = world_model.encoder(obs_tensor)                  # (1, embed_dim)

                # Advance RSSM state with the real observation
                h, z, _, _ = world_model.rssm.observe_step(h, z, prev_action, embed)
                latent = world_model.rssm.get_latent(h, z)              # (1, latent_dim)

                # Sample action from actor
                action_type, x, y, _, _ = actor.sample(latent)          # scalars

                action_int = int(action_type.item())
                x_int = int(x.item())
                y_int = int(y.item())

                obs, reward, terminated, truncated, info = env.step(action_int, x_int, y_int)
                ep_return += reward
                steps += 1

                prev_action = action_type  # (1,) long tensor

                if terminated or truncated:
                    if info.get("state") == "WIN":
                        wins += 1
                    break

            episode_returns.append(ep_return)
            episode_steps.append(steps)
            levels_completed_list.append(info.get("levels_completed", 0))

            logger.info(
                "Eval ep %d/%d: return=%.2f steps=%d state=%s",
                ep + 1, num_episodes,
                ep_return, steps,
                info.get("state", "UNKNOWN"),
            )

    actor.train()

    results = {
        "mean_return":          float(np.mean(episode_returns)),
        "max_return":           float(np.max(episode_returns)),
        "mean_steps":           float(np.mean(episode_steps)),
        "max_levels_completed": int(np.max(levels_completed_list)),
        "win_rate":             wins / max(num_episodes, 1),
    }

    logger.info(
        "Eval summary: mean_return=%.2f max_return=%.2f mean_steps=%.1f "
        "max_levels=%d win_rate=%.0f%%",
        results["mean_return"],
        results["max_return"],
        results["mean_steps"],
        results["max_levels_completed"],
        results["win_rate"] * 100,
    )

    return results


def log_training_stats(iteration: int, stats: dict) -> None:
    """Log a dict of training statistics as a single structured log line.

    Formats float values to 4 decimal places and integer/string values as-is.
    Intended for use in the main training loop at ``log_interval``.

    Args:
        iteration: Current training iteration number.
        stats: Dict of metric names to scalar values.
    """
    parts = [f"iter={iteration}"]
    for key, value in stats.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    logger.info(" | ".join(parts))
