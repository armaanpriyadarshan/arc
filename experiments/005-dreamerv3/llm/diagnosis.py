"""Stuck-agent diagnosis via LLM.

When the agent's episode returns stop improving for a sustained period, this
module sends a diagnostic summary to Claude and asks for actionable suggestions.

The suggestions are logged and returned as a string for the agent to include in
subsequent log analysis. The agent itself does not act on the suggestions
programmatically — they are intended as hints for the human experimenter and as
context for future LLM calls.

Call at most every ``llm_diagnosis_patience`` episodes to bound API costs.
"""

import logging

import anthropic
import numpy as np

from .reward_shaping import format_frame_compact

logger = logging.getLogger(__name__)


def diagnose_stuck_agent(
    game_name: str,
    best_trajectory: list[tuple],
    episode_returns: list[float],
    model: str = "claude-sonnet-4-20250514",
) -> str:
    """Ask Claude to diagnose why the agent is stuck and suggest improvements.

    Summarises recent performance statistics and, if available, a sample of
    the best trajectory the agent has found so far. Sends the summary to
    Claude and returns its diagnostic text.

    Args:
        game_name: ARC-AGI-3 game identifier, included as context.
        best_trajectory: List of (obs, action, reward) tuples from the highest-
                         return episode seen so far. At most 10 steps are sent
                         to keep the prompt compact.
        episode_returns: Full list of episode returns seen during training,
                         used to compute trend statistics.
        model: Anthropic model identifier to call.

    Returns:
        Diagnostic text from Claude as a plain string, or a fallback message
        if the API call fails.
    """
    client = anthropic.Anthropic()

    # --- Compute return statistics ---
    returns_arr = np.array(episode_returns, dtype=np.float32)
    n = len(returns_arr)
    recent_n = min(20, n)
    recent_returns = returns_arr[-recent_n:]
    older_returns  = returns_arr[:-recent_n] if n > recent_n else returns_arr

    stats_lines = [
        f"Total episodes: {n}",
        f"All-time return: min={returns_arr.min():.2f}, "
        f"max={returns_arr.max():.2f}, mean={returns_arr.mean():.2f}",
        f"Recent {recent_n} episodes: "
        f"min={recent_returns.min():.2f}, "
        f"max={recent_returns.max():.2f}, "
        f"mean={recent_returns.mean():.2f}",
    ]
    if len(older_returns) > 0:
        trend = recent_returns.mean() - older_returns.mean()
        stats_lines.append(
            f"Trend vs earlier episodes: {trend:+.3f} "
            f"({'improving' if trend > 0 else 'declining or flat'})"
        )

    # --- Summarise best trajectory ---
    traj_sample = best_trajectory[:10]
    action_names = {
        0: "RESET", 1: "UP", 2: "DOWN", 3: "LEFT",
        4: "RIGHT", 5: "INTERACT", 6: "CLICK",
    }
    traj_lines = []
    for step_idx, entry in enumerate(traj_sample):
        obs, action, reward = entry
        action_name = action_names.get(int(action), f"ACTION{action}")
        obs_str = format_frame_compact(np.asarray(obs)) if obs is not None else "(none)"
        traj_lines.append(
            f"Step {step_idx}: action={action_name}, reward={reward:.2f}\n"
            f"Observation after action (16x16):\n{obs_str}"
        )

    traj_section = (
        "Best trajectory sample:\n" + "\n\n".join(traj_lines)
        if traj_lines
        else "No trajectory data available."
    )

    prompt = f"""You are an expert RL researcher analysing a stuck learning agent.
The agent is playing an unknown grid game called '{game_name}'.

Performance statistics:
{chr(10).join(stats_lines)}

{traj_section}

The agent has stopped improving. Based on the above information:

1. What do you think is going wrong? (1-2 sentences)
2. What specific changes to the reward shaping or exploration strategy might help?
3. Is there any pattern in the best trajectory that suggests what the game objective might be?

Be concise and actionable. Focus on practical suggestions for the RL training loop.
"""

    logger.info(
        "LLM diagnosis: calling %s for game=%s (episodes=%d)",
        model, game_name, n,
    )
    try:
        response = client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        diagnosis = response.content[0].text.strip()
        logger.info("LLM diagnosis:\n%s", diagnosis)
        return diagnosis
    except Exception as exc:
        msg = f"LLM diagnosis failed: {exc}"
        logger.warning(msg)
        return msg
