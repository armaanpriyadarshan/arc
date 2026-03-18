"""Action effect analysis via LLM.

After a short random exploration phase (~50 steps), this module sends a
representative sample of transitions to Claude and asks which actions appear
most useful. The result biases initial exploration in the training loop but
does not hard-filter any actions — the RL agent can still discover that an
action is useful even if Claude initially rated it low.

One API call per game.
"""

import json
import logging
from typing import Optional

import anthropic
import numpy as np

from .reward_shaping import format_frame_compact

logger = logging.getLogger(__name__)

# Default usefulness score for actions Claude did not mention
_DEFAULT_SCORE = 0.5


def analyze_action_effects(
    transitions: list[tuple],
    game_name: str,
    model: str = "claude-sonnet-4-20250514",
) -> dict[int, float]:
    """Ask the LLM which actions appear useful after a random exploration phase.

    Sends a compact summary of sampled (obs, action, next_obs) transitions
    to Claude and requests a JSON mapping of action indices to estimated
    usefulness scores.

    This is a single API call per game; the result is used to weight initial
    exploration but never to disable actions entirely.

    Args:
        transitions: List of (obs, action, next_obs) tuples where ``obs`` and
                     ``next_obs`` are (64, 64) numpy arrays and ``action`` is
                     an integer 0-6.
        game_name: ARC-AGI-3 game identifier, included as prompt context.
        model: Anthropic model identifier to call.

    Returns:
        Dict mapping action index (int) to usefulness score in [0.0, 1.0].
        All 7 actions (0-6) are present in the returned dict. Actions not
        mentioned by the LLM receive a neutral score of 0.5.
    """
    client = anthropic.Anthropic()

    # Sample up to 20 transitions to keep the prompt manageable
    sample = transitions[:20] if len(transitions) > 20 else transitions

    # Group transitions by action to show the LLM effect patterns
    by_action: dict[int, list[tuple]] = {}
    for obs, action, next_obs in sample:
        by_action.setdefault(int(action), []).append((obs, next_obs))

    action_summaries = []
    action_names = {
        0: "RESET",
        1: "UP(ACTION1)",
        2: "DOWN(ACTION2)",
        3: "LEFT(ACTION3)",
        4: "RIGHT(ACTION4)",
        5: "INTERACT(ACTION5)",
        6: "CLICK(ACTION6)",
    }

    for action_idx in sorted(by_action.keys()):
        pairs = by_action[action_idx]
        n = len(pairs)
        changed = sum(
            1 for obs, next_obs in pairs
            if np.any(obs != next_obs)
        )
        change_rate = changed / n if n > 0 else 0.0

        # Show one example pair (downsampled)
        obs, next_obs = pairs[0]
        before_str = format_frame_compact(obs)
        after_str  = format_frame_compact(next_obs)

        action_summaries.append(
            f"Action {action_idx} ({action_names.get(action_idx, 'UNKNOWN')}):\n"
            f"  Tried {n} times, changed grid {changed}/{n} times "
            f"({change_rate:.0%} change rate)\n"
            f"  Example before (16x16):\n{before_str}\n"
            f"  Example after  (16x16):\n{after_str}"
        )

    if not action_summaries:
        logger.warning("action_analysis: no transitions to analyse, returning defaults")
        return {i: _DEFAULT_SCORE for i in range(7)}

    prompt = f"""You are analyzing an unknown grid game called '{game_name}'.
An agent took random actions for a short exploration phase. Here are the results:

{chr(10).join(action_summaries)}

Based on these observations, rate how useful each action seems for making progress
in this game. Actions that change the grid and seem to produce meaningful state
transitions are more useful than actions that have no effect.

Return a JSON object with integer action indices as keys and float scores 0.0-1.0
as values. Include all actions 0-6. Example format:
{{"0": 0.1, "1": 0.9, "2": 0.8, "3": 0.7, "4": 0.7, "5": 0.5, "6": 0.3}}

Return ONLY the JSON object, no explanation.
"""

    logger.info("LLM action analysis: calling %s for game=%s", model, game_name)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("action_analysis: JSON parse failed: %s — raw: %r", exc, raw)
        return {i: _DEFAULT_SCORE for i in range(7)}
    except Exception as exc:
        logger.warning("action_analysis: API call failed: %s", exc)
        return {i: _DEFAULT_SCORE for i in range(7)}

    # Parse and validate — ensure all 7 actions are present and scores are clipped
    result: dict[int, float] = {}
    for i in range(7):
        key = str(i)
        try:
            score = float(data.get(key, _DEFAULT_SCORE))
            result[i] = max(0.0, min(1.0, score))
        except (TypeError, ValueError):
            result[i] = _DEFAULT_SCORE

    logger.info("LLM action analysis: scores=%s", result)
    return result
