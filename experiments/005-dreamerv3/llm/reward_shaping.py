"""LLM-generated reward shaping functions.

Sends a compact representation of early gameplay frames to Claude and asks it
to write a Python reward shaping function. The returned callable is then used
throughout training to augment the sparse environment reward.

Usage pattern — call once per game, not per step:

    shaped_fn = generate_reward_shaping_function(initial_frames, game_name)
    env.set_shaped_reward(shaped_fn)
"""

import logging

import anthropic
import numpy as np

logger = logging.getLogger(__name__)


def generate_reward_shaping_function(
    initial_frames: list[np.ndarray],
    game_name: str,
    model: str = "claude-sonnet-4-20250514",
) -> callable:
    """Send 3-5 initial observation frames to Claude and get a reward shaping function.

    Encodes the frames as compact 16x16 hex grids, prompts Claude to analyse
    the game and write a Python reward shaping function, then compiles and
    returns it. The compiled function is cached — call this once per game.

    Args:
        initial_frames: List of 3-5 (64, 64) numpy arrays (color index 0-15),
                        taken from the first few steps of random exploration.
        game_name: ARC-AGI-3 game identifier (e.g. ``"ls20"``). Used only as
                   context in the prompt.
        model: Anthropic model identifier to call.

    Returns:
        A callable with signature:
            ``shaped_reward(prev_obs, action, curr_obs, env_reward) -> float``
        where ``prev_obs`` and ``curr_obs`` are (64, 64) numpy arrays,
        ``action`` is an integer 0-6, and ``env_reward`` is the raw reward.

    Raises:
        anthropic.APIError: On API failures (let the caller handle or fall back).
    """
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

    # Limit to 5 frames to keep the prompt compact
    frames = initial_frames[:5]
    frame_descriptions = [
        f"Frame {i}:\n{format_frame_compact(f)}"
        for i, f in enumerate(frames)
    ]

    prompt = f"""You are analyzing frames from an interactive grid game called '{game_name}'.
Each frame is a 64x64 grid where each cell is a color (0-15).

Here are the first few frames from gameplay (each downsampled to 16x16 for readability,
hex digits represent color values):
{chr(10).join(frame_descriptions)}

The agent can take these actions:
  0=RESET, 1=UP(ACTION1), 2=DOWN(ACTION2), 3=LEFT(ACTION3), 4=RIGHT(ACTION4),
  5=INTERACT(ACTION5), 6=CLICK(ACTION6, with x/y coords)

Analyze what might be happening in this game based on the visual patterns. Then write
a Python reward shaping function that provides intermediate rewards to help an RL agent
learn from sparse environment signals.

The function should:
- Reward state changes (grid changed — something happened)
- Reward novel states (seeing new grid configurations for the first time)
- Penalise no-ops (action had zero effect on the grid)
- Optionally reward apparent progress indicators if you can identify any
- Scale extra rewards in the range [-0.1, +0.5] to avoid dominating the sparse env reward

Return ONLY the Python function, no explanation or markdown. Use only numpy (imported as np):

def shaped_reward(prev_obs: np.ndarray, action: int, curr_obs: np.ndarray, env_reward: float) -> float:
    # prev_obs and curr_obs are (64, 64) numpy arrays with integer values 0-15
    # action is an integer 0-6
    # env_reward is the environment's native reward (e.g. 10.0 for win, -1.0 for game over)
    ...
"""

    logger.info("LLM reward shaping: calling %s for game=%s", model, game_name)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    code = extract_python_code(response.content[0].text)
    logger.info("LLM reward shaping: received %d chars of code", len(code))

    namespace: dict = {"np": np}
    exec(code, namespace)  # noqa: S102

    if "shaped_reward" not in namespace:
        raise ValueError("LLM did not produce a 'shaped_reward' function")

    return namespace["shaped_reward"]


def format_frame_compact(obs: np.ndarray) -> str:
    """Downsample a 64x64 grid to 16x16 for efficient LLM consumption.

    Each 4x4 block is reduced to its most common color (mode). The result is
    formatted as space-separated hex digits (0-f) per row.

    Args:
        obs: (64, 64) numpy array with integer color values 0-15.

    Returns:
        16-line string, each line containing 16 space-separated hex digits.
    """
    downsampled = np.zeros((16, 16), dtype=int)
    for i in range(16):
        for j in range(16):
            block = obs[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4]
            values, counts = np.unique(block, return_counts=True)
            downsampled[i, j] = values[np.argmax(counts)]

    lines = []
    for row in downsampled:
        lines.append(" ".join(f"{c:x}" for c in row))
    return "\n".join(lines)


def extract_python_code(text: str) -> str:
    """Extract a Python code block from an LLM response.

    Handles three formats:
    - Fenced block: ````python ... ````
    - Plain fenced block: ```` ``` ... ``` ````
    - Raw code with no fencing

    Args:
        text: Full LLM response text.

    Returns:
        Stripped Python source code string.
    """
    if "```python" in text:
        code = text.split("```python")[1].split("```")[0]
    elif "```" in text:
        code = text.split("```")[1].split("```")[0]
    else:
        code = text
    return code.strip()


class ExplorationRewardShaper:
    """Stateful reward shaper combining position-based novelty and curiosity.

    Reward components:
    - Environment rewards (WIN/GAME_OVER/level completion) passed through amplified
    - Position novelty: +0.2 for visiting a new 4x4 grid region
    - State novelty: +0.05 for a never-before-seen grid configuration
    - Systematic exploration bonus: +0.05 for changing direction (avoids wall-hugging)
    - No-op penalty: -0.05 for actions with no effect (walls, invalid moves)
    """

    def __init__(self):
        self._seen_states: set[int] = set()
        self._visited_positions: set[tuple[int, int]] = set()
        self._prev_action: int = -1
        self._step_count: int = 0

    def __call__(
        self,
        prev_obs: np.ndarray,
        action: int,
        curr_obs: np.ndarray,
        env_reward: float,
    ) -> float:
        self._step_count += 1

        # Pass through terminal rewards (amplified)
        if env_reward > 0:
            return env_reward * 2.0
        if env_reward < 0:
            return env_reward

        if prev_obs is None:
            return 0.0

        diff_mask = prev_obs != curr_obs
        diff = int(np.sum(diff_mask))

        # Strong no-op penalty: hitting walls should be costly
        if diff == 0:
            return -0.05

        reward = 0.0

        # Position tracking via change centroid
        changed_rows, changed_cols = np.where(diff_mask)
        if len(changed_rows) > 0:
            pos = (int(np.median(changed_rows)) // 4, int(np.median(changed_cols)) // 4)
            if pos not in self._visited_positions:
                self._visited_positions.add(pos)
                reward += 0.2  # significant reward for new positions

        # State novelty (smaller than position novelty)
        state_hash = hash(curr_obs.tobytes())
        if state_hash not in self._seen_states:
            self._seen_states.add(state_hash)
            reward += 0.05

        # Direction change bonus
        if action != self._prev_action and self._prev_action >= 0:
            reward += 0.05

        self._prev_action = action
        return reward

    def reset_episode(self):
        """Reset per-episode state. Keep global novelty for cross-episode learning."""
        self._visited_positions.clear()
        self._prev_action = -1
        self._step_count = 0


def default_shaped_reward(
    prev_obs: np.ndarray,
    action: int,
    curr_obs: np.ndarray,
    env_reward: float,
) -> float:
    """Simple stateless fallback reward shaping.

    Rewards grid changes proportional to the number of differing cells, and
    penalises no-ops. If the environment provides a non-zero reward (e.g. WIN
    or GAME_OVER), that value is returned unchanged.

    Args:
        prev_obs: (64, 64) numpy array before the action.
        action: Integer action index (unused in this implementation).
        curr_obs: (64, 64) numpy array after the action.
        env_reward: Raw reward from the environment.

    Returns:
        Shaped scalar reward.
    """
    if env_reward != 0.0:
        return env_reward

    if prev_obs is None:
        return 0.0

    diff = int(np.sum(prev_obs != curr_obs))
    if diff == 0:
        return -0.02  # penalise no-op

    return 0.02  # small reward for any change
