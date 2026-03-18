"""Gymnasium-style wrapper around the ARC-AGI-3 environment.

Presents a clean gym-like interface (reset / step) regardless of the
underlying SDK details.  Handles observation preprocessing and optional
reward shaping.

Key design decisions
--------------------
* Observations are returned as (64, 64) numpy int arrays (color index 0-15).
* ``preprocess_observation`` converts that to a (16, 64, 64) float one-hot
  tensor ready for the CNN encoder.
* Rewards are derived from environment state transitions. An optional
  ``shaped_reward_fn`` can be attached to augment or replace the reward.
* ``step()`` accepts an integer action index (0-6) plus optional (x, y)
  coordinates for click-type actions. It translates these into ``GameAction``
  objects using the real SDK API (``set_data`` for coordinates).

SDK notes (from experiments/003-tool-use and ARC-AGI-3-Agents reference):
- ``arcade.make(game_id)`` returns an ``EnvironmentWrapper``.
- ``env.step(action)`` returns ``FrameDataRaw | None``.
- ``FrameDataRaw.frame`` is a ``list[ndarray]`` (private attr via property).
- Color indices live in the first channel of each 64×64 ndarray layer.
- ``GameAction.ACTION6.set_data({'game_id': ..., 'x': int, 'y': int})``
  attaches spatial coordinates before stepping.
"""

import logging
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from arcengine import FrameData, FrameDataRaw, GameAction, GameState

logger = logging.getLogger(__name__)

# Mapping from integer action indices to GameAction enum members.
# Index 0 = RESET, indices 1-4 = directional actions.
# Additional actions (ACTION5-7) can be added per-game if available_actions includes them.
ACTION_MAP: dict[int, GameAction] = {
    0: GameAction.RESET,
    1: GameAction.ACTION1,
    2: GameAction.ACTION2,
    3: GameAction.ACTION3,
    4: GameAction.ACTION4,
}

# Click action support (ACTION6) is disabled by default — ls20 doesn't use it.
# For games that support click, extend ACTION_MAP and set CLICK_ACTION_IDX.
CLICK_ACTION_IDX = -1  # No click action in current config


def preprocess_observation(obs: np.ndarray) -> torch.Tensor:
    """Convert a (64, 64) integer color grid to a (16, 64, 64) float one-hot tensor.

    Each cell's integer color index (0-15) is expanded into a 16-dimensional
    one-hot vector, producing a tensor suitable as CNN encoder input.

    Args:
        obs: (64, 64) int64 NumPy array with color values in [0, 15].

    Returns:
        (16, 64, 64) float32 torch.Tensor.
    """
    one_hot = np.eye(16, dtype=np.float32)[obs]          # (64, 64, 16)
    return torch.from_numpy(one_hot.transpose(2, 0, 1))  # (16, 64, 64)


class ARCEnvWrapper:
    """Gym-style wrapper around the arc_agi SDK environment.

    Manages a single ``EnvironmentWrapper`` instance obtained from
    ``Arcade.make()``.  Translates between integer action indices and
    ``GameAction`` enum values, extracts 64×64 grid observations from
    ``FrameDataRaw``, and delegates optional shaped rewards to a user-
    supplied callable.

    Args:
        game_id: ARC-AGI-3 game identifier (e.g. ``"ls20"``).
        shaped_reward_fn: Optional callable with signature
            ``(prev_obs, action_type, obs, env_reward) -> float`` that
            provides shaped rewards. If None, the default environment
            reward (WIN=+10, GAME_OVER=-1, otherwise 0) is used.
        scorecard_id: Optional scorecard ID for tracking runs.
    """

    def __init__(
        self,
        game_id: str,
        shaped_reward_fn: Optional[Callable] = None,
        scorecard_id: Optional[str] = None,
    ) -> None:
        from arc_agi import Arcade

        self.game_id = game_id
        self.shaped_reward_fn = shaped_reward_fn

        self._arcade = Arcade()
        self._env = self._arcade.make(game_id, scorecard_id=scorecard_id)
        if self._env is None:
            raise RuntimeError(
                f"arc_agi.Arcade.make() returned None for game_id={game_id!r}. "
                "Check that the environment file exists and ARC_API_KEY is set."
            )

        self._prev_obs: Optional[np.ndarray] = None
        self._last_frame: Optional[FrameData] = None
        self._prev_levels_completed: int = 0
        # Set of valid action indices discovered from first frame's available_actions.
        # Populated on first reset; used to filter random exploration.
        self._valid_action_indices: Optional[set[int]] = None

        logger.info("ARCEnvWrapper: game=%s scorecard=%s", game_id, scorecard_id)

    # ------------------------------------------------------------------
    # Core gym-like interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial observation.

        Sends ``GameAction.RESET`` and converts the resulting frame into a
        (64, 64) integer NumPy array.

        Returns:
            obs: (64, 64) int64 NumPy array with color indices 0-15.
        """
        raw = self._safe_step(GameAction.RESET)
        frame = self._to_frame_data(raw)
        obs = self._extract_grid(frame)
        self._prev_obs = obs
        self._last_frame = frame

        # Discover valid actions from the first frame.
        # available_actions is a list of ints (GameAction enum values, e.g. [1,2,3,4]).
        if self._valid_action_indices is None and frame.available_actions:
            self._valid_action_indices = {0}  # RESET is always valid
            for action_val in frame.available_actions:
                val = action_val.value if hasattr(action_val, 'value') else int(action_val)
                for idx, mapped_ga in ACTION_MAP.items():
                    if mapped_ga.value == val:
                        self._valid_action_indices.add(idx)
                        break
            logger.info(
                "Discovered valid action indices: %s (from env available_actions=%s)",
                sorted(self._valid_action_indices), frame.available_actions,
            )

        return obs

    @property
    def num_valid_actions(self) -> int:
        """Number of valid actions for this game (excluding unavailable ones)."""
        if self._valid_action_indices is not None:
            return len(self._valid_action_indices)
        return len(ACTION_MAP)

    @property
    def valid_action_indices(self) -> list[int]:
        """Sorted list of valid action indices for this game."""
        if self._valid_action_indices is not None:
            return sorted(self._valid_action_indices)
        return list(range(len(ACTION_MAP)))

    def step(
        self,
        action_type: int,
        x: int = 0,
        y: int = 0,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one action and return the transition tuple.

        Args:
            action_type: Integer 0-6 mapping to ``ACTION_MAP``.
            x: Grid x-coordinate (0-63) for ACTION6; ignored otherwise.
            y: Grid y-coordinate (0-63) for ACTION6; ignored otherwise.

        Returns:
            obs:        (64, 64) int64 NumPy array.
            reward:     Scalar float reward (shaped if ``shaped_reward_fn`` set).
            terminated: True when episode has ended (WIN or GAME_OVER).
            truncated:  Always False (no time-limit truncation at this level).
            info:       Dict with ``state``, ``levels_completed``, ``env_reward``.
        """
        game_action = self._build_game_action(action_type, x, y)
        raw = self._safe_step(game_action)
        frame = self._to_frame_data(raw)
        obs = self._extract_grid(frame)

        env_reward = self._compute_env_reward(frame)
        if self.shaped_reward_fn is not None:
            try:
                reward = float(
                    self.shaped_reward_fn(self._prev_obs, action_type, obs, env_reward)
                )
            except Exception as exc:
                logger.warning("shaped_reward_fn raised: %s", exc)
                reward = env_reward
        else:
            reward = env_reward

        terminated = frame.state in (GameState.WIN, GameState.GAME_OVER)
        info = {
            "state": frame.state.name if frame.state else "UNKNOWN",
            "levels_completed": frame.levels_completed,
            "env_reward": env_reward,
        }

        self._prev_obs = obs
        self._last_frame = frame
        return obs, reward, terminated, False, info

    def set_shaped_reward(self, fn: Callable) -> None:
        """Replace or attach a reward shaping callable.

        Args:
            fn: Callable with signature
                ``(prev_obs, action_type, curr_obs, env_reward) -> float``.
        """
        self.shaped_reward_fn = fn
        logger.info("ARCEnvWrapper: shaped_reward_fn updated")

    def close(self) -> None:
        """Close the scorecard and release resources."""
        try:
            self._arcade.close_scorecard()
        except Exception as exc:
            logger.warning("ARCEnvWrapper.close(): %s", exc)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _safe_step(self, game_action: GameAction) -> Optional[FrameDataRaw]:
        """Call env.step() and catch exceptions, returning None on failure.

        Args:
            game_action: GameAction enum value to execute.

        Returns:
            FrameDataRaw from the SDK, or None on error.
        """
        try:
            return self._env.step(game_action)
        except Exception as exc:
            logger.warning("env.step(%s) raised: %s", game_action, exc)
            return None

    def _to_frame_data(self, raw: Optional[FrameDataRaw]) -> FrameData:
        """Convert a raw SDK frame to a FrameData object.

        Falls back to the last known frame or a minimal stub if ``raw`` is
        None (SDK returned nothing).

        Args:
            raw: FrameDataRaw from env.step(), possibly None.

        Returns:
            Populated FrameData instance.
        """
        if raw is None:
            logger.warning("env.step() returned None; reusing last frame or stub.")
            if self._last_frame is not None:
                return self._last_frame
            return FrameData(levels_completed=0)

        return FrameData(
            game_id=getattr(raw, "game_id", self.game_id),
            frame=[arr.tolist() for arr in raw.frame],
            state=raw.state,
            levels_completed=raw.levels_completed,
            win_levels=getattr(raw, "win_levels", None),
            guid=getattr(raw, "guid", ""),
            full_reset=getattr(raw, "full_reset", False),
            available_actions=getattr(raw, "available_actions", []),
        )

    def _extract_grid(self, frame: FrameData) -> np.ndarray:
        """Extract a (64, 64) integer color grid from a FrameData object.

        ``FrameData.frame`` is a ``list[list[list[int]]]`` after the
        ``tolist()`` conversion in ``_to_frame_data``.  The last element is
        the most recently rendered layer.

        Args:
            frame: FrameData as produced by ``_to_frame_data``.

        Returns:
            (64, 64) int64 NumPy array with color values in [0, 15].
        """
        if not frame.frame:
            return np.zeros((64, 64), dtype=np.int64)

        layer = frame.frame[-1]
        grid = np.array(layer, dtype=np.int64)

        # Guard against multi-channel layers: keep only the first channel.
        if grid.ndim == 3:
            grid = grid[:, :, 0]

        return grid  # (64, 64)

    def _compute_env_reward(self, frame: FrameData) -> float:
        """Derive a scalar reward from the environment state.

        Rewards:
            +10.0 on WIN
            +5.0 per new level completed
            -1.0 on GAME_OVER
            0.0 otherwise

        Args:
            frame: FrameData from the current step.

        Returns:
            Scalar reward.
        """
        reward = 0.0

        if frame.state == GameState.WIN:
            reward += 10.0
        elif frame.state == GameState.GAME_OVER:
            reward -= 1.0

        # Big reward for level completion progress
        current_levels = frame.levels_completed or 0
        if current_levels > self._prev_levels_completed:
            levels_gained = current_levels - self._prev_levels_completed
            reward += 5.0 * levels_gained
            logger.info(
                "LEVEL PROGRESS: %d -> %d (+%.1f reward)",
                self._prev_levels_completed, current_levels, 5.0 * levels_gained,
            )
            self._prev_levels_completed = current_levels

        return reward

    def _build_game_action(self, action_type: int, x: int = 0, y: int = 0) -> GameAction:
        """Translate an integer action index to a GameAction.

        Args:
            action_type: Integer action index. Clamped to valid ACTION_MAP range.
            x: X-coordinate for click actions (unused in most games).
            y: Y-coordinate for click actions (unused in most games).

        Returns:
            GameAction enum value.
        """
        idx = max(0, min(action_type, len(ACTION_MAP) - 1))
        return ACTION_MAP[idx]
