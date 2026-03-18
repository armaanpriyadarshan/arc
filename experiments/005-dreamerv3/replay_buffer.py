"""Replay buffer storing complete episodes for sequence sampling.

DreamerV3 trains on contiguous sequences from past experience. This buffer
stores episodes as dictionaries of stacked numpy arrays and provides two
sampling methods:

* ``sample_sequences`` — draws (batch_size, seq_len) contiguous slices for
  world model training, returning torch tensors with a validity mask.
* ``sample_states`` — draws individual (64, 64) observations for seeding
  imagined rollouts in the actor-critic update.

Capacity is measured in total environment steps. The oldest episodes are
evicted when the buffer exceeds ``capacity`` steps.
"""

import logging
import random
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """FIFO episode buffer with step-count-based capacity enforcement.

    Episodes are stored in insertion order as dicts of stacked arrays.
    When total stored steps exceed ``capacity``, the oldest episode is
    evicted until the constraint is satisfied.

    Args:
        capacity: Maximum total environment steps to retain across all
                  stored episodes.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.episodes: list[dict[str, Any]] = []
        self.total_steps: int = 0

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add_episode(
        self,
        observations: list[np.ndarray],
        actions: list[int],
        rewards: list[float],
        dones: list[bool],
    ) -> None:
        """Store a completed episode.

        Args:
            observations: Length-T+1 list of (64, 64) int arrays (includes the
                          initial observation and the final observation after the
                          last action).
            actions: Length-T list of integer action indices.
            rewards: Length-T list of scalar rewards.
            dones: Length-T list of done flags (True = episode ended).

        Notes:
            Episodes with fewer than 2 observations are silently discarded
            as they cannot form any valid transition.
        """
        if len(observations) < 2:
            logger.warning(
                "Replay buffer: episode too short (%d obs), skipping.",
                len(observations),
            )
            return

        ep_len = len(actions)
        episode: dict[str, Any] = {
            "observations": np.stack(observations, axis=0).astype(np.int64),  # (T+1, 64, 64)
            "actions": np.array(actions, dtype=np.int64),                     # (T,)
            "rewards": np.array(rewards, dtype=np.float32),                   # (T,)
            "dones": np.array(dones, dtype=np.float32),                       # (T,)
        }

        self.episodes.append(episode)
        self.total_steps += ep_len

        # Evict oldest episodes until we are within the step budget.
        while self.total_steps > self.capacity and len(self.episodes) > 1:
            removed = self.episodes.pop(0)
            self.total_steps -= len(removed["actions"])

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_sequences(
        self,
        batch_size: int,
        sequence_length: int,
    ) -> dict[str, torch.Tensor]:
        """Sample a batch of contiguous observation-action sequences.

        Each sequence is drawn independently: a random episode is chosen
        (weighted by length so longer episodes are sampled proportionally),
        then a random start index within that episode is chosen. Slices that
        extend past the end of the episode are zero-padded; a boolean mask
        tensor indicates which timesteps are real (True) vs. padded (False).

        Args:
            batch_size: Number of sequences (B) in the returned batch.
            sequence_length: Length (T) of each returned sequence.

        Returns:
            Dictionary with the following torch.Tensor values:
                ``observations`` — (B, T+1, 64, 64) int64, T+1 frames per seq
                ``actions``      — (B, T) int64
                ``rewards``      — (B, T) float32
                ``dones``        — (B, T) float32 (1.0 = done, 0.0 = not done)
                ``mask``         — (B, T) bool (True = real, False = padded)
        """
        if not self.episodes:
            raise RuntimeError("Cannot sample from an empty ReplayBuffer.")

        # Build length weights for episode-proportional sampling.
        lengths = np.array([len(ep["actions"]) for ep in self.episodes], dtype=np.float64)
        weights = lengths / lengths.sum()

        obs_seqs: list[np.ndarray] = []
        act_seqs: list[np.ndarray] = []
        rew_seqs: list[np.ndarray] = []
        done_seqs: list[np.ndarray] = []
        masks: list[np.ndarray] = []

        for _ in range(batch_size):
            # Sample an episode proportional to its length.
            ep_idx = int(np.random.choice(len(self.episodes), p=weights))
            ep = self.episodes[ep_idx]

            ep_len = len(ep["actions"])  # T for this episode

            # Random start index: start can be anywhere from 0 to ep_len-1.
            start = random.randint(0, max(0, ep_len - 1))
            end = start + sequence_length  # exclusive end for actions/rewards/dones

            # Observation slice: we need sequence_length+1 frames starting at start.
            # Frames are indexed 0..T (T+1 total in the stored episode).
            obs_end = start + sequence_length + 1
            real_obs_len = min(obs_end, len(ep["observations"])) - start
            real_act_len = min(end, ep_len) - start

            # Allocate output arrays (zero-padded by default).
            obs_seq = np.zeros((sequence_length + 1, 64, 64), dtype=np.int64)
            act_seq = np.zeros(sequence_length, dtype=np.int64)
            rew_seq = np.zeros(sequence_length, dtype=np.float32)
            done_seq = np.zeros(sequence_length, dtype=np.float32)
            mask = np.zeros(sequence_length, dtype=bool)

            # Fill real observations.
            obs_seq[:real_obs_len] = ep["observations"][start : start + real_obs_len]
            # Repeat last real observation to fill any observation padding.
            if real_obs_len < sequence_length + 1:
                last_real_obs = ep["observations"][start + real_obs_len - 1]
                obs_seq[real_obs_len:] = last_real_obs

            # Fill real transitions.
            act_seq[:real_act_len] = ep["actions"][start : start + real_act_len]
            rew_seq[:real_act_len] = ep["rewards"][start : start + real_act_len]
            done_seq[:real_act_len] = ep["dones"][start : start + real_act_len]
            mask[:real_act_len] = True

            obs_seqs.append(obs_seq)
            act_seqs.append(act_seq)
            rew_seqs.append(rew_seq)
            done_seqs.append(done_seq)
            masks.append(mask)

        return {
            "observations": torch.from_numpy(np.stack(obs_seqs, axis=0)),  # (B, T+1, 64, 64)
            "actions":      torch.from_numpy(np.stack(act_seqs, axis=0)),  # (B, T)
            "rewards":      torch.from_numpy(np.stack(rew_seqs, axis=0)),  # (B, T)
            "dones":        torch.from_numpy(np.stack(done_seqs, axis=0)), # (B, T)
            "mask":         torch.from_numpy(np.stack(masks, axis=0)),     # (B, T)
        }

    def sample_states(self, batch_size: int) -> torch.Tensor:
        """Sample individual observation frames for seeding imagination rollouts.

        Picks random (episode, timestep) pairs uniformly and returns the
        corresponding (64, 64) observation grid.

        Args:
            batch_size: Number of observation frames to return.

        Returns:
            (batch_size, 64, 64) int64 torch.Tensor.

        Raises:
            RuntimeError: If the buffer contains no episodes.
        """
        if not self.episodes:
            raise RuntimeError("Cannot sample from an empty ReplayBuffer.")

        frames: list[np.ndarray] = []
        for _ in range(batch_size):
            ep = random.choice(self.episodes)
            # observations has shape (T+1, 64, 64); any index is valid.
            t = random.randint(0, len(ep["observations"]) - 1)
            frames.append(ep["observations"][t])

        return torch.from_numpy(np.stack(frames, axis=0))  # (B, 64, 64)

    # ------------------------------------------------------------------
    # Dunders / properties
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of stored environment steps."""
        return self.total_steps
