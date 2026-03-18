"""DQN agent with learned exploration for ARC-AGI-3.

Architecture:
1. Shared CNN encoder maps 64x64 one-hot grids to compact embeddings
2. Q-network predicts action values from embeddings
3. Action-effect predictor: binary classifier predicting if action changes state
4. SimHash state counter: learned hash → visitation count → exploration bonus
5. Reward = env_reward + novelty_bonus * (1/sqrt(visit_count))

This is designed to work where DreamerV3 failed:
- No world model (too much capacity wasted on reconstruction)
- Intrinsic reward from state counting (doesn't decay like RND)
- Action-effect prediction biases exploration toward useful actions
"""

import copy
import logging
import random
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class SharedEncoder(nn.Module):
    """CNN encoder shared by Q-network and hash network."""

    def __init__(self, in_channels: int = 16, embed_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # 64x64 input → 4x4 after convs → flatten = 1024
        self.fc = nn.Linear(1024, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).flatten(1))


class QNetwork(nn.Module):
    """DQN: encoder → Q-values per action."""

    def __init__(self, encoder: SharedEncoder, num_actions: int = 5):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder.embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns (B, num_actions) Q-values."""
        return self.head(self.encoder(obs))


class ActionEffectPredictor(nn.Module):
    """Predicts P(action changes state) — binary classifier.

    Input: observation embedding + action one-hot
    Output: sigmoid probability that the action will change the grid
    """

    def __init__(self, embed_dim: int = 256, num_actions: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + num_actions, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, embed: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        """Returns (B,) logits for P(change)."""
        return self.net(torch.cat([embed, action_onehot], dim=-1)).squeeze(-1)


class SimHasher(nn.Module):
    """Learned hash function for state counting.

    Maps embeddings to binary hash codes via a learned projection.
    States with similar embeddings get the same hash → counted together.
    """

    def __init__(self, embed_dim: int = 256, hash_bits: int = 32):
        super().__init__()
        self.proj = nn.Linear(embed_dim, hash_bits, bias=False)
        # Freeze — the random projection is the hash function
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        """Returns (B, hash_bits) binary codes."""
        return (self.proj(embed) > 0).float()

    def hash_key(self, embed: torch.Tensor) -> int:
        """Returns a hashable integer key for a single embedding."""
        bits = self.forward(embed.unsqueeze(0))[0]
        bit_list = bits.cpu().numpy().astype(int).flatten().tolist()
        return int("".join(str(b) for b in bit_list), 2)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class DQNReplayBuffer:
    """Simple replay buffer for DQN transitions."""

    def __init__(self, capacity: int = 50000):
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.stack(obs),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_obs),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(obs: np.ndarray) -> np.ndarray:
    """(64,64) int → (16,64,64) float32 one-hot."""
    return np.eye(16, dtype=np.float32)[obs].transpose(2, 0, 1)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """DQN agent with count-based exploration and action-effect prediction.

    Args:
        game_id: ARC-AGI-3 game identifier.
        config: Hyperparameter dict.
    """

    def __init__(self, game_id: str, config: Optional[dict] = None):
        from config import CONFIG
        self.config = {**CONFIG, **(config or {})}
        self.game_id = game_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_actions = self.config.get("num_base_actions", 5)

        # Models
        self.encoder = SharedEncoder(embed_dim=256).to(self.device)
        self.q_net = QNetwork(self.encoder, num_actions).to(self.device)
        self.target_q_net = copy.deepcopy(self.q_net).to(self.device)
        self.effect_pred = ActionEffectPredictor(256, num_actions).to(self.device)
        self.hasher = SimHasher(256, hash_bits=32).to(self.device)

        # Optimizers
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.effect_optimizer = torch.optim.Adam(self.effect_pred.parameters(), lr=1e-3)

        # Replay
        self.replay = DQNReplayBuffer(capacity=50000)
        self.effect_data: list[tuple] = []  # (obs, action, changed: bool)

        # State counting
        self.state_counts: dict[tuple, int] = {}

        # Exploration — slow decay to keep exploring
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.998

        # Hyperparams
        self.gamma = 0.99
        self.batch_size = 32
        self.target_update_freq = 100
        self.novelty_coef = 1.0  # stronger exploration bonus
        self.max_episode_steps = self.config.get("max_episode_steps", 500)
        self.num_episodes = self.config.get("max_iterations", 200)

        self.train_steps = 0

    def run(self):
        """Main training loop."""
        from env_wrapper import ARCEnvWrapper

        logger.info("DQNAgent starting: game=%s device=%s episodes=%d",
                    self.game_id, self.device, self.num_episodes)

        env = ARCEnvWrapper(self.game_id)

        # Phase 1: Collect random experience for action-effect pretraining
        logger.info("Phase 1: Random data collection (10 episodes)")
        for ep in range(10):
            self._collect_episode(env, random_policy=True)

        # Pretrain action-effect predictor
        logger.info("Phase 2: Pretrain action-effect predictor (%d samples)", len(self.effect_data))
        self._train_effect_predictor(epochs=20)

        # Phase 3: DQN training with exploration
        logger.info("Phase 3: DQN training (%d episodes)", self.num_episodes)
        best_levels = 0
        for ep in range(self.num_episodes):
            ep_return, ep_steps, levels, info = self._train_episode(env)

            if levels > best_levels:
                best_levels = levels
                logger.info("NEW BEST: %d levels at episode %d!", levels, ep)

            if ep % 10 == 0:
                logger.info(
                    "Ep %d/%d: return=%.2f steps=%d levels=%d eps=%.3f "
                    "replay=%d effect_acc=%.2f%% unique_states=%d",
                    ep, self.num_episodes, ep_return, ep_steps, levels,
                    self.epsilon, len(self.replay),
                    self._effect_accuracy() * 100,
                    len(self.state_counts),
                )

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        env.close()
        logger.info("DQN training complete. Best levels: %d", best_levels)

    def _collect_episode(self, env, random_policy: bool = False) -> tuple:
        """Collect one episode, storing transitions and effect data."""
        obs = env.reset()
        valid_actions = [a for a in env.valid_action_indices if a != 0]
        if not valid_actions:
            valid_actions = env.valid_action_indices
        num_actions = self.config.get("num_base_actions", 5)

        total_reward = 0.0
        steps = 0
        levels = 0

        for _ in range(self.max_episode_steps):
            if random_policy:
                action = valid_actions[np.random.randint(len(valid_actions))]
            else:
                action = self._select_action(obs, valid_actions)

            prev_obs = obs.copy()
            next_obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            levels = info.get("levels_completed", 0)

            # Track action effect
            changed = not np.array_equal(prev_obs, next_obs)
            self.effect_data.append((preprocess(prev_obs), action, changed))

            # Compute intrinsic reward from state counting
            obs_tensor = torch.from_numpy(preprocess(next_obs)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embed = self.encoder(obs_tensor)
                hash_key = self.hasher.hash_key(embed)

            self.state_counts[hash_key] = self.state_counts.get(hash_key, 0) + 1
            visit_count = self.state_counts[hash_key]
            novelty_bonus = self.novelty_coef / (visit_count ** 0.5)

            # Penalize no-ops (wall hits)
            if not changed:
                novelty_bonus = -0.1

            total_reward_step = reward + novelty_bonus

            # Store transition
            self.replay.add(
                preprocess(prev_obs), action, total_reward_step,
                preprocess(next_obs), terminated or truncated,
            )

            total_reward += total_reward_step
            obs = next_obs

            if terminated or truncated:
                if info.get("state") == "GAME_OVER":
                    obs = env.reset()
                    # Reset state counts on GAME_OVER — new life, explore afresh
                    self.state_counts.clear()
                elif info.get("state") == "WIN":
                    break
                else:
                    obs = env.reset()

        return total_reward, steps, levels, info

    def _select_action(self, obs: np.ndarray, valid_actions: list[int]) -> int:
        """Epsilon-greedy with action-effect bias."""
        num_actions = self.config.get("num_base_actions", 5)

        if np.random.random() < self.epsilon:
            # Exploration: bias toward actions predicted to cause changes
            obs_tensor = torch.from_numpy(preprocess(obs)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embed = self.encoder(obs_tensor)
                # Get effect predictions for all valid actions
                probs = []
                for a in valid_actions:
                    a_onehot = torch.zeros(1, num_actions, device=self.device)
                    a_onehot[0, a] = 1.0
                    logit = self.effect_pred(embed, a_onehot)
                    probs.append(torch.sigmoid(logit).item())

            # Softmax over effect probabilities to bias exploration
            probs = np.array(probs)
            probs = np.exp(probs * 3) / np.exp(probs * 3).sum()  # temperature=1/3
            idx = np.random.choice(len(valid_actions), p=probs)
            return valid_actions[idx]
        else:
            # Exploitation: Q-values
            obs_tensor = torch.from_numpy(preprocess(obs)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(obs_tensor)[0]
            # Mask invalid actions
            valid_mask = torch.full_like(q_values, -1e9)
            for a in valid_actions:
                valid_mask[a] = 0
            q_values = q_values + valid_mask
            return int(q_values.argmax().item())

    def _train_episode(self, env) -> tuple:
        """Run one episode with DQN training interleaved."""
        result = self._collect_episode(env, random_policy=False)

        # Train Q-network from replay buffer
        if len(self.replay) >= self.batch_size:
            for _ in range(4):  # 4 gradient steps per episode
                self._train_q_step()

        # Periodically retrain effect predictor
        if self.train_steps % 50 == 0 and len(self.effect_data) > 100:
            self._train_effect_predictor(epochs=5)

        # Update target network
        if self.train_steps % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.train_steps += 1
        return result

    def _train_q_step(self):
        """One DQN gradient step."""
        obs, actions, rewards, next_obs, dones = self.replay.sample(self.batch_size)

        obs_t = torch.from_numpy(obs).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_obs_t = torch.from_numpy(next_obs).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)

        # Current Q-values
        q_values = self.q_net(obs_t)
        q_selected = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values (Double DQN)
        with torch.no_grad():
            next_q = self.q_net(next_obs_t)
            best_actions = next_q.argmax(dim=1)
            next_q_target = self.target_q_net(next_obs_t)
            next_q_selected = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target = rewards_t + self.gamma * next_q_selected * (1 - dones_t)

        loss = F.smooth_l1_loss(q_selected, target)
        self.q_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.q_optimizer.step()

    def _train_effect_predictor(self, epochs: int = 10):
        """Train the action-effect predictor on collected data."""
        if len(self.effect_data) < 50:
            return

        num_actions = self.config.get("num_base_actions", 5)
        # Sample up to 2000 recent samples
        data = self.effect_data[-2000:]
        dataset_size = len(data)

        for epoch in range(epochs):
            indices = np.random.permutation(dataset_size)
            total_loss = 0.0
            batches = 0

            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                batch_idx = indices[start:end]

                obs_batch = np.stack([data[i][0] for i in batch_idx])
                actions_batch = [data[i][1] for i in batch_idx]
                labels_batch = np.array([float(data[i][2]) for i in batch_idx], dtype=np.float32)

                obs_t = torch.from_numpy(obs_batch).to(self.device)
                labels_t = torch.from_numpy(labels_batch).to(self.device)

                # One-hot encode actions
                a_onehot = torch.zeros(len(batch_idx), num_actions, device=self.device)
                for i, a in enumerate(actions_batch):
                    a_onehot[i, a] = 1.0

                with torch.no_grad():
                    embed = self.encoder(obs_t)

                logits = self.effect_pred(embed.detach(), a_onehot)
                loss = F.binary_cross_entropy_with_logits(logits, labels_t)

                self.effect_optimizer.zero_grad()
                loss.backward()
                self.effect_optimizer.step()

                total_loss += loss.item()
                batches += 1

        avg_loss = total_loss / max(batches, 1)
        logger.debug("Effect predictor: epoch %d, loss=%.4f", epochs, avg_loss)

    def _effect_accuracy(self) -> float:
        """Compute accuracy of action-effect predictor on recent data."""
        if len(self.effect_data) < 50:
            return 0.0

        num_actions = self.config.get("num_base_actions", 5)
        data = self.effect_data[-200:]
        obs_batch = np.stack([d[0] for d in data])
        actions_batch = [d[1] for d in data]
        labels = np.array([float(d[2]) for d in data])

        obs_t = torch.from_numpy(obs_batch).to(self.device)
        a_onehot = torch.zeros(len(data), num_actions, device=self.device)
        for i, a in enumerate(actions_batch):
            a_onehot[i, a] = 1.0

        with torch.no_grad():
            embed = self.encoder(obs_t)
            logits = self.effect_pred(embed, a_onehot)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(float)

        return float(np.mean(preds == labels))
