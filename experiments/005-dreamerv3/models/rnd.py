"""Random Network Distillation (RND) for intrinsic motivation.

RND trains a predictor network to match the outputs of a fixed, randomly
initialized target network. States that the predictor hasn't seen before
will have high prediction error, providing an exploration bonus.

Reference: Burda et al., "Exploration by Random Network Distillation" (2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNDTarget(nn.Module):
    """Fixed randomly-initialized target network (never trained).

    Maps observations to an embedding. The randomness of the initialization
    means that similar observations produce similar outputs, so the prediction
    error naturally decreases for familiar states.
    """

    def __init__(self, in_channels: int = 16, embed_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
        )
        # Output size: 64 channels * 4 * 4 = 1024 (for 64x64 input)
        self.fc = nn.Linear(1024, embed_dim)

        # Freeze: never train this network
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map one-hot observation to embedding.

        Args:
            x: (B, 16, 64, 64) one-hot observation tensor.

        Returns:
            (B, embed_dim) target embedding.
        """
        features = self.conv(x)
        return self.fc(features.flatten(start_dim=1))


class RNDPredictor(nn.Module):
    """Trainable predictor network that learns to match RNDTarget outputs.

    Same architecture as the target but with trainable parameters. The MSE
    between predictor and target outputs serves as the intrinsic reward.
    """

    def __init__(self, in_channels: int = 16, embed_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map one-hot observation to predicted embedding.

        Args:
            x: (B, 16, 64, 64) one-hot observation tensor.

        Returns:
            (B, embed_dim) predicted embedding.
        """
        features = self.conv(x)
        return self.fc(features.flatten(start_dim=1))


class RNDModule:
    """Manages RND target + predictor and computes intrinsic rewards.

    The intrinsic reward for a state is the MSE between the target and
    predictor embeddings. High error = unfamiliar state = high reward.

    The predictor is trained on observed states, so its error decreases
    for frequently-visited states and remains high for novel ones.
    """

    def __init__(self, in_channels: int = 16, embed_dim: int = 128, lr: float = 1e-3):
        self.target = RNDTarget(in_channels, embed_dim)
        self.predictor = RNDPredictor(in_channels, embed_dim)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)

        # Running statistics for reward normalization
        self._reward_mean = 0.0
        self._reward_var = 1.0
        self._count = 0

    def to(self, device: torch.device) -> "RNDModule":
        """Move both networks to a device."""
        self.target = self.target.to(device)
        self.predictor = self.predictor.to(device)
        return self

    def compute_intrinsic_reward(self, obs_onehot: torch.Tensor) -> torch.Tensor:
        """Compute intrinsic reward for a batch of observations.

        Returns raw (unnormalized) prediction error scaled to a reasonable range.
        No running statistics to avoid instability.

        Args:
            obs_onehot: (B, 16, 64, 64) one-hot encoded observations.

        Returns:
            (B,) intrinsic reward in [0, 1] range.
        """
        with torch.no_grad():
            target_embed = self.target(obs_onehot)
            pred_embed = self.predictor(obs_onehot)
            # Per-sample MSE
            raw_reward = ((target_embed - pred_embed) ** 2).mean(dim=-1)

        # Soft normalization: tanh squashes to [0, 1) without running stats
        return torch.tanh(raw_reward)

    def train_step(self, obs_onehot: torch.Tensor) -> float:
        """Train the predictor to match the target on observed states.

        Args:
            obs_onehot: (B, 16, 64, 64) one-hot encoded observations.

        Returns:
            Scalar MSE loss for logging.
        """
        self.optimizer.zero_grad()
        target_embed = self.target(obs_onehot).detach()
        pred_embed = self.predictor(obs_onehot)
        loss = F.mse_loss(pred_embed, target_embed)
        loss.backward()
        self.optimizer.step()
        return loss.item()
