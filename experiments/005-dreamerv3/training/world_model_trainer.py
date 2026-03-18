"""World model trainer.

Implements the DreamerV3 world model loss, which combines:

1. Reconstruction loss  — cross-entropy between decoded and real observations.
2. Reward prediction loss — symlog MSE on predicted vs real rewards.
3. Continue prediction loss — binary cross-entropy on episode-continue signal.
4. KL loss (representation loss) — encourages posterior to match prior.

The KL loss uses free-bits clipping to avoid posterior collapse and balances
the prior and posterior gradients per the DreamerV3 paper.
"""

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.utils import symlog, kl_divergence_categorical

logger = logging.getLogger(__name__)


class WorldModelTrainer:
    """Manages world model optimisation for one gradient update per call.

    Accepts batches from the replay buffer, runs the world model's ``observe``
    method, computes all four loss terms, backpropagates, and clips gradients.

    Args:
        world_model: The ``WorldModel`` instance to train.
        config: Experiment config dict. Relevant keys:

            ``world_model_lr``       — Adam learning rate (default 1e-4).
            ``max_grad_norm``        — Gradient clip norm (default 100.0).
            ``kl_free_bits``         — Free nats per categorical (default 1.0).
            ``kl_balance_prior``     — Weight on prior KL term (default 0.8).
            ``kl_balance_posterior`` — Weight on posterior KL term (default 0.5).
    """

    def __init__(self, world_model: nn.Module, config: dict) -> None:
        self.world_model = world_model
        self.config = config

        self.optimizer = torch.optim.Adam(
            world_model.parameters(),
            lr=config.get("world_model_lr", 1e-4),
            eps=1e-8,
        )
        self._step: int = 0

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Run one gradient update on a batch of sequences from the replay buffer.

        Steps:
        1. Preprocess observations to one-hot: (B, T, 64, 64) → (B, T, 16, 64, 64).
        2. Run ``world_model.observe(observations, actions)`` to get latent states
           and predictions.
        3. Compute reconstruction, KL, reward, and continue losses.
        4. Backpropagate the total loss and clip gradients.

        Args:
            batch: Output of ``ReplayBuffer.sample_sequences``. Expected keys:
                ``observations`` — (B, T+1, 64, 64) int64 torch.Tensor
                ``actions``      — (B, T) int64 torch.Tensor
                ``rewards``      — (B, T) float32 torch.Tensor
                ``dones``        — (B, T) float32 torch.Tensor
                ``mask``         — (B, T) bool torch.Tensor (optional, unused in loss)

        Returns:
            Dict mapping loss component names to scalar float values for logging:
                ``wm/total_loss``, ``wm/recon_loss``, ``wm/rew_loss``,
                ``wm/cont_loss``, ``wm/kl_loss``.
        """
        device = next(self.world_model.parameters()).device

        def _to(x: Any, dtype: torch.dtype) -> torch.Tensor:
            """Move tensor or array to device with the correct dtype."""
            if isinstance(x, torch.Tensor):
                return x.to(device=device, dtype=dtype)
            return torch.tensor(x, dtype=dtype, device=device)

        # Move batch to device.
        # observations: (B, T+1, 64, 64) — first T frames are inputs; frame T+1
        # is the next observation used for reconstruction of the last step.
        obs_raw = _to(batch["observations"], torch.long)
        actions = _to(batch["actions"], torch.long)
        rewards = _to(batch["rewards"], torch.float32)
        dones   = _to(batch["dones"],   torch.float32)

        B, T = actions.shape

        # Use the first T frames as observation inputs to the world model.
        obs_int = obs_raw[:, :T]  # (B, T, 64, 64)

        # One-hot encode: (B, T, 64, 64) → (B, T, 16, 64, 64)
        obs_onehot = F.one_hot(obs_int, num_classes=16).float()   # (B, T, 64, 64, 16)
        obs_onehot = obs_onehot.permute(0, 1, 4, 2, 3)           # (B, T, 16, 64, 64)

        self.optimizer.zero_grad()

        # Forward pass through world model.
        outputs = self.world_model.observe(obs_onehot, actions)
        # outputs keys: latents, prior_logits, posterior_logits,
        #               decoded_obs, predicted_rewards, predicted_continues.

        # ------------------------------------------------------------------
        # 1. Reconstruction loss — cross-entropy per cell.
        # decoded_obs: (B, T, 16, 64, 64) logits vs obs_int: (B, T, 64, 64)
        # ------------------------------------------------------------------
        decoded  = outputs["decoded_obs"]             # (B, T, 16, 64, 64)
        dec_flat = decoded.view(B * T, 16, 64, 64)    # (B*T, 16, 64, 64)
        tgt_flat = obs_int.reshape(B * T, 64, 64)     # (B*T, 64, 64)
        recon_loss = F.cross_entropy(dec_flat, tgt_flat)

        # ------------------------------------------------------------------
        # 2. Reward prediction loss — MSE in symlog space.
        # predicted_rewards: (B, T), rewards: (B, T)
        # ------------------------------------------------------------------
        pred_rew = outputs["predicted_rewards"]       # (B, T)
        rew_loss = F.mse_loss(pred_rew, symlog(rewards))

        # ------------------------------------------------------------------
        # 3. Continue prediction loss — binary cross-entropy.
        # continue = 1 − done; predicted_continues are pre-sigmoid logits.
        # ------------------------------------------------------------------
        pred_cont    = outputs["predicted_continues"] # (B, T)
        cont_targets = 1.0 - dones                   # (B, T), float32
        cont_loss = F.binary_cross_entropy_with_logits(pred_cont, cont_targets)

        # ------------------------------------------------------------------
        # 4. KL loss (representation loss) — balanced with free bits.
        # prior_logits, posterior_logits: (B, T, stochastic_dim)
        # Reshape to (B*T, num_categories, num_classes) for the utility fn.
        # ------------------------------------------------------------------
        num_cats = self.world_model.rssm.num_categories
        num_cls  = self.world_model.rssm.num_classes

        prior_flat = outputs["prior_logits"].view(B * T, num_cats, num_cls)
        post_flat  = outputs["posterior_logits"].view(B * T, num_cats, num_cls)

        kl_loss = kl_divergence_categorical(
            posterior_logits=post_flat,
            prior_logits=prior_flat,
            free_bits=self.config.get("kl_free_bits", 1.0),
            balance_prior=self.config.get("kl_balance_prior", 0.8),
            balance_posterior=self.config.get("kl_balance_posterior", 0.5),
        )

        # ------------------------------------------------------------------
        # Total loss and backprop.
        # ------------------------------------------------------------------
        total_loss = recon_loss + rew_loss + cont_loss + kl_loss
        total_loss.backward()

        nn.utils.clip_grad_norm_(
            self.world_model.parameters(),
            max_norm=self.config.get("max_grad_norm", 100.0),
        )
        self.optimizer.step()
        self._step += 1

        return {
            "wm/total_loss": total_loss.item(),
            "wm/recon_loss": recon_loss.item(),
            "wm/rew_loss":   rew_loss.item(),
            "wm/cont_loss":  cont_loss.item(),
            "wm/kl_loss":    kl_loss.item(),
        }
