"""Actor-critic trainer (imagination-based policy optimisation).

The actor and critic are trained entirely on imagined trajectories produced
by the world model. No real environment interaction happens here.

Algorithm outline (DreamerV3-style):
1. Encode seed observations into latent states (world model encoder + RSSM).
2. Roll out the world model for ``imagination_horizon`` steps using the
   current actor policy (``world_model.imagine``).
3. Compute lambda-return targets using the slow (EMA) critic for bootstrapping.
4. Normalise returns via running percentile estimates.
5. Update the critic to minimise symlog MSE to the lambda-returns.
6. Update the actor to maximise normalised returns, with entropy regularisation.
7. Soft-update the slow critic toward the live critic.

The world model is frozen during actor-critic training (no gradients flow
into its parameters). The actor receives gradients through the imagined
trajectory via the straight-through RSSM samples.
"""

import logging
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.utils import (
    symlog,
    symexp,
    compute_lambda_returns,
    compute_return_normalization,
)
from env_wrapper import preprocess_observation

logger = logging.getLogger(__name__)


class ActorCriticTrainer:
    """Trains the actor and critic on imagined trajectories from the world model.

    Args:
        actor:       ``HierarchicalActor`` to optimise.
        critic:      Live ``Critic`` to optimise.
        slow_critic: ``SlowCritic`` (EMA target) used for bootstrapping lambda-returns.
        world_model: ``WorldModel`` used for imagination rollouts (frozen during training).
        config:      Experiment config dict. Relevant keys:

            ``actor_lr``                   — Adam LR for actor (default 3e-5).
            ``critic_lr``                  — Adam LR for critic (default 3e-5).
            ``max_grad_norm``              — Gradient clip norm (default 100.0).
            ``imagination_horizon``        — Rollout length H (default 8).
            ``gamma``                      — Discount factor (default 0.997).
            ``lambda_``                    — TD(λ) mixing coefficient (default 0.95).
            ``actor_entropy_coef``         — Entropy bonus weight (default 3e-3).
            ``return_norm_percentile_low`` — Low percentile (default 5).
            ``return_norm_percentile_high``— High percentile (default 95).
            ``return_norm_decay``          — EMA decay for running percentiles (default 0.99).
    """

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        slow_critic: object,
        world_model: nn.Module,
        config: dict,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.slow_critic = slow_critic
        self.world_model = world_model
        self.config = config

        self.actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=config.get("actor_lr", 3e-5),
            eps=1e-8,
        )
        self.critic_optimizer = torch.optim.Adam(
            critic.parameters(),
            lr=config.get("critic_lr", 3e-5),
            eps=1e-8,
        )

        # Running percentile estimates for return normalisation.
        self.running_return_low: Optional[torch.Tensor] = None
        self.running_return_high: Optional[torch.Tensor] = None

        self._step: int = 0

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        initial_states: Union[np.ndarray, torch.Tensor],
    ) -> dict[str, float]:
        """One full actor-critic update from imagined trajectories.

        Steps:
        1. Encode ``initial_states`` (seed observations) into latent (h, z) using
           the world model encoder + one RSSM observe step.
        2. Imagine forward H steps using ``world_model.imagine`` and the actor.
        3. Predict rewards and continue probabilities along the trajectory.
        4. Compute lambda-returns using the slow critic for bootstrapping.
        5. Normalise returns (percentile-based EMA).
        6. Update the critic (symlog MSE to lambda-return targets).
        7. Update the actor (policy gradient + entropy bonus).
        8. Soft-update the slow critic.

        The world model is kept frozen throughout (``torch.no_grad()`` for all
        world model forward passes).

        Args:
            initial_states: (B, 64, 64) int64 seed observations sampled from the
                            replay buffer. Both numpy arrays and torch tensors are
                            accepted.

        Returns:
            Dict mapping metric names to scalar float values for logging:
                ``ac/actor_loss``, ``ac/critic_loss``, ``ac/mean_return``,
                ``ac/mean_entropy``.
        """
        device = next(self.actor.parameters()).device
        config = self.config
        horizon = config.get("imagination_horizon", 8)
        gamma   = config.get("gamma", 0.997)
        lam     = config.get("lambda_", 0.95)
        ent_c   = config.get("actor_entropy_coef", 3e-3)

        # ------------------------------------------------------------------
        # 1. Encode seed observations into latent starting states.
        # ------------------------------------------------------------------
        if isinstance(initial_states, torch.Tensor):
            obs_np = initial_states.cpu().numpy().astype(np.int64)
        else:
            obs_np = np.asarray(initial_states, dtype=np.int64)

        B = obs_np.shape[0]

        # Preprocess: (B, 64, 64) → (B, 16, 64, 64) one-hot float tensor.
        obs_tensors = torch.stack(
            [preprocess_observation(obs_np[i]) for i in range(B)]
        ).to(device)  # (B, 16, 64, 64)

        with torch.no_grad():
            embeds = self.world_model.encoder(obs_tensors)          # (B, embed_dim)
            h0, z0 = self.world_model.rssm.initial_state(B, device)
            dummy_action = torch.zeros(B, dtype=torch.long, device=device)
            h0, z0, _, _ = self.world_model.rssm.observe_step(
                h0, z0, dummy_action, embeds
            )

        # ------------------------------------------------------------------
        # 2. Imagine H steps forward using the actor policy.
        # World model is frozen; actor gradients flow through latents.
        # ------------------------------------------------------------------
        imagination = self.world_model.imagine(h0, z0, self.actor, horizon)
        # imagination keys:
        #   latents            : (B, H, latent_dim) — straight-through gradients
        #   action_types       : (B, H) int64
        #   action_xs          : (B, H) int64
        #   action_ys          : (B, H) int64
        #   log_probs          : (B, H)
        #   predicted_rewards  : (B, H) in symlog space
        #   predicted_continues: (B, H) pre-sigmoid logits

        latents      = imagination["latents"]               # (B, H, latent_dim)
        pred_rews    = imagination["predicted_rewards"]     # (B, H) symlog space
        cont_logits  = imagination["predicted_continues"]   # (B, H)

        rewards   = symexp(pred_rews)                       # (B, H) original scale
        continues = torch.sigmoid(cont_logits)              # (B, H) in [0, 1]

        # ------------------------------------------------------------------
        # 3. Compute lambda-returns using slow critic for bootstrapping.
        # All slow-critic evaluations are detached (no gradients).
        # ------------------------------------------------------------------
        with torch.no_grad():
            # Value estimates at all H steps from the slow critic: (B, H)
            lat_flat      = latents.view(B * horizon, -1)
            slow_vals_flat = self.slow_critic.forward(lat_flat)  # (B*H,) symlog
            slow_vals     = symexp(slow_vals_flat).view(B, horizon)

            # Bootstrap value at the step AFTER the last imagined step.
            # We take the slow critic's estimate at the final latent as V_T.
            bootstrap_v = slow_vals[:, -1]  # (B,)

            # Build (B, T+1) values tensor: [V_1, ..., V_H, V_H] (repeat last for bootstrap).
            values_tp1 = torch.cat([slow_vals, bootstrap_v.unsqueeze(1)], dim=1)  # (B, H+1)

        lambda_returns = compute_lambda_returns(
            rewards=rewards,
            values=values_tp1,
            continues=continues,
            gamma=gamma,
            lambda_=lam,
        )  # (B, H)

        # ------------------------------------------------------------------
        # 4. Normalise returns via running percentiles.
        # ------------------------------------------------------------------
        normed_returns, self.running_return_low, self.running_return_high = (
            compute_return_normalization(
                returns=lambda_returns,
                percentile_low=config.get("return_norm_percentile_low", 5),
                percentile_high=config.get("return_norm_percentile_high", 95),
                decay=config.get("return_norm_decay", 0.99),
                running_low=self.running_return_low,
                running_high=self.running_return_high,
            )
        )

        # ------------------------------------------------------------------
        # 5. Critic update — symlog MSE on detached lambda-return targets.
        # ------------------------------------------------------------------
        self.critic_optimizer.zero_grad()

        critic_input = latents.detach().view(B * horizon, -1)  # detach world model grads
        critic_pred  = self.critic(critic_input)               # (B*H,) in symlog space
        critic_target = symlog(lambda_returns.detach()).view(B * horizon)
        critic_loss = F.mse_loss(critic_pred, critic_target)

        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            max_norm=config.get("max_grad_norm", 100.0),
        )
        self.critic_optimizer.step()

        # ------------------------------------------------------------------
        # 6. Actor update — policy gradient loss + entropy bonus.
        # Re-evaluate log_probs and entropy with fresh actor parameters so
        # gradients flow from the loss to the actor.
        # latents are detached from the world model but still carry the
        # straight-through sample graph from the RSSM (via world_model.imagine).
        # ------------------------------------------------------------------
        self.actor_optimizer.zero_grad()

        lat_flat_actor = latents.detach().view(B * horizon, -1)
        act_types_flat = imagination["action_types"].view(B * horizon)
        act_xs_flat    = imagination["action_xs"].view(B * horizon)
        act_ys_flat    = imagination["action_ys"].view(B * horizon)

        # Re-compute log-probs for the sampled actions under current policy.
        log_prob_flat = self.actor.log_prob(
            lat_flat_actor, act_types_flat, act_xs_flat, act_ys_flat
        )  # (B*H,)

        # Sample entropy estimate (use a fresh sample for the entropy term).
        _, _, _, _, entropy_flat = self.actor.sample(lat_flat_actor)  # (B*H,)

        log_probs_fresh = log_prob_flat.view(B, horizon)
        entropies       = entropy_flat.view(B, horizon)

        # Policy gradient: maximise (normed_returns * log_prob) + entropy.
        actor_loss = (
            -(normed_returns.detach() * log_probs_fresh).mean()
            - ent_c * entropies.mean()
        )

        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            max_norm=config.get("max_grad_norm", 100.0),
        )
        self.actor_optimizer.step()

        # ------------------------------------------------------------------
        # 7. Soft-update slow critic.
        # ------------------------------------------------------------------
        self.slow_critic.update()

        self._step += 1

        return {
            "ac/actor_loss":   actor_loss.item(),
            "ac/critic_loss":  critic_loss.item(),
            "ac/mean_return":  lambda_returns.mean().item(),
            "ac/mean_entropy": entropies.mean().item(),
        }
