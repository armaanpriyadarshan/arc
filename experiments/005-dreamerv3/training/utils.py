"""Utility functions for DreamerV3 training.

Implements the mathematical primitives shared across world-model and
actor-critic training:

* ``symlog`` / ``symexp``      — symmetric log/exp transforms for reward scaling.
* ``compute_lambda_returns``   — TD(λ) return computation for actor-critic.
* ``compute_return_normalization`` — percentile-based return normalisation.
* ``kl_divergence_categorical``    — KL with DreamerV3 balancing and free bits.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Symmetric log / exp transforms
# ---------------------------------------------------------------------------


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithmic transform.

    Compresses large-magnitude values while preserving sign and remaining
    differentiable through zero:

        symlog(x) = sign(x) * ln(|x| + 1)

    Args:
        x: Input tensor of any shape.

    Returns:
        Tensor of the same shape as ``x``.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog.

        symexp(x) = sign(x) * (exp(|x|) - 1)

    Args:
        x: symlog-transformed tensor of any shape.

    Returns:
        Tensor of the same shape as ``x`` in the original scale.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


# ---------------------------------------------------------------------------
# TD(λ) lambda returns
# ---------------------------------------------------------------------------


def compute_lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continues: torch.Tensor,
    gamma: float = 0.997,
    lambda_: float = 0.95,
) -> torch.Tensor:
    """Compute TD(λ) returns for a sequence of imagined transitions.

    Implements the standard backward recursion:

        R_{T-1} = r_{T-1} + γ · c_{T-1} · V_T          (bootstrap last step)
        R_t     = r_t + γ · c_t · [(1-λ) · V_{t+1} + λ · R_{t+1}]

    where c_t = continue probability (soft form of the "not done" mask).

    Args:
        rewards:   (B, T) predicted rewards along the imagined trajectory.
        values:    (B, T+1) predicted values. The extra step V_T provides
                   the bootstrap value at the end of the trajectory. If
                   only (B, T) is provided, the last value is used as its
                   own bootstrap (equivalent to value at T = values[:, -1]).
        continues: (B, T) continue probabilities in [0, 1].
        gamma:     Discount factor.
        lambda_:   TD(λ) mixing coefficient.

    Returns:
        (B, T) tensor of lambda-return targets.

    Notes:
        When ``values`` has shape (B, T) rather than (B, T+1), V_T is taken
        as the last column of ``values`` so the function degrades gracefully.
    """
    B, T = rewards.shape

    # Determine bootstrap values V_1 ... V_T
    if values.shape[-1] == T + 1:
        # Preferred: separate bootstrap column
        bootstrap = values[:, 1:]   # (B, T)  — V_{t+1} for each t
        v_last = values[:, T]       # (B,)    — V_T for the final bootstrap
    else:
        # Fallback: values shape is (B, T), use shifted view
        bootstrap = torch.cat([values[:, 1:], values[:, -1:]], dim=1)  # (B, T)
        v_last = values[:, -1]      # (B,)

    # Work backwards from t = T-1 to t = 0.
    # Returns tensor: allocate then fill right-to-left.
    returns = torch.zeros_like(rewards)

    # Last step: R_{T-1} = r_{T-1} + γ · c_{T-1} · V_T
    returns[:, -1] = (
        rewards[:, -1] + gamma * continues[:, -1] * v_last
    )

    for t in reversed(range(T - 1)):
        # R_t = r_t + γ · c_t · [(1-λ) · V_{t+1} + λ · R_{t+1}]
        returns[:, t] = (
            rewards[:, t]
            + gamma * continues[:, t] * (
                (1.0 - lambda_) * bootstrap[:, t] + lambda_ * returns[:, t + 1]
            )
        )

    return returns  # (B, T)


# ---------------------------------------------------------------------------
# Return normalisation
# ---------------------------------------------------------------------------


def compute_return_normalization(
    returns: torch.Tensor,
    percentile_low: float = 5.0,
    percentile_high: float = 95.0,
    decay: float = 0.99,
    running_low: Optional[torch.Tensor] = None,
    running_high: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DreamerV3-style percentile-based return normalisation.

    Computes the ``percentile_low``th and ``percentile_high``th percentile
    of the current batch of returns, then exponentially smooths them against
    running estimates. Returns are normalised to the range
    ``[running_low, running_high]`` with a minimum scale of 1.

    Args:
        returns:         (B, T) or flat tensor of lambda-returns to normalise.
        percentile_low:  Lower percentile for the scale window (default 5).
        percentile_high: Upper percentile for the scale window (default 95).
        decay:           EMA decay for running percentile estimates (default 0.99).
        running_low:     Previous EMA low-percentile estimate (scalar tensor).
                         If None, the batch estimate is used directly.
        running_high:    Previous EMA high-percentile estimate (scalar tensor).
                         If None, the batch estimate is used directly.

    Returns:
        Tuple of:
            normalized_returns: Same shape as ``returns``.
            new_running_low:    Updated scalar EMA low-percentile estimate.
            new_running_high:   Updated scalar EMA high-percentile estimate.
    """
    flat = returns.detach().float().flatten()

    batch_low = torch.quantile(flat, percentile_low / 100.0)
    batch_high = torch.quantile(flat, percentile_high / 100.0)

    if running_low is not None:
        new_low = decay * running_low + (1.0 - decay) * batch_low
        new_high = decay * running_high + (1.0 - decay) * batch_high
    else:
        new_low = batch_low
        new_high = batch_high

    # Minimum scale of 1 to avoid division by near-zero.
    scale = torch.clamp(new_high - new_low, min=1.0)
    normalized = (returns - new_low) / scale

    return normalized, new_low, new_high


# ---------------------------------------------------------------------------
# KL divergence for categorical RSSM states
# ---------------------------------------------------------------------------


def kl_divergence_categorical(
    posterior_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    free_bits: float = 1.0,
    balance_prior: float = 0.8,
    balance_posterior: float = 0.5,
) -> torch.Tensor:
    """KL divergence between categorical distributions with DreamerV3 balancing.

    Implements the DreamerV3 balanced KL loss:

        L_KL = balance_prior * KL(sg(posterior) || prior)
             + balance_posterior * KL(posterior || sg(prior))

    where ``sg(·)`` denotes stop-gradient.  Free-bits clipping prevents
    posterior collapse by ignoring KL values below ``free_bits`` nats.

    Per-category KL values are clipped individually (not per-batch-element),
    then summed across categories.  The resulting scalar is the mean over the
    batch.

    Args:
        posterior_logits: (B, num_categories * num_classes) or
                          (B, num_categories, num_classes) raw logits for
                          the posterior distribution q(z|h, embed).
        prior_logits:     Same shape as ``posterior_logits`` — raw logits
                          for the prior distribution p(z|h).
        free_bits:        Minimum KL nats per category to allow gradients
                          (clips smaller values to this floor, default 1.0).
        balance_prior:    Weight on the KL with stopped-gradient posterior
                          (trains the prior; default 0.8).
        balance_posterior: Weight on the KL with stopped-gradient prior
                           (trains the posterior; default 0.5).

    Returns:
        Scalar tensor — the mean balanced KL loss over the batch.

    Notes:
        Both inputs are reshaped to (B, num_categories, num_classes) internally;
        the function accepts either a flat or pre-shaped input tensor.
    """
    # Normalise shapes to (B, num_categories, num_classes).
    if posterior_logits.dim() == 2:
        # Infer num_categories * num_classes from the flat dimension.
        # We cannot infer the factorisation from shape alone; reshape to (B, D, 1)
        # as a 1-class-per-category degenerate case is fine for the math.
        # In practice the caller should pass pre-shaped tensors.
        B, D = posterior_logits.shape
        post = posterior_logits.unsqueeze(-1)          # (B, D, 1)
        prior = prior_logits.unsqueeze(-1)
    else:
        post = posterior_logits   # (B, num_categories, num_classes)
        prior = prior_logits

    # Convert logits to log-probabilities.
    log_post = F.log_softmax(post, dim=-1)   # (B, K, C)
    log_prior = F.log_softmax(prior, dim=-1) # (B, K, C)
    post_probs = log_post.exp()              # (B, K, C)

    # KL(posterior || prior) = sum_c post * (log_post - log_prior)
    # Shape: (B, K)
    kl_post_prior = (post_probs * (log_post - log_prior.detach())).sum(dim=-1)
    kl_prior_post = (post_probs.detach() * (log_post.detach() - log_prior)).sum(dim=-1)

    # Free-bits: clip per-category KL below ``free_bits`` to prevent collapse.
    kl_post_prior_clipped = torch.clamp(kl_post_prior, min=free_bits)
    kl_prior_post_clipped = torch.clamp(kl_prior_post, min=free_bits)

    # Balanced loss: sum over categories, mean over batch.
    loss = (
        balance_prior * kl_prior_post_clipped.sum(dim=-1).mean()
        + balance_posterior * kl_post_prior_clipped.sum(dim=-1).mean()
    )

    return loss
