"""Transposed CNN decoder: reconstructs grid observations from latent states.

Mirrors the Encoder architecture, producing per-cell color logits for the
cross-entropy reconstruction loss.
"""

import torch
import torch.nn as nn
from typing import List


class Decoder(nn.Module):
    """Transposed CNN: (B, latent_dim) -> (B, 16, 64, 64) logits.

    Mirrors the Encoder. Projects the latent vector to a 8x8 feature map then
    upsamples with transposed convolutions:
        8x8 -> 16x16 -> 32x32 -> 64x64

    Output logits (no softmax) are intended for use with
    ``nn.CrossEntropyLoss`` against integer color targets.

    Args:
        latent_dim: Dimensionality of the input latent vector
                    (gru_hidden_dim + stochastic_dim).
        channels: Channel widths for each transposed conv layer (in order,
                  before the final output layer).
        out_channels: Number of output channels (16 for one-hot color classes).
    """

    def __init__(
        self,
        latent_dim: int,
        channels: List[int] = None,
        out_channels: int = 16,
    ):
        super().__init__()
        if channels is None:
            channels = [128, 64, 32]

        # Spatial size before first deconv
        self._init_channels = channels[0]
        self._init_spatial = 8
        self._flatten_size = self._init_channels * self._init_spatial * self._init_spatial

        self.fc = nn.Linear(latent_dim, self._flatten_size)

        deconv_layers: List[nn.Module] = []
        prev_channels = channels[0]
        for i, ch in enumerate(channels[1:]):
            deconv_layers.extend([
                nn.ConvTranspose2d(
                    prev_channels, ch,
                    kernel_size=4, stride=2, padding=1,
                ),
                nn.ReLU(inplace=True),
            ])
            prev_channels = ch

        # Final layer: upsample to 64x64 and produce output logits (no activation)
        deconv_layers.append(
            nn.ConvTranspose2d(
                prev_channels, out_channels,
                kernel_size=4, stride=2, padding=1,
            )
        )

        self.deconv = nn.Sequential(*deconv_layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a batch of latent vectors into grid observation logits.

        Args:
            z: Float tensor of shape (B, latent_dim).

        Returns:
            Logit tensor of shape (B, 16, 64, 64). Pass to
            ``nn.CrossEntropyLoss`` with integer color targets.
        """
        flat = self.fc(z)                                              # (B, C*8*8)
        spatial = flat.view(                                           # (B, C, 8, 8)
            -1, self._init_channels,
            self._init_spatial, self._init_spatial,
        )
        return self.deconv(spatial)                                    # (B, 16, 64, 64)
