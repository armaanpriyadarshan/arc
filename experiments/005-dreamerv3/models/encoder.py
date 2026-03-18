"""CNN encoder for 64x64x16 one-hot grid observations.

Converts raw game frames into a compact embedding vector used by the RSSM.
"""

import torch
import torch.nn as nn
from typing import List


class Encoder(nn.Module):
    """CNN encoder: (B, 16, 64, 64) -> (B, embed_dim).

    Takes one-hot encoded grid observations (16 color channels, 64x64 spatial)
    and produces a flat embedding vector. Uses kernel_size=4, stride=2, padding=1
    which halves spatial dimensions at each layer:
        64x64 -> 32x32 -> 16x16 -> 8x8

    Args:
        in_channels: Number of input channels (16 for one-hot color encoding).
        channels: List of channel widths for each conv layer.
        embed_dim: Size of the output embedding vector.
    """

    def __init__(
        self,
        in_channels: int = 16,
        channels: List[int] = None,
        embed_dim: int = 1024,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128]

        conv_layers: List[nn.Module] = []
        prev_channels = in_channels
        for out_channels in channels:
            conv_layers.extend([
                nn.Conv2d(
                    prev_channels, out_channels,
                    kernel_size=4, stride=2, padding=1,
                ),
                nn.ReLU(inplace=True),
            ])
            prev_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)

        # After 3 halving layers: 64 -> 32 -> 16 -> 8
        # Flatten size = final_channels * 8 * 8
        self._flatten_size = channels[-1] * 8 * 8
        self.fc = nn.Linear(self._flatten_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of grid observations.

        Args:
            x: Float tensor of shape (B, 16, 64, 64). Should be one-hot or
               normalized color values.

        Returns:
            Embedding tensor of shape (B, embed_dim).
        """
        features = self.conv(x)                  # (B, C, 8, 8)
        flat = features.flatten(start_dim=1)     # (B, C*8*8)
        return self.fc(flat)                     # (B, embed_dim)
