"""Wave-U-Net (Stoller, Ewert & Dixon, 2018) adapted to single-source enhancement.

Architecture, as in the paper:
- ``num_layers`` downsampling blocks: Conv1d(kernel 15) + LeakyReLU, then decimation by 2,
  layer ``i`` has ``num_filters * i`` channels,
- bottleneck Conv1d(kernel 15),
- ``num_layers`` upsampling blocks: linear interpolation x2, concatenation with the skip
  connection, Conv1d(kernel 5) + LeakyReLU,
- the raw input is concatenated before the final 1x1 convolution with ``tanh``.

Unlike the original (``valid`` convolutions and context cropping) this version uses ``same``
padding and interpolates to the skip length, so it accepts any input length. This is the
common choice in speech-enhancement ports (e.g. Macartney & Weyde, 2018).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from src.denoising.base import DenoisingModel


class _DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = F.leaky_relu(self.conv(x), 0.2)
        return skip[..., ::2], skip


class _UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels + skip_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=True)
        return F.leaky_relu(self.conv(torch.cat([x, skip], dim=1)), 0.2)


@dataclass(frozen=True)
class WaveUNetConfig:
    model_sample_rate: int = 4000
    num_layers: int = 9
    num_filters: int = 24
    down_kernel_size: int = 15
    up_kernel_size: int = 5
    #: scale every segment to a peak of 0.95 before the network (it ends with ``tanh``)
    peak_normalize: bool = True


class WaveUNet(DenoisingModel):
    """Waveform-in, waveform-out Wave-U-Net. :meth:`forward` takes ``(batch, 1, samples)``;
    :meth:`enhance` adds the peak normalisation."""

    def __init__(self, config: WaveUNetConfig | None = None) -> None:
        super().__init__()
        self.config = config = config or WaveUNetConfig()
        num_layers, num_filters = config.num_layers, config.num_filters
        down_kernel_size, up_kernel_size = config.down_kernel_size, config.up_kernel_size
        channels = [1] + [num_filters * i for i in range(1, num_layers + 1)]
        self.down = nn.ModuleList(
            _DownBlock(channels[i], channels[i + 1], down_kernel_size) for i in range(num_layers)
        )
        bottleneck_channels = num_filters * (num_layers + 1)
        self.bottleneck = nn.Conv1d(
            channels[-1],
            bottleneck_channels,
            down_kernel_size,
            padding=down_kernel_size // 2,
        )
        up_in = [bottleneck_channels, *channels[:1:-1]]
        self.up = nn.ModuleList(
            _UpBlock(up_in[i], channels[num_layers - i], channels[num_layers - i], up_kernel_size)
            for i in range(num_layers)
        )
        self.output = nn.Conv1d(channels[1] + 1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        skips = []
        hidden = x
        for block in self.down:
            hidden, skip = block(hidden)
            skips.append(skip)
        hidden = F.leaky_relu(self.bottleneck(hidden), 0.2)
        for block, skip in zip(self.up, reversed(skips), strict=True):
            hidden = block(hidden, skip)
        return torch.tanh(self.output(torch.cat([hidden, x], dim=1)))

    @staticmethod
    def peak_scale(waveform: torch.Tensor) -> torch.Tensor:
        """Per-row gain bringing the peak to 0.95."""
        return 0.95 / waveform.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)

    def enhance(self, noisy: torch.Tensor) -> torch.Tensor:
        if not self.config.peak_normalize:
            return cast(torch.Tensor, self(noisy.unsqueeze(1))).squeeze(1)
        scale = self.peak_scale(noisy)
        return cast(torch.Tensor, self((noisy * scale).unsqueeze(1))).squeeze(1) / scale
