"""MANNER: Multi-view Attention Network for Noise ERasure (Park et al., ICASSP 2022).

Port of the official implementation (https://github.com/winddori2002/MANNER, MIT licence).
Module and parameter names are kept identical, so the official pretrained weights
(``manner_small.pth``, ``manner_base.pth``, ``manner_large.pth``; 16 kHz, VoiceBank+DEMAND)
load with ``strict=True``.

Architecture (time domain, U-Net with masking):
- ``in_conv`` (Conv1d k=3) lifts the waveform to ``hidden`` channels; its output is later
  multiplied by the estimated mask,
- ``depth`` encoder layers: strided Conv1d (``kernel_size``, ``stride``) -> Residual Conformer
  block (pointwise conv + GLU, depthwise conv k=31 + Swish, pointwise conv; doubles channels)
  -> Multi-view Attention block,
- a linear layer over channels at the bottleneck,
- ``depth`` decoder layers (skip connections are added): Residual Conformer block (halves
  channels) -> ConvTranspose1d -> Multi-view Attention block,
- ``MaskGate`` (tanh * sigmoid -> ReLU) produces the mask, ``out_conv`` maps back to 1 channel.

The Multi-view Attention block splits channels into three branches: channel attention
(squeeze-excitation from max and average pooling), global attention (self-attention between
overlapping chunks of ``segment_len`` frames) and local attention (depthwise conv k=31
inside each chunk followed by a spatial attention map). The branches are merged by a 1x1
convolution and a tanh/sigmoid gate on top of a residual shortcut.

The input is divided by its standard deviation and right-padded to a length the strided
convolutions accept; the output is rescaled and trimmed back. Variants from the paper:
``small`` (attention only in the deepest encoder/decoder layer), ``base`` and ``large``
(``hidden=120``).

Deviations from the official code that do not change results: chunking is stateless,
``squeeze()`` in channel attention became ``squeeze(-1)`` (the original drops the batch
dimension for batch size 1, which only worked through broadcasting) and the deprecated
``F.sigmoid`` is replaced by ``torch.sigmoid``. ``growth`` is fixed to 2, the only value the
original channel bookkeeping supports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from src.denoising.base import DenoisingModel

Variant = Literal["small", "base", "large"]

#: paper configurations: channel count and whether every layer has multi-view attention
_VARIANTS: dict[str, dict[str, object]] = {
    "small": {"hidden": 60, "attention_everywhere": False},
    "base": {"hidden": 60, "attention_everywhere": True},
    "large": {"hidden": 120, "attention_everywhere": True},
}


@dataclass(frozen=True)
class MANNERConfig:
    """``variant`` selects the paper configuration; ``hidden`` overrides its channel count."""

    model_sample_rate: int = 4000
    variant: Variant = "base"
    hidden: int | None = None
    depth: int = 4
    kernel_size: int = 8
    stride: int = 4
    head: int = 1
    segment_len: int = 64

    def __post_init__(self) -> None:
        if self.variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {sorted(_VARIANTS)}, got {self.variant!r}.")


# --------------------------------------------------------------------------- building blocks
class BasicConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        relu: bool = True,
        bn: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Swish(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class DepthwiseConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int, padding: int = 0) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, groups=channels, padding=padding, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.conv(x))


class PointwiseConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.conv(x))


class ResConBlock(nn.Module):
    """Residual Conformer block; ``growth2`` = 2 in the encoder, 1/2 in the decoder."""

    def __init__(
        self, in_channels: int, kernel_size: int = 31, growth1: float = 2, growth2: float = 2
    ) -> None:
        super().__init__()
        out_channels1 = int(in_channels * growth1)
        out_channels2 = int(in_channels * growth2)
        self.point_conv1 = nn.Sequential(
            PointwiseConv(in_channels, out_channels1),
            nn.BatchNorm1d(out_channels1),
            nn.GLU(dim=1),
        )
        self.depth_conv = nn.Sequential(
            DepthwiseConv(in_channels, kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(in_channels),
            Swish(),
        )
        self.point_conv2 = nn.Sequential(
            PointwiseConv(in_channels, out_channels2),
            nn.BatchNorm1d(out_channels2),
            Swish(),
        )
        self.conv = BasicConv(out_channels2, out_channels2, kernel_size=1, relu=False)
        self.shortcut = BasicConv(in_channels, out_channels2, kernel_size=1, relu=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.point_conv1(x)
        out = self.depth_conv(out)
        out = self.point_conv2(out)
        out = self.conv(out)
        return F.relu(out + self.shortcut(x))


# --------------------------------------------------------------------------- attention
class DualPathProcessing(nn.Module):
    """Overlapping chunking (chunk ``chunk_size``, hop ``hop_size``), as in Asteroid."""

    def __init__(self, chunk_size: int, hop_size: int) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.hop_size = hop_size

    def unfold(self, x: torch.Tensor) -> torch.Tensor:
        """``(batch, channels, frames)`` -> ``(batch, channels, chunk_size, n_chunks)``."""
        batch, channels, _ = x.shape
        unfolded = F.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )
        return unfolded.reshape(batch, channels, self.chunk_size, -1)

    def fold(self, x: torch.Tensor, frames: int) -> torch.Tensor:
        """Overlap-add back to ``(batch, channels, frames)``."""
        batch, channels, chunk_size, n_chunks = x.shape
        folded = F.fold(
            x.reshape(batch, channels * chunk_size, n_chunks),
            (frames, 1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )
        folded = folded / (float(self.chunk_size) / self.hop_size)
        return folded.reshape(batch, channels, frames)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 2), nn.ReLU(), nn.Linear(channels // 2, channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_max = self.fc(F.adaptive_max_pool1d(x, 1).squeeze(-1))
        attn_avg = self.fc(F.adaptive_avg_pool1d(x, 1).squeeze(-1))
        return x * torch.sigmoid(attn_max + attn_avg).unsqueeze(-1)


class GlobalAttention(nn.Module):
    """Multi-head self-attention between chunks (sequence = chunks, features = chunk)."""

    def __init__(self, n_head: int, d_model: int, d_k: int, d_v: int) -> None:
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.temperature = d_k**0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, chunks, _ = x.shape
        q = self.w_qs(x).view(batch, chunks, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_ks(x).view(batch, chunks, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_vs(x).view(batch, chunks, self.n_head, self.d_v).transpose(1, 2)
        attn = torch.softmax(torch.matmul(q / self.temperature, k.transpose(2, 3)), dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, chunks, -1)
        return cast(torch.Tensor, self.fc(out))


class LocalAttention(nn.Module):
    """Depthwise conv inside each chunk, then a spatial attention map from max/mean pooling."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        kernel_size1, kernel_size2 = 31, 7
        self.depth_conv = nn.Sequential(
            DepthwiseConv(channels, kernel_size1, padding=(kernel_size1 - 1) // 2),
            nn.BatchNorm1d(channels),
            Swish(),
        )
        self.conv = BasicConv(2, 1, kernel_size2, padding=(kernel_size2 - 1) // 2, relu=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, chunks, chunk_size = x.shape
        attn = x.permute(0, 2, 1, 3).reshape(batch * chunks, channels, chunk_size)
        attn = self.depth_conv(attn)
        attn = torch.cat([attn.max(dim=1, keepdim=True)[0], attn.mean(dim=1, keepdim=True)], dim=1)
        attn = torch.sigmoid(self.conv(attn))
        attn = attn.view(batch, chunks, 1, chunk_size).permute(0, 2, 1, 3)
        return x * attn


class MultiviewAttentionBlock(nn.Module):
    def __init__(self, channels: int, segment_len: int, head: int) -> None:
        super().__init__()
        self.inter = int(channels / 3)
        d_k = int(segment_len * head)
        self.dsp = DualPathProcessing(segment_len, segment_len // 2)

        self.in_branch0 = BasicConv(channels, self.inter, kernel_size=1)
        self.in_branch1 = BasicConv(channels, self.inter, kernel_size=1)
        self.in_branch2 = BasicConv(channels, self.inter, kernel_size=1)

        self.channel_attn = ChannelAttention(self.inter)
        self.global_attn = GlobalAttention(head, segment_len, d_k, d_k)
        self.local_attn = LocalAttention(self.inter)

        self.out_branch0 = BasicConv(self.inter, self.inter, kernel_size=3, padding=1)
        self.out_branch1 = BasicConv(self.inter, self.inter, kernel_size=3, padding=1)
        self.out_branch2 = BasicConv(self.inter, self.inter, kernel_size=3, padding=1)

        self.conv = BasicConv(self.inter * 3, channels, kernel_size=1, relu=False)
        self.shortcut = BasicConv(channels, channels, kernel_size=1, relu=False)

        self.output_tanh = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1), nn.Tanh())
        self.output_sigmoid = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1), nn.Sigmoid()
        )
        self.gate_conv = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frames = x.shape[-1]
        x0 = self.channel_attn(self.in_branch0(x))

        # (B, N/3, T) -> (B, N/3, chunks, chunk_size)
        x1 = self.dsp.unfold(self.in_branch1(x)).transpose(2, 3)
        x2 = self.dsp.unfold(self.in_branch2(x)).transpose(2, 3)
        batch, channels, chunks, chunk_size = x1.shape

        x1 = self.global_attn(x1.reshape(batch * channels, chunks, chunk_size))
        x1 = x1.view(batch, channels, chunks, chunk_size)
        x2 = self.local_attn(x2)

        x1 = self.dsp.fold(x1.transpose(2, 3), frames)
        x2 = self.dsp.fold(x2.transpose(2, 3), frames)

        out = torch.cat([self.out_branch0(x0), self.out_branch1(x1), self.out_branch2(x2)], dim=1)
        out = self.conv(out)
        gated = self.output_tanh(out) * self.output_sigmoid(out)
        return cast(torch.Tensor, self.shortcut(x) + self.gate_conv(gated))


# --------------------------------------------------------------------------- network
class MaskGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.output = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1), nn.Tanh())
        self.output_gate = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1), nn.Sigmoid())
        self.mask = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.mask(self.output(x) * self.output_gate(x)))


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        segment_len: int,
        head: int,
        attention: bool,
    ) -> None:
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size, stride),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
        )
        self.conv_block = ResConBlock(in_channels, growth1=2, growth2=2)
        self.attn_block = (
            MultiviewAttentionBlock(out_channels, segment_len, head) if attention else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(self.down_conv(x))
        return self.attn_block(x) if self.attn_block is not None else x


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        segment_len: int,
        head: int,
        attention: bool,
    ) -> None:
        super().__init__()
        self.conv_block = ResConBlock(in_channels, growth1=2, growth2=1 / 2)
        self.up_conv = nn.Sequential(
            nn.ConvTranspose1d(out_channels, out_channels, kernel_size, stride),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        self.attn_block = (
            MultiviewAttentionBlock(out_channels, segment_len, head) if attention else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_conv(self.conv_block(x))
        return self.attn_block(x) if self.attn_block is not None else x


class MANNERBlock(nn.Module):
    def __init__(
        self,
        hidden: int,
        depth: int,
        kernel_size: int,
        stride: int,
        head: int,
        segment_len: int,
        attention_everywhere: bool,
    ) -> None:
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.out_conv = nn.Sequential(nn.Conv1d(hidden, 1, kernel_size=3, stride=1, padding=1))

        encoder, decoder = [], []
        for layer in range(depth):
            channels, wider = hidden * 2**layer, hidden * 2 ** (layer + 1)
            attention = attention_everywhere or layer == depth - 1
            encoder.append(
                Encoder(channels, wider, kernel_size, stride, segment_len, head, attention)
            )
            decoder.append(
                Decoder(wider, channels, kernel_size, stride, segment_len, head, attention)
            )
        decoder.reverse()
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)

        bottleneck = hidden * 2**depth
        self.linear = nn.Sequential(nn.Linear(bottleneck, bottleneck, bias=False), nn.ReLU())
        self.mask_gate = MaskGate(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        enc_out = x
        skips = []
        for encoder in self.encoder:
            x = encoder(x)
            skips.append(x)
        x = self.linear(x.transpose(1, 2)).transpose(1, 2)
        for decoder in self.decoder:
            skip = skips.pop()
            x = decoder(x + skip[..., : x.shape[-1]])
        return cast(torch.Tensor, self.out_conv(enc_out * self.mask_gate(x)))


def _rescale_conv_weights(module: nn.Module, reference: float) -> None:
    """Weight rescaling applied at construction in the official code (as in Demucs)."""
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            scale = (sub.weight.std().detach() / reference) ** 0.5
            sub.weight.data /= scale
            if sub.bias is not None:
                sub.bias.data /= scale


class MANNER(DenoisingModel):
    """Waveform-in, waveform-out MANNER. :meth:`forward` takes ``(batch, 1, samples)`` and
    normalises by the input standard deviation itself.

    The project default runs the model at 4 kHz (the pipeline rate). The official pretrained
    weights are 16 kHz speech models; to start from them (as Tzeng et al., 2025, did) build
    ``MANNERConfig(variant="base", model_sample_rate=16000)`` and use
    :func:`~src.denoising.models.load_weights`.
    """

    eps = 1e-3
    rescale = 0.1

    def __init__(self, config: MANNERConfig | None = None) -> None:
        super().__init__()
        self.config = config = config or MANNERConfig()
        variant = _VARIANTS[config.variant]
        hidden = config.hidden if config.hidden is not None else cast(int, variant["hidden"])
        self.kernel_size = config.kernel_size
        self.stride = config.stride
        self.depth = config.depth
        self.manner_block = MANNERBlock(
            hidden,
            config.depth,
            config.kernel_size,
            config.stride,
            config.head,
            config.segment_len,
            cast(bool, variant["attention_everywhere"]),
        )
        _rescale_conv_weights(self, self.rescale)

    def valid_length(self, length: int) -> int:
        """Smallest length >= ``length`` that the strided convolutions map back exactly."""
        for _ in range(self.depth):
            length = max(math.ceil((length - self.kernel_size) / self.stride) + 1, 1)
        for _ in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        return int(length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        std = x.mean(dim=1, keepdim=True).std(dim=-1, keepdim=True)
        x = x / (self.eps + std)
        length = x.shape[-1]
        x = F.pad(x, (0, self.valid_length(length) - length))
        x = self.manner_block(x)[..., :length]
        return std * x

    def enhance(self, noisy: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self(noisy.unsqueeze(1))).squeeze(1)
