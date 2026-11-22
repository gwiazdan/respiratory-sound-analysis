"""CMGAN: Conformer-based Metric GAN for speech enhancement (Cao, Abdulatif & Yang, 2022).

Port of the official implementation (https://github.com/ruizhecao96/CMGAN, MIT licence):
``models/generator.py`` (``TSCNet``), ``models/conformer.py`` and ``models/discriminator.py``.
Module and parameter names are kept identical, so the official generator checkpoint
(``src/best_ckpt/ckpt``; 16 kHz, VoiceBank+DEMAND, ``n_fft=400``, ``hop=100``) loads with
``strict=True``. Tzeng et al. (2025) fine-tuned exactly these weights.

Pipeline of one segment (time-frequency domain):

1. the waveform is scaled to unit RMS (``c = sqrt(L / sum(x^2))``, undone at the end),
2. STFT (Hamming window) and power-law compression of the magnitude (``|X|^0.3``, phase
   kept), giving a ``(batch, 2, frames, bins)`` real/imaginary input,
3. the generator (``TSCNet``):

   - ``DenseEncoder``: 1x1 conv over [magnitude, real, imag] -> dilated DenseNet (4 layers,
     dilation 1/2/4/8 in time) -> conv with stride 2 in frequency,
   - 4 x ``TSCB`` (two-stage conformer block): a conformer over time for every frequency
     bin, then a conformer over frequency for every frame,
   - ``MaskDecoder``: magnitude mask (PReLU output, one slope per bin) applied to the noisy
     magnitude with the noisy phase,
   - ``ComplexDecoder``: a complex residual added to the masked spectrum (phase refinement),

4. decompression, inverse STFT, rescaling.

The metric discriminator (:class:`Discriminator`) learns to predict a quality score of the
enhanced magnitude given the clean one; the generator is trained to make it predict the
maximum. The original uses PESQ, which is defined only for speech at 8/16 kHz; see
:class:`~src.denoising.training.CMGANRecipe` for the score used here.

Frequency resolution: the encoder halves the number of bins and the decoder restores
``2k - 1`` of them, so ``n_fft // 2 + 1`` must be odd, i.e. ``n_fft`` divisible by 4.
The project default at 4 kHz is ``n_fft=100, hop=25`` - the same 25 ms window and 6.25 ms hop
as the original 400/100 at 16 kHz.

Deviations from the official code that do not change results: ``einops`` rearrangements are
written with ``permute``/``reshape``, the magnitude in :meth:`TSCNet.forward` gets a tiny
epsilon inside the square root (the gradient of ``sqrt`` at 0 is infinite).
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from src.denoising.base import DenoisingModel

COMPRESS_EXPONENT = 0.3


# --------------------------------------------------------------------------- #
# Conformer (official models/conformer.py, after lucidrains/conformer)
# --------------------------------------------------------------------------- #
def calc_same_padding(kernel_size: int) -> tuple[int, int]:
    pad = kernel_size // 2
    return pad, pad - (kernel_size + 1) % 2


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class Transpose(nn.Module):
    """``einops.Rearrange('b n c -> b c n')`` and back (no parameters)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2)


class DepthWiseConv1d(nn.Module):
    def __init__(
        self, chan_in: int, chan_out: int, kernel_size: int, padding: tuple[int, int]
    ) -> None:
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.conv(F.pad(x, self.padding)))


class Scale(nn.Module):
    def __init__(self, scale: float, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.fn(x)) * self.scale


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.fn(self.norm(x)))


class Attention(nn.Module):
    """Multi-head self-attention with Shaw's relative positional embedding."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        max_pos_emb: int = 512,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n, _ = x.shape
        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        q, k, v = (t.reshape(batch, n, self.heads, -1).transpose(1, 2) for t in (q, k, v))
        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale

        seq = torch.arange(n, device=x.device)
        dist = (seq[:, None] - seq[None, :]).clamp(-self.max_pos_emb, self.max_pos_emb)
        rel_pos_emb = self.rel_pos_emb(dist + self.max_pos_emb).to(q)
        dots = dots + torch.einsum("bhnd,nrd->bhnr", q, rel_pos_emb) * self.scale

        attn = dots.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).reshape(batch, n, -1)
        return cast(torch.Tensor, self.dropout(self.to_out(out)))


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(x))


class ConformerConvModule(nn.Module):
    def __init__(
        self, dim: int, expansion_factor: int = 2, kernel_size: int = 31, dropout: float = 0.0
    ) -> None:
        super().__init__()
        inner_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose(),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size, calc_same_padding(kernel_size)),
            nn.BatchNorm1d(inner_dim),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Transpose(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(x))


class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        conv_expansion_factor: int = 2,
        conv_kernel_size: int = 31,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        conv_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ff1 = Scale(0.5, PreNorm(dim, FeedForward(dim, ff_mult, ff_dropout)))
        self.attn = PreNorm(dim, Attention(dim, heads, dim_head, attn_dropout))
        self.conv = ConformerConvModule(dim, conv_expansion_factor, conv_kernel_size, conv_dropout)
        self.ff2 = Scale(0.5, PreNorm(dim, FeedForward(dim, ff_mult, ff_dropout)))
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ff1(x) + x
        x = self.attn(x) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        return cast(torch.Tensor, self.post_norm(x))


# --------------------------------------------------------------------------- #
# Generator (official models/generator.py)
# --------------------------------------------------------------------------- #
class DilatedDenseNet(nn.Module):
    def __init__(self, depth: int = 4, in_channels: int = 64) -> None:
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2**i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, f"pad{i + 1}", nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0))
            setattr(
                self,
                f"conv{i + 1}",
                nn.Conv2d(
                    self.in_channels * (i + 1),
                    self.in_channels,
                    kernel_size=self.kernel_size,
                    dilation=(dil, 1),
                ),
            )
            setattr(self, f"norm{i + 1}", nn.InstanceNorm2d(in_channels, affine=True))
            setattr(self, f"prelu{i + 1}", nn.PReLU(self.in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        out = x
        for i in range(self.depth):
            out = getattr(self, f"pad{i + 1}")(skip)
            out = getattr(self, f"conv{i + 1}")(out)
            out = getattr(self, f"norm{i + 1}")(out)
            out = getattr(self, f"prelu{i + 1}")(out)
            skip = torch.cat([out, skip], dim=1)
        return out


class DenseEncoder(nn.Module):
    def __init__(self, in_channel: int, channels: int = 64) -> None:
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, (1, 1), (1, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.dilated_dense = DilatedDenseNet(depth=4, in_channels=channels)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.conv_2(self.dilated_dense(self.conv_1(x))))


class TSCB(nn.Module):
    """Two-stage conformer: time conformer per frequency bin, then frequency per frame."""

    def __init__(self, num_channel: int = 64) -> None:
        super().__init__()
        options = {
            "dim": num_channel,
            "dim_head": num_channel // 4,
            "heads": 4,
            "conv_kernel_size": 31,
            "attn_dropout": 0.2,
            "ff_dropout": 0.2,
        }
        self.time_conformer = ConformerBlock(**options)  # type: ignore[arg-type]
        self.freq_conformer = ConformerBlock(**options)  # type: ignore[arg-type]

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        b, c, t, f = x_in.size()
        x_t: torch.Tensor = x_in.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x_t = self.time_conformer(x_t) + x_t
        x_f: torch.Tensor = x_t.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x_f = self.freq_conformer(x_f) + x_f
        return x_f.view(b, t, f, c).permute(0, 3, 1, 2)


class SPConvTranspose2d(nn.Module):
    """Sub-pixel convolution: upsamples the frequency axis by ``r``."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: tuple[int, int], r: int = 1
    ) -> None:
        super().__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.conv(self.pad1(x))
        batch_size, nchannels, height, width = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, height, width))
        out = out.permute(0, 2, 3, 4, 1)
        return out.contiguous().view((batch_size, nchannels // self.r, height, -1))


class MaskDecoder(nn.Module):
    def __init__(self, num_features: int, num_channel: int = 64, out_channel: int = 1) -> None:
        super().__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.conv_1 = nn.Conv2d(num_channel, out_channel, (1, 2))
        self.norm = nn.InstanceNorm2d(out_channel, affine=True)
        self.prelu = nn.PReLU(out_channel)
        self.final_conv = nn.Conv2d(out_channel, out_channel, (1, 1))
        self.prelu_out = nn.PReLU(num_features, init=-0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.conv_1(x)
        x = self.prelu(self.norm(x))
        x = self.final_conv(x).permute(0, 3, 2, 1).squeeze(-1)
        return cast(torch.Tensor, self.prelu_out(x)).permute(0, 2, 1).unsqueeze(1)


class ComplexDecoder(nn.Module):
    def __init__(self, num_channel: int = 64) -> None:
        super().__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.prelu = nn.PReLU(num_channel)
        self.norm = nn.InstanceNorm2d(num_channel, affine=True)
        self.conv = nn.Conv2d(num_channel, 2, (1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.prelu(self.norm(x))
        return cast(torch.Tensor, self.conv(x))


class TSCNet(nn.Module):
    """CMGAN generator. Input ``(batch, 2, frames, bins)`` compressed real/imag spectrum;
    returns the enhanced compressed real and imaginary parts, each ``(batch, 1, frames, bins)``.
    """

    def __init__(self, num_channel: int = 64, num_features: int = 201) -> None:
        super().__init__()
        self.dense_encoder = DenseEncoder(in_channel=3, channels=num_channel)
        self.TSCB_1 = TSCB(num_channel=num_channel)
        self.TSCB_2 = TSCB(num_channel=num_channel)
        self.TSCB_3 = TSCB(num_channel=num_channel)
        self.TSCB_4 = TSCB(num_channel=num_channel)
        self.mask_decoder = MaskDecoder(num_features, num_channel=num_channel, out_channel=1)
        self.complex_decoder = ComplexDecoder(num_channel=num_channel)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2 + 1e-12).unsqueeze(1)
        noisy_phase = torch.atan2(x[:, 1, :, :], x[:, 0, :, :]).unsqueeze(1)
        x_in = torch.cat([mag, x], dim=1)

        out = self.dense_encoder(x_in)
        out = self.TSCB_1(out)
        out = self.TSCB_2(out)
        out = self.TSCB_3(out)
        out = self.TSCB_4(out)

        mask = self.mask_decoder(out)
        out_mag = mask * mag
        complex_out = self.complex_decoder(out)
        final_real = out_mag * torch.cos(noisy_phase) + complex_out[:, 0:1, :, :]
        final_imag = out_mag * torch.sin(noisy_phase) + complex_out[:, 1:2, :, :]
        return final_real, final_imag


# --------------------------------------------------------------------------- #
# Metric discriminator (official models/discriminator.py)
# --------------------------------------------------------------------------- #
class LearnableSigmoid(nn.Module):
    def __init__(self, in_features: int, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.beta * torch.sigmoid(self.slope * x)


class Discriminator(nn.Module):
    """Predicts a quality score of ``y`` (enhanced magnitude) relative to ``x`` (clean)."""

    def __init__(self, ndf: int = 16, in_channel: int = 2) -> None:
        super().__init__()
        sn = nn.utils.spectral_norm
        self.layers = nn.Sequential(
            sn(nn.Conv2d(in_channel, ndf, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.PReLU(ndf),
            sn(nn.Conv2d(ndf, ndf * 2, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.PReLU(2 * ndf),
            sn(nn.Conv2d(ndf * 2, ndf * 4, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.PReLU(4 * ndf),
            sn(nn.Conv2d(ndf * 4, ndf * 8, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.PReLU(8 * ndf),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            sn(nn.Linear(ndf * 8, ndf * 4)),
            nn.Dropout(0.3),
            nn.PReLU(4 * ndf),
            sn(nn.Linear(ndf * 4, 1)),
            LearnableSigmoid(1, beta=2.0),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.layers(torch.cat([x, y], dim=1)))


# --------------------------------------------------------------------------- #
# Spectral front end shared by inference and training
# --------------------------------------------------------------------------- #
def power_compress(spec: torch.Tensor) -> torch.Tensor:
    """Complex ``(batch, bins, frames)`` -> compressed real/imag ``(batch, 2, bins, frames)``."""
    mag = spec.abs() ** COMPRESS_EXPONENT
    phase = spec.angle()
    return torch.stack([mag * torch.cos(phase), mag * torch.sin(phase)], dim=1)


def power_uncompress(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    """Compressed real/imag ``(batch, 1, bins, frames)`` -> complex ``(batch, bins, frames)``."""
    spec = torch.complex(real.float(), imag.float()).squeeze(1)
    mag = spec.abs() ** (1.0 / COMPRESS_EXPONENT)
    return torch.polar(mag, spec.angle())


class CMGANFrontEnd:
    """STFT/compression around :class:`TSCNet`, with the official normalisation."""

    def __init__(self, n_fft: int, hop_length: int) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length

    @staticmethod
    def rms_scale(waveform: torch.Tensor) -> torch.Tensor:
        """``c`` such that ``waveform * c`` has unit RMS, shape ``(batch, 1)``."""
        energy = waveform.float().pow(2).sum(dim=-1, keepdim=True)
        return torch.sqrt(waveform.shape[-1] / energy.clamp_min(1e-10))

    def spectrum(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compressed spectrum ``(batch, 2, bins, frames)`` of an already scaled waveform."""
        window = torch.hamming_window(self.n_fft, device=waveform.device)
        spec = torch.stft(
            waveform.float(),
            self.n_fft,
            self.hop_length,
            window=window,
            onesided=True,
            return_complex=True,
        )
        return power_compress(spec)

    def waveform(self, real: torch.Tensor, imag: torch.Tensor, length: int) -> torch.Tensor:
        """Inverse of :meth:`spectrum` for ``(batch, 1, bins, frames)`` real/imag parts."""
        window = torch.hamming_window(self.n_fft, device=real.device)
        return torch.istft(
            power_uncompress(real, imag),
            self.n_fft,
            self.hop_length,
            window=window,
            onesided=True,
            length=length,
        )

    def run(
        self,
        model: nn.Module,
        scaled: torch.Tensor,
        autocast: Callable[[], AbstractContextManager[object]] = nullcontext,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Enhanced compressed real/imag ``(batch, 1, bins, frames)`` and the enhanced (still
        scaled) waveform. Only the network runs under ``autocast``; STFTs stay in float32."""
        spec = self.spectrum(scaled).permute(0, 1, 3, 2)  # (batch, 2, frames, bins)
        with autocast():
            est_real, est_imag = model(spec)
        est_real = est_real.float().permute(0, 1, 3, 2)
        est_imag = est_imag.float().permute(0, 1, 3, 2)
        return est_real, est_imag, self.waveform(est_real, est_imag, scaled.shape[-1])


@dataclass(frozen=True)
class CMGANConfig:
    model_sample_rate: int = 4000
    n_fft: int = 100
    hop_length: int = 25
    num_channel: int = 64

    def __post_init__(self) -> None:
        if self.n_fft % 4:
            raise ValueError(
                f"n_fft must be divisible by 4 (odd number of bins), got {self.n_fft}."
            )


class CMGAN(TSCNet, DenoisingModel):
    """CMGAN generator with its STFT front end.

    The project default runs the model at 4 kHz with a 25 ms window (``n_fft=100``,
    ``hop_length=25``). The parameter names are those of :class:`TSCNet`, so the official
    16 kHz speech weights load into ``CMGANConfig(model_sample_rate=16000, n_fft=400,
    hop_length=100)`` with :func:`~src.denoising.models.load_weights`.
    """

    def __init__(self, config: CMGANConfig | None = None) -> None:
        config = config or CMGANConfig()
        super().__init__(num_channel=config.num_channel, num_features=config.n_fft // 2 + 1)
        self.config = config

    @property
    def front_end(self) -> CMGANFrontEnd:
        return CMGANFrontEnd(self.config.n_fft, self.config.hop_length)

    def enhance(self, noisy: torch.Tensor) -> torch.Tensor:
        length = noisy.shape[-1]
        # the STFT needs a few frames; very short tails are zero-padded and trimmed back
        minimum = 2 * self.config.n_fft
        if length < minimum:
            noisy = F.pad(noisy, (0, minimum - length))
        scale = CMGANFrontEnd.rms_scale(noisy)
        _, _, enhanced = self.front_end.run(self, noisy * scale)
        return (enhanced / scale)[..., :length]
