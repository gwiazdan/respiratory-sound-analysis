"""Model-specific training recipes: loss, optimiser, learning-rate schedule and defaults.

The training loop (notebook ``05_denoising_training.ipynb``) is shared by all models; only
what differs between them lives here. :func:`recipe_for` returns the recipe for a network's
class, or :class:`TrainingRecipe` (L1 on the enhanced waveform, Adam) for models that have
none yet.

Every recipe's :meth:`~TrainingRecipe.loss` takes ``(clean, noisy)`` batches of shape
``(batch, samples)`` at the model sample rate and returns the loss together with the enhanced
signal in the original amplitude scale, so validation metrics are comparable between models.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from typing import ClassVar, cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from src.denoising.base import DenoisingModel
from src.denoising.cmgan import CMGAN, CMGANFrontEnd, Discriminator
from src.denoising.manner import MANNER
from src.denoising.wave_u_net import WaveUNet

Autocast = Callable[[], AbstractContextManager[object]]


class MultiResolutionSTFTLoss(nn.Module):
    """Spectral convergence + log-magnitude L1 over several STFT resolutions.

    Default resolutions are those of Parallel WaveGAN / facebookresearch/denoiser, used by
    MANNER, defined at 16 kHz: FFT (1024, 2048, 512), hop (120, 240, 50), window
    (600, 1200, 240). For other sample rates they are scaled proportionally, so each
    resolution covers the same duration in milliseconds.
    """

    _REFERENCE = ((1024, 120, 600), (2048, 240, 1200), (512, 50, 240))

    def __init__(
        self,
        sample_rate: int = 16000,
        factor_sc: float = 1.0,
        factor_mag: float = 1.0,
    ) -> None:
        super().__init__()
        ratio = sample_rate / 16000
        self.resolutions = [
            (max(16, round(fft * ratio)), max(1, round(hop * ratio)), max(8, round(win * ratio)))
            for fft, hop, win in self._REFERENCE
        ]
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    @staticmethod
    def _magnitude(x: torch.Tensor, fft: int, hop: int, win: int) -> torch.Tensor:
        window = torch.hann_window(win, device=x.device)
        spectrum = torch.stft(
            x.float(), fft, hop_length=hop, win_length=win, window=window, return_complex=True
        )
        return torch.sqrt(torch.clamp(spectrum.real**2 + spectrum.imag**2, min=1e-7))

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        convergence = estimate.new_zeros(())
        magnitude = estimate.new_zeros(())
        for fft, hop, win in self.resolutions:
            est = self._magnitude(estimate, fft, hop, win)
            ref = self._magnitude(target, fft, hop, win)
            convergence = convergence + torch.linalg.norm(ref - est) / torch.linalg.norm(ref)
            magnitude = magnitude + F.l1_loss(est.log(), ref.log())
        count = len(self.resolutions)
        return (self.factor_sc * convergence + self.factor_mag * magnitude) / count


class TrainingRecipe:
    """Default recipe: L1 between the enhanced and clean waveform, Adam, constant lr."""

    #: Short description of where the settings come from (shown in the notebook).
    source: ClassVar[str] = "domyślny przepis: L1 na fali, Adam, stałe lr"
    batch_size: ClassVar[int] = 16
    lr: ClassVar[float] = 1e-4
    #: 50 epochs for every model, as in the project's runs (the validation SI-SDR of
    #: Wave-U-Net and MANNER levels off around epoch 45)
    epochs: ClassVar[int] = 50

    def __init__(self, network: DenoisingModel) -> None:
        self.sample_rate = network.sample_rate

    def defaults(self) -> dict[str, object]:
        return {"batch_size": self.batch_size, "lr": self.lr, "epochs": self.epochs}

    def loss(
        self, model: nn.Module, clean: torch.Tensor, noisy: torch.Tensor, autocast: Autocast
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with autocast():
            enhanced = cast(DenoisingModel, model).enhance(noisy)
        enhanced = enhanced.float()
        return F.l1_loss(enhanced, clean), enhanced

    def optimizer(self, parameters: Iterable[nn.Parameter], lr: float) -> torch.optim.Optimizer:
        return torch.optim.Adam(parameters, lr=lr)

    def scheduler(
        self, optimizer: torch.optim.Optimizer, steps_per_epoch: int, epochs: int, lr: float
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Scheduler stepped after every optimiser step, or ``None`` for a constant lr."""
        return None

    def to(self, device: torch.device) -> TrainingRecipe:
        for value in vars(self).values():
            if isinstance(value, nn.Module):
                value.to(device)
        return self


class WaveUNetRecipe(TrainingRecipe):
    """Stoller et al. (2018): Adam lr 1e-4; L1 on the waveform.

    The loss is computed on the peak-normalised signal the network actually sees (the same
    scaling as :meth:`WaveUNet.enhance`), optionally with a multi-resolution STFT term.
    Batch 64 (the paper used 16) and 50 epochs are the settings of the project's run.
    """

    source = (
        "Stoller i in. (2018): Adam, lr 1e-4, L1 na fali po normalizacji szczytu; "
        "batch 64 i 50 epok jak w naszych przebiegach"
    )
    batch_size = 64

    def __init__(self, network: DenoisingModel, stft_loss_weight: float = 0.0) -> None:
        if not isinstance(network, WaveUNet):
            raise TypeError("WaveUNetRecipe needs a WaveUNet.")
        super().__init__(network)
        self.peak_normalize = network.config.peak_normalize
        self.stft_loss_weight = stft_loss_weight
        self.stft_loss = MultiResolutionSTFTLoss(self.sample_rate) if stft_loss_weight > 0 else None

    def loss(
        self, model: nn.Module, clean: torch.Tensor, noisy: torch.Tensor, autocast: Autocast
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = WaveUNet.peak_scale(noisy) if self.peak_normalize else torch.ones_like(noisy[:, :1])
        target = clean * scale
        with autocast():
            prediction = model((noisy * scale).unsqueeze(1)).squeeze(1)
        prediction = prediction.float()
        loss = F.l1_loss(prediction, target)
        if self.stft_loss is not None:
            loss = loss + self.stft_loss_weight * self.stft_loss(prediction, target)
        return loss, prediction / scale


class MANNERRecipe(TrainingRecipe):
    """Official MANNER training (github.com/winddori2002/MANNER, ``src/train.py``).

    - loss: L1 + multi-resolution STFT loss (factors 0.5/0.5) for both the clean estimate and
      the removed noise, mixed per example by the clean-energy ratio (``WeightedLoss``),
    - Adam with OneCycleLR (``max_lr=1e-3``); the official batch size is 4.

    The official run uses 350 epochs on VoiceBank+DEMAND, Tzeng et al. (2025) trained for 30
    epochs from scratch. The defaults here, batch 32 and 50 epochs, are the settings of the
    project's best run (batch 8 is about 3x slower per epoch). For fine-tuning the official
    pretrained weights they used lr 5e-5 for 20 epochs (set ``lr`` and ``epochs`` and pass
    ``one_cycle=False`` for a constant lr).
    """

    source = (
        "oficjalny kod MANNER: L1 + MR-STFT (0,5/0,5) dla mowy i szumu z wagą energii, "
        "Adam + OneCycleLR (max lr 1e-3); batch 32 i 50 epok jak w naszych przebiegach "
        "(oryginał: batch 4)"
    )
    batch_size = 32
    lr = 1e-3  # peak learning rate of OneCycleLR

    def __init__(self, network: DenoisingModel, one_cycle: bool = True) -> None:
        if not isinstance(network, MANNER):
            raise TypeError("MANNERRecipe needs a MANNER network.")
        super().__init__(network)
        self.one_cycle = one_cycle
        self.stft_loss = MultiResolutionSTFTLoss(self.sample_rate, factor_sc=0.5, factor_mag=0.5)

    def _branch_loss(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # per-example L1 plus a batch-level STFT term, as in the official code
        stft_term = cast(torch.Tensor, self.stft_loss(estimate, target))
        return (estimate - target).abs().mean(dim=-1) + stft_term

    def loss(
        self, model: nn.Module, clean: torch.Tensor, noisy: torch.Tensor, autocast: Autocast
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with autocast():
            estimate = model(noisy.unsqueeze(1)).squeeze(1)
        estimate = estimate.float()
        noise_label = noisy - clean
        noise_estimate = noisy - estimate

        clean_loss = self._branch_loss(estimate, clean)
        noise_loss = self._branch_loss(noise_estimate, noise_label)
        clean_energy = clean.pow(2).sum(dim=1)
        weight = clean_energy / (clean_energy + noise_label.pow(2).sum(dim=1) + 2e-7)
        loss = torch.mean(weight * clean_loss + (1 - weight) * noise_loss)
        return loss, estimate

    def optimizer(self, parameters: Iterable[nn.Parameter], lr: float) -> torch.optim.Optimizer:
        # OneCycleLR overrides the initial lr (it starts at max_lr / 25)
        return torch.optim.Adam(parameters, lr=lr)

    def scheduler(
        self, optimizer: torch.optim.Optimizer, steps_per_epoch: int, epochs: int, lr: float
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        if not self.one_cycle:
            return None
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epochs
        )


def _si_sdr(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Per-example SI-SDR in dB for ``(batch, samples)`` tensors."""
    reference = reference - reference.mean(dim=-1, keepdim=True)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    energy = reference.pow(2).sum(-1, keepdim=True).clamp_min(1e-10)
    target = (estimate * reference).sum(-1, keepdim=True) / energy * reference
    ratio = target.pow(2).sum(-1) / (estimate - target).pow(2).sum(-1).clamp_min(1e-10)
    return 10 * torch.log10(ratio.clamp_min(1e-10))


class CMGANRecipe(TrainingRecipe):
    """Official CMGAN training (github.com/ruizhecao96/CMGAN, ``src/train.py``).

    - generator loss: ``0.1 * L_RI + 0.9 * L_mag + 0.2 * L_time + 0.05 * L_GAN`` - MSE of the
      compressed real/imaginary parts, MSE of the compressed magnitude, L1 of the waveform and
      the metric-GAN term (MSE between the discriminator's score of the estimate and 1),
    - metric discriminator: MSE to 1 for (clean, clean) and to the normalised quality score
      for (clean, estimate); AdamW with twice the generator's lr,
    - AdamW lr 5e-4, step decay x0.5, random 2 s crops of every training segment
      (``cut_len``), inputs scaled to unit RMS; batch 8 here (4 in the original).

    CMGAN is slow: about 0.5 s per step at batch 8 on an RTX 4070 Ti SUPER, i.e. ~30 min per
    full epoch of the project's data. ``TRAIN["max_steps"]`` in the notebook limits an epoch.

    Differences from the original, forced by the task:

    - the quality score is **SI-SDR** mapped linearly from ``metric_range_db`` to [0, 1]
      instead of PESQ (normalised as ``(PESQ - 1) / 3.5``), because PESQ models human
      perception of speech and is defined only for 8/16 kHz,
    - the official run decays the lr every 30 of 120 epochs; here every quarter of
      ``epochs``,
    - the discriminator step runs right before the generator step (the original runs it right
      after) on the same, detached estimate, so the loop shared by all models is unchanged.
      Its weights are not stored in checkpoints: resuming restarts the discriminator.

    ``discriminator=False`` trains the generator only (the "generator only" ablation of
    Tzeng et al., 2025, Table 9).
    """

    source = (
        "oficjalny kod CMGAN: 0,1·RI + 0,9·mag + 0,2·czas + 0,05·GAN, dyskryminator metryczny "
        "z SI-SDR zamiast PESQ, AdamW lr 5e-4, spadek lr o połowę co 1/4 treningu, "
        "wycinki 2 s; batch 8, 50 epok (oryginał: batch 4)"
    )
    batch_size = 8
    lr = 5e-4

    def __init__(
        self,
        network: DenoisingModel,
        discriminator: bool = True,
        loss_weights: tuple[float, float, float, float] = (0.1, 0.9, 0.2, 0.05),
        cut_seconds: float | None = 2.0,
        metric_range_db: tuple[float, float] = (-5.0, 35.0),
    ) -> None:
        if not isinstance(network, CMGAN):
            raise TypeError("CMGANRecipe needs a CMGAN network.")
        super().__init__(network)
        self.front_end: CMGANFrontEnd = network.front_end
        self.loss_weights = loss_weights
        rate = self.sample_rate
        self.cut = None if cut_seconds is None else round(cut_seconds * rate)
        self.metric_range_db = metric_range_db
        self.discriminator = Discriminator(ndf=16) if discriminator else None
        self.discriminator_optimizer: torch.optim.Optimizer | None = None
        self.discriminator_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    def optimizer(self, parameters: Iterable[nn.Parameter], lr: float) -> torch.optim.Optimizer:
        if self.discriminator is not None:
            self.discriminator_optimizer = torch.optim.AdamW(
                self.discriminator.parameters(), lr=2 * lr
            )
        return torch.optim.AdamW(parameters, lr=lr)

    def scheduler(
        self, optimizer: torch.optim.Optimizer, steps_per_epoch: int, epochs: int, lr: float
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        step_size = max(1, epochs // 4) * steps_per_epoch
        if self.discriminator_optimizer is not None:
            self.discriminator_scheduler = torch.optim.lr_scheduler.StepLR(
                self.discriminator_optimizer, step_size=step_size, gamma=0.5
            )
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)

    def quality(self, estimate: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        """Discriminator target in [0, 1] (the role of normalised PESQ)."""
        low, high = self.metric_range_db
        return ((_si_sdr(estimate, clean) - low) / (high - low)).clamp(0.0, 1.0)

    @staticmethod
    def _magnitude(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(real**2 + imag**2 + 1e-12)

    def loss(
        self, model: nn.Module, clean: torch.Tensor, noisy: torch.Tensor, autocast: Autocast
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if model.training and self.cut is not None and noisy.shape[-1] > self.cut:
            start = int(torch.randint(0, noisy.shape[-1] - self.cut + 1, ()).item())
            clean = clean[:, start : start + self.cut]
            noisy = noisy[:, start : start + self.cut]
        scale = CMGANFrontEnd.rms_scale(noisy)
        clean, noisy = clean.float() * scale, noisy.float() * scale

        est_real, est_imag, est_audio = self.front_end.run(model, noisy, autocast)
        clean_spec = self.front_end.spectrum(clean)
        clean_real, clean_imag = clean_spec[:, 0:1], clean_spec[:, 1:2]
        est_mag = self._magnitude(est_real, est_imag)
        clean_mag = self._magnitude(clean_real, clean_imag)

        w_ri, w_mag, w_time, w_gan = self.loss_weights
        loss = (
            w_ri * (F.mse_loss(est_real, clean_real) + F.mse_loss(est_imag, clean_imag))
            + w_mag * F.mse_loss(est_mag, clean_mag)
            + w_time * F.l1_loss(est_audio, clean)
        )
        if model.training and self.discriminator is not None:
            if self.discriminator_optimizer is None:
                raise RuntimeError("Call recipe.optimizer(...) before training.")
            discriminator = self.discriminator.train()
            ones = torch.ones(clean.shape[0], device=clean.device)
            target = self.quality(est_audio.detach(), clean)
            d_loss = F.mse_loss(discriminator(clean_mag, clean_mag).flatten(), ones)
            d_loss = d_loss + F.mse_loss(
                discriminator(clean_mag, est_mag.detach()).flatten(), target
            )
            self.discriminator_optimizer.zero_grad(set_to_none=True)
            d_loss.backward()
            self.discriminator_optimizer.step()
            if self.discriminator_scheduler is not None:
                self.discriminator_scheduler.step()
            # the generator's backward also fills the discriminator's .grad; they are
            # cleared before the next discriminator step
            adversarial = F.mse_loss(discriminator(clean_mag, est_mag).flatten(), ones)
            loss = loss + w_gan * adversarial
        return loss, est_audio / scale


_RECIPES: dict[type[DenoisingModel], type[TrainingRecipe]] = {
    WaveUNet: WaveUNetRecipe,
    MANNER: MANNERRecipe,
    CMGAN: CMGANRecipe,
}


def recipe_for(network: DenoisingModel, **options: object) -> TrainingRecipe:
    """Recipe for the network's class (the default recipe for other models)."""
    recipe_class = _RECIPES.get(type(network), TrainingRecipe)
    return recipe_class(network, **options)  # type: ignore[arg-type]
