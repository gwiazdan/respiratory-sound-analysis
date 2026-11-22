"""Neural denoising of respiratory recordings (Wave-U-Net, MANNER, CMGAN).

- :mod:`~src.denoising.models` - ``build_model`` / ``load_model``; to add a model, write a
  :class:`~src.denoising.base.DenoisingModel` subclass with a config dataclass and add it to
  ``MODELS``,
- :mod:`~src.denoising.training` - model-specific losses and optimisers,
- :mod:`~src.denoising.inference` - denoising whole recordings and datasets,
- :mod:`~src.denoising.training_data`, :mod:`~src.denoising.noise`,
  :mod:`~src.denoising.musan`, :mod:`~src.denoising.audio_tagging` - noisy/clean training pairs.
"""

from src.denoising.base import DenoisingModel
from src.denoising.cmgan import CMGAN, CMGANConfig
from src.denoising.inference import enhance_signal, save_processed_audio
from src.denoising.manner import MANNER, MANNERConfig
from src.denoising.models import MODELS, build_model, load_model, load_weights, model_config
from src.denoising.wave_u_net import WaveUNet, WaveUNetConfig

__all__ = [
    "CMGAN",
    "MANNER",
    "MODELS",
    "CMGANConfig",
    "DenoisingModel",
    "MANNERConfig",
    "WaveUNet",
    "WaveUNetConfig",
    "build_model",
    "enhance_signal",
    "load_model",
    "load_weights",
    "model_config",
    "save_processed_audio",
]
