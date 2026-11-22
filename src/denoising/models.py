"""Available denoising models and their checkpoints.

A checkpoint written by the training notebook is a dict
``{"model": <name>, "config": {...}, "state_dict": {...}, ...}``::

    model = build_model("wave-u-net", num_layers=12)   # new, untrained network
    model = load_model("models/wave_u_net/best.pt")    # trained network, evaluation mode
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from src.denoising.base import DenoisingModel, load_checkpoint
from src.denoising.cmgan import CMGAN, CMGANConfig
from src.denoising.manner import MANNER, MANNERConfig
from src.denoising.wave_u_net import WaveUNet, WaveUNetConfig

#: name -> (network class, configuration class)
MODELS: dict[str, tuple[type[DenoisingModel], type[Any]]] = {
    "wave-u-net": (WaveUNet, WaveUNetConfig),
    "manner": (MANNER, MANNERConfig),
    "cmgan": (CMGAN, CMGANConfig),
}


def build_model(name: str, **options: Any) -> DenoisingModel:
    """Untrained network ``name`` with configuration ``options`` (see its ``*Config``)."""
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}. Available: {', '.join(MODELS)}")
    network, config = MODELS[name]
    return network(config(**options))


def model_config(model: DenoisingModel) -> dict[str, Any]:
    """Configuration to store in a checkpoint."""
    return asdict(model.config)


def load_model(path: str | Path, device: str | torch.device = "cpu") -> DenoisingModel:
    """Network saved by the training notebook, on ``device`` and in evaluation mode."""
    state = load_checkpoint(path)
    model = build_model(state["model"], **state["config"])
    model.load_state_dict(state["state_dict"])
    return model.to(device).eval()


def load_weights(model: DenoisingModel, path: str | Path) -> DenoisingModel:
    """Load weights into ``model`` from our checkpoint or a bare ``state_dict`` (e.g. the
    official pretrained weights used to initialise fine-tuning)."""
    state = load_checkpoint(path)
    if isinstance(state, dict):
        state = state.get("state_dict", state.get("model", state))
    model.load_state_dict(state)
    return model
