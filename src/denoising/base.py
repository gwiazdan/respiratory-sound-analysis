"""What every denoising network shares: its configuration and the ``enhance`` method."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class DenoisingModel(nn.Module):
    """Base class of the denoising networks.

    Subclasses keep their configuration (a dataclass with at least ``model_sample_rate``) in
    ``self.config`` and implement :meth:`enhance`, which includes any normalisation the model
    needs, so training and inference process a signal in exactly the same way.
    """

    config: Any

    @property
    def sample_rate(self) -> int:
        return int(self.config.model_sample_rate)

    def enhance(self, noisy: torch.Tensor) -> torch.Tensor:
        """``(batch, samples)`` at :attr:`sample_rate` -> enhanced ``(batch, samples)``."""
        raise NotImplementedError


def load_checkpoint(path: str | Path) -> Any:
    """``torch.load`` restricted to tensors and plain data (``weights_only=True``).

    Numpy scalars and dtypes are allowed as well, because third-party training scripts often
    store metrics such as ``best_pesq`` as ``numpy.float64`` next to the weights.
    """
    from numpy._core.multiarray import scalar  # private module, but the pickled name

    numpy_globals: list[Any] = [np.dtype, scalar]
    numpy_globals += [type(np.dtype(kind)) for kind in (np.float32, np.float64, np.int64)]
    with torch.serialization.safe_globals(numpy_globals):
        return torch.load(path, map_location="cpu", weights_only=True)
