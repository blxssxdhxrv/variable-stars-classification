from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Series:
    path: Path
    time: np.ndarray
    quality: np.ndarray
    shape: tuple[int, int]
    time_reference: float
    time_scale: str
    flux_unit: str


@dataclass(frozen=True)
class Source:
    x: float
    y: float
