from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Aperture:
    radius: float = 2.0
    sky_inner: float = 3.0
    sky_outer: float = 5.0
    max_shift: float = 1.0
    quality_mask: int = -1

    def __post_init__(self):
        if not 0 < self.radius < self.sky_inner < self.sky_outer:
            raise ValueError("Радиусы апертуры и кольца должны возрастать.")
        if not 0 <= self.max_shift <= 3:
            raise ValueError("Допустимое смещение должно быть от 0 до 3 пикселей.")


@dataclass(frozen=True)
class Measurement:
    time: float
    flux: float
    error: float
    background: float
    x: float
    y: float
    quality: int
    reason: str = ""


def relative_magnitude(flux, reference):
    return -2.5 * np.log10(np.asarray(flux) / reference)
