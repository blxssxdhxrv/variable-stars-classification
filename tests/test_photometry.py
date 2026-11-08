import numpy as np
import pytest

from starclass.core.observations import Source
from starclass.core.photometry import Aperture
from starclass.infrastructure.imaging import measure


def test_aperture_recovers_signal_over_background():
    image = np.full((15, 15), 20.0)
    image[7, 7] += 100
    flux, sigma, background, x, y = measure(
        image, np.ones_like(image), Source(7, 7), Aperture()
    )
    assert flux == pytest.approx(100)
    assert sigma > 0
    assert (background, x, y) == pytest.approx((20, 7, 7))


def test_aperture_at_edge_is_rejected():
    image = np.ones((15, 15))
    with pytest.raises(ValueError, match="пересекает край"):
        measure(image, image, Source(0, 7), Aperture(max_shift=0))
