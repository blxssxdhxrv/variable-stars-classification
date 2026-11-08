import warnings

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture
from photutils.centroids import centroid_com
from photutils.detection import DAOStarFinder

from starclass.core.observations import Source


def detect(image, fwhm=2.0, threshold=5.0):
    _, background, noise = sigma_clipped_stats(image, mask=~np.isfinite(image))
    if not np.isfinite(noise) or noise <= 0:
        return []
    mask = ~np.isfinite(image)
    sources = DAOStarFinder(fwhm=fwhm, threshold=threshold * noise)(
        np.where(mask, 0, image - background), mask=mask
    )
    if sources is None:
        return []
    x_key = "x_centroid" if "x_centroid" in sources.colnames else "xcentroid"
    y_key = "y_centroid" if "y_centroid" in sources.colnames else "ycentroid"
    return [Source(float(row[x_key]), float(row[y_key])) for row in sources]


def measure(image, error, source, config, neighbors=()):
    height, width = image.shape
    x, y = source.x, source.y
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("Источник находится за пределами кадра.")
    yy, xx = np.indices(image.shape)
    distance = np.hypot(xx - x, yy - y)
    sky = (distance >= config.sky_inner) & (distance <= config.sky_outer)
    for neighbor in neighbors:
        if np.hypot(neighbor.x - x, neighbor.y - y) > 1:
            sky &= np.hypot(xx - neighbor.x, yy - neighbor.y) > config.radius
    sky &= np.isfinite(image)
    if np.count_nonzero(sky) < 12:
        raise ValueError("Недостаточно пикселей для оценки фона.")
    _, background, scatter = sigma_clipped_stats(image[sky])
    if config.max_shift:
        valid = (distance <= config.radius) & np.isfinite(image)
        signal = np.where(valid, np.maximum(image - background, 0), 0)
        if not signal.sum():
            raise ValueError("Источник не найден на кадре.")
        cx, cy = centroid_com(signal)
        if not np.isfinite(cx + cy) or np.hypot(cx - x, cy - y) > config.max_shift:
            raise ValueError("Источник сместился за пределы поиска.")
        x, y = cx, cy
    radius = config.radius
    if min(x + 0.5, y + 0.5, width - 0.5 - x, height - 0.5 - y) < radius:
        raise ValueError("Апертура пересекает край. Нужна большая вырезка.")
    weights = (
        CircularAperture((x, y), r=radius).to_mask(method="exact").to_image(image.shape)
    )
    inside = weights > 0
    if not np.isfinite(image[inside]).all():
        raise ValueError("Пропущенные пиксели внутри апертуры.")
    w = weights[inside]
    flux = float(w @ (image[inside] - background))
    sigma = error[inside]
    uncertainty = np.nan
    if np.all(np.isfinite(sigma) & (sigma > 0)):
        uncertainty = float(
            np.hypot(
                np.linalg.norm(w * sigma),
                w.sum() * 1.2533 * scatter / np.sqrt(sky.sum()),
            )
        )
    return flux, uncertainty, float(background), float(x), float(y)


def reference_image(meta, frame):
    good = np.flatnonzero(np.isfinite(meta.time) & (meta.quality == 0))
    if not len(good):
        raise ValueError("Нет пригодных кадров для опорного изображения.")
    indices = good[np.linspace(0, len(good) - 1, min(24, len(good)), dtype=int)]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return np.nanmedian([frame(i)[0] for i in indices], axis=0)
