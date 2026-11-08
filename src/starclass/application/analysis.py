import math

from starclass.core.photometry import Measurement


def inspect_series(path, reader, reference, detector):
    with reader(path) as (meta, frame):
        image = reference(meta, frame)
        return meta, image, detector(image)


def extract(
    path, source, aperture, reader, measure, neighbors=(), progress=None, cancelled=None
):
    results = []
    with reader(path) as (meta, frame):
        seen = set()
        for index, (time, quality) in enumerate(zip(meta.time, meta.quality)):
            if cancelled and cancelled():
                raise InterruptedError("Измерения отменены.")
            reason = ""
            flux = error = background = x = y = math.nan
            if not math.isfinite(time):
                reason = "invalid_time"
            elif int(quality) & aperture.quality_mask:
                reason = "quality"
            elif time in seen:
                reason = "duplicate_time"
            else:
                try:
                    flux, error, background, x, y = measure(
                        *frame(index), source, aperture, neighbors
                    )
                    seen.add(time)
                    if flux <= 0:
                        reason = "nonpositive_flux"
                except ValueError as exc:
                    reason = str(exc)
            results.append(
                Measurement(time, flux, error, background, x, y, int(quality), reason)
            )
            if progress:
                progress(index + 1, len(meta.time))
    return meta, results
