import csv
import os
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from starclass.core.photometry import relative_magnitude


def export_curve(path, measurements):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    good = [row.flux for row in measurements if not row.reason]
    reference = np.median(good) if good else np.nan
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False, newline=""
        ) as handle:
            temporary = handle.name
            fields = list(asdict(measurements[0])) + ["relative_flux", "mag", "mag_err"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in measurements:
                extra = {}
                if not row.reason:
                    extra = {
                        "relative_flux": row.flux / reference,
                        "mag": relative_magnitude(row.flux, reference),
                        "mag_err": 2.5 / np.log(10) * row.error / row.flux,
                    }
                writer.writerow(asdict(row) | extra)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
