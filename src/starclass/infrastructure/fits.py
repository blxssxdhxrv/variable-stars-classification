from contextlib import contextmanager
from pathlib import Path

import numpy as np
from astropy.io import fits

from starclass.core.observations import Series


@contextmanager
def open_series(path):
    path = Path(path)
    with fits.open(path, memmap=True) as hdus:
        table = next(
            (
                hdu
                for hdu in hdus
                if isinstance(hdu, fits.BinTableHDU)
                and {"TIME", "FLUX", "FLUX_ERR", "QUALITY"} <= set(hdu.columns.names)
            ),
            None,
        )
        if table is None or hdus[0].header.get("TELESCOP", "").upper() != "TESS":
            raise ValueError("Ожидается FITS серии TESS TESScut или TPF.")
        time = np.asarray(table.data["TIME"], dtype=float)
        flux = table.data["FLUX"]
        error = table.data["FLUX_ERR"]
        quality = np.asarray(table.data["QUALITY"], dtype=int)
        if flux.ndim != 3 or not len(time) or error.shape != flux.shape:
            raise ValueError("Неверные размеры данных FITS.")
        order = np.argsort(time, kind="stable")
        meta = Series(
            path,
            time[order],
            quality[order],
            flux.shape[1:],
            table.header.get("BJDREFI", 0) + table.header.get("BJDREFF", 0),
            table.header.get("TIMESYS", ""),
            table.columns["FLUX"].unit or "",
        )

        def frame(index):
            row = order[index]
            return np.asarray(flux[row], dtype=float), np.asarray(
                error[row], dtype=float
            )

        yield meta, frame
