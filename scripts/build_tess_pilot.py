"""Build the local TESS training manifest from a Gaia candidate catalog."""

import argparse
import csv
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

from astropy.io import fits
from astropy.wcs import WCS

from starclass.application.analysis import extract, inspect_series
from starclass.core.observations import Source
from starclass.core.photometry import Aperture
from starclass.infrastructure.classifier import features
from starclass.infrastructure.fits import open_series
from starclass.infrastructure.imaging import detect, measure, reference_image
from starclass.infrastructure.persistence import export_curve

API = "https://mast.stsci.edu/tesscut/api/v0.1/astrocut"
MAX_ARCHIVE_BYTES = 18 * 1024 * 1024


def acquire(row, output, sector):
    object_id = str(int(row["object_id"]))
    ra, dec = float(row["ra"]), float(row["dec"])
    url = f"{API}?{urlencode({'ra': ra, 'dec': dec, 'x': 15, 'y': 15, 'units': 'px', 'sector': sector})}"
    path = output / f"{object_id}-s{sector:04d}.fits"
    if not path.exists():
        with urlopen(url, timeout=90) as response:
            payload = response.read(MAX_ARCHIVE_BYTES + 1)
        if len(payload) > MAX_ARCHIVE_BYTES:
            raise ValueError("Archive exceeds download limit")
        with ZipFile(io.BytesIO(payload)) as archive:
            entries = [
                entry
                for entry in archive.infolist()
                if entry.filename.endswith(".fits")
            ]
            if len(entries) != 1 or entries[0].file_size > 3 * MAX_ARCHIVE_BYTES:
                raise ValueError("Unexpected FITS archive")
            temporary = path.with_suffix(".part")
            temporary.write_bytes(archive.read(entries[0]))
            temporary.replace(path)
    with fits.open(path) as hdus:
        x, y = WCS(hdus["APERTURE"].header).world_to_pixel_values(ra, dec)
    source = Source(float(x), float(y))
    _, _, neighbors = inspect_series(path, open_series, reference_image, detect)
    _, measurements = extract(path, source, Aperture(), open_series, measure, neighbors)
    features(measurements)
    good = sum(not measurement.reason for measurement in measurements)
    if good < 100 or good < len(measurements) / 2:
        raise ValueError(f"Insufficient QC coverage: {good}/{len(measurements)}")
    export_curve(path.with_suffix(".csv"), measurements)
    with path.open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "path": path.name,
        "object_id": object_id,
        "label": row["label"],
        "x": source.x,
        "y": source.y,
        "ra": ra,
        "dec": dec,
        "sector": sector,
        "usable_frames": good,
        "total_frames": len(measurements),
        "sha256": checksum,
        "source_url": url,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sector", type=int, default=14)
    parser.add_argument("--per-class", type=int, default=12)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=6)
    args = parser.parse_args()
    if args.per_class < 1:
        parser.error("--per-class must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    with args.catalog.open(newline="") as handle:
        candidates = list(csv.DictReader(handle))
    successes = []
    rejected = []

    def process(row):
        try:
            return acquire(row, args.out, args.sector), None
        except (OSError, ValueError, KeyError, BadZipFile) as exc:
            return None, {
                "object_id": row["object_id"],
                "label": row["label"],
                "reason": str(exc),
            }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for label in sorted({row["label"] for row in candidates}):
            rows = [row for row in candidates if row["label"] == label]
            count = 0
            for start in range(0, len(rows), args.workers):
                batch = rows[start : start + min(args.workers, args.per_class - count)]
                for result, error in pool.map(process, batch):
                    if result:
                        successes.append(result)
                        count += 1
                        print(f"{label}: {count} accepted", flush=True)
                    else:
                        rejected.append(error)
                        print(f"{label}: {error['reason']}", flush=True)
                if count == args.per_class:
                    break
                # Retry unconsumed rows if the quota limited the previous batch.
                if len(batch) < args.workers:
                    for result, error in pool.map(
                        process, rows[start + len(batch) : start + args.workers]
                    ):
                        if result and count < args.per_class:
                            successes.append(result)
                            count += 1
                        elif error:
                            rejected.append(error)
                    if count == args.per_class:
                        break
    fields = (
        list(successes[0]) if successes else ["path", "object_id", "label", "x", "y"]
    )
    with (args.out / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(successes)
    (args.out / "rejected.json").write_text(
        json.dumps(rejected, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
