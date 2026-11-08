import argparse
import json
from functools import partial
from pathlib import Path

from starclass.application.analysis import extract, inspect_series
from starclass.core.observations import Source
from starclass.core.photometry import Aperture
from starclass.infrastructure.fits import open_series
from starclass.infrastructure.imaging import detect, measure, reference_image
from starclass.infrastructure.persistence import export_curve


def main(argv=None):
    parser = argparse.ArgumentParser(prog="starclass")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("gui")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("fits", type=Path)
    for name in ("extract", "predict"):
        command = sub.add_parser(name)
        command.add_argument("fits", type=Path)
        command.add_argument("--xy", type=float, nargs=2, required=True)
        command.add_argument("--radius", type=float, default=2.0)
        command.add_argument("--out", type=Path, required=name == "extract")
        if name == "predict":
            command.add_argument("--trusted-model", type=Path, required=True)
    training = sub.add_parser("train")
    training.add_argument("manifest", type=Path)
    training.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    scan = partial(
        inspect_series, reader=open_series, reference=reference_image, detector=detect
    )
    run = partial(extract, reader=open_series, measure=measure)
    from starclass.infrastructure.classifier import predict, train

    def train_model(manifest, output, cancelled=None):
        def extract_row(parent, row):
            path = parent / row["path"]
            _, _, sources = scan(path)
            return run(
                path,
                Source(float(row["x"]), float(row["y"])),
                Aperture(),
                neighbors=sources,
                cancelled=cancelled,
            )[1]

        return train(manifest, output, extract_row, cancelled=cancelled)

    try:
        if args.command == "gui":
            from starclass.presentation.gui import launch

            return launch(scan, run, open_series, export_curve, predict, train_model)
        if args.command == "inspect":
            meta, _, sources = scan(args.fits)
            print(
                json.dumps(
                    {
                        "frames": len(meta.time),
                        "shape": meta.shape,
                        "sources": [vars(source) for source in sources],
                    },
                    indent=2,
                )
            )
        elif args.command == "train":
            metadata = train_model(args.manifest, args.out)
            print(json.dumps(metadata["metrics"], indent=2))
        else:
            _, _, sources = scan(args.fits)
            _, rows = run(
                args.fits,
                Source(*args.xy),
                Aperture(
                    radius=args.radius,
                    sky_inner=args.radius + 1,
                    sky_outer=args.radius + 3,
                ),
                neighbors=sources,
            )
            if args.out:
                export_curve(args.out, rows)
            if args.command == "predict":
                if args.radius != Aperture().radius:
                    raise ValueError(
                        "Модель обучена для стандартной апертуры 2 пикселя."
                    )
                print(
                    json.dumps(
                        predict(args.trusted_model, rows), ensure_ascii=False, indent=2
                    )
                )
            elif not any(not row.reason for row in rows):
                parser.exit(2, "Нет пригодных кадров. Причины сохранены в CSV.\n")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")
    return 0
