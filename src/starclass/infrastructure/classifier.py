import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from astropy.timeseries import LombScargle
from scipy.stats import kurtosis, skew
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCHEMA = "time_series_v2"
FEATURES = (
    "amplitude",
    "scatter",
    "skewness",
    "kurtosis",
    "period",
    "power",
    "harmonic_ratio",
)


def features(rows):
    good = [row for row in rows if not row.reason]
    if len(good) < 30:
        raise ValueError("Для классификации нужны хотя бы 30 пригодных кадров.")
    time = np.array([row.time for row in good])
    flux = np.array([row.flux for row in good])
    error = np.array([row.error for row in good])
    time -= time.min()
    signal = flux / np.median(flux) - 1
    baseline = np.ptp(time)
    cadence = np.median(np.diff(time))
    if not baseline or cadence <= 0 or np.std(signal) < 1e-12:
        raise ValueError("Недостаточная длительность или вариативность сигнала.")
    low, high = 2 / baseline, 0.5 / cadence
    if low >= high:
        raise ValueError("Недостаточное покрытие для поиска периода.")
    dy = error / np.median(flux) if np.all(np.isfinite(error) & (error > 0)) else None
    periodogram = LombScargle(time, signal, dy)
    grid = np.linspace(low, high, min(20000, max(256, int(5 * baseline * high))))
    power = periodogram.power(grid)
    peak = int(np.nanargmax(power))
    frequency = grid[peak]
    harmonic = float(periodogram.power(2 * frequency)) if 2 * frequency <= high else 0.0
    quantiles = np.quantile(signal, [0.05, 0.95])
    vector = [
        np.ptp(quantiles),
        np.median(np.abs(signal - np.median(signal))),
        skew(signal),
        kurtosis(signal),
        1 / frequency,
        power[peak],
        harmonic / max(power[peak], 1e-12),
    ]
    if not np.isfinite(vector).all():
        raise ValueError("Не удалось вычислить признаки.")
    return vector


def train(manifest, destination, extract_row, seed=42, cancelled=None):
    manifest = Path(manifest)
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))
    vectors, labels, groups, rejected = [], [], [], []
    object_labels = {}
    for row in records:
        if cancelled and cancelled():
            raise InterruptedError("Обучение отменено.")
        group, label = row["object_id"].strip(), row["label"].strip()
        if not group or not label:
            raise ValueError("object_id и label обязательны.")
        if object_labels.setdefault(group, label) != label:
            raise ValueError(f"Несогласованные метки для {group}.")
        try:
            vectors.append(features(extract_row(manifest.parent, row)))
        except InterruptedError:
            raise
        except (ValueError, OSError) as exc:
            rejected.append(
                {"object_id": group, "path": row["path"], "reason": str(exc)}
            )
            continue
        labels.append(label)
        groups.append(group)
    ids = sorted(set(groups))
    classes, counts = np.unique(
        [object_labels[group] for group in ids], return_counts=True
    )
    if len(classes) < 2 or min(counts) < 10:
        raise ValueError(
            f"Нужно минимум 10 пригодных звёзд каждого из двух или более классов. Исключено серий: {len(rejected)}."
        )
    rest, test = train_test_split(
        ids,
        test_size=0.2,
        random_state=seed,
        stratify=[object_labels[group] for group in ids],
    )
    train_ids, validation = train_test_split(
        rest,
        test_size=0.25,
        random_state=seed,
        stratify=[object_labels[group] for group in rest],
    )
    x, y = np.asarray(vectors), np.asarray(labels)
    masks = [np.isin(groups, subset) for subset in (train_ids, validation, test)]
    train_mask, val_mask, test_mask = masks
    estimators = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier()),
        "rf": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=seed
        ),
    }
    metrics = {}
    for name, estimator in estimators.items():
        estimator.fit(x[train_mask], y[train_mask])
        metrics[name] = {}
        for subset, mask in (("validation", val_mask), ("test", test_mask)):
            prediction = estimator.predict(x[mask])
            metrics[name][subset] = {
                "report": classification_report(
                    y[mask],
                    prediction,
                    labels=classes,
                    output_dict=True,
                    zero_division=0,
                ),
                "confusion_matrix": confusion_matrix(
                    y[mask], prediction, labels=classes
                ).tolist(),
            }
    meta = {
        "schema": SCHEMA,
        "features": FEATURES,
        "classes": classes.tolist(),
        "sklearn": sklearn.__version__,
        "seed": seed,
        "threshold": 0.7,
        "threshold_status": "provisional",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "split": {"train": train_ids, "validation": validation, "test": test},
        "metrics": metrics,
        "rejected": rejected,
    }
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    joblib.dump(estimators["rf"], destination / "pipeline.joblib")
    (destination / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )
    return meta


def predict(bundle, rows):
    bundle = Path(bundle)
    meta = json.loads((bundle / "metadata.json").read_text())
    if (
        meta["schema"] != SCHEMA
        or tuple(meta["features"]) != FEATURES
        or meta["sklearn"] != sklearn.__version__
    ):
        raise ValueError(
            "Модель несовместима с установленной версией или схемой признаков."
        )
    vector = features(rows)
    estimator = joblib.load(bundle / "pipeline.joblib")
    scores = estimator.predict_proba([vector])[0]
    label = str(estimator.classes_[np.argmax(scores)])
    return {
        "label": label if max(scores) >= meta["threshold"] else None,
        "candidate": label,
        "scores": dict(zip(estimator.classes_.tolist(), scores.tolist())),
        "global_importance": dict(
            zip(FEATURES, estimator.feature_importances_.tolist())
        ),
    }
