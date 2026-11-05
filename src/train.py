"""Обучение Random Forest + метрики."""
import os, time, joblib, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from .features import extract_features
from .simulation import CLASSES

def build_dataset(records):
    X, y = [], []
    for r in records:
        try:
            X.append(extract_features(r["t"], r["mag"])); y.append(r["cls"])
        except Exception:
            continue
    return np.vstack(X), np.array(y)

def train_rf(X, y, n_estimators=300, random_state=42):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=random_state, stratify=y)
    model = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced",
                                   random_state=random_state, n_jobs=-1)
    t0 = time.time(); model.fit(Xtr, ytr); tt = time.time()-t0
    yp = model.predict(Xte)
    return model, {
        "accuracy": accuracy_score(yte, yp),
        "macro_f1": f1_score(yte, yp, average="macro"),
        "confusion_matrix": confusion_matrix(yte, yp, labels=CLASSES),
        "report": classification_report(yte, yp, labels=CLASSES, zero_division=0),
        "train_time": tt,
    }

def save_model(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)
