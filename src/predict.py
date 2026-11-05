"""Классификация одной кривой."""
import numpy as np
from .features import extract_features

FEATURE_NAMES = ["f_top1","f_top2","f_top3","a_top1","a_top2","a_top3","w_top1","w_top2","w_top3",
                 "mean_mag","std_mag","skew_mag","kurt_mag","amp_mag","spec_entropy","spec_centroid",
                 "spec_width","n_peaks","ac_lag1","ac_val1","spec_mean","spec_std","spec_max"]

def predict_one(model, t, mag, top_k=5):
    feat = extract_features(t, mag)
    proba = model.predict_proba([feat])[0]
    idx = int(np.argmax(proba))
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1][:top_k]
    return {
        "label": model.classes_[idx],
        "probabilities": {c: float(p) for c, p in zip(model.classes_, proba)},
        "confidence": float(proba[idx]),
        "top_features": [(FEATURE_NAMES[i], float(imp[i])) for i in order],
    }
