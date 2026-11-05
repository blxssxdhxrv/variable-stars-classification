"""CLI: generate / train / predict."""
import argparse, os, sys
import numpy as np, pandas as pd
from .simulation import generate_dataset, generate_curve, CLASSES
from .train import build_dataset, train_rf, save_model
from .predict import predict_one

def _ens(p):
    if p: os.makedirs(os.path.dirname(p) or ".", exist_ok=True)

def cmd_generate(a):
    data = generate_dataset(a.per_class, a.noise)
    rows = [{"id": i, "cls": r["cls"], "t": float(t), "mag": float(m)}
            for i, r in enumerate(data) for t, m in zip(r["t"], r["mag"])]
    _ens(a.out); pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"Сгенерировано {len(data)} кривых -> {a.out}")

def _read_csv(p):
    df = pd.read_csv(p); recs = []
    for _, g in df.groupby("id"):
        g = g.sort_values("t")
        recs.append({"cls": g["cls"].iloc[0], "t": g["t"].to_numpy(), "mag": g["mag"].to_numpy()})
    return recs

def cmd_train(a):
    recs = _read_csv(a.data); print(f"Кривых: {len(recs)}")
    X, y = build_dataset(recs); print(f"X: {X.shape}")
    model, m = train_rf(X, y)
    print(f"accuracy={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}  время={m['train_time']:.2f} c")
    print("Матрица ошибок:\n", m["confusion_matrix"])
    print(m["report"]); save_model(model, a.model); print(f"Модель -> {a.model}")

def cmd_predict(a):
    import joblib
    model = joblib.load(a.model)
    if a.file:
        df = pd.read_csv(a.file); t, mag = df["t"].to_numpy(), df["mag"].to_numpy()
    else:
        t, mag = generate_curve(a.cls)
    r = predict_one(model, t, mag)
    print(f"Класс: {r['label']}  уверенность={r['confidence']:.4f}")
    for c, p in sorted(r["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {c:18s} {p:.4f}")
    for n, i in r["top_features"]:
        print(f"  {n:14s} {i:.4f}")

def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate"); g.add_argument("--out", default="data/dataset.csv")
    g.add_argument("--per-class", type=int, default=200); g.add_argument("--noise", type=float, default=0.02)
    g.set_defaults(func=cmd_generate)
    t = sub.add_parser("train"); t.add_argument("--data", required=True)
    t.add_argument("--model", default="models/rf.joblib"); t.set_defaults(func=cmd_train)
    p = sub.add_parser("predict"); p.add_argument("--model", default="models/rf.joblib")
    gr = p.add_mutually_exclusive_group()
    gr.add_argument("--file"); gr.add_argument("--cls", choices=CLASSES)
    p.set_defaults(func=cmd_predict)
    a = ap.parse_args(argv); a.func(a); return 0

if __name__ == "__main__":
    sys.exit(main())
