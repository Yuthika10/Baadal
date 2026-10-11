"""Compare TabPFN against simple baselines on a time-based split.

    python scripts/evaluate.py

Train on the older 80% of hours, test on the most recent 20% (no peeking at the future).
Writes results/metrics.json and results/comparison.png for your DEV post.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config  # noqa: E402
from app.features import FEATURES, LABEL  # noqa: E402
from app.predictor import candidate_backends, make_model  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
MAX_TEST = 1500  # keeps TabPFN API usage modest


def balanced_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    rain, dry = df[df[LABEL] == 1], df[df[LABEL] == 0]
    k = min(len(rain), n // 2)
    return pd.concat([rain.sample(k, random_state=seed), dry.sample(n - k, random_state=seed)])


def recalibrate(p, true_rate, sample_rate):
    num = p * true_rate / sample_rate
    return num / (num + (1 - p) * (1 - true_rate) / (1 - sample_rate))


def main() -> None:
    df = pd.read_csv(config.TRAINING_CSV).dropna(subset=FEATURES + [LABEL])
    df = df.sort_values("time").reset_index(drop=True)
    cut = int(len(df) * 0.8)
    train, test = df.iloc[:cut], df.iloc[cut:]
    if len(test) > MAX_TEST:
        test = test.sample(MAX_TEST, random_state=0).sort_values("time")

    true_rate = float(train[LABEL].mean())
    train_s = balanced_sample(train, min(config.MAX_TRAIN_ROWS, len(train)))
    sample_rate = float(train_s[LABEL].mean())
    Xtr, ytr = train_s[FEATURES].to_numpy(), train_s[LABEL].astype(int).to_numpy()
    Xte, yte = test[FEATURES].to_numpy(), test[LABEL].astype(int).to_numpy()

    print(f"train {len(train_s)} rows (balanced from {len(train)}), test {len(test)} rows")
    print(f"test period: {test['time'].iloc[0]} .. {test['time'].iloc[-1]}, rain rate {yte.mean():.1%}")

    models = {
        "Climatology (always base rate)": None,
        "Logistic regression": make_model("baseline")[0],
        "Gradient boosting": HistGradientBoostingClassifier(random_state=0),
    }
    for b in candidate_backends():
        if b.startswith("tabpfn"):
            try:
                models[f"TabPFN ({b})"] = make_model(b)[0]
                break
            except Exception as exc:
                print(f"  skipping {b}: {exc}")

    results = []
    for name, model in models.items():
        t0 = time.time()
        if model is None:
            p = np.full(len(yte), true_rate)
        else:
            model.fit(Xtr, ytr)
            p = recalibrate(model.predict_proba(Xte)[:, 1], true_rate, sample_rate)
        secs = time.time() - t0
        auc = roc_auc_score(yte, p) if len(set(yte)) > 1 else float("nan")
        res = {
            "model": name,
            "roc_auc": round(float(auc), 4),
            "brier": round(float(brier_score_loss(yte, p)), 4),
            "f1_at_0.5": round(float(f1_score(yte, p >= 0.5, zero_division=0)), 4),
            "seconds": round(secs, 2),
        }
        results.append(res)
        print(f"  {name:34s} AUC {res['roc_auc']:.3f}  Brier {res['brier']:.3f}  "
              f"F1 {res['f1_at_0.5']:.3f}  ({secs:.1f}s)")

    RESULTS.mkdir(exist_ok=True)
    meta = {
        "train_rows": len(train_s),
        "test_rows": len(test),
        "test_rain_rate": float(yte.mean()),
        "horizon_hours": config.HORIZON_HOURS,
        "threshold_mm": config.RAIN_THRESHOLD_MM,
    }
    (RESULTS / "metrics.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["model"] for r in results]
    aucs = [r["roc_auc"] for r in results]
    colors = ["#2a78d6" if n.startswith("TabPFN") else "#9aa4b2" for n in names]
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    bars = ax.barh(names, aucs, color=colors)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("ROC-AUC on the most recent 20% of hours (higher is better)")
    ax.set_title(f"Will it rain in the next {config.HORIZON_HOURS} hours?", loc="left", fontsize=11)
    for bar, v in zip(bars, aucs):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)
    ax.invert_yaxis()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "comparison.png")
    print(f"wrote {RESULTS/'metrics.json'} and {RESULTS/'comparison.png'}")


if __name__ == "__main__":
    main()
