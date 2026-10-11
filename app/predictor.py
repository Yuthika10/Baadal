"""Rain-probability model.

Backends, in the order `auto` tries them:
  1. tabpfn-client  - Prior Labs' hosted TabPFN (set TABPFN_TOKEN). Light enough for Render.
  2. tabpfn         - the open-source model running locally (pip install tabpfn; needs torch).
  3. baseline       - logistic regression, used ONLY if TabPFN is unavailable, and the
                      API/UI say so explicitly. Never silently pretend it is TabPFN.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import FEATURES, LABEL

log = logging.getLogger("cloud-oracle.predictor")


def load_training(max_rows: int | None = None, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(config.TRAINING_CSV)
    df = df.dropna(subset=FEATURES + [LABEL])
    max_rows = max_rows or config.MAX_TRAIN_ROWS
    if len(df) > max_rows:
        # Keep every rain row we can (rain hours are rare), fill the rest with dry rows.
        rain = df[df[LABEL] == 1]
        dry = df[df[LABEL] == 0]
        n_rain = min(len(rain), max_rows // 2)
        rain = rain.sample(n_rain, random_state=seed)
        dry = dry.sample(max_rows - n_rain, random_state=seed)
        df = pd.concat([rain, dry]).sample(frac=1, random_state=seed)
    return df


def make_model(backend: str):
    """Return (estimator, backend_name). Raises if the requested backend is unavailable."""
    if backend == "tabpfn-client":
        from tabpfn_client import TabPFNClassifier, set_access_token

        if config.TABPFN_TOKEN:
            set_access_token(config.TABPFN_TOKEN)
        return TabPFNClassifier(), "tabpfn-client"
    if backend == "tabpfn":
        from tabpfn import TabPFNClassifier

        return TabPFNClassifier(device="cpu"), "tabpfn"
    if backend == "baseline":
        return (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
            "baseline",
        )
    raise ValueError(f"unknown backend {backend}")


def candidate_backends() -> list[str]:
    choice = config.PREDICTOR_BACKEND
    if choice != "auto":
        return [choice]
    order = []
    if config.TABPFN_TOKEN:
        order.append("tabpfn-client")
    order.append("tabpfn")
    order.append("baseline")
    return order


class RainPredictor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.model = None
        self.backend = "not-ready"
        self.error: str | None = None
        self.train_rows = 0
        self.base_rate = None
        # Rain prior in the training data; TabPFN is trained on a balanced sample, so
        # we re-calibrate its output back to the real-world rate (see `predict`).
        self._true_rate = None
        self._sample_rate = None

    def ensure_ready(self) -> None:
        with self._lock:
            if self.model is not None:
                return
            full = pd.read_csv(config.TRAINING_CSV).dropna(subset=FEATURES + [LABEL])
            self._true_rate = float(full[LABEL].mean())
            df = load_training()
            self._sample_rate = float(df[LABEL].mean())
            X, y = df[FEATURES].to_numpy(), df[LABEL].astype(int).to_numpy()
            errors = []
            for backend in candidate_backends():
                try:
                    model, name = make_model(backend)
                    model.fit(X, y)
                    self.model, self.backend = model, name
                    self.train_rows = len(df)
                    self.base_rate = self._true_rate
                    self.error = "; ".join(errors) or None
                    log.info("predictor ready: %s on %d rows", name, len(df))
                    return
                except Exception as exc:  # try the next backend, but remember why
                    errors.append(f"{backend}: {type(exc).__name__}: {exc}")
                    log.warning("backend %s unavailable: %s", backend, exc)
            self.error = "; ".join(errors)
            raise RuntimeError(f"No predictor backend could start: {self.error}")

    def _recalibrate(self, p: np.ndarray) -> np.ndarray:
        """Undo the class re-balancing so probabilities match the real rain frequency."""
        if not self._true_rate or not self._sample_rate:
            return p
        a, b = self._true_rate, self._sample_rate
        num = p * a / b
        den = num + (1 - p) * (1 - a) / (1 - b)
        return num / den

    def predict(self, rows: list[dict]) -> list[float]:
        self.ensure_ready()
        X = np.array([[float(r[f]) for f in FEATURES] for r in rows])
        proba = self.model.predict_proba(X)[:, 1]
        return [float(v) for v in self._recalibrate(proba)]

    def status(self) -> dict:
        return {
            "backend": self.backend,
            "is_tabpfn": self.backend.startswith("tabpfn"),
            "train_rows": self.train_rows,
            "base_rate": self.base_rate,
            "error": self.error,
        }


predictor = RainPredictor()
