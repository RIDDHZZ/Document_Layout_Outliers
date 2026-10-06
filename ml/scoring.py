"""Loads the trained artefacts and scores feature tables.

Used by BOTH evaluate.py and the FastAPI backend, so offline results and
website results always come from the same code path.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class Detector:
    def __init__(self, models_dir: str | Path):
        d = Path(models_dir)
        self.pre = joblib.load(d / "scaler.pkl")          # Pipeline(median imputer, StandardScaler)
        self.model = joblib.load(d / "anomaly_model.pkl")  # IsolationForest or LocalOutlierFactor (config model.type)
        with open(d / "feature_config.json", "r", encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.features: list[str] = self.cfg["features"]
        self.groups: dict[str, str] = self.cfg["feature_groups"]
        self.thr_raw: float = self.cfg["threshold"]["raw"]
        self.lo: float = self.cfg["score_norm"]["lo"]
        self.hi: float = self.cfg["score_norm"]["hi"]

    def matrix(self, df: pd.DataFrame) -> np.ndarray:
        x = df.reindex(columns=self.features).astype(float)
        return x.replace([np.inf, -np.inf], np.nan).to_numpy()

    def zscores(self, df: pd.DataFrame) -> np.ndarray:
        """Standardised features z = (x - mu) / sigma using TRAINING statistics."""
        return self.pre.transform(self.matrix(df))

    def raw_from_z(self, z: np.ndarray) -> np.ndarray:
        # score_samples: higher = more normal. We flip it so higher = more anomalous.
        return -self.model.score_samples(z)

    def raw(self, df: pd.DataFrame) -> np.ndarray:
        return self.raw_from_z(self.zscores(df))

    def normalize(self, raw: np.ndarray) -> np.ndarray:
        """Relative unusualness in [0, 1]; the calibrated threshold maps to 0.5."""
        return np.clip((np.asarray(raw) - self.lo) / (self.hi - self.lo), 0.0, 1.0)

    def is_anomalous(self, raw: np.ndarray) -> np.ndarray:
        return np.asarray(raw) > self.thr_raw

    @property
    def threshold_norm(self) -> float:
        return float(self.normalize(np.array([self.thr_raw]))[0])

    def group_deviation(self, z_row: np.ndarray, cap: float = 3.0) -> dict[str, float]:
        out = {}
        for g in sorted(set(self.groups.values())):
            idx = [i for i, f in enumerate(self.features) if self.groups.get(f) == g]
            if idx:
                out[g] = float(min(1.0, np.mean(np.abs(z_row[idx])) / cap))
        return out
