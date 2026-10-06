"""Shared configuration helpers for the ML pipeline and the backend.

`config.yaml` (from Block A) is merged OVER the defaults below, so any key you
already have keeps its value and any key that is missing gets a sane default.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

DEFAULTS: dict = {
    "random_state": 42,
    "paths": {
        "data_dir": "data",
        "models_dir": "models",
        "reports_dir": "reports",
        "annotations": "data/metadata/annotations.csv",
        "features_table": "data/features/page_features.csv",
        "elements_table": "data/features/elements.csv",
    },
    # Split is done on source_doc_id, so an original and ALL its altered variants
    # always land in the same split (leakage control).
    "split": {"train": 0.6, "val": 0.2, "test": 0.2},
    "model": {
        "type": "isolation_forest",   # "isolation_forest" | "lof"  (the other one is used for --compare)
        "lof_neighbors": 20,
        "n_estimators": 300,
        "max_samples": "auto",
        "contamination": "auto",   # only affects .predict(); we threshold ourselves
        "max_features": 1.0,
    },
    "threshold": {"method": "percentile", "percentile": 95},
    "grid": {"rows": 4, "cols": 4},
    "regions": {
        "top_k": 3,
        "z_cap": 6.0,          # z-scores are capped here when converting to [0, 1]
        "std_floor": 0.02,     # minimum robust scale (normalised page units)
        "flag_z": 4.0,         # a region is "flagged" if its z >= this
        "neighbor_radius": 0.12,
        "min_reference": 10,   # min reference elements to use a per-kind reference
        "iou_hit": 0.3,        # IoU needed to count a localisation as a hit
    },
    "annotations": {"y_origin": "top"},   # set to "bottom" if your ground-truth boxes use ReportLab coordinates
    "deviation_cap": 3.0,      # mean |z| that maps to a 100% group deviation
    # First matching group wins, so the more specific groups come first.
    "feature_groups": {
        "signature": ["sig", "nearest_text", "nontext", "non_text", "image", "local_density"],
        "margin": ["margin"],
        "bbox": ["bbox"],
        "density": ["density"],
        "count": ["count", "n_"],
    },
    "upload": {"max_mb": 20, "max_pages": 30, "ttl_seconds": 1800, "render_dpi": 110,
               "timeout_seconds": 120},
}

# Columns of the page-level feature table that are NOT model features.
META_COLS = [
    "document_id", "source_doc_id", "page_number", "variant", "label",
    "anomaly_type", "mode", "page_w", "page_h",
    "gt_x1", "gt_y1", "gt_x2", "gt_y2",
]


def deep_update(base: dict, upd: dict) -> dict:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    cfg = copy.deepcopy(DEFAULTS)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            deep_update(cfg, yaml.safe_load(f) or {})
    return cfg


def get_seed(cfg: dict) -> int:
    return int(cfg.get("random_state", cfg.get("seed", 42)))


def resolve(cfg: dict, key: str) -> Path:
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p


def feature_columns(df) -> list[str]:
    import numpy as np
    return [c for c in df.columns
            if c not in META_COLS and np.issubdtype(df[c].dtype, np.number)]


def assign_groups(cols: list[str], cfg: dict) -> dict[str, str]:
    out = {}
    for c in cols:
        low = c.lower()
        out[c] = "other"
        for g, subs in cfg["feature_groups"].items():
            if any(s in low for s in subs):
                out[c] = g
                break
    return out
