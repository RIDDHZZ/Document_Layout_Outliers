"""Train the anomaly detector on NORMAL documents only.

Leakage control
  * split by source_doc_id -> an original and all its variants share a split
  * imputer + StandardScaler fitted on train-normal only
  * IsolationForest fitted on train-normal only
  * threshold chosen on validation-normal only
  * test split is never touched here

Run:  python -m ml.train
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.common import (assign_groups, feature_columns, get_seed, load_config,
                       resolve)
from ml.regions import (build_cell_reference, build_element_reference,
                        element_features, grid_density, normalize_elements)


def split_ids(ids, fr: dict, seed: int):
    ids = sorted(set(ids))
    np.random.RandomState(seed).shuffle(ids)
    n = len(ids)
    ntr, nva = int(round(fr["train"] * n)), int(round(fr["val"] * n))
    return ids[:ntr], ids[ntr:ntr + nva], ids[ntr + nva:]


def elements_by_page(elems: pd.DataFrame) -> dict:
    return {k: g for k, g in elems.groupby(["document_id", "page_number"])}


def fit(feats: pd.DataFrame, elems: pd.DataFrame, cfg: dict) -> dict:
    seed = get_seed(cfg)
    tr_ids, va_ids, te_ids = split_ids(feats["source_doc_id"].unique(), cfg["split"], seed)

    normal = feats[feats["variant"] == "normal"]
    dtr = normal[normal["source_doc_id"].isin(tr_ids)]
    dva = normal[normal["source_doc_id"].isin(va_ids)]
    if len(dtr) < 10 or len(dva) < 5:
        raise ValueError(f"Not enough normal pages (train={len(dtr)}, val={len(dva)}).")

    fcols = [c for c in feature_columns(feats) if dtr[c].replace([np.inf, -np.inf], np.nan).notna().any()]
    clean = lambda d: d[fcols].astype(float).replace([np.inf, -np.inf], np.nan).to_numpy()

    pre = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())]).fit(clean(dtr))
    ztr, zva = pre.transform(clean(dtr)), pre.transform(clean(dva))

    mc = cfg["model"]
    model = IsolationForest(n_estimators=int(mc["n_estimators"]), max_samples=mc["max_samples"],
                            contamination=mc["contamination"], max_features=float(mc["max_features"]),
                            random_state=seed).fit(ztr)

    raw_tr, raw_va = -model.score_samples(ztr), -model.score_samples(zva)
    pct = float(cfg["threshold"]["percentile"])
    thr = float(np.percentile(raw_va, pct))

    # Score normalisation: piecewise-linear so the calibrated threshold maps to 0.5.
    lo = float(raw_tr.min())
    span = max(thr - lo, 1e-6)
    hi = lo + 2 * span

    # reference statistics for the Features page
    ref_stats = {}
    raw_feat = dtr[fcols].astype(float)
    for c in fcols:
        v = raw_feat[c].dropna().to_numpy()
        ref_stats[c] = {"mean": float(v.mean()), "std": float(v.std()),
                        "p5": float(np.percentile(v, 5)), "p50": float(np.percentile(v, 50)),
                        "p95": float(np.percentile(v, 95))}

    # region + cell references from train-normal pages only
    rc, gc = cfg["regions"], cfg["grid"]
    ebp = elements_by_page(elems)
    frames, grids = [], []
    for _, r in dtr.iterrows():
        g = ebp.get((r["document_id"], r["page_number"]))
        if g is None or not len(g):
            continue
        en = normalize_elements(g, r["page_w"], r["page_h"])
        frames.append(element_features(en, rc["neighbor_radius"]))
        grids.append(grid_density(en, gc["rows"], gc["cols"]))
    region_ref = build_element_reference(frames, rc["std_floor"])
    cell_ref = build_cell_reference(grids)

    feature_config = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_state": seed,
        "features": fcols,
        "feature_groups": assign_groups(fcols, cfg),
        "preprocessing": {"imputer": "median", "scaler": "StandardScaler", "fitted_on": "train-normal pages"},
        "model": {"type": "IsolationForest", **{k: mc[k] for k in mc}},
        "threshold": {"method": cfg["threshold"]["method"], "percentile": pct,
                      "raw": thr, "fitted_on": "validation-normal pages"},
        "score_norm": {"lo": lo, "hi": hi,
                       "formula": "clip((raw - lo) / (hi - lo), 0, 1); hi = lo + 2*(threshold_raw - lo); threshold -> 0.5",
                       "raw_definition": "raw = -IsolationForest.score_samples(z)"},
        "reference_stats": ref_stats,
        "region_reference": region_ref,
        "cell_reference": cell_ref,
        "grid": gc, "region_params": rc, "deviation_cap": cfg["deviation_cap"],
        "dataset": {
            "train_normal_pages": int(len(dtr)), "val_normal_pages": int(len(dva)),
            "source_docs": {"train": len(tr_ids), "val": len(va_ids), "test": len(te_ids)},
            "n_features": len(fcols),
        },
    }
    return {"pre": pre, "model": model, "feature_config": feature_config,
            "split": {"train": tr_ids, "val": va_ids, "test": te_ids},
            "raw_train": raw_tr, "raw_val": raw_va}


def save(art: dict, cfg: dict):
    out = resolve(cfg, "models_dir")
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(art["pre"], out / "scaler.pkl")
    joblib.dump(art["model"], out / "anomaly_model.pkl")
    with open(out / "feature_config.json", "w", encoding="utf-8") as f:
        json.dump(art["feature_config"], f, indent=2)
    with open(out / "split.json", "w", encoding="utf-8") as f:
        json.dump(art["split"], f, indent=2)
    with open(out / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    from ml.pipeline import load_tables
    feats, elems = load_tables(cfg)
    art = fit(feats, elems, cfg)
    save(art, cfg)
    fc = art["feature_config"]
    print(f"Trained on {fc['dataset']['train_normal_pages']} normal pages, "
          f"{fc['dataset']['n_features']} features.")
    print(f"Validation {fc['threshold']['percentile']:.0f}th percentile threshold (raw) = {fc['threshold']['raw']:.4f}")
    print(f"Saved models to {resolve(cfg, 'models_dir')}")


if __name__ == "__main__":
    main()
