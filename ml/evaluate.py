"""Evaluate the trained detector on the held-out TEST split.

Normal class  = pages of documents in data/normal/ (test split only)
Anomalous     = pages flagged is_anomalous=1 in the altered variants (test split only)
(Unchanged pages of altered documents are excluded: they are ambiguous.)

Outputs reports/metrics.json, reports/experiments.csv and PNG plots.
Run:  python -m ml.evaluate [--compare]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)

from ml.common import feature_columns, load_config, resolve
from ml.regions import (element_features, iou, normalize_elements,
                        score_regions)
from ml.scoring import Detector
from ml.train import elements_by_page


def binary_metrics(y, raw, thr) -> dict:
    y, raw = np.asarray(y), np.asarray(raw)
    pred = (raw > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    both = len(set(y.tolist())) == 2
    return {
        "n_normal": int((y == 0).sum()), "n_anomalous": int((y == 1).sum()),
        "accuracy": float((tp + tn) / len(y)),
        "balanced_accuracy": float((tp / max(tp + fn, 1) + tn / max(tn + fp, 1)) / 2),
        "precision": float(p), "recall": float(r), "f1": float(f),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "roc_auc": float(roc_auc_score(y, raw)) if both else None,
        "average_precision": float(average_precision_score(y, raw)) if both else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def cohens_d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Standardised mean difference per column (anomalous b vs normal a)."""
    va, vb = a.var(axis=0, ddof=1), b.var(axis=0, ddof=1)
    pooled = np.sqrt((va + vb) / 2)
    pooled[pooled < 1e-12] = np.nan
    return (b.mean(axis=0) - a.mean(axis=0)) / pooled


def gt_norm(row, y_origin: str):
    x1, y1, x2, y2 = row["gt_x1"], row["gt_y1"], row["gt_x2"], row["gt_y2"]
    pw, ph = row["page_w"], row["page_h"]
    if y_origin == "bottom":
        y1, y2 = ph - y2, ph - y1
    return [min(x1, x2) / pw, min(y1, y2) / ph, max(x1, x2) / pw, max(y1, y2) / ph]


def region_eval(rows: pd.DataFrame, ebp: dict, ref: dict, cfg: dict) -> dict:
    rc = cfg["regions"]
    top1, best_k, hits_k, n = [], [], 0, 0
    for _, r in rows.iterrows():
        g = ebp.get((r["document_id"], r["page_number"]))
        if g is None or not len(g) or pd.isna(r["gt_x1"]):
            continue
        ef = element_features(normalize_elements(g, r["page_w"], r["page_h"]), rc["neighbor_radius"])
        regs = score_regions(ef, ref, rc["top_k"], rc["z_cap"], rc["min_reference"], rc["flag_z"])
        if not regs:
            continue
        gt = gt_norm(r, cfg["annotations"]["y_origin"])
        ious = [iou(x["bbox_norm"], gt) for x in regs]
        n += 1
        top1.append(ious[0])
        best_k.append(max(ious))
        hits_k += int(max(ious) >= rc["iou_hit"])
    if n == 0:
        return {"n_pages": 0}
    return {"n_pages": n, "top_k": rc["top_k"], "iou_threshold": rc["iou_hit"],
            "mean_iou_top1": float(np.mean(top1)),
            "mean_best_iou_topk": float(np.mean(best_k)),
            "hit_rate_top1": float(np.mean(np.array(top1) >= rc["iou_hit"])),
            "hit_rate_topk": float(hits_k / n)}


def run(cfg: dict, feats: pd.DataFrame | None = None, elems: pd.DataFrame | None = None,
        compare: bool = False, plots: bool = True) -> dict:
    if feats is None:
        from ml.pipeline import load_tables
        feats, elems = load_tables(cfg)
    models_dir, rep_dir = resolve(cfg, "models_dir"), resolve(cfg, "reports_dir")
    rep_dir.mkdir(parents=True, exist_ok=True)
    det = Detector(models_dir)
    with open(models_dir / "split.json") as f:
        split = json.load(f)
    test = feats[feats["source_doc_id"].isin(split["test"])].copy()
    assert not set(test["source_doc_id"]) & (set(split["train"]) | set(split["val"])), "LEAKAGE"

    is_norm = test["variant"] == "normal"
    ev = test[is_norm | (test["label"] == 1)].copy()
    ev["raw"] = det.raw(ev)
    ev["y"] = (ev["label"] == 1).astype(int)
    thr = det.thr_raw

    experiments, rows = {}, []
    for name, sel in [("overall", ev["variant"].isin(["normal", "margin", "signature"])),
                      ("margin_alteration", ev["variant"].isin(["normal", "margin"])),
                      ("signature_insertion", ev["variant"].isin(["normal", "signature"]))]:
        s = ev[sel]
        if s["y"].nunique() < 2:
            continue
        m = binary_metrics(s["y"], s["raw"], thr)
        experiments[name] = m
        rows.append({"Experiment": name, "Normal": m["n_normal"], "Anomalous": m["n_anomalous"],
                     "Precision": round(m["precision"], 3), "Recall": round(m["recall"], 3),
                     "F1": round(m["f1"], 3), "ROC-AUC": round(m["roc_auc"], 3) if m["roc_auc"] is not None else None})
    pd.DataFrame(rows).to_csv(rep_dir / "experiments.csv", index=False)

    # document level: max page score, any page anomalous
    dd = ev.groupby("document_id").agg(raw=("raw", "max"), y=("y", "max"), variant=("variant", "first")).reset_index()
    doc_level = binary_metrics(dd["y"], dd["raw"], thr) if dd["y"].nunique() == 2 else {}

    # recall by alteration mode (only if the annotations carry one)
    by_mode = {}
    if "mode" in ev.columns:
        for mode, s in ev[(ev["y"] == 1) & (ev["mode"].fillna("") != "")].groupby("mode"):
            by_mode[str(mode)] = {"n": int(len(s)), "recall": float((s["raw"] > thr).mean())}

    # which features move most (Cohen's d, normal vs each anomaly type)
    fcols = det.features
    nz = det.matrix(ev[ev["y"] == 0])
    top_features = {}
    for v in ("margin", "signature"):
        sub = ev[(ev["variant"] == v) & (ev["y"] == 1)]
        if len(sub) < 2:
            continue
        a = det.matrix(sub)
        d = cohens_d(np.nan_to_num(nz), np.nan_to_num(a))
        order = np.argsort(-np.nan_to_num(np.abs(d)))[:10]
        top_features[v] = [{"feature": fcols[i], "cohens_d": float(d[i])} for i in order]

    # region-level localisation on inserted-signature pages
    ebp = elements_by_page(elems)
    sig_rows = ev[(ev["variant"] == "signature") & (ev["y"] == 1)]
    region = region_eval(sig_rows, ebp, det.cfg["region_reference"], cfg)
    if region.get("n_pages"):
        region["among_detected_pages_only"] = region_eval(
            sig_rows[sig_rows["raw"] > thr], ebp, det.cfg["region_reference"], cfg)

    result = {
        "split_sizes": {k: len(v) for k, v in split.items()},
        "threshold": {"raw": thr, "normalized": det.threshold_norm,
                      "method": det.cfg["threshold"]["method"], "percentile": det.cfg["threshold"]["percentile"]},
        "test_pages": {"normal": int((ev["y"] == 0).sum()), "anomalous": int((ev["y"] == 1).sum())},
        "experiments": experiments, "document_level": doc_level,
        "recall_by_mode": by_mode, "top_features": top_features, "region_level": region,
        "score_summary": {
            "normal_mean": float(ev[ev.y == 0].raw.mean()), "anomalous_mean": float(ev[ev.y == 1].raw.mean())},
    }

    if compare:
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        tr = feats[(feats["variant"] == "normal") & feats["source_doc_id"].isin(split["train"])]
        va = feats[(feats["variant"] == "normal") & feats["source_doc_id"].isin(split["val"])]
        mc = det.cfg["model"]
        if mc.get("type", "isolation_forest") == "lof":
            other = IsolationForest(n_estimators=int(mc["n_estimators"]), max_samples=mc["max_samples"],
                                    contamination=mc["contamination"], max_features=float(mc["max_features"]),
                                    random_state=int(det.cfg["random_state"])).fit(det.zscores(tr))
            key = "comparison_isolation_forest"
        else:
            other = LocalOutlierFactor(n_neighbors=int(mc.get("lof_neighbors", 20)), novelty=True).fit(det.zscores(tr))
            key = "comparison_lof"
        r_va = -other.score_samples(det.zscores(va))
        r_ev = -other.score_samples(det.zscores(ev))
        result[key] = binary_metrics(ev["y"], r_ev, float(np.percentile(r_va, det.cfg["threshold"]["percentile"])))
        result["primary_model"] = mc.get("type", "isolation_forest")

        # Model selection on VALIDATION documents only (labeled altered pages are used for comparison,
        # never for fitting), so the choice between detectors does not depend on the test set.
        from sklearn.metrics import roc_auc_score
        vl = feats[feats["source_doc_id"].isin(split["val"])]
        vl = vl[(vl["variant"] == "normal") | (vl["label"] == 1)]
        yv = (vl["label"] == 1).astype(int)
        if yv.nunique() == 2:
            result["validation_comparison"] = {
                "n_normal": int((yv == 0).sum()), "n_anomalous": int((yv == 1).sum()),
                "primary_model": mc.get("type", "isolation_forest"),
                "primary_roc_auc": float(roc_auc_score(yv, det.raw(vl))),
                "other_model": "isolation_forest" if key == "comparison_isolation_forest" else "lof",
                "other_roc_auc": float(roc_auc_score(yv, -other.score_samples(det.zscores(vl)))),
            }

    with open(rep_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    if plots:
        _plots(ev, thr, experiments, rep_dir)
    return result


def _plots(ev, thr, experiments, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = experiments["overall"]["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, mat[i, j], ha="center", va="center")
    ax.set_xticks([0, 1], ["Pred normal", "Pred anomalous"])
    ax.set_yticks([0, 1], ["Normal", "Anomalous"])
    ax.set_title("Confusion matrix (test)")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=150); plt.close(fig)

    fpr, tpr, _ = roc_curve(ev["y"], ev["raw"])
    fig, ax = plt.subplots(figsize=(4, 3.4))
    ax.plot(fpr, tpr); ax.plot([0, 1], [0, 1], "--", c="grey")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("ROC (test)")
    fig.tight_layout(); fig.savefig(out / "roc_curve.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    for v, c in [("normal", "tab:blue"), ("margin", "tab:orange"), ("signature", "tab:red")]:
        s = ev[ev["variant"] == v]["raw"]
        if len(s):
            ax.hist(s, bins=20, alpha=.55, label=v, color=c)
    ax.axvline(thr, color="k", ls="--", label="threshold")
    ax.set_xlabel("raw anomaly score"); ax.legend(); ax.set_title("Score distribution (test)")
    fig.tight_layout(); fig.savefig(out / "score_hist.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--compare", action="store_true", help="also evaluate Local Outlier Factor")
    a = ap.parse_args()
    cfg = load_config(a.config)
    res = run(cfg, compare=a.compare)
    print(json.dumps({k: res[k] for k in ("threshold", "test_pages", "experiments", "region_level")}, indent=2))


if __name__ == "__main__":
    main()
