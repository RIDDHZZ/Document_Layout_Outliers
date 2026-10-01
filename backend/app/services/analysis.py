"""Analysis service: PDF -> page records -> scores -> localisation -> JSON-able result.

`analyze_records` is a pure function (no PyMuPDF / FastAPI), so it can be unit-tested.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.errors import AppError
from ml.regions import (cell_deviation, element_features, grid_density,
                        normalize_elements, score_regions)

log = logging.getLogger("analysis")

DISCLAIMER = ("This tool detects potential layout anomalies (margins, bounding-box spread, spatial density, "
              "unexpected elements). It does not assess legal authenticity or prove forgery.")
GROUP_LABEL = {"margin": "margin", "bbox": "bounding-box", "density": "spatial-density",
               "signature": "signature-related", "count": "element-count", "other": "other"}


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if (np.isnan(x) or np.isinf(x)) else x


def _position(b) -> str:
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    v = "top" if cy < 1 / 3 else "middle" if cy < 2 / 3 else "bottom"
    h = "left" if cx < 1 / 3 else "center" if cx < 2 / 3 else "right"
    return "center" if (v, h) == ("middle", "center") else f"{v}-{h}".replace("middle-", "")


def _explain(page, score, thr, anomalous, groups, top_features, regions) -> str:
    if not anomalous:
        return f"Page {page} is consistent with the normal layout pattern (score {score:.2f}, threshold {thr:.2f})."
    g = max(groups, key=groups.get) if groups else "other"
    txt = (f"Page {page} shows a potential layout anomaly (score {score:.2f}, threshold {thr:.2f}), "
           f"mainly in {GROUP_LABEL.get(g, g)} features")
    if top_features:
        tf = top_features[0]
        txt += f" (largest deviation: {tf['feature']}, {tf['z']:+.1f} SD from the normal reference)"
    flagged = [r for r in regions if r["flagged"]]
    if flagged:
        r = flagged[0]
        what = "non-text (image-like) element" if r["kind"] == "image" else "text region"
        txt += f". The most suspicious region is a {what} in the {_position(r['bbox_norm'])} of the page"
    return txt + "."


def analyze_records(records: list[dict], det, cfg: dict, aid: str, filename: str,
                    n_pages: int) -> tuple[dict, list[dict], list[dict]]:
    df = pd.DataFrame([r["features"] for r in records])
    missing = [f for f in det.features if f not in df.columns]
    if len(missing) > 0.2 * len(det.features):
        raise AppError(500, "feature_mismatch", "The analysis could not be completed because of a model configuration problem.")

    z = det.zscores(df)
    raw = det.raw_from_z(z)
    norm = det.normalize(raw)
    flags = det.is_anomalous(raw)
    thr = det.threshold_norm

    rc, gc, cap = cfg["regions"], det.cfg["grid"], det.cfg.get("deviation_cap", 3.0)
    ref = det.cfg["region_reference"]
    ref_stats = det.cfg["reference_stats"]
    pages, feat_tables, top_regions = [], [], []

    for i, rec in enumerate(records):
        pw, ph = rec["page_w"], rec["page_h"]
        en = normalize_elements(rec["elements"], pw, ph)
        ef = element_features(en, rc["neighbor_radius"])
        grid = grid_density(en, gc["rows"], gc["cols"])
        zgrid = cell_deviation(grid, det.cfg["cell_reference"])
        regs = score_regions(ef, ref, rc["top_k"], rc["z_cap"], rc["min_reference"], rc["flag_z"])
        for r in regs:
            b = r["bbox_norm"]
            r["bbox"] = [b[0] * pw, b[1] * ph, b[2] * pw, b[3] * ph]
            r["type"] = "potential_inserted_element" if r["kind"] == "image" else "spatial_anomaly"
        groups = det.group_deviation(z[i], cap)
        order = np.argsort(-np.abs(z[i]))[:5]
        top_f = [{"feature": det.features[j], "group": det.groups[det.features[j]],
                  "value": _num(df.iloc[i].get(det.features[j])), "z": float(z[i][j]),
                  "reference_median": ref_stats[det.features[j]]["p50"]} for j in order]
        page_no = rec["page_number"]
        page = {
            "page": page_no, "page_width": pw, "page_height": ph,
            "status": "anomalous" if flags[i] else "normal",
            "is_anomalous": bool(flags[i]), "score": float(norm[i]), "raw_score": float(raw[i]),
            "group_deviation": groups, "top_features": top_f,
            "elements": [{"bbox_norm": [float(r.nx1), float(r.ny1), float(r.nx2), float(r.ny2)], "kind": r.kind}
                         for r in en.itertuples()],
            "density_grid": {"rows": gc["rows"], "cols": gc["cols"], "values": grid.tolist(),
                             "z_vs_normal": zgrid.tolist()},
            "regions": regs,
            "explanation": _explain(page_no, float(norm[i]), thr, bool(flags[i]), groups, top_f, regs),
            "image_url": f"/api/analysis/{aid}/page/{page_no}/image",
        }
        pages.append(page)
        feat_tables.append({"page": page_no, "features": [
            {"name": f, "group": det.groups[f], "value": _num(df.iloc[i].get(f)), "z": float(z[i][k]),
             "ref_p5": ref_stats[f]["p5"], "ref_p50": ref_stats[f]["p50"], "ref_p95": ref_stats[f]["p95"]}
            for k, f in enumerate(det.features)]})
        if flags[i]:
            for r in regs:
                top_regions.append({"page": page_no, "bbox": r["bbox"], "bbox_norm": r["bbox_norm"],
                                    "score": r["score"], "type": r["type"], "flagged": r["flagged"]})

    worst = int(np.argmax(norm))
    gd = pages[worst]["group_deviation"]
    anomalous_pages = [p["page"] for p in pages if p["is_anomalous"]]
    result = {
        "analysis_id": aid, "filename": filename, "pages": n_pages, "pages_analyzed": len(pages),
        "overall_status": "potential_anomaly" if anomalous_pages else "normal",
        "overall_score": float(norm.max()), "threshold": thr,
        "anomalous_pages": anomalous_pages, "regions": top_regions,
        "feature_summary": {"margin_deviation": gd.get("margin"), "bbox_variance": gd.get("bbox"),
                            "spatial_density_deviation": gd.get("density"),
                            "signature_related_deviation": gd.get("signature")},
        "page_summaries": [{"page": p["page"], "status": p["status"], "score": p["score"]} for p in pages],
        "score_note": "Relative unusualness score (0 = typical, 1 = highly unusual). It is not a probability.",
        "disclaimer": DISCLAIMER,
    }
    return result, pages, feat_tables


def render_pages(pdf_path: Path, out_dir: Path, dpi: int) -> None:
    import fitz
    doc = fitz.open(str(pdf_path))
    try:
        for i, page in enumerate(doc, 1):
            page.get_pixmap(dpi=dpi).save(str(out_dir / f"page_{i}.png"))
    finally:
        doc.close()


def run_analysis(store, aid: str, pdf_path: Path, filename: str, n_pages: int, det, cfg: dict) -> None:
    """Background job. Always deletes the uploaded PDF, whatever happens."""
    from ml.pipeline import extract_page_records
    t0, limit = time.time(), cfg["upload"]["timeout_seconds"]

    def check_time():
        if time.time() - t0 > limit:
            raise AppError(504, "timeout", "The analysis took too long and was stopped.")

    try:
        store.update(aid, status="processing", step="extracting")
        render_pages(pdf_path, store.image_dir(aid), cfg["upload"]["render_dpi"])
        records = extract_page_records(pdf_path, cfg)
        check_time()
        if not records or all(len(r["elements"]) == 0 for r in records):
            raise AppError(422, "no_layout", "The uploaded PDF could not be processed because no readable layout elements were detected.")
        store.update(aid, step="scoring")
        result, pages, feats = analyze_records(records, det, cfg, aid, filename, n_pages)
        check_time()
        store.update(aid, status="done", step="completed", result=result, page_results=pages, features=feats)
    except AppError as e:
        store.update(aid, status="failed", error={"code": e.code, "message": e.message})
    except Exception:
        log.exception("analysis %s failed", aid)      # stack trace stays in the server log only
        store.update(aid, status="failed", error={
            "code": "processing_failed",
            "message": "The document could not be analysed. It may be corrupted or use an unsupported layout."})
    finally:
        pdf_path.unlink(missing_ok=True)
