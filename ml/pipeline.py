"""ADAPTER between Block A (extract.py / features.py) and everything else.

This is the ONLY file that talks to your Block A code. If function names or
return types differ from what is guessed below, fix `extract_page_records()`
and nothing else.

CONTRACT: extract_page_records(pdf_path) must return one dict per page:
    {
      "page_number": int,          # 1-based
      "page_w": float, "page_h": float,      # PDF points (PyMuPDF page.rect)
      "features": {name: float},   # the 83 page-level features from features.py
      "elements": DataFrame[x1, y1, x2, y2, kind]   # points; kind in {"text","image"}
    }

Check your adapter with:   python -m ml.pipeline --check data/normal/doc_001.pdf
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ml.common import ROOT, load_config, resolve

REQUIRED_EL_COLS = ["x1", "y1", "x2", "y2", "kind"]


def extract_page_records(pdf_path, cfg=None, force_ocr=False):
    from ml import extract as ex
    from ml import features as ft
    cfg = cfg or load_config()

    pages = ex.extract_document(str(pdf_path), cfg)
    records = []
    for p in pages:
        W, H = float(p.width_pt), float(p.height_pt)
        rows = [{
            "x1": e.x1 * W, "y1": e.y1 * H, "x2": e.x2 * W, "y2": e.y2 * H,
            "kind": "text" if e.kind == "text" else "image",   # 'graphic' -> 'image'
        } for e in p.elements]
        el = pd.DataFrame(rows, columns=["x1", "y1", "x2", "y2", "kind"])

        feats = ft.extract_page_features(p, cfg)
        records.append({
            "page_number": int(p.page_number),
            "page_w": W, "page_h": H,
            "features": {k: float(v) for k, v in feats.items()},
            "elements": el,
        })
    return records


def validate_record(rec: dict) -> list[str]:
    problems = []
    for k in ("page_number", "page_w", "page_h", "features", "elements"):
        if k not in rec:
            problems.append(f"missing key '{k}'")
    if problems:
        return problems
    if not isinstance(rec["features"], dict) or not rec["features"]:
        problems.append("'features' must be a non-empty dict")
    el = rec["elements"]
    miss = [c for c in REQUIRED_EL_COLS if c not in el.columns]
    if miss:
        problems.append(f"'elements' is missing columns {miss}")
    elif len(el):
        if el[["x1", "y1", "x2", "y2"]].max().max() <= 1.0 + 1e-6:
            problems.append("element coordinates look normalised; they must be in PDF points")
    return problems


# --------------------------------------------------------------------------
def _find_pdfs(data_dir: Path) -> dict[str, Path]:
    return {p.stem: p for p in data_dir.glob("*/*.pdf")}


def _strip_variant(doc_id: str) -> str:
    return re.sub(r"_(margin|signature)$", "", doc_id)


def build_tables(cfg: dict | None = None, limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Block A over the whole dataset -> page_features.csv + elements.csv."""
    cfg = cfg or load_config()
    ann = pd.read_csv(resolve(cfg, "annotations"))
    pdfs = _find_pdfs(resolve(cfg, "data_dir"))
    base = int(ann["page_number"].min())   # annotations may be 0- or 1-based
    mode_col = next((c for c in ("alteration_mode", "mode", "subtype", "margin_mode") if c in ann.columns), None)

    frows, erows = [], []
    doc_ids = list(ann["document_id"].unique())[: limit or None]
    for n, doc in enumerate(doc_ids, 1):
        if doc not in pdfs:
            print(f"  [skip] no PDF found for {doc}")
            continue
        variant = pdfs[doc].parent.name                       # normal / altered_margin / inserted_signature
        a_doc = ann[ann["document_id"] == doc]
        src = a_doc["source_doc_id"].iloc[0] if "source_doc_id" in a_doc else _strip_variant(doc)
        for rec in extract_page_records(pdfs[doc], cfg):
            page_key = rec["page_number"] - 1 + base
            a_pg = a_doc[a_doc["page_number"] == page_key]
            anom = a_pg[a_pg["is_anomalous"].astype(int) == 1]
            row = {
                "document_id": doc, "source_doc_id": src, "page_number": page_key,
                "variant": {"normal": "normal", "altered_margin": "margin",
                            "inserted_signature": "signature"}.get(variant, variant),
                "label": int(len(anom) > 0),
                "anomaly_type": (anom["anomaly_type"].iloc[0] if len(anom) else "none"),
                "mode": (anom[mode_col].iloc[0] if (mode_col and len(anom)) else ""),
                "page_w": rec["page_w"], "page_h": rec["page_h"],
                "gt_x1": np.nan, "gt_y1": np.nan, "gt_x2": np.nan, "gt_y2": np.nan,
            }
            if len(anom) and anom[["x1", "y1", "x2", "y2"]].notna().all(axis=1).any():
                b = anom.dropna(subset=["x1", "y1", "x2", "y2"]).iloc[0]
                row.update(gt_x1=b["x1"], gt_y1=b["y1"], gt_x2=b["x2"], gt_y2=b["y2"])
            row.update(rec["features"])
            frows.append(row)
            el = rec["elements"][REQUIRED_EL_COLS].copy()
            el.insert(0, "page_number", page_key)
            el.insert(0, "document_id", doc)
            erows.append(el)
        if n % 20 == 0:
            print(f"  processed {n}/{len(doc_ids)} documents")

    feats = pd.DataFrame(frows)
    elems = pd.concat(erows, ignore_index=True) if erows else pd.DataFrame()
    fp, ep = resolve(cfg, "features_table"), resolve(cfg, "elements_table")
    fp.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(fp, index=False)
    elems.to_csv(ep, index=False)
    print(f"Wrote {fp} ({len(feats)} pages) and {ep} ({len(elems)} elements)")
    return feats, elems


def load_tables(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    fp, ep = resolve(cfg, "features_table"), resolve(cfg, "elements_table")
    if not fp.exists() or not ep.exists():
        raise FileNotFoundError("Feature tables not found. Run:  python -m ml.pipeline --build")
    return pd.read_csv(fp), pd.read_csv(ep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="extract features for the whole dataset")
    ap.add_argument("--check", metavar="PDF", help="validate the adapter on one PDF")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config()
    if a.check:
        recs = extract_page_records(a.check, cfg)
        bad = False
        for r in recs:
            probs = validate_record(r)
            print(f"page {r['page_number']}: {len(r['features'])} features, "
                  f"{len(r['elements'])} elements  ->  {'OK' if not probs else probs}")
            bad |= bool(probs)
        sys.exit(1 if bad else 0)
    if a.build:
        build_tables(cfg, a.limit)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
