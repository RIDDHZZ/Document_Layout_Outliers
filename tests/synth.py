"""Fabricated feature/element tables to test the ML machinery without Block A.
Numbers produced from this data are NOT results; they only prove the code runs."""
import numpy as np
import pandas as pd

PW, PH = 612.0, 792.0


def _page_elements(rng, shift=0.0, sig=None, genuine_sig=False):
    n = int(rng.integers(18, 30))
    left = 0.12 + rng.normal(0, .004)
    rows = []
    y = 0.10
    for _ in range(n):
        w = rng.uniform(.45, .76)
        rows.append([(left + shift) * PW, y * PH, (left + shift + w) * PW, (y + .022) * PH, "text"])
        y += rng.uniform(.026, .034)
    if genuine_sig:                      # genuine signature: near end of text, left half
        rows.append([.14 * PW, (y + .01) * PH, .34 * PW, (y + .07) * PH, "image"])
    if sig is not None:
        rows.append([sig[0] * PW, sig[1] * PH, sig[2] * PW, sig[3] * PH, "image"])
    return pd.DataFrame(rows, columns=["x1", "y1", "x2", "y2", "kind"])


def _features(el):
    t = el[el.kind == "text"]
    img = el[el.kind == "image"]
    w = (t.x2 - t.x1) / PW
    f = {
        "left_margin_mean": t.x1.mean() / PW, "left_margin_std": t.x1.std() / PW,
        "right_margin_mean": 1 - t.x2.mean() / PW, "top_margin_mean": t.y1.min() / PH,
        "bottom_margin_mean": 1 - t.y2.max() / PH,
        "bbox_width_mean": w.mean(), "bbox_width_variance": w.var(),
        "n_text_lines": float(len(t)), "sig_n_images": float(len(img)),
        "sig_max_area": float(((img.x2 - img.x1) * (img.y2 - img.y1)).max() / (PW * PH)) if len(img) else 0.0,
        "sig_min_dist_bottom": float((1 - img.y2.max() / PH)) if len(img) else 1.0,
        "sig_max_dist_right": float((1 - img.x2.min() / PW)) if len(img) else 0.0,
    }
    cx = ((el.x1 + el.x2) / 2 / PW * 4).astype(int).clip(0, 3)
    cy = ((el.y1 + el.y2) / 2 / PH * 4).astype(int).clip(0, 3)
    g = np.zeros((4, 4))
    np.add.at(g, (cy, cx), 1)
    g /= g.sum()
    for i, v in enumerate(g.ravel(), 1):
        f[f"density_{i}"] = v
    f["density_variance"] = g.var()
    return f


def make_tables(n_src=60, seed=0):
    rng = np.random.default_rng(seed)
    frows, erows = [], []
    for i in range(n_src):
        src = f"doc_{i:03d}"
        for variant in ("normal", "margin", "signature"):
            doc = src if variant == "normal" else f"{src}_{variant}"
            for pg in (1, 2):
                r = np.random.default_rng(rng.integers(1e9))
                gen = r.random() < .6
                shift, sig, gt = 0.0, None, (np.nan,) * 4
                if variant == "margin":
                    shift = r.choice([-1, 1]) * r.uniform(.06, .12)
                if variant == "signature":
                    x = r.uniform(.55, .8); y = r.uniform(.55, .9)
                    sig = (x, y, x + r.uniform(.12, .2), y + r.uniform(.04, .07))
                    gt = (sig[0] * PW, sig[1] * PH, sig[2] * PW, sig[3] * PH)
                el = _page_elements(r, shift, sig, gen)
                lab = int(variant != "normal")
                row = {"document_id": doc, "source_doc_id": src, "page_number": pg, "variant": variant,
                       "label": lab, "anomaly_type": {"normal": "none", "margin": "margin_alteration",
                                                      "signature": "inserted_signature"}[variant],
                       "mode": "left_shift" if variant == "margin" else "", "page_w": PW, "page_h": PH,
                       "gt_x1": gt[0], "gt_y1": gt[1], "gt_x2": gt[2], "gt_y2": gt[3]}
                row.update(_features(el))
                frows.append(row)
                e = el.copy(); e.insert(0, "page_number", pg); e.insert(0, "document_id", doc)
                erows.append(e)
    return pd.DataFrame(frows), pd.concat(erows, ignore_index=True)
