"""
Phase 3/4 - Pattern representation.

`extract_page_features(layout, cfg)` turns one PageLayout into a fixed-length numeric vector (dict).
`element_features(layout, cfg)` gives one row per element (used later for region-level localisation).

All geometry is in page-normalised coordinates (x/page_width, y/page_height), origin top-left.
Variance/std are POPULATION statistics (ddof=0), i.e.  sigma^2 = (1/n) * sum (x_i - mu)^2.

Feature groups (so training can ablate them):
  margin    per-side distribution of element edges (mean/median/std/var/min/max) + 2 balance ratios   (26)
  bbox      width / height / area / aspect / centre-x / centre-y statistics                            (17)
  density   4x4 normalised cell densities + mean(count) / std / var / max / min / entropy               (22)
  counts    n_elements, n_text_elements                                                                 (2)
  signature min/max aggregates over non-text elements: size, position, gap to text, overlap, density  (16)

Design choices worth documenting
  * Margins are computed from TEXT elements only (falls back to all elements if a page has no text), so a stray
    image cannot masquerade as a margin change; images are captured by the signature group and by bbox/density.
  * density_mean is the mean RAW count per cell (= n_elements / n_cells). The mean of the *normalised* grid is
    always 1/n_cells (a constant), so it carries no information.
  * Signature features aggregate ALL non-text elements with order-independent min/max. A single "most suspicious
    element" rule was tried first and failed: an inserted signature overlapping body text (gap = 0) lost to the
    genuine signature, so it was invisible to the position features. Pages without non-text elements get NaN for
    position-type features (imputed later with TRAINING-set medians only) and 0 for count/area features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .extract import PageLayout

EPS = 1e-9
STATS = ["mean", "median", "std", "var", "min", "max"]
SIDES = ["left", "right", "top", "bottom"]
SIGNATURE_FEATURES = [
    "sig_n_nontext", "sig_area_ratio_total", "sig_rel_area_max", "sig_rel_area_min",
    "sig_center_x_min", "sig_center_x_max", "sig_center_y_min", "sig_center_y_max",
    "sig_dist_nearest_text_min", "sig_dist_nearest_text_max",
    "sig_dist_bottom_margin_min", "sig_dist_bottom_margin_max",
    "sig_dist_right_margin_min", "sig_dist_right_margin_max",
    "sig_text_overlap_max", "sig_local_density_max"]


def feature_groups(cfg: dict) -> dict[str, list[str]]:
    fc = cfg["features"]
    n_cells = fc["grid_rows"] * fc["grid_cols"]
    return {
        "margin": [f"{s}_margin_{st}" for s in SIDES for st in STATS] + ["margin_ratio_lr", "margin_ratio_tb"],
        "bbox": ["bbox_width_mean", "bbox_width_std", "bbox_width_variance",
                 "bbox_height_mean", "bbox_height_std", "bbox_height_variance",
                 "bbox_area_mean", "bbox_area_std", "bbox_area_variance",
                 "bbox_aspect_ratio_mean", "bbox_aspect_ratio_std",
                 "center_x_mean", "center_x_std", "center_x_variance",
                 "center_y_mean", "center_y_std", "center_y_variance"],
        "density": [f"density_cell_{i:02d}" for i in range(n_cells)]
                   + ["density_mean", "density_std", "density_variance", "density_max", "density_min",
                      "density_entropy"],
        "counts": ["n_elements", "n_text_elements"],
        "signature": list(SIGNATURE_FEATURES),
    }


def feature_names(cfg: dict) -> list[str]:
    return [n for g in feature_groups(cfg).values() for n in g]


# --------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------
def boxes_and_kinds(layout: PageLayout) -> tuple[np.ndarray, np.ndarray]:
    boxes = np.array([[e.x1, e.y1, e.x2, e.y2] for e in layout.elements], dtype=float).reshape(-1, 4)
    is_text = np.array([e.kind == "text" for e in layout.elements], dtype=bool)
    return boxes, is_text


def pairwise_gap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Edge-to-edge distance between every box in a (n,4) and b (m,4); 0 when they touch/overlap."""
    dx = np.maximum(np.maximum(b[None, :, 0] - a[:, None, 2], a[:, None, 0] - b[None, :, 2]), 0)
    dy = np.maximum(np.maximum(b[None, :, 1] - a[:, None, 3], a[:, None, 1] - b[None, :, 3]), 0)
    return np.hypot(dx, dy)


def pairwise_overlap_area(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix = np.maximum(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0)
    iy = np.maximum(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0)
    return ix * iy


def cell_indices(centers: np.ndarray, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    c = np.clip((centers[:, 0] * cols).astype(int), 0, cols - 1)
    r = np.clip((centers[:, 1] * rows).astype(int), 0, rows - 1)
    return r, c


def density_grid(boxes: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Raw element counts per cell (element assigned to the cell containing its centre)."""
    grid = np.zeros((rows, cols))
    if len(boxes):
        centers = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
        r, c = cell_indices(centers, rows, cols)
        np.add.at(grid, (r, c), 1)
    return grid


def _stats(prefix: str, v: np.ndarray) -> dict[str, float]:
    return {f"{prefix}_mean": float(np.mean(v)), f"{prefix}_median": float(np.median(v)),
            f"{prefix}_std": float(np.std(v)), f"{prefix}_var": float(np.var(v)),
            f"{prefix}_min": float(np.min(v)), f"{prefix}_max": float(np.max(v))}


# --------------------------------------------------------------------------------------
# page-level features
# --------------------------------------------------------------------------------------
def extract_page_features(layout: PageLayout, cfg: dict) -> dict[str, float]:
    fc = cfg["features"]
    rows, cols, radius = fc["grid_rows"], fc["grid_cols"], fc["local_density_radius"]
    names = feature_names(cfg)
    f: dict[str, float] = {n: np.nan for n in names}

    boxes, is_text = boxes_and_kinds(layout)
    n = len(boxes)
    f["n_elements"], f["n_text_elements"] = float(n), float(is_text.sum())
    f.update({"sig_n_nontext": 0.0, "sig_area_ratio_total": 0.0, "sig_rel_area_max": 0.0})
    if n == 0:
        return {k: f[k] for k in names}
    W, H = layout.width_pt, layout.height_pt

    # ---- margins (text elements)
    mb = boxes[is_text] if is_text.any() else boxes
    edges = {"left": mb[:, 0], "right": 1 - mb[:, 2], "top": mb[:, 1], "bottom": 1 - mb[:, 3]}
    for side, v in edges.items():
        f.update(_stats(f"{side}_margin", v))
    L, R, T, B = (f[f"{s}_margin_min"] for s in SIDES)      # page margins = closest text edge to each side
    f["margin_ratio_lr"] = L / (L + R + EPS)                # horizontal balance; right share = 1 - value
    f["margin_ratio_tb"] = T / (T + B + EPS)                # vertical balance;   bottom share = 1 - value

    # ---- bounding-box statistics (all elements)
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    area = w * h
    aspect = (w * W) / np.maximum(h * H, EPS)               # physical aspect ratio (points), not normalised units
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    for name, v in (("width", w), ("height", h), ("area", area)):
        f[f"bbox_{name}_mean"], f[f"bbox_{name}_std"], f[f"bbox_{name}_variance"] = v.mean(), v.std(), v.var()
    f["bbox_aspect_ratio_mean"], f["bbox_aspect_ratio_std"] = aspect.mean(), aspect.std()
    for name, v in (("center_x", cx), ("center_y", cy)):
        f[f"{name}_mean"], f[f"{name}_std"], f[f"{name}_variance"] = v.mean(), v.std(), v.var()

    # ---- spatial density
    raw = density_grid(boxes, rows, cols)
    dens = raw / raw.sum()
    for i, v in enumerate(dens.ravel()):
        f[f"density_cell_{i:02d}"] = float(v)
    f["density_mean"] = float(raw.mean())
    f["density_std"], f["density_variance"] = float(dens.std()), float(dens.var())
    f["density_max"], f["density_min"] = float(dens.max()), float(dens.min())
    p = dens.ravel()[dens.ravel() > 0]
    f["density_entropy"] = float(-(p * np.log(p)).sum() / np.log(rows * cols))

    # ---- signature-related: order-independent min/max aggregates over ALL non-text elements
    nt = np.where(~is_text)[0]
    f["sig_n_nontext"] = float(len(nt))
    if len(nt):
        ntb, a_nt = boxes[nt], area[nt]
        f["sig_area_ratio_total"] = float(a_nt.sum())
        f["sig_rel_area_max"], f["sig_rel_area_min"] = float(a_nt.max()), float(a_nt.min())
        f["sig_center_x_min"], f["sig_center_x_max"] = float(cx[nt].min()), float(cx[nt].max())
        f["sig_center_y_min"], f["sig_center_y_max"] = float(cy[nt].min()), float(cy[nt].max())
        tb = boxes[is_text]
        if len(tb):
            gap = pairwise_gap(ntb, tb).min(axis=1)
            d_bottom = tb[:, 3].max() - ntb[:, 3]          # signed: < 0 => element lies below the text block
            d_right = tb[:, 2].max() - ntb[:, 2]           # signed: < 0 => element lies right of the text block
            ov = np.minimum(pairwise_overlap_area(ntb, tb).sum(axis=1) / np.maximum(a_nt, EPS), 1.0)
            f["sig_dist_nearest_text_min"], f["sig_dist_nearest_text_max"] = float(gap.min()), float(gap.max())
            f["sig_dist_bottom_margin_min"], f["sig_dist_bottom_margin_max"] = float(d_bottom.min()), float(d_bottom.max())
            f["sig_dist_right_margin_min"], f["sig_dist_right_margin_max"] = float(d_right.min()), float(d_right.max())
            f["sig_text_overlap_max"] = float(ov.max())
        dd = np.hypot(cx[nt][:, None] - cx[None, :], cy[nt][:, None] - cy[None, :])
        f["sig_local_density_max"] = float(((dd <= radius).sum(axis=1) - 1).max() / n)

    return {k: float(f[k]) for k in names}


def features_dataframe(layouts: list[PageLayout], cfg: dict) -> pd.DataFrame:
    return pd.DataFrame([extract_page_features(l, cfg) for l in layouts])


# --------------------------------------------------------------------------------------
# element-level features (region localisation, Block B)
# --------------------------------------------------------------------------------------
def element_features(layout: PageLayout, cfg: dict) -> pd.DataFrame:
    fc = cfg["features"]
    rows, cols, radius = fc["grid_rows"], fc["grid_cols"], fc["local_density_radius"]
    boxes, is_text = boxes_and_kinds(layout)
    n = len(boxes)
    if n == 0:
        return pd.DataFrame()
    W, H = layout.width_pt, layout.height_pt
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    r, c = cell_indices(np.stack([cx, cy], axis=1), rows, cols)

    gap_all = pairwise_gap(boxes, boxes)
    np.fill_diagonal(gap_all, np.inf)
    gap_text = gap_all[:, is_text] if is_text.any() else np.full((n, 1), np.inf)
    d = np.hypot(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :])
    np.fill_diagonal(d, np.inf)
    overlap = pairwise_overlap_area(boxes, boxes[is_text]) if is_text.any() else np.zeros((n, 1))
    tidx = np.where(is_text)[0]
    for j, t in enumerate(tidx):        # an element does not overlap itself
        overlap[t, j] = 0.0

    def _finite_min(m):
        v = m.min(axis=1)
        return np.where(np.isfinite(v), v, 1.0)

    return pd.DataFrame({
        "page": layout.page_number, "idx": np.arange(n), "kind": [e.kind for e in layout.elements],
        "x1": boxes[:, 0], "y1": boxes[:, 1], "x2": boxes[:, 2], "y2": boxes[:, 3],
        "width": w, "height": h, "center_x": cx, "center_y": cy, "area": w * h,
        "aspect_ratio": (w * W) / np.maximum(h * H, EPS),
        "cell_row": r, "cell_col": c, "cell_id": r * cols + c,
        "local_density": (d <= radius).sum(axis=1) / n,
        "dist_nearest_element": _finite_min(gap_all),
        "dist_nearest_text": _finite_min(gap_text),
        "dist_left": boxes[:, 0], "dist_right": 1 - boxes[:, 2], "dist_top": boxes[:, 1], "dist_bottom": 1 - boxes[:, 3],
        "text_overlap": np.minimum(overlap.sum(axis=1) / np.maximum(w * h, EPS), 1.0),
    })
