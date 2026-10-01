"""Region-level localisation.

Every extracted element (text line or image block) is described by a small set
of spatial features in normalised page units. Each element is compared with a
reference distribution built from NORMAL training pages using a robust
z-score (median / MAD). Elements are ranked by their largest deviation.

Also provides the 4x4 spatial-density grid and per-cell deviation used by the
heatmap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ELEM_FEATS = ["cx", "cy", "w", "h", "area", "d_left", "d_right", "d_top",
              "d_bottom", "nn_gap", "local_density"]


def normalize_elements(el: pd.DataFrame, pw: float, ph: float) -> pd.DataFrame:
    """Points -> normalised [0,1] coordinates. Expects columns x1,y1,x2,y2,kind."""
    out = pd.DataFrame({
        "nx1": np.clip(el["x1"].to_numpy(float) / pw, 0, 1),
        "ny1": np.clip(el["y1"].to_numpy(float) / ph, 0, 1),
        "nx2": np.clip(el["x2"].to_numpy(float) / pw, 0, 1),
        "ny2": np.clip(el["y2"].to_numpy(float) / ph, 0, 1),
    })
    out["kind"] = el["kind"].astype(str).to_numpy() if "kind" in el else "text"
    return out


def element_features(en: pd.DataFrame, radius: float = 0.12) -> pd.DataFrame:
    cols = ["nx1", "ny1", "nx2", "ny2", "kind"] + ELEM_FEATS
    n = len(en)
    if n == 0:
        return pd.DataFrame(columns=cols)
    x1, y1, x2, y2 = (en[c].to_numpy(float) for c in ["nx1", "ny1", "nx2", "ny2"])
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if n > 1:
        dx = np.maximum(0, np.maximum(x1[:, None], x1[None, :]) - np.minimum(x2[:, None], x2[None, :]))
        dy = np.maximum(0, np.maximum(y1[:, None], y1[None, :]) - np.minimum(y2[:, None], y2[None, :]))
        gap = np.hypot(dx, dy)
        np.fill_diagonal(gap, np.inf)
        nn = gap.min(axis=1)
        dc = np.hypot(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :])
        np.fill_diagonal(dc, np.inf)
        dens = (dc <= radius).sum(axis=1).astype(float)
    else:
        nn, dens = np.ones(1), np.zeros(1)
    out = pd.DataFrame({
        "nx1": x1, "ny1": y1, "nx2": x2, "ny2": y2, "kind": en["kind"].to_numpy(),
        "cx": cx, "cy": cy, "w": w, "h": h, "area": w * h,
        "d_left": x1, "d_right": 1 - x2, "d_top": y1, "d_bottom": 1 - y2,
        "nn_gap": nn, "local_density": dens,
    })
    return out[cols]


def grid_density(en: pd.DataFrame, rows: int = 4, cols: int = 4) -> np.ndarray:
    """density(cell) = elements whose centre is in the cell / total elements."""
    g = np.zeros((rows, cols))
    if len(en) == 0:
        return g
    cx = (en["nx1"].to_numpy(float) + en["nx2"].to_numpy(float)) / 2
    cy = (en["ny1"].to_numpy(float) + en["ny2"].to_numpy(float)) / 2
    ci = np.clip((cx * cols).astype(int), 0, cols - 1)
    ri = np.clip((cy * rows).astype(int), 0, rows - 1)
    np.add.at(g, (ri, ci), 1)
    return g / g.sum()


# ---------- reference distributions ----------
def _robust(df: pd.DataFrame, floor: float) -> dict:
    x = df[ELEM_FEATS].to_numpy(float)
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0) * 1.4826
    std = x.std(axis=0)
    scale = np.maximum.reduce([mad, 0.25 * std, np.full_like(mad, floor)])
    return {"n": int(len(df)), "median": dict(zip(ELEM_FEATS, map(float, med))),
            "scale": dict(zip(ELEM_FEATS, map(float, scale)))}


def build_element_reference(frames: list[pd.DataFrame], floor: float = 0.02) -> dict:
    frames = [f for f in frames if len(f)]
    if not frames:
        return {}
    allf = pd.concat(frames, ignore_index=True)
    ref = {"all": _robust(allf, floor)}
    for kind, sub in allf.groupby("kind"):
        ref[str(kind)] = _robust(sub, floor)
    return ref


def score_regions(ef: pd.DataFrame, ref: dict, top_k: int = 3, z_cap: float = 6.0,
                  min_ref: int = 10, flag_z: float = 4.0) -> list[dict]:
    if len(ef) == 0 or not ref:
        return []
    zs = np.zeros((len(ef), len(ELEM_FEATS)))
    refs = []
    for i, kind in enumerate(ef["kind"].to_numpy()):
        r = ref.get(str(kind))
        if r is None or r["n"] < min_ref:
            r = ref["all"]
        refs.append(r)
        med = np.array([r["median"][f] for f in ELEM_FEATS])
        sc = np.array([r["scale"][f] for f in ELEM_FEATS])
        zs[i] = np.abs(ef.iloc[i][ELEM_FEATS].to_numpy(float) - med) / sc
    zmax = zs.max(axis=1)
    out = []
    for i in np.argsort(-zmax)[:top_k]:
        order = np.argsort(-zs[i])[:3]
        r = refs[i]
        out.append({
            "bbox_norm": [float(ef.iloc[i][c]) for c in ("nx1", "ny1", "nx2", "ny2")],
            "kind": str(ef.iloc[i]["kind"]),
            "z": float(zmax[i]),
            "score": float(min(1.0, zmax[i] / z_cap)),
            "flagged": bool(zmax[i] >= flag_z),
            "top_features": [{"feature": ELEM_FEATS[j], "z": float(zs[i, j]),
                              "value": float(ef.iloc[i][ELEM_FEATS[j]]),
                              "reference_median": r["median"][ELEM_FEATS[j]]} for j in order],
        })
    return out


def build_cell_reference(grids: list[np.ndarray], floor: float = 0.01) -> dict:
    if not grids:
        return {}
    a = np.stack(grids)
    return {"mean": a.mean(0).tolist(), "std": np.maximum(a.std(0), floor).tolist()}


def cell_deviation(grid: np.ndarray, ref: dict) -> np.ndarray:
    """Signed z-score of each cell's density against normal training pages."""
    if not ref:
        return np.zeros_like(grid)
    return (grid - np.array(ref["mean"])) / np.array(ref["std"])


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0
