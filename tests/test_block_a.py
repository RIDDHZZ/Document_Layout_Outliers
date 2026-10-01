import random
import shutil

import numpy as np
import pandas as pd
import pymupdf
import pytest

from ml.config import load_config
from ml.extract import ExtractionError, extract_document, render_page
from ml.features import element_features, extract_page_features, feature_groups, feature_names
from ml.generate_dataset import build_spec, generate, insert_signature, render_pdf

CFG = load_config()


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    ann = generate(CFG, n=6, root=root)
    return root, ann


def iou(a, b):
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def test_annotations_schema_and_groups(mini):
    root, ann = mini
    for c in ["document_id", "source_doc_id", "page_number", "anomaly_type", "x1", "y1", "x2", "y2", "is_anomalous"]:
        assert c in ann.columns
    assert set(ann.category) == {"normal", "altered_margin", "inserted_signature"}
    # every variant shares source_doc_id with its original (needed for leakage-free splits)
    assert (ann.groupby("source_doc_id").category.nunique() == 3).all()
    assert (ann[ann.category == "normal"].is_anomalous == 0).all()
    var = ann[ann.category != "normal"].groupby("document_id").is_anomalous.sum()
    assert (var >= 1).all()   # every variant has at least one anomalous page


def test_generation_is_reproducible(tmp_path):
    a = generate(CFG, n=3, root=tmp_path / "a")
    b = generate(CFG, n=3, root=tmp_path / "b")
    pd.testing.assert_frame_equal(a, b)


def test_native_extraction_and_feature_vector(mini):
    root, ann = mini
    layouts = extract_document(root / "normal/doc_001.pdf", CFG)
    assert layouts[0].source == "native" and layouts[0].elements
    names = feature_names(CFG)
    assert len(names) == len(set(names)) == sum(len(v) for v in feature_groups(CFG).values())
    f = extract_page_features(layouts[0], CFG)
    assert list(f) == names
    # density grid is a probability distribution over cells
    cells = [f[n] for n in names if n.startswith("density_cell_")]
    assert len(cells) == CFG["features"]["grid_rows"] * CFG["features"]["grid_cols"]
    assert abs(sum(cells) - 1) < 1e-9
    # everything except signature position features must be defined on a text page
    nan = [n for n, v in f.items() if np.isnan(v)]
    assert all(n.startswith("sig_") for n in nan)
    assert 0 <= f["left_margin_min"] < 0.5


def test_inserted_signature_bbox_matches_ground_truth(mini):
    root, ann = mini
    for _, r in ann[ann.anomaly_type == "inserted_signature"].iterrows():
        lay = extract_document(root / r.file, CFG)[int(r.page_number) - 1]
        gt = [r.x1 / r.page_width, r.y1 / r.page_height, r.x2 / r.page_width, r.y2 / r.page_height]
        best = max(iou(gt, [e.x1, e.y1, e.x2, e.y2]) for e in lay.elements if e.kind == "image")
        assert best > 0.98


def test_margin_alteration_is_measurable(mini):
    root, ann = mini
    rows = ann[ann.anomaly_type == "altered_margin"]
    assert len(rows) > 0
    moved = 0
    for _, r in rows.iterrows():
        lay = extract_document(root / r.file, CFG)[int(r.page_number) - 1]
        boxes = np.array([[e.x1, e.y1, e.x2, e.y2] for e in lay.elements if e.kind == "text"])
        # ground-truth bbox was measured on the same page at generation time (text + images union)
        gt = np.array([r.x1 / r.page_width, r.y1 / r.page_height, r.x2 / r.page_width, r.y2 / r.page_height])
        assert abs(boxes[:, 0].min() - gt[0]) < 0.05
        moved += 1
    assert moved == len(rows)


def test_explicit_left_margin_increase_moves_left_margin_feature(tmp_path):
    spec = build_spec("doc_x", random.Random(3), CFG)
    n, _ = render_pdf(spec, tmp_path / "n.pdf")
    alt = dict(mode="left_increase", pages=set(range(n)), left=60.0, right=0, top=0, bottom=0, dx=0, dy=0)
    render_pdf(spec, tmp_path / "a.pdf", alt)
    fn = extract_page_features(extract_document(tmp_path / "n.pdf", CFG)[0], CFG)
    fa = extract_page_features(extract_document(tmp_path / "a.pdf", CFG)[0], CFG)
    assert fa["left_margin_min"] - fn["left_margin_min"] == pytest.approx(60.0 / spec.page_w, abs=0.01)


def test_rotated_page_boxes_stay_inside_page(mini, tmp_path):
    root, _ = mini
    d = pymupdf.open(root / "normal/doc_001.pdf")
    d[0].set_rotation(90)
    d.save(tmp_path / "rot.pdf")
    lay = extract_document(tmp_path / "rot.pdf", CFG)[0]
    b = np.array([[e.x1, e.y1, e.x2, e.y2] for e in lay.elements])
    assert b.min() >= 0 and b.max() <= 1 and lay.width_pt > lay.height_pt


def test_element_table(mini):
    root, _ = mini
    lay = extract_document(root / "normal/doc_001.pdf", CFG)[0]
    et = element_features(lay, CFG)
    assert len(et) == len(lay.elements)
    assert et[["local_density", "dist_nearest_element", "dist_nearest_text", "cell_id"]].notna().all().all()
    assert et.cell_id.between(0, 15).all()


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_ocr_fallback_on_scanned_pdf(mini, tmp_path):
    root, _ = mini
    src = pymupdf.open(root / "normal/doc_001.pdf")
    scan = pymupdf.open()
    for p in src:                                   # image-only PDF = no text layer
        pg = scan.new_page(width=p.rect.width, height=p.rect.height)
        pg.insert_image(pg.rect, pixmap=p.get_pixmap(dpi=150))
    scan.save(tmp_path / "scan.pdf")
    lay = extract_document(tmp_path / "scan.pdf", CFG)[0]
    nat = extract_document(root / "normal/doc_001.pdf", CFG)[0]
    assert lay.source == "ocr"
    n_ocr = sum(e.kind == "text" for e in lay.elements)
    n_nat = sum(e.kind == "text" for e in nat.elements)
    assert abs(n_ocr - n_nat) <= max(3, 0.25 * n_nat)


def test_error_handling(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    with pytest.raises(ExtractionError):
        extract_document(bad, CFG)
    blank = pymupdf.open()
    blank.new_page()
    blank.save(tmp_path / "blank.pdf")
    with pytest.raises(ExtractionError, match="no readable layout elements"):
        extract_document(tmp_path / "blank.pdf", CFG, force_ocr=False)
