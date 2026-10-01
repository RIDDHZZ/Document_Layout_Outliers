import json, tempfile
from pathlib import Path
import pandas as pd
from ml.common import load_config, feature_columns
from ml import train
from ml.scoring import Detector
from backend.app.services import analysis as A
from backend.app.services.store import AnalysisStore
from tests.synth import make_tables
from tests.test_ml_synthetic import _cfg


def _records(feats, elems, doc):
    out = []
    for _, r in feats[feats.document_id == doc].iterrows():
        el = elems[(elems.document_id == doc) & (elems.page_number == r.page_number)][["x1", "y1", "x2", "y2", "kind"]]
        out.append({"page_number": int(r.page_number), "page_w": r.page_w, "page_h": r.page_h,
                    "features": {c: r[c] for c in feature_columns(feats)}, "elements": el.reset_index(drop=True)})
    return out


def test_service():
    feats, elems = make_tables()
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(tmp)
        art = train.fit(feats, elems, cfg); train.save(art, cfg)
        det = Detector(cfg["paths"]["models_dir"])
        test_src = art["split"]["test"][0]

        recs = _records(feats, elems, f"{test_src}_signature")
        res, pages, ft = A.analyze_records(recs, det, cfg, "abc", "x.pdf", 2)
        json.dumps([res, pages, ft])                                  # everything must be JSON-serialisable
        assert res["overall_status"] in ("normal", "potential_anomaly")
        assert 0 <= res["overall_score"] <= 1 and len(pages) == 2
        assert pages[0]["regions"] and "bbox" in pages[0]["regions"][0]
        print(pages[0]["explanation"])

        # background job: failure path must not leak a stack trace and must delete the upload
        store = AnalysisStore(60)
        aid = store.create("x.pdf", 1)
        pdf = Path(tmp) / "up.pdf"; pdf.write_bytes(b"%PDF-1.4 fake")
        A.render_pages = lambda *a, **k: None
        import ml.pipeline as P
        P.extract_page_records = lambda *a, **k: []
        A.run_analysis(store, aid, pdf, "x.pdf", 1, det, cfg)
        rec = store.get(aid)
        assert rec["status"] == "failed" and rec["error"]["code"] == "no_layout"
        assert not pdf.exists()

        # success path
        P.extract_page_records = lambda *a, **k: recs
        aid2 = store.create("y.pdf", 2); pdf.write_bytes(b"%PDF-1.4 fake")
        A.run_analysis(store, aid2, pdf, "y.pdf", 2, det, cfg)
        assert store.get(aid2)["status"] == "done" and not pdf.exists()
        store.close()


if __name__ == "__main__":
    test_service(); print("BACKEND SERVICE OK")
