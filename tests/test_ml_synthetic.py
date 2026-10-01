import json, tempfile
from pathlib import Path
import numpy as np
from ml.common import load_config
from ml import train, evaluate
from ml.scoring import Detector
from ml.regions import iou
from tests.synth import make_tables


def _cfg(tmp):
    cfg = load_config()
    cfg["paths"]["models_dir"] = str(Path(tmp) / "models")
    cfg["paths"]["reports_dir"] = str(Path(tmp) / "reports")
    return cfg


def test_end_to_end():
    feats, elems = make_tables()
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(tmp)
        art = train.fit(feats, elems, cfg)
        train.save(art, cfg)
        sp = art["split"]
        # no source document appears in two splits
        assert not (set(sp["train"]) & set(sp["val"]) | set(sp["train"]) & set(sp["test"]) | set(sp["val"]) & set(sp["test"]))
        det = Detector(cfg["paths"]["models_dir"])
        # threshold maps to 0.5 by construction
        assert abs(det.threshold_norm - 0.5) < 1e-9
        # scaler was fitted on train-normal only
        tr = feats[(feats.variant == "normal") & feats.source_doc_id.isin(sp["train"])]
        assert np.allclose(det.pre.named_steps["scaler"].mean_, tr[det.features].mean().to_numpy(), atol=1e-9)
        res = evaluate.run(cfg, feats, elems, compare=True, plots=True)
        assert res["experiments"]["margin_alteration"]["roc_auc"] > 0.8   # sanity floor only; data is synthetic
        assert res["experiments"]["overall"]["roc_auc"] > 0.6
        assert 0 <= res["experiments"]["overall"]["false_positive_rate"] <= 0.3
        assert res["region_level"]["n_pages"] > 0
        print(json.dumps({k: res[k] for k in ("experiments", "region_level")}, indent=1))
        # determinism
        art2 = train.fit(feats, elems, cfg)
        assert art2["feature_config"]["threshold"]["raw"] == art["feature_config"]["threshold"]["raw"]


def test_iou():
    assert iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert iou([0, 0, .5, .5], [.5, .5, 1, 1]) == 0.0


if __name__ == "__main__":
    test_iou(); test_end_to_end(); print("ALL OK")
