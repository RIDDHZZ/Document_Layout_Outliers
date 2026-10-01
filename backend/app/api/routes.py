import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Request, UploadFile
from fastapi.responses import FileResponse

from backend.app.errors import AppError
from backend.app.schemas import AnalysisStatus, AnalyzeAccepted
from backend.app.services.analysis import run_analysis
from backend.app.services.validation import inspect_pdf, read_upload, safe_name

router = APIRouter(prefix="/api")


def _det(request: Request):
    det = request.app.state.detector
    if det is None:
        raise AppError(503, "model_missing", "The anomaly model has not been trained yet. Run: python -m ml.train")
    return det


def _done(request: Request, aid: str) -> dict:
    rec = request.app.state.store.get(aid)
    if rec["status"] == "failed":
        raise AppError(422, rec["error"]["code"], rec["error"]["message"])
    if rec["status"] != "done":
        raise AppError(409, "not_ready", "The analysis is still running.")
    return rec


@router.get("/health")
def health(request: Request):
    return {"status": "ok", "model_loaded": request.app.state.detector is not None}


@router.post("/analyze", status_code=202, response_model=AnalyzeAccepted)
async def analyze(request: Request, background: BackgroundTasks, file: UploadFile = File(...)):
    st, cfg = request.app.state, request.app.state.cfg
    det = _det(request)
    st.store.sweep()
    data = await read_upload(file, cfg["upload"]["max_mb"] * 1024 * 1024)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=st.tmp_dir) as tmp:
        tmp.write(data)
    path = Path(tmp.name)
    try:
        n_pages = inspect_pdf(path, cfg["upload"]["max_pages"])
        aid = st.store.create(safe_name(file.filename), n_pages)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    background.add_task(run_analysis, st.store, aid, path, safe_name(file.filename), n_pages, det, cfg)
    return AnalyzeAccepted(analysis_id=aid, status="queued", pages=n_pages)


@router.get("/analysis/{aid}", response_model=AnalysisStatus)
def get_analysis(request: Request, aid: str):
    rec = request.app.state.store.get(aid)
    return AnalysisStatus(**{k: rec[k] for k in ("analysis_id", "status", "step", "filename", "pages", "error", "result")})


@router.get("/analysis/{aid}/page/{page}")
def get_page(request: Request, aid: str, page: int):
    rec = _done(request, aid)
    for p in rec["page_results"]:
        if p["page"] == page:
            return p
    raise AppError(404, "page_not_found", f"Page {page} does not exist in this analysis.")


@router.get("/analysis/{aid}/page/{page}/image")
def get_page_image(request: Request, aid: str, page: int):
    request.app.state.store.get(aid)
    img = request.app.state.store.image_dir(aid) / f"page_{int(page)}.png"
    if not img.exists():
        raise AppError(404, "image_not_found", "Page preview not available.")
    return FileResponse(img, media_type="image/png")


@router.get("/analysis/{aid}/features")
def get_features(request: Request, aid: str):
    rec = _done(request, aid)
    return {"analysis_id": aid, "threshold": rec["result"]["threshold"], "pages": rec["features"]}


@router.get("/analysis/{aid}/visualization")
def get_visualization(request: Request, aid: str):
    rec = _done(request, aid)
    return {"analysis_id": aid, "pages": [
        {"page": p["page"], "image_url": p["image_url"], "page_width": p["page_width"],
         "page_height": p["page_height"], "status": p["status"], "score": p["score"],
         "elements": p["elements"], "regions": p["regions"], "density_grid": p["density_grid"]}
        for p in rec["page_results"]]}


@router.get("/evaluation")
def get_evaluation(request: Request):
    f = request.app.state.reports_dir / "metrics.json"
    if not f.exists():
        raise AppError(404, "no_evaluation", "No evaluation report found. Run: python -m ml.evaluate")
    return json.loads(f.read_text(encoding="utf-8"))


@router.get("/model-info")
def model_info(request: Request):
    d = _det(request).cfg
    return {k: d[k] for k in ("created_utc", "random_state", "model", "preprocessing", "threshold",
                              "score_norm", "dataset", "grid")} | {"n_features": len(d["features"]),
                                                                     "features": d["features"]}
