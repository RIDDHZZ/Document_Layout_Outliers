"""FastAPI entry point.

Run from the project root:

    uvicorn backend.app.main:app --reload --port 8000
"""

import logging
import shutil
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from backend.app.api.routes import router
from backend.app.errors import AppError
from backend.app.services.store import AnalysisStore
from ml.common import load_config, resolve
from ml.scoring import Detector

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    app.state.cfg = cfg
    app.state.reports_dir = resolve(cfg, "reports_dir")
    app.state.store = AnalysisStore(cfg["upload"]["ttl_seconds"])
    app.state.tmp_dir = tempfile.mkdtemp(prefix="pad_uploads_")

    try:
        app.state.detector = Detector(resolve(cfg, "models_dir"))
        log.info("Model loaded (%d features)", len(app.state.detector.features))
    except FileNotFoundError:
        app.state.detector = None
        log.warning(
            "No trained model found in %s. Run `python -m ml.train`.",
            resolve(cfg, "models_dir"),
        )

    yield

    app.state.store.close()
    shutil.rmtree(app.state.tmp_dir, ignore_errors=True)


app = FastAPI(
    title="Document Layout Anomaly Detection API",
    version="1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)


# ---------------------------------------------------------
# Frontend
# ---------------------------------------------------------

FRONTEND_CANDIDATES = [
    ROOT / "frontend" / "index.html",
    ROOT / "frontend" / "frontend" / "index.html",
]


@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the DocuTrace frontend."""

    for index_file in FRONTEND_CANDIDATES:
        if index_file.exists():
            return FileResponse(index_file)

    return JSONResponse(
        status_code=404,
        content={
            "error": "Frontend index.html not found",
            "checked": [str(path) for path in FRONTEND_CANDIDATES],
        },
    )


# ---------------------------------------------------------
# Error handlers
# ---------------------------------------------------------

@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
            }
        },
    )


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception):
    log.exception("Unhandled error")

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "Something went wrong while processing the request.",
            }
        },
    )

