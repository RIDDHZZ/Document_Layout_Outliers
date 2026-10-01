"""In-memory analysis store with a time-to-live.

Uploaded PDFs are deleted right after processing. Only derived results and
low-resolution page previews (needed for the overlay view) are kept, and they
are purged after `ttl_seconds`. Nothing is written to a database.
"""
from __future__ import annotations

import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

from backend.app.errors import AppError


class AnalysisStore:
    def __init__(self, ttl_seconds: int = 1800):
        self.ttl = ttl_seconds
        self.base = Path(tempfile.mkdtemp(prefix="pad_analyses_"))
        self._d: dict[str, dict] = {}
        self._lock = threading.Lock()

    def create(self, filename: str, pages: int) -> str:
        aid = uuid.uuid4().hex[:12]
        with self._lock:
            self._d[aid] = {"analysis_id": aid, "filename": filename, "pages": pages,
                            "status": "queued", "step": "uploaded", "error": None,
                            "result": None, "page_results": [], "features": [],
                            "created": time.time()}
        (self.base / aid).mkdir(parents=True, exist_ok=True)
        return aid

    def get(self, aid: str) -> dict:
        with self._lock:
            rec = self._d.get(aid)
        if rec is None:
            raise AppError(404, "not_found", "Analysis not found or it has expired.")
        return rec

    def update(self, aid: str, **kw) -> None:
        with self._lock:
            self._d[aid].update(kw)

    def image_dir(self, aid: str) -> Path:
        return self.base / aid

    def sweep(self) -> None:
        now = time.time()
        with self._lock:
            dead = [k for k, v in self._d.items() if now - v["created"] > self.ttl]
            for k in dead:
                del self._d[k]
        for k in dead:
            shutil.rmtree(self.base / k, ignore_errors=True)

    def close(self) -> None:
        shutil.rmtree(self.base, ignore_errors=True)
