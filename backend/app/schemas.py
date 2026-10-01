from typing import Any, Optional

from pydantic import BaseModel


class AnalyzeAccepted(BaseModel):
    analysis_id: str
    status: str
    pages: int


class AnalysisStatus(BaseModel):
    analysis_id: str
    status: str                      # queued | processing | done | failed
    step: str                        # uploaded | extracting | scoring | completed
    filename: str
    pages: int
    error: Optional[dict[str, str]] = None
    result: Optional[dict[str, Any]] = None
