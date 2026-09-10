"""Request and response shapes.

Every response carries the model version, the threshold that was applied, and whether the
item was abstained on. A classification API that returns only a label gives an operator no
way to tell a confident answer from a coin flip, and no way to know which model produced it
when something goes wrong.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(5, ge=1, le=20)
    explain: bool = Field(False, description="include the nearest training neighbours")


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=256)
    top_k: int = Field(3, ge=1, le=20)


class Prediction(BaseModel):
    label: str
    confidence: float


class ClassifyResponse(BaseModel):
    text: str
    prediction: str
    confidence: float
    abstain: bool
    reason: str
    alternatives: List[Prediction]
    neighbours: Optional[List[Dict[str, object]]] = None
    model_version: str
    threshold: float
    timing_ms: Dict[str, float]
