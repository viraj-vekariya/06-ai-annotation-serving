"""Serve the annotation model.

WHAT IS SERVED, and why. The ablation showed the language model does not earn its forward
passes: the retriever alone scores 0.8920 against the LM's 0.6040, and fusion's optimum
weight on the LM is zero. So the LM is NOT on the serving path. It stays in the pipeline as
the thing that was measured and rejected, which is the honest outcome of an ablation.

The abstention threshold is not a constant in this file. It is read from
artifacts/serving.json, where the pipeline wrote the value it DERIVED from
coverage-at-precision on held-out data. A threshold hard-coded here would drift away from
the analysis that justified it the first time either changed.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from .health import Health
from .schemas import (BatchRequest, ClassifyRequest, ClassifyResponse, Prediction)

logging.basicConfig(level=os.environ.get("ANNOTATION_LOG_LEVEL", "INFO"))
log = logging.getLogger("annotation.serve")

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = Path(os.environ.get("ANNOTATION_ARTIFACTS", ROOT / "artifacts"))
UI = ROOT / "dashboard" / "index.html"


class Service:
    def __init__(self) -> None:
        self.config: Dict[str, object] = {}
        self.embedder = None
        self.matrix: Optional[np.ndarray] = None
        self.labels: List[str] = []
        self.texts: List[str] = []
        self.temperature = 1.0
        self.threshold = 0.5
        self.version = "unknown"
        self.encoder_backend = "unknown"
        self.requests = 0
        self.abstentions = 0
        self.latency_ms: List[float] = []
        self._lock = threading.Lock()

    def load(self) -> None:
        config_path = ARTIFACTS / "serving.json"
        index_path = ARTIFACTS / "index.npz"
        if not config_path.exists() or not index_path.exists():
            raise FileNotFoundError(
                f"missing artifacts in {ARTIFACTS}; run: python3 -m src.report")

        self.config = json.loads(config_path.read_text())
        self.temperature = float(self.config["temperature"])
        self.threshold = float(self.config["abstain_threshold"])
        self.version = str(self.config.get("embed_model", "unknown")).split("/")[-1]

        blob = np.load(index_path, allow_pickle=True)
        self.matrix = blob["matrix"]
        self.labels = [str(x) for x in blob["labels"]]
        self.texts = [str(x) for x in blob["texts"]]

        # Prefer the traced encoder: it needs only torch and `tokenizers`, skipping the
        # `transformers` import that measurement showed costs 215MB - more than the model
        # itself. Falls back to the full path when the artifact is absent, so local
        # development is unchanged.
        from .light_embedder import get_serving_embedder
        self.embedder, self.encoder_backend = get_serving_embedder(
            ARTIFACTS, self.config["embed_model"])
        log.info("encoder backend: %s", self.encoder_backend)

    def warm(self) -> float:
        """The first forward pass is materially slower than the thousandth. Serving
        before warming means the load balancer's first requests are the slowest the
        process will ever produce - exactly when an autoscaler starts adding instances."""
        started = time.perf_counter()
        for _ in range(3):
            self.classify("warmup message about my card", top_k=3)
        with self._lock:
            self.requests = 0
            self.abstentions = 0
            self.latency_ms.clear()
        return (time.perf_counter() - started) * 1000

    def classify(self, text: str, top_k: int = 5,
                 explain: bool = False) -> Dict[str, object]:
        started = time.perf_counter()
        vector = self.embedder.encode([text])[0]
        embed_ms = (time.perf_counter() - started) * 1000

        t1 = time.perf_counter()
        similarity = self.matrix @ vector
        k = min(24, len(similarity))
        idx = np.argpartition(-similarity, k - 1)[:k]
        idx = idx[np.argsort(-similarity[idx])]

        votes: Dict[str, float] = {}
        for i in idx:
            votes[self.labels[i]] = votes.get(self.labels[i], 0.0) + float(similarity[i])

        ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
        total = sum(v for _, v in ranked) or 1.0
        raw = np.array([v / total for _, v in ranked])

        # The same temperature the pipeline fitted on held-out data. Serving raw scores
        # would report a confidence the calibration report says is wrong - and the
        # abstention threshold was derived on CALIBRATED confidences, so applying it to
        # uncalibrated ones would silently change the operating point.
        scaled = raw / max(self.temperature, 1e-6)
        exp = np.exp(scaled - scaled.max())
        probabilities = exp / exp.sum()
        search_ms = (time.perf_counter() - t1) * 1000

        confidence = float(probabilities[0])
        abstain = confidence < self.threshold
        total_ms = (time.perf_counter() - started) * 1000

        with self._lock:
            self.requests += 1
            self.abstentions += int(abstain)
            self.latency_ms.append(total_ms)
            if len(self.latency_ms) > 5000:
                del self.latency_ms[:2500]

        body: Dict[str, object] = {
            "text": text,
            "prediction": ranked[0][0],
            "confidence": round(confidence, 4),
            "abstain": abstain,
            "reason": (f"confidence {confidence:.3f} is below the {self.threshold:.3f} "
                       f"threshold derived for "
                       f"{float(self.config.get('target_precision', 0.95)):.0%} precision "
                       f"- route to a human"
                       if abstain else
                       f"confidence {confidence:.3f} clears the {self.threshold:.3f} "
                       f"threshold - auto-accept"),
            "alternatives": [{"label": label, "confidence": round(float(p), 4)}
                             for (label, _), p in zip(ranked[:top_k], probabilities[:top_k])],
            "model_version": self.version,
            "threshold": round(self.threshold, 4),
            "timing_ms": {"total": round(total_ms, 3), "embed": round(embed_ms, 3),
                          "search": round(search_ms, 3)},
        }
        if explain:
            body["neighbours"] = [
                {"text": self.texts[i], "label": self.labels[i],
                 "similarity": round(float(similarity[i]), 4)} for i in idx[:5]]
        return body

    def stats(self) -> Dict[str, object]:
        with self._lock:
            n, abstentions = self.requests, self.abstentions
            samples = sorted(self.latency_ms)

        def pct(p: float) -> float:
            return round(samples[min(len(samples) - 1, int(len(samples) * p))], 3) \
                if samples else 0.0

        return {"encoder_backend": self.encoder_backend,
                "requests": n, "abstentions": abstentions,
                "abstention_rate": round(abstentions / n, 4) if n else 0.0,
                "expected_coverage": self.config.get("expected_coverage"),
                "p50_ms": pct(0.5), "p95_ms": pct(0.95), "p99_ms": pct(0.99),
                "index_size": len(self.labels), "classes": len(self.config.get("classes", [])),
                "temperature": round(self.temperature, 4),
                "threshold": round(self.threshold, 4)}


service = Service()
health = Health()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        service.load()
        health.index_loaded = health.model_loaded = True
        health.warmup_ms = round(service.warm(), 1)
        health.warmed = True
        log.info("ready: %d indexed messages, %d classes, threshold %.4f",
                 len(service.labels), len(service.config.get("classes", [])),
                 service.threshold)
    except Exception as exc:                      # noqa: BLE001
        # Start but never become READY, so an orchestrator keeps the previous version
        # serving rather than crash-looping this one.
        health.error = str(exc)
        log.exception("startup failed; the service will report NOT READY")
    yield


app = FastAPI(title="Annotation Service", version="1.0.0", lifespan=lifespan)


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    if not health.warmed:
        raise HTTPException(503, f"not ready: {health.error or 'still loading'}")
    return service.classify(req.text, req.top_k, req.explain)


@app.post("/classify/batch")
async def classify_batch(req: BatchRequest):
    if not health.warmed:
        raise HTTPException(503, "not ready")
    started = time.perf_counter()
    results = [service.classify(t, req.top_k) for t in req.texts]
    total_ms = (time.perf_counter() - started) * 1000
    return {"count": len(results),
            "abstentions": sum(1 for r in results if r["abstain"]),
            "timing_ms": {"total": round(total_ms, 3),
                          "per_item": round(total_ms / len(results), 4)},
            "results": results}


@app.get("/health/live")
async def live():
    return health.live()


@app.get("/health/ready")
async def ready():
    body = health.ready()
    # 503 is what removes the instance from the load balancer. 200 with ready:false
    # would keep traffic arriving at a service that cannot serve it.
    return JSONResponse(body, status_code=200 if body["ready"] else 503)


@app.get("/health/startup")
async def startup():
    return health.startup()


@app.get("/metrics")
async def metrics():
    return {"service": service.stats(), "health": health.ready()}


@app.get("/config")
async def config():
    return {"config": service.config,
            "note": ("The threshold was DERIVED by coverage-at-precision on held-out "
                     "data, not chosen. The language model is deliberately not on the "
                     "serving path: the ablation measured it at 0.6040 against the "
                     "retriever's 0.8920.")}


@app.get("/results")
async def results():
    path = ROOT / "outputs" / "results.json"
    if not path.exists():
        raise HTTPException(503, "no results; run python3 -m src.report")
    return json.loads(path.read_text())


@app.get("/", response_class=HTMLResponse)
async def index():
    return UI.read_text() if UI.exists() else "<h1>Annotation Service</h1><p>UI not built.</p>"
