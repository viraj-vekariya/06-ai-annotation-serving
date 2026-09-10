"""Liveness, readiness and startup - three signals, because they mean opposite things.

  live     - is the process alive? A failure KILLS the container, so it must never depend
             on a model or a disk; a slow dependency would otherwise cause a restart loop.
  ready    - can this instance serve? A failure removes it from the load balancer WITHOUT
             killing it. "index loaded and warmed" belongs here.
  startup  - has booting finished? Suppresses liveness during a slow start so a 20-second
             model load is not repeatedly killed at 10.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Health:
    started_at: float = field(default_factory=time.time)
    index_loaded: bool = False
    model_loaded: bool = False
    warmed: bool = False
    error: str = ""
    warmup_ms: Optional[float] = None

    def live(self) -> Dict[str, object]:
        return {"status": "alive",
                "uptime_sec": round(time.time() - self.started_at, 1)}

    def ready(self) -> Dict[str, object]:
        ok = self.index_loaded and self.model_loaded and self.warmed
        return {"status": "ready" if ok else "not_ready", "ready": ok,
                "index_loaded": self.index_loaded, "model_loaded": self.model_loaded,
                "warmed": self.warmed, "warmup_ms": self.warmup_ms, "error": self.error}

    def startup(self) -> Dict[str, object]:
        return {"status": "started" if self.model_loaded else "starting",
                "elapsed_sec": round(time.time() - self.started_at, 1)}
