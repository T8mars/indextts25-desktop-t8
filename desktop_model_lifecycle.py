"""Thread-safe lazy loading and release policy for the desktop IndexTTS model."""

from __future__ import annotations

import gc
import threading
import time
from collections.abc import Callable
from typing import Any

import torch


class DesktopModelLifecycle:
    """Own one model instance without touching models owned by other applications."""

    def __init__(self, model: Any, factory: Callable[[], Any]) -> None:
        self._model = model
        self._factory = factory
        self._guard = threading.RLock()
        self._idle_timer: threading.Timer | None = None
        self._completed_generations = 0
        self._loaded_at = time.time() if model is not None else None

    def get(self) -> Any:
        with self._guard:
            self._cancel_timer_locked()
            if self._model is None:
                self._model = self._factory()
                self._loaded_at = time.time()
            return self._model

    def replace(self, model: Any) -> None:
        with self._guard:
            self._cancel_timer_locked()
            previous = self._model
            self._model = model
            self._loaded_at = time.time() if model is not None else None
        if previous is not None and previous is not model:
            self._dispose(previous)

    def release(self, reason: str = "manual") -> dict[str, Any]:
        with self._guard:
            self._cancel_timer_locked()
            previous = self._model
            self._model = None
            self._loaded_at = None
        if previous is not None:
            self._dispose(previous)
        return {**self.status(), "released": previous is not None, "reason": reason}

    def after_generation(
        self,
        *,
        release_after_generation: bool = False,
        idle_seconds: float = 0,
        recycle_after_generations: int = 0,
    ) -> dict[str, Any]:
        with self._guard:
            self._completed_generations += 1
            should_recycle = (
                int(recycle_after_generations) > 0
                and self._completed_generations % int(recycle_after_generations) == 0
            )
        if release_after_generation or should_recycle:
            reason = "after_generation" if release_after_generation else "recycle_threshold"
            return self.release(reason)
        self.schedule_idle(idle_seconds)
        return self.status()

    def schedule_idle(self, idle_seconds: float) -> None:
        seconds = max(0.0, float(idle_seconds))
        with self._guard:
            self._cancel_timer_locked()
            if seconds <= 0 or self._model is None:
                return
            timer = threading.Timer(seconds, self.release, kwargs={"reason": "idle_timeout"})
            timer.daemon = True
            self._idle_timer = timer
            timer.start()

    def status(self) -> dict[str, Any]:
        with self._guard:
            loaded = self._model is not None
            payload = {
                "loaded": loaded,
                "loaded_at": self._loaded_at,
                "completed_generations": self._completed_generations,
                "idle_release_scheduled": self._idle_timer is not None,
            }
        cuda = {"available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            cuda.update(
                allocated_mb=round(torch.cuda.memory_allocated() / (1024**2), 2),
                reserved_mb=round(torch.cuda.memory_reserved() / (1024**2), 2),
                max_allocated_mb=round(torch.cuda.max_memory_allocated() / (1024**2), 2),
            )
        payload["cuda"] = cuda
        return payload

    def _cancel_timer_locked(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    @staticmethod
    def _dispose(model: Any) -> None:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["DesktopModelLifecycle"]
