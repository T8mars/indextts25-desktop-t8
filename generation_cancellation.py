"""Cooperative, model-step cancellation shared by Desktop UI and API."""

from __future__ import annotations

import threading
from typing import Any


class GenerationStopped(InterruptedError):
    pass


class GenerationCancellation:
    def __init__(self) -> None:
        self._guard = threading.RLock()
        self._local = threading.local()
        self._stop_generation = 0

    def begin(self, request_event: threading.Event | None = None) -> None:
        with self._guard:
            self._local.stop_generation = self._stop_generation
            self._local.request_event = request_event

    def stop(self) -> None:
        with self._guard:
            self._stop_generation += 1

    def checkpoint(self, *_args) -> None:
        with self._guard:
            stopped = getattr(self._local, "stop_generation", self._stop_generation) != self._stop_generation
            request_event = getattr(self._local, "request_event", None)
        if stopped or (request_event is not None and request_event.is_set()):
            raise GenerationStopped("generation cancelled")

    def attach(self, model: Any) -> Any:
        if getattr(model, "_t8_generation_cancellation", None) is self:
            return model
        modules = []
        gpt = getattr(model, "gpt", None)
        modules.append(getattr(gpt, "inference_model", None))
        s2mel = getattr(model, "s2mel", None)
        try:
            modules.append(s2mel.models["cfm"].estimator)
        except (AttributeError, KeyError, TypeError):
            pass
        modules.extend((getattr(model, "bigvgan", None), getattr(model, "semantic_model", None)))
        handles = []
        seen: set[int] = set()
        for module in modules:
            if module is None or id(module) in seen or not hasattr(module, "register_forward_pre_hook"):
                continue
            seen.add(id(module))
            handles.append(module.register_forward_pre_hook(self.checkpoint))
        model._t8_generation_cancellation = self
        model._t8_generation_cancellation_handles = handles
        return model


generation_cancellation = GenerationCancellation()


__all__ = ["GenerationCancellation", "GenerationStopped", "generation_cancellation"]
