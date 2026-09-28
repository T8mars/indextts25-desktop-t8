from __future__ import annotations

import threading

import pytest

from generation_cancellation import GenerationCancellation, GenerationStopped


class FakeModule:
    def __init__(self):
        self.hook = None

    def register_forward_pre_hook(self, hook):
        self.hook = hook
        return object()


class FakeModel:
    def __init__(self):
        self.gpt = type("GPT", (), {"inference_model": FakeModule()})()
        estimator = FakeModule()
        self.s2mel = type("S2Mel", (), {"models": {"cfm": type("CFM", (), {"estimator": estimator})()}})()
        self.bigvgan = FakeModule()
        self.semantic_model = FakeModule()


def test_stop_interrupts_attached_model_at_next_forward_step():
    cancellation = GenerationCancellation()
    model = cancellation.attach(FakeModel())
    cancellation.begin()
    model.gpt.inference_model.hook(model.gpt.inference_model, ())
    cancellation.stop()
    with pytest.raises(GenerationStopped):
        model.bigvgan.hook(model.bigvgan, ())


def test_api_cancel_event_is_checked_inside_model_step():
    cancellation = GenerationCancellation()
    model = cancellation.attach(FakeModel())
    request_cancel = threading.Event()
    cancellation.begin(request_cancel)
    request_cancel.set()
    with pytest.raises(GenerationStopped):
        model.semantic_model.hook(model.semantic_model, ())


def test_attach_is_idempotent():
    cancellation = GenerationCancellation()
    model = FakeModel()
    assert cancellation.attach(model) is model
    handles = model._t8_generation_cancellation_handles
    assert cancellation.attach(model) is model
    assert model._t8_generation_cancellation_handles is handles


def test_request_cancel_events_are_isolated_between_generation_threads():
    cancellation = GenerationCancellation()
    first_cancel = threading.Event()
    second_cancel = threading.Event()
    first_ready = threading.Event()
    second_done = threading.Event()
    outcomes = []

    def first_generation():
        cancellation.begin(first_cancel)
        first_ready.set()
        second_done.wait(timeout=2)
        first_cancel.set()
        try:
            cancellation.checkpoint()
        except GenerationStopped:
            outcomes.append("first-stopped")

    def second_generation():
        first_ready.wait(timeout=2)
        cancellation.begin(second_cancel)
        cancellation.checkpoint()
        outcomes.append("second-running")
        second_done.set()

    first = threading.Thread(target=first_generation)
    second = threading.Thread(target=second_generation)
    first.start()
    second.start()
    first.join(timeout=3)
    second.join(timeout=3)
    assert sorted(outcomes) == ["first-stopped", "second-running"]


def test_global_stop_epoch_does_not_cancel_generation_started_after_stop():
    cancellation = GenerationCancellation()
    cancellation.begin()
    cancellation.stop()
    with pytest.raises(GenerationStopped):
        cancellation.checkpoint()
    cancellation.begin()
    cancellation.checkpoint()
