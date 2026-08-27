from __future__ import annotations

from desktop_model_lifecycle import DesktopModelLifecycle


def test_lifecycle_lazily_reloads_after_manual_release():
    created = []
    lifecycle = DesktopModelLifecycle(object(), lambda: created.append(object()) or created[-1])

    report = lifecycle.release()
    assert report["released"] is True
    assert lifecycle.status()["loaded"] is False
    loaded = lifecycle.get()

    assert loaded is created[0]
    assert lifecycle.status()["loaded"] is True


def test_lifecycle_recycles_at_configured_generation_threshold():
    lifecycle = DesktopModelLifecycle(object(), object)

    first = lifecycle.after_generation(recycle_after_generations=2)
    second = lifecycle.after_generation(recycle_after_generations=2)

    assert first["loaded"] is True
    assert second["released"] is True
    assert lifecycle.status()["loaded"] is False


def test_lifecycle_idle_timer_releases_model():
    import time

    lifecycle = DesktopModelLifecycle(object(), object)
    lifecycle.schedule_idle(0.01)
    time.sleep(0.05)

    assert lifecycle.status()["loaded"] is False
