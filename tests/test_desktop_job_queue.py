from __future__ import annotations

import pytest

from desktop_job_queue import DesktopJobQueue


def test_queue_persists_orders_and_retries_jobs(tmp_path):
    path = tmp_path / "task_queue.json"
    queue = DesktopJobQueue(path)
    first = queue.enqueue("single", {"inputs": ["voice.wav", "你好"]}, "你好")
    second = queue.enqueue("srt", {"inputs": ["srt", "1\n..."]}, "字幕 1 条")

    assert [job["job_id"] for job in queue.pending()] == [first["job_id"], second["job_id"]]
    queue.update(first["job_id"], "failed", error="boom")
    queue.retry(first["job_id"])
    assert queue.get(first["job_id"])["status"] == "pending"
    assert DesktopJobQueue(path).get(second["job_id"])["kind"] == "srt"


def test_queue_recovers_interrupted_running_job(tmp_path):
    path = tmp_path / "task_queue.json"
    queue = DesktopJobQueue(path)
    job = queue.enqueue("dialogue", {"inputs": []}, "两条台词")
    queue.update(job["job_id"], "running")

    recovered = DesktopJobQueue(path).get(job["job_id"])
    assert recovered["status"] == "pending"
    assert "恢复" in recovered["error"]


def test_queue_rejects_corrupt_or_oversized_payload(tmp_path):
    path = tmp_path / "task_queue.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="格式无效"):
        DesktopJobQueue(path).jobs()

    path.unlink()
    queue = DesktopJobQueue(path)
    with pytest.raises(ValueError, match="2 MiB"):
        queue.enqueue("single", {"text": "字" * (2 * 1024 * 1024)}, "large")
    assert queue.jobs() == []
