from __future__ import annotations

import pytest

from desktop_tasks import create_task, load_task, set_task_status, task_choices, update_task_line


TASK_ID = "dialogue_20260824_120000_abcdef12"


def test_task_manifest_survives_restart_and_tracks_each_line(tmp_path):
    task = create_task(
        tmp_path,
        TASK_ID,
        script_type="batch",
        script="旁白|测试|ZH|1.0",
        settings={"seed": 7},
        line_count=2,
    )
    update_task_line(
        tmp_path,
        task,
        1,
        status="completed",
        file=str(tmp_path / TASK_ID / "0001_旁白.wav"),
        report={"duration_ms": 1000},
    )
    set_task_status(tmp_path, task, "failed", error="second line failed")

    restored = load_task(tmp_path, TASK_ID)
    assert restored["status"] == "failed"
    assert restored["lines"]["1"]["status"] == "completed"
    assert task_choices(tmp_path)[0][1] == TASK_ID


def test_task_ids_cannot_escape_output_directory(tmp_path):
    with pytest.raises(ValueError, match="格式无效"):
        load_task(tmp_path, "../outside")
