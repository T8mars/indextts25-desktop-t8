import json
import zipfile
from pathlib import Path

import pytest

from desktop_project_bundle import export_project, import_project
from desktop_tasks import create_task, load_task, set_task_status, update_task_line
from desktop_voice_library import VoiceLibrary


def _completed_project(tmp_path: Path):
    output = tmp_path / "outputs"
    data = tmp_path / "data"
    output.mkdir()
    voice_audio = tmp_path / "voice.wav"
    voice_audio.write_bytes(b"voice")
    voices = VoiceLibrary(data)
    voices.save("旁白", voice_audio, tags="旁白", quality={"score": 90})
    task_id = "dialogue_20260829_120000_1234abcd"
    task = create_task(
        output,
        task_id,
        script_type="batch",
        script="旁白|你好|ZH|1.0",
        settings={
            "default_role": "旁白",
            "default_language": "ZH",
            "timeline_rows": [[1, "旁白", "ZH", 0, 1000, 1.0, "你好", ""]],
        },
        line_count=1,
    )
    session = output / task_id
    clip = session / "0001_旁白.wav"
    clip.write_bytes(b"clip")
    report = session / "report.json"
    report.write_text(json.dumps({"lines": [{"role": "旁白", "file": str(clip)}]}), encoding="utf-8")
    subtitle = session / "rewritten.srt"
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\n你好\n", encoding="utf-8")
    combined = output / f"{task_id}.wav"
    combined.write_bytes(b"combined")
    task = update_task_line(
        output,
        task,
        1,
        status="completed",
        file=str(clip),
        report={"role": "旁白", "file": str(clip)},
    )
    task.update(
        combined_file=str(combined),
        report_file=str(report),
        rewritten_srt_file=str(subtitle),
    )
    set_task_status(output, task, "completed")
    return output, voices, task_id


def test_project_bundle_round_trip_restores_task_audio_and_voices(tmp_path: Path):
    output, voices, task_id = _completed_project(tmp_path)
    project = export_project(output, task_id, voices, tmp_path / "episode")
    assert project.name.endswith(".indextts-project.zip")

    restored_output = tmp_path / "restored-output"
    restored_voices = VoiceLibrary(tmp_path / "restored-data")
    result = import_project(project, restored_output, restored_voices)
    assert result["task_id"] != task_id
    assert result["imported_voices"] == ["旁白"]
    task = load_task(restored_output, result["task_id"])
    assert task["script"] == "旁白|你好|ZH|1.0"
    assert Path(task["lines"]["1"]["file"]).read_bytes() == b"clip"
    assert Path(task["combined_file"]).read_bytes() == b"combined"
    assert Path(task["report_file"]).is_file()
    assert restored_voices.get("旁白").quality["score"] == 90


def test_project_bundle_detects_tampered_asset(tmp_path: Path):
    output, voices, task_id = _completed_project(tmp_path)
    project = export_project(output, task_id, voices, tmp_path / "episode")
    tampered = tmp_path / "tampered.indextts-project.zip"
    with zipfile.ZipFile(project) as source, zipfile.ZipFile(tampered, "w") as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename.endswith("0001_旁白.wav"):
                data = b"changed"
            target.writestr(item, data)
    with pytest.raises(ValueError, match="大小不匹配|校验失败"):
        import_project(tampered, tmp_path / "other", VoiceLibrary(tmp_path / "voices2"))


def test_project_bundle_renames_conflicting_voice_and_task_roles(tmp_path: Path):
    output, voices, task_id = _completed_project(tmp_path)
    project = export_project(output, task_id, voices, tmp_path / "episode")
    restored_voices = VoiceLibrary(tmp_path / "restored-data")
    existing_audio = tmp_path / "existing.wav"
    existing_audio.write_bytes(b"existing")
    restored_voices.save("旁白", existing_audio)

    restored_output = tmp_path / "restored-output"
    result = import_project(
        project,
        restored_output,
        restored_voices,
        voice_conflict="rename",
    )
    task = load_task(restored_output, result["task_id"])
    assert result["imported_voices"] == ["旁白（导入 2）"]
    assert task["script"].startswith("旁白（导入 2）|")
    assert task["settings"]["default_role"] == "旁白（导入 2）"
    assert task["settings"]["timeline_rows"][0][1] == "旁白（导入 2）"
    assert task["lines"]["1"]["report"]["role"] == "旁白（导入 2）"
    assert Path(restored_voices.get("旁白（导入 2）").audio_path).read_bytes() == b"voice"


def test_project_bundle_rejects_files_missing_from_manifest(tmp_path: Path):
    output, voices, task_id = _completed_project(tmp_path)
    project = export_project(output, task_id, voices, tmp_path / "episode")
    malicious = tmp_path / "unlisted.indextts-project.zip"
    with zipfile.ZipFile(project) as source, zipfile.ZipFile(malicious, "w") as target:
        for item in source.infolist():
            target.writestr(item, source.read(item.filename))
        target.writestr("task/unlisted.bin", b"not-in-project-manifest")

    with pytest.raises(ValueError, match="未列入清单"):
        import_project(
            malicious,
            tmp_path / "restored-output",
            VoiceLibrary(tmp_path / "restored-data"),
        )


def test_project_bundle_rolls_back_task_and_voices_on_voice_import_failure(
    tmp_path: Path,
    monkeypatch,
):
    output, voices, task_id = _completed_project(tmp_path)
    project = export_project(output, task_id, voices, tmp_path / "episode")
    restored_output = tmp_path / "restored-output"
    restored_voices = VoiceLibrary(tmp_path / "restored-data")
    original_import = VoiceLibrary._import_bundle_in_place

    def import_then_fail(self, *args, **kwargs):
        original_import(self, *args, **kwargs)
        raise RuntimeError("simulated late failure")

    monkeypatch.setattr(VoiceLibrary, "_import_bundle_in_place", import_then_fail)

    with pytest.raises(RuntimeError, match="simulated late failure"):
        import_project(project, restored_output, restored_voices)

    assert restored_voices.list() == []
    assert not list(restored_output.glob("dialogue_*"))
    assert not list(restored_output.glob("dialogue_*.wav"))
