from __future__ import annotations

from pathlib import Path

import pytest

from audiocpp_backend import build_audiocpp_command, run_audiocpp


def _paths(tmp_path: Path):
    executable = tmp_path / "audiocpp_cli.exe"
    model = tmp_path / "IndexTTS2.5-GGUF"
    speaker = tmp_path / "speaker.wav"
    output = tmp_path / "output.wav"
    executable.write_bytes(b"binary")
    model.mkdir()
    speaker.write_bytes(b"wave")
    return executable, model, speaker, output


def test_command_uses_index_25_language_speed_and_memory_options(tmp_path):
    executable, model, speaker, output = _paths(tmp_path)
    command = build_audiocpp_command(
        executable,
        model,
        speaker,
        output,
        "今天测试。",
        "ZH",
        backend="cuda",
        duration_factor=0.8,
        emotion_text="平静而坚定",
        emotion_alpha=0.6,
    )

    assert command[:5] == [str(executable), "--task", "clon", "--family", "index_tts2"]
    assert "language=zh" in command
    assert "duration_factor=0.8" in command
    assert "index_tts2.mem_saver=true" in command
    assert "use_emotion_text=true" in command
    assert "emotion_alpha=0.6" in command


def test_command_rejects_unknown_backend(tmp_path):
    executable, model, speaker, output = _paths(tmp_path)
    with pytest.raises(ValueError, match="未知 audio.cpp 后端"):
        build_audiocpp_command(
            executable, model, speaker, output, "test", "EN", backend="unknown"
        )


def test_runner_never_uses_shell_and_requires_created_output(tmp_path, monkeypatch):
    executable, model, speaker, output = _paths(tmp_path)
    seen = {}

    def fake_run(command, **kwargs):
        seen.update(kwargs)
        output.write_bytes(b"RIFF")
        return type("Completed", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr("audiocpp_backend.subprocess.run", fake_run)
    report = run_audiocpp(executable, model, speaker, output, "test", "EN")

    assert seen["shell"] is False
    assert report["experimental"] is True
