"""Isolated optional connector for the official audio.cpp CLI."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Sequence


AUDIOCPP_REPOSITORY = "https://github.com/0xShug0/audio.cpp"
AUDIOCPP_MODEL_REPOSITORY = "https://huggingface.co/audio-cpp/audio.cpp-gguf"
AUDIOCPP_LANGUAGES = {"AUTO": "auto", "ZH": "zh", "EN": "en", "JA": "ja", "ES": "es", "AR": "ar"}


def _required_path(value: str | Path, label: str, *, file: bool | None = None) -> Path:
    path = Path(str(value or "").strip()).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"{label}不存在：{path}")
    if file is True and not path.is_file():
        raise ValueError(f"{label}必须是文件：{path}")
    if file is False and not path.is_dir():
        raise ValueError(f"{label}必须是目录：{path}")
    return path


def probe_audiocpp(executable: str | Path, *, timeout: float = 15.0) -> dict[str, Any]:
    try:
        binary = _required_path(executable, "audio.cpp 可执行文件", file=True)
        completed = subprocess.run(
            [str(binary), "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(1.0, float(timeout)),
            check=False,
            shell=False,
        )
        output = (completed.stdout + "\n" + completed.stderr).strip()
        compatible = "--family" in output and "--voice-ref" in output
        return {
            "available": completed.returncode == 0 and compatible,
            "executable": str(binary),
            "returncode": completed.returncode,
            "compatible_cli": compatible,
            "summary": output[:2000],
            "repository": AUDIOCPP_REPOSITORY,
        }
    except Exception as exc:
        return {
            "available": False,
            "executable": str(executable or ""),
            "error": f"{type(exc).__name__}: {exc}",
            "repository": AUDIOCPP_REPOSITORY,
        }


def build_audiocpp_command(
    executable: str | Path,
    model_dir: str | Path,
    speaker_wav: str | Path,
    output_wav: str | Path,
    text: str,
    language: str,
    *,
    backend: str = "cuda",
    duration_factor: float = 1.0,
    interval_silence_ms: int = 200,
    memory_saver: bool = True,
    emotion_text: str = "",
    emotion_audio: str | Path | None = None,
    emotion_vector: Sequence[float] | None = None,
    emotion_alpha: float = 1.0,
) -> list[str]:
    binary = _required_path(executable, "audio.cpp 可执行文件", file=True)
    model = _required_path(model_dir, "audio.cpp IndexTTS2.5 GGUF 模型")
    speaker = _required_path(speaker_wav, "音色参考音频", file=True)
    output = Path(output_wav).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source = str(text or "").strip()
    if not source:
        raise ValueError("待合成文本不能为空。")
    language_code = AUDIOCPP_LANGUAGES.get(str(language).upper())
    if language_code is None:
        raise ValueError(f"audio.cpp 不支持该语言：{language}")
    if backend not in {"cuda", "cpu", "vulkan", "hip", "metal"}:
        raise ValueError(f"未知 audio.cpp 后端：{backend}")
    if not 0.5 <= float(duration_factor) <= 2.0:
        raise ValueError("时长系数必须在 0.5–2.0。")
    command = [
        str(binary),
        "--task",
        "clon",
        "--family",
        "index_tts2",
        "--model",
        str(model),
        "--backend",
        backend,
        "--text",
        source,
        "--voice-ref",
        str(speaker),
        "--out",
        str(output),
        "--request-option",
        f"language={language_code}",
        "--request-option",
        f"duration_factor={float(duration_factor):.6g}",
        "--request-option",
        f"interval_silence_ms={max(0, int(interval_silence_ms))}",
        "--session-option",
        f"index_tts2.mem_saver={'true' if memory_saver else 'false'}",
    ]
    if str(emotion_text or "").strip():
        command.extend(
            [
                "--emotion",
                str(emotion_text).strip(),
                "--request-option",
                "use_emotion_text=true",
            ]
        )
    if emotion_audio:
        command.extend(
            ["--audio", str(_required_path(emotion_audio, "情感参考音频", file=True))]
        )
    if emotion_vector is not None:
        values = [max(0.0, float(item)) for item in emotion_vector]
        if len(values) != 8:
            raise ValueError("audio.cpp 情感向量必须正好包含 8 个数值。")
        command.extend(
            ["--request-option", "emotion_vector=" + ",".join(f"{item:.6g}" for item in values)]
        )
    if emotion_text or emotion_audio or emotion_vector is not None:
        command.extend(
            ["--request-option", f"emotion_alpha={max(0.0, min(1.0, float(emotion_alpha))):.6g}"]
        )
    return command


def run_audiocpp(*args, timeout: float = 3600.0, **kwargs) -> dict[str, Any]:
    command = build_audiocpp_command(*args, **kwargs)
    output = Path(command[command.index("--out") + 1])
    safe_command = list(command)
    text_value_index = safe_command.index("--text") + 1
    safe_command[text_value_index] = "<text>"
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=max(30.0, float(timeout)),
        check=False,
        shell=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0 or not output.is_file():
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"audio.cpp 推理失败（exit={completed.returncode}）：{detail[-3000:]}"
        )
    return {
        "output_path": str(output),
        "elapsed_seconds": round(elapsed, 3),
        "backend": kwargs.get("backend", "cuda"),
        "family": "index_tts2",
        "experimental": True,
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-2000:],
        "command": safe_command,
        "repository": AUDIOCPP_REPOSITORY,
        "model_repository": AUDIOCPP_MODEL_REPOSITORY,
    }


__all__ = [
    "AUDIOCPP_LANGUAGES",
    "AUDIOCPP_MODEL_REPOSITORY",
    "AUDIOCPP_REPOSITORY",
    "build_audiocpp_command",
    "probe_audiocpp",
    "run_audiocpp",
]
