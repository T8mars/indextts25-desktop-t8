"""Persistent named presets for the T8star-Aix desktop WebUI."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


PRESET_SCHEMA_VERSION = 1


def _preset_root(data_dir: Path) -> Path:
    return Path(data_dir) / "presets"


def _preset_id(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:24]


def _manifest_path(data_dir: Path, name: str) -> Path:
    return _preset_root(data_dir) / f"{_preset_id(name)}.json"


def _audio_path(data_dir: Path, name: str, kind: str, suffix: str) -> Path:
    suffix = suffix.lower() if suffix else ".wav"
    if not suffix.startswith(".") or len(suffix) > 10:
        suffix = ".wav"
    return _preset_root(data_dir) / f"{_preset_id(name)}-{kind}{suffix}"


def _read_manifest(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("name"), str):
        return None
    return data


def list_presets(data_dir: Path) -> list[str]:
    root = _preset_root(data_dir)
    if not root.is_dir():
        return []
    names = []
    for path in root.glob("*.json"):
        data = _read_manifest(path)
        if data:
            names.append(data["name"])
    return sorted(set(names), key=str.casefold)


def save_preset(
    data_dir: Path,
    name: str,
    settings: dict,
    *,
    prompt_audio: str | None = None,
    emotion_audio: str | None = None,
) -> dict:
    name = str(name or "").strip()
    if not name:
        raise ValueError("预设名称不能为空。")
    if len(name) > 80:
        raise ValueError("预设名称不能超过 80 个字符。")
    root = _preset_root(data_dir)
    root.mkdir(parents=True, exist_ok=True)

    preset_key = _preset_id(name)
    for stale in root.glob(f"{preset_key}-prompt.*"):
        stale.unlink(missing_ok=True)
    for stale in root.glob(f"{preset_key}-emotion.*"):
        stale.unlink(missing_ok=True)

    stored_audio = {}
    for kind, source_value in (("prompt", prompt_audio), ("emotion", emotion_audio)):
        if not source_value:
            continue
        source = Path(source_value)
        if not source.is_file():
            raise FileNotFoundError(f"预设引用音频不存在：{source}")
        target = _audio_path(data_dir, name, kind, source.suffix)
        shutil.copy2(source, target)
        stored_audio[kind] = target.name

    manifest = {
        "schemaVersion": PRESET_SCHEMA_VERSION,
        "name": name,
        "settings": dict(settings),
        "audio": stored_audio,
    }
    _manifest_path(data_dir, name).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def load_preset(data_dir: Path, name: str) -> dict | None:
    name = str(name or "").strip()
    if not name:
        return None
    manifest = _read_manifest(_manifest_path(data_dir, name))
    if not manifest or manifest.get("name") != name:
        return None
    root = _preset_root(data_dir)
    resolved_audio = {}
    for kind, filename in (manifest.get("audio") or {}).items():
        candidate = root / Path(str(filename)).name
        resolved_audio[kind] = str(candidate) if candidate.is_file() else None
    manifest["audio"] = resolved_audio
    return manifest


def delete_preset(data_dir: Path, name: str) -> bool:
    name = str(name or "").strip()
    if not name:
        return False
    root = _preset_root(data_dir)
    manifest_path = _manifest_path(data_dir, name)
    existed = manifest_path.is_file()
    manifest_path.unlink(missing_ok=True)
    preset_key = _preset_id(name)
    for kind in ("prompt", "emotion"):
        for path in root.glob(f"{preset_key}-{kind}.*"):
            path.unlink(missing_ok=True)
    return existed
