"""Persistent, user-writable character voice library used by the desktop bundle."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


_SAFE_NAME = re.compile(r"[^\w\u3400-\u9fff.-]+", re.UNICODE)
_EMOTION_MODES = {"speaker", "reference_audio", "vector", "text"}
_EMPTY_EMOTION_VECTOR = (0.0,) * 8


def _emotion_vector(values) -> tuple[float, ...]:
    if values is None:
        return _EMPTY_EMOTION_VECTOR
    result = tuple(max(0.0, min(1.0, float(value))) for value in values)
    if len(result) != 8:
        raise ValueError("八维情感向量必须正好包含 8 个数值。")
    total = sum(result)
    if total > 0.8:
        scale = 0.8 / total
        result = tuple(value * scale for value in result)
    return result


@dataclass(frozen=True, slots=True)
class VoiceProfile:
    profile_id: str
    name: str
    audio_path: str
    language: str = "ZH"
    emotion_mode: str = "speaker"
    emotion_text: str = ""
    emotion_strength: float = 1.0
    emotion_audio_path: str = ""
    emotion_vector: tuple[float, ...] = _EMPTY_EMOTION_VECTOR
    emotion_use_random: bool = False
    pronunciation_dictionary: str = ""

    def __post_init__(self) -> None:
        mode = str(self.emotion_mode).strip()
        if mode not in _EMOTION_MODES:
            raise ValueError(f"不支持的角色情感模式：{mode}")
        object.__setattr__(self, "emotion_mode", mode)
        object.__setattr__(self, "emotion_vector", _emotion_vector(self.emotion_vector))
        object.__setattr__(
            self,
            "emotion_strength",
            max(0.0, min(1.0, float(self.emotion_strength))),
        )
        object.__setattr__(self, "emotion_use_random", bool(self.emotion_use_random))

    def to_dict(self) -> dict:
        return asdict(self)


class VoiceLibrary:
    def __init__(self, data_dir: str | Path):
        self.root = Path(data_dir).expanduser().resolve() / "voices"
        self.audio_dir = self.root / "audio"
        self.manifest_path = self.root / "library.json"

    def _load(self) -> dict[str, dict]:
        if not self.manifest_path.exists():
            return {}
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _save(self, payload: dict[str, dict]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.manifest_path)

    def list(self) -> list[VoiceProfile]:
        result: list[VoiceProfile] = []
        for value in self._load().values():
            try:
                profile = VoiceProfile(**value)
            except (TypeError, ValueError):
                continue
            if Path(profile.audio_path).exists():
                result.append(profile)
        return sorted(result, key=lambda item: item.name.casefold())

    def get(self, name_or_id: str) -> VoiceProfile:
        needle = str(name_or_id).strip().casefold()
        for profile in self.list():
            if profile.profile_id.casefold() == needle or profile.name.casefold() == needle:
                return profile
        raise KeyError(f"角色音色不存在：{name_or_id}")

    def save(
        self,
        name: str,
        source_audio: str | Path,
        language: str = "ZH",
        *,
        emotion_mode: str = "speaker",
        emotion_text: str = "",
        emotion_strength: float = 1.0,
        emotion_audio: str | Path | None = None,
        emotion_vector=None,
        emotion_use_random: bool = False,
        pronunciation_dictionary: str = "",
        replace_name_or_id: str | None = None,
    ) -> VoiceProfile:
        clean_name = str(name).strip()
        source = Path(source_audio).expanduser().resolve()
        if not clean_name:
            raise ValueError("角色名称不能为空。")
        if not source.is_file():
            raise FileNotFoundError(f"参考音频不存在：{source}")
        language = str(language).upper()
        if language not in {"ZH", "EN", "JA", "ES", "AR"}:
            raise ValueError(f"不支持的语言：{language}")
        emotion_mode = str(emotion_mode).strip()
        if emotion_mode not in _EMOTION_MODES:
            raise ValueError(f"不支持的角色情感模式：{emotion_mode}")
        normalized_vector = _emotion_vector(emotion_vector)
        emotion_source: Path | None = None
        if emotion_mode == "reference_audio":
            if not emotion_audio:
                raise ValueError("情感参考音频模式需要提供情感参考音频。")
            emotion_source = Path(emotion_audio).expanduser().resolve()
            if not emotion_source.is_file():
                raise FileNotFoundError(f"情感参考音频不存在：{emotion_source}")
        payload = self._load()
        replace_needle = str(replace_name_or_id or "").strip().casefold()
        replace_id = next(
            (
                item_id
                for item_id, value in payload.items()
                if replace_needle
                and (
                    item_id.casefold() == replace_needle
                    or str(value.get("name", "")).casefold() == replace_needle
                )
            ),
            None,
        )
        if replace_needle and replace_id is None:
            raise KeyError(f"要修改的角色音色不存在：{replace_name_or_id}")
        name_match_id = next(
            (
                item_id
                for item_id, value in payload.items()
                if str(value.get("name", "")).casefold() == clean_name.casefold()
            ),
            None,
        )
        if replace_id and name_match_id and replace_id != name_match_id:
            raise ValueError(f"角色名称已存在：{clean_name}")
        profile_id = replace_id or name_match_id or hashlib.sha256(
            clean_name.casefold().encode("utf-8")
        ).hexdigest()[:16]
        safe_stem = _SAFE_NAME.sub("_", clean_name).strip("._")[:48] or profile_id
        suffix = source.suffix.lower() or ".wav"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        target = self.audio_dir / f"{safe_stem}-{profile_id}{suffix}"
        if source != target:
            shutil.copy2(source, target)
        emotion_target = ""
        if emotion_source is not None:
            emotion_suffix = emotion_source.suffix.lower() or ".wav"
            emotion_target_path = self.audio_dir / f"{safe_stem}-{profile_id}-emotion{emotion_suffix}"
            if emotion_source != emotion_target_path:
                shutil.copy2(emotion_source, emotion_target_path)
            emotion_target = str(emotion_target_path)
        profile = VoiceProfile(
            profile_id=profile_id,
            name=clean_name,
            audio_path=str(target),
            language=language,
            emotion_mode=emotion_mode,
            emotion_text=str(emotion_text),
            emotion_strength=max(0.0, min(1.0, float(emotion_strength))),
            emotion_audio_path=emotion_target,
            emotion_vector=normalized_vector,
            emotion_use_random=bool(emotion_use_random),
            pronunciation_dictionary=str(pronunciation_dictionary),
        )
        previous = payload.get(profile_id)
        payload[profile_id] = profile.to_dict()
        self._save(payload)
        if previous:
            current_paths = {str(target), emotion_target}
            for field_name in ("audio_path", "emotion_audio_path"):
                old_value = str(previous.get(field_name, ""))
                if not old_value or old_value in current_paths:
                    continue
                old_path = Path(old_value)
                if old_path.is_file() and self.audio_dir in old_path.resolve().parents:
                    old_path.unlink(missing_ok=True)
        return profile

    def delete(self, name_or_id: str) -> VoiceProfile:
        profile = self.get(name_or_id)
        payload = self._load()
        payload.pop(profile.profile_id, None)
        self._save(payload)
        for value in (profile.audio_path, profile.emotion_audio_path):
            audio = Path(value) if value else None
            if audio and audio.is_file() and self.audio_dir in audio.resolve().parents:
                audio.unlink(missing_ok=True)
        return profile


__all__ = ["VoiceLibrary", "VoiceProfile"]
