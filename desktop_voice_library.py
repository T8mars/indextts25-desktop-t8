"""Persistent, user-writable character voice library used by the desktop bundle."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


_SAFE_NAME = re.compile(r"[^\w\u3400-\u9fff.-]+", re.UNICODE)
_EMOTION_MODES = {"speaker", "reference_audio", "vector", "text"}
_EMPTY_EMOTION_VECTOR = (0.0,) * 8
VOICE_BUNDLE_SCHEMA_VERSION = 1
VOICE_BUNDLE_SUFFIX = ".t8voice.zip"
MAX_VOICE_BUNDLE_MEMBER_BYTES = 2 * 1024**3
MAX_VOICE_BUNDLE_TOTAL_BYTES = 4 * 1024**3
MAX_VOICE_BUNDLE_MEMBERS = 1024
MAX_VOICE_PROFILES = 256


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


def _tags(values: Iterable[str] | str | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = re.split(r"[,，;；\n]+", values)
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        tag = str(value).strip()
        key = tag.casefold()
        if tag and key not in seen:
            seen.add(key)
            result.append(tag[:32])
    return tuple(result[:32])


def _safe_archive_name(value: str) -> str:
    cleaned = _SAFE_NAME.sub("_", str(value)).strip("._")
    return cleaned[:80] or "voice"


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
    tags: tuple[str, ...] = ()
    favorite: bool = False
    notes: str = ""
    quality: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        mode = str(self.emotion_mode).strip()
        if mode not in _EMOTION_MODES:
            raise ValueError(f"不支持的角色情感模式：{mode}")
        object.__setattr__(self, "emotion_mode", mode)
        object.__setattr__(self, "emotion_vector", _emotion_vector(self.emotion_vector))
        object.__setattr__(self, "tags", _tags(self.tags))
        object.__setattr__(self, "favorite", bool(self.favorite))
        object.__setattr__(self, "notes", str(self.notes or "").strip())
        object.__setattr__(
            self,
            "quality",
            dict(self.quality) if isinstance(self.quality, dict) else {},
        )
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

    def search(
        self,
        query: str = "",
        *,
        tags: Iterable[str] | str | None = None,
        favorites_only: bool = False,
    ) -> list[VoiceProfile]:
        needle = str(query or "").strip().casefold()
        required_tags = {item.casefold() for item in _tags(tags)}
        result: list[VoiceProfile] = []
        for profile in self.list():
            profile_tags = {item.casefold() for item in profile.tags}
            searchable = "\n".join(
                (profile.name, profile.language, profile.notes, *profile.tags)
            ).casefold()
            if needle and needle not in searchable:
                continue
            if favorites_only and not profile.favorite:
                continue
            if required_tags and not required_tags.issubset(profile_tags):
                continue
            result.append(profile)
        return result

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
        tags: Iterable[str] | str | None = None,
        favorite: bool = False,
        notes: str = "",
        quality: dict[str, Any] | None = None,
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
        previous = payload.get(profile_id)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
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
            tags=_tags(tags),
            favorite=bool(favorite),
            notes=str(notes or "").strip(),
            quality=dict(quality or {}),
            created_at=str((previous or {}).get("created_at") or now),
            updated_at=now,
        )
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

    def update_metadata(
        self,
        name_or_id: str,
        *,
        tags: Iterable[str] | str | None = None,
        favorite: bool | None = None,
        notes: str | None = None,
        quality: dict[str, Any] | None = None,
    ) -> VoiceProfile:
        current = self.get(name_or_id)
        payload = self._load()
        value = dict(payload[current.profile_id])
        if tags is not None:
            value["tags"] = _tags(tags)
        if favorite is not None:
            value["favorite"] = bool(favorite)
        if notes is not None:
            value["notes"] = str(notes).strip()
        if quality is not None:
            value["quality"] = dict(quality)
        value["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        profile = VoiceProfile(**value)
        payload[current.profile_id] = profile.to_dict()
        self._save(payload)
        return profile

    def set_favorite(self, name_or_id: str, favorite: bool) -> VoiceProfile:
        return self.update_metadata(name_or_id, favorite=favorite)

    def export_bundle(
        self,
        destination: str | Path,
        names: Iterable[str] | None = None,
    ) -> Path:
        selected = list(self.list())
        if names is not None:
            needles = {str(item).strip().casefold() for item in names if str(item).strip()}
            selected = [
                item
                for item in selected
                if item.name.casefold() in needles or item.profile_id.casefold() in needles
            ]
        if not selected:
            raise ValueError("没有可导出的角色音色。")
        target = Path(destination).expanduser().resolve()
        if not str(target).lower().endswith(VOICE_BUNDLE_SUFFIX):
            target = target.with_name(target.name + VOICE_BUNDLE_SUFFIX)
        target.parent.mkdir(parents=True, exist_ok=True)
        manifest_profiles: list[dict[str, Any]] = []
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for profile in selected:
                value = profile.to_dict()
                for field_name, suffix in (
                    ("audio_path", "speaker"),
                    ("emotion_audio_path", "emotion"),
                ):
                    source_value = str(value.get(field_name) or "")
                    if not source_value:
                        value[field_name] = ""
                        continue
                    source = Path(source_value).resolve()
                    if not source.is_file():
                        raise FileNotFoundError(f"音色包文件不存在：{source}")
                    archive_name = (
                        f"audio/{profile.profile_id}-{suffix}-"
                        f"{_safe_archive_name(source.name)}"
                    )
                    archive.write(source, archive_name)
                    value[field_name] = archive_name
                manifest_profiles.append(value)
            archive.writestr(
                "manifest.json",
                json.dumps(
                    {
                        "schemaVersion": VOICE_BUNDLE_SCHEMA_VERSION,
                        "format": "T8star-Aix IndexTTS 2.5 Voice Bundle",
                        "createdAt": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "profiles": manifest_profiles,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        return target

    def import_bundle(
        self,
        source: str | Path,
        *,
        conflict: str = "rename",
    ) -> list[VoiceProfile]:
        archive_path = Path(source).expanduser().resolve()
        if conflict not in {"rename", "replace", "skip"}:
            raise ValueError("音色冲突策略必须是 rename、replace 或 skip。")
        if not archive_path.is_file():
            raise FileNotFoundError(f"音色包不存在：{archive_path}")
        imported: list[VoiceProfile] = []
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            if len(members) > MAX_VOICE_BUNDLE_MEMBERS:
                raise ValueError("音色包文件数量超过安全上限。")
            if any(member.file_size > MAX_VOICE_BUNDLE_MEMBER_BYTES for member in members):
                raise ValueError("音色包包含超过 2 GiB 的单个文件。")
            if sum(member.file_size for member in members) > MAX_VOICE_BUNDLE_TOTAL_BYTES:
                raise ValueError("音色包解压后总大小超过 4 GiB。")
            for member in members:
                candidate = Path(member.filename.replace("\\", "/"))
                if candidate.is_absolute() or ".." in candidate.parts:
                    raise ValueError(f"音色包包含不安全路径：{member.filename}")
            try:
                manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
            except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("音色包 manifest.json 缺失或损坏。") from exc
            if manifest.get("schemaVersion") != VOICE_BUNDLE_SCHEMA_VERSION:
                raise ValueError("不支持的音色包版本。")
            profiles = manifest.get("profiles")
            if not isinstance(profiles, list) or not profiles:
                raise ValueError("音色包中没有角色。")
            if len(profiles) > MAX_VOICE_PROFILES:
                raise ValueError("单个音色包最多包含 256 个角色。")
            with tempfile.TemporaryDirectory(prefix="t8_voice_import_") as temporary:
                temporary_root = Path(temporary).resolve()
                archive.extractall(temporary_root)
                for raw in profiles:
                    if not isinstance(raw, dict):
                        raise ValueError("音色包角色数据格式无效。")
                    name = str(raw.get("name") or "").strip()
                    if not name:
                        raise ValueError("音色包包含空角色名称。")
                    existing = None
                    try:
                        existing = self.get(name)
                    except KeyError:
                        pass
                    if existing and conflict == "skip":
                        continue
                    if existing and conflict == "rename":
                        base = name
                        index = 2
                        while True:
                            candidate_name = f"{base}（导入 {index}）"
                            try:
                                self.get(candidate_name)
                            except KeyError:
                                name = candidate_name
                                break
                            index += 1

                    def member_path(field_name: str, required: bool) -> Path | None:
                        relative = str(raw.get(field_name) or "")
                        if not relative:
                            if required:
                                raise ValueError(f"角色“{name}”缺少音色参考音频。")
                            return None
                        resolved = (temporary_root / relative).resolve()
                        if temporary_root not in resolved.parents or not resolved.is_file():
                            raise ValueError(f"角色“{name}”的音频路径无效。")
                        return resolved

                    speaker = member_path("audio_path", True)
                    emotion = member_path("emotion_audio_path", False)
                    imported.append(
                        self.save(
                            name,
                            speaker,
                            raw.get("language", "ZH"),
                            emotion_mode=raw.get("emotion_mode", "speaker"),
                            emotion_text=raw.get("emotion_text", ""),
                            emotion_strength=raw.get("emotion_strength", 1.0),
                            emotion_audio=emotion,
                            emotion_vector=raw.get("emotion_vector"),
                            emotion_use_random=raw.get("emotion_use_random", False),
                            pronunciation_dictionary=raw.get("pronunciation_dictionary", ""),
                            tags=raw.get("tags"),
                            favorite=raw.get("favorite", False),
                            notes=raw.get("notes", ""),
                            quality=raw.get("quality"),
                            replace_name_or_id=(
                                existing.profile_id
                                if existing is not None and conflict == "replace"
                                else None
                            ),
                        )
                    )
        return imported

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


__all__ = [
    "VOICE_BUNDLE_SCHEMA_VERSION",
    "VOICE_BUNDLE_SUFFIX",
    "VoiceLibrary",
    "VoiceProfile",
]
