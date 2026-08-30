"""Persistent, user-writable character voice library used by the desktop bundle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
import unicodedata
import uuid
import wave
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from functools import wraps
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator


_SAFE_NAME = re.compile(r"[^\w\u3400-\u9fff.-]+", re.UNICODE)
_EMOTION_MODES = {"speaker", "reference_audio", "vector", "text"}
_EMPTY_EMOTION_VECTOR = (0.0,) * 8
VOICE_BUNDLE_SCHEMA_VERSION = 1
VOICE_BUNDLE_SUFFIX = ".t8voice.zip"
MAX_VOICE_BUNDLE_MEMBER_BYTES = 2 * 1024**3
MAX_VOICE_BUNDLE_TOTAL_BYTES = 4 * 1024**3
MAX_VOICE_BUNDLE_MEMBERS = 1024
MAX_VOICE_PROFILES = 256
MAX_VOICE_MANIFEST_BYTES = 4 * 1024**2
VOICE_DISK_RESERVE_BYTES = 512 * 1024**2
_VOICE_LIBRARY_LOCK = threading.RLock()


def _synchronized(method):
    @wraps(method)
    def wrapped(*args, **kwargs):
        with _VOICE_LIBRARY_LOCK:
            return method(*args, **kwargs)

    return wrapped


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


def safe_voice_file_stem(value: str, fallback: str = "voice") -> str:
    """Return a display-name-derived stem that cannot create path components."""

    cleaned = _SAFE_NAME.sub("_", str(value)).strip("._")
    while ".." in cleaned:
        cleaned = cleaned.replace("..", "_")
    return cleaned[:80] or str(fallback)


def _safe_member(value: str) -> PurePosixPath:
    candidate = PurePosixPath(str(value).replace("\\", "/"))
    if (
        candidate.is_absolute()
        or not candidate.parts
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError(f"音色包包含不安全路径：{value}")
    return candidate


def _portable_member_key(value: str) -> str:
    member = _safe_member(value)
    normalized_parts = []
    for part in member.parts:
        normalized = unicodedata.normalize("NFC", part).rstrip(" .").casefold()
        if not normalized:
            raise ValueError(f"音色包包含 Windows 不兼容路径：{value}")
        normalized_parts.append(normalized)
    return "/".join(normalized_parts)


def _validated_bundle_manifest(
    archive: zipfile.ZipFile,
    *,
    bundle_name: str,
) -> dict[str, Any]:
    members = archive.infolist()
    if len(members) > MAX_VOICE_BUNDLE_MEMBERS:
        raise ValueError("音色包文件数量超过安全上限。")
    if any(member.file_size > MAX_VOICE_BUNDLE_MEMBER_BYTES for member in members):
        raise ValueError("音色包包含超过 2 GiB 的单个文件。")
    if (
        sum(max(0, member.file_size) for member in members)
        > MAX_VOICE_BUNDLE_TOTAL_BYTES
    ):
        raise ValueError("音色包解压后总大小超过 4 GiB。")
    normalized_names = [
        _safe_member(member.filename).as_posix()
        for member in members
        if not member.is_dir()
    ]
    if len(normalized_names) != len(set(normalized_names)):
        raise ValueError("音色包包含重复文件名。")
    portable_names = [_portable_member_key(name) for name in normalized_names]
    if len(portable_names) != len(set(portable_names)):
        raise ValueError("音色包包含 Windows 下会互相覆盖的文件名。")
    try:
        manifest_info = archive.getinfo("manifest.json")
    except KeyError as exc:
        raise ValueError("音色包 manifest.json 缺失或损坏。") from exc
    if manifest_info.file_size > MAX_VOICE_MANIFEST_BYTES:
        raise ValueError("音色包 manifest.json 超过 4 MiB。")
    try:
        manifest = json.loads(archive.read(manifest_info).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("音色包 manifest.json 缺失或损坏。") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schemaVersion") != VOICE_BUNDLE_SCHEMA_VERSION
    ):
        raise ValueError("不支持的音色包版本。")
    profiles = manifest.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("音色包中没有角色。")
    if len(profiles) > MAX_VOICE_PROFILES:
        raise ValueError("单个音色包最多包含 256 个角色。")
    expected = {"manifest.json"}
    profile_names: set[str] = set()
    profile_ids: set[str] = set()
    for raw in profiles:
        if not isinstance(raw, dict):
            raise ValueError("音色包角色数据格式无效。")
        profile_name = str(raw.get("name") or "").strip()
        if not profile_name:
            raise ValueError("音色包包含空角色名称。")
        profile_id = str(raw.get("profile_id") or "").strip()
        name_key = unicodedata.normalize("NFC", profile_name).casefold()
        id_key = unicodedata.normalize("NFC", profile_id).casefold()
        if name_key in profile_names or (id_key and id_key in profile_ids):
            raise ValueError("音色包包含重复角色名称或角色 ID。")
        profile_names.add(name_key)
        if id_key:
            profile_ids.add(id_key)
        speaker = str(raw.get("audio_path") or "")
        if not speaker:
            raise ValueError("音色包角色缺少音色参考音频。")
        expected.add(_safe_member(speaker).as_posix())
        emotion = str(raw.get("emotion_audio_path") or "")
        if emotion:
            expected.add(_safe_member(emotion).as_posix())
    actual = set(normalized_names)
    if actual != expected:
        extra = sorted(actual - expected)
        missing = sorted(expected - actual)
        detail = []
        if extra:
            detail.append("未列入清单：" + "、".join(extra[:5]))
        if missing:
            detail.append("缺少：" + "、".join(missing[:5]))
        raise ValueError(f"音色包文件清单不一致（{'; '.join(detail)}）：{bundle_name}")
    return manifest


def _copy_or_link(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _write_portable_wav(source: Path, target: Path) -> None:
    """Decode and atomically normalize supported audio to mono 24 kHz PCM WAV."""

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        try:
            import av
        except ImportError as exc:
            raise RuntimeError("转换音色音频需要 PyAV；请修复整合包音频依赖。") from exc
        try:
            written_bytes = 0
            with (
                av.open(str(source)) as container,
                wave.open(str(temporary), "wb") as output,
            ):
                audio_streams = [
                    stream for stream in container.streams if stream.type == "audio"
                ]
                if not audio_streams:
                    raise ValueError("文件中没有可解码的音频流。")
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(24000)
                resampler = av.AudioResampler(format="s16", layout="mono", rate=24000)
                for frame in container.decode(audio=0):
                    for converted in resampler.resample(frame):
                        payload = converted.to_ndarray().tobytes()
                        output.writeframes(payload)
                        written_bytes += len(payload)
                for converted in resampler.resample(None):
                    payload = converted.to_ndarray().tobytes()
                    output.writeframes(payload)
                    written_bytes += len(payload)
            if written_bytes <= 0:
                raise ValueError("音频中没有可用采样。")
        except Exception as exc:
            raise ValueError(
                f"无法把参考音频转换成便携 WAV：{source.name}（{exc}）"
            ) from exc
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


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
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.manifest_path)

    def _rebase_audio_paths(self, old_root: Path, new_root: Path) -> None:
        old_root = old_root.resolve()
        new_root = new_root.resolve()
        payload = self._load()
        changed = False
        for value in payload.values():
            if not isinstance(value, dict):
                continue
            for field_name in ("audio_path", "emotion_audio_path"):
                raw = str(value.get(field_name) or "")
                if not raw:
                    continue
                candidate = Path(raw).resolve()
                if candidate == old_root or old_root in candidate.parents:
                    value[field_name] = str(new_root / candidate.relative_to(old_root))
                    changed = True
        if changed:
            self._save(payload)

    @contextmanager
    def transaction(self) -> Iterator["VoiceLibrary"]:
        """Yield an isolated clone and publish it only after the caller succeeds."""

        with _VOICE_LIBRARY_LOCK:
            self.root.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=".t8_voice_transaction_", dir=self.root.parent
            ) as temporary:
                transaction_data = Path(temporary) / "data"
                staged = VoiceLibrary(transaction_data)
                if self.root.is_dir():
                    shutil.copytree(
                        self.root,
                        staged.root,
                        copy_function=_copy_or_link,
                    )
                    staged._rebase_audio_paths(self.audio_dir, staged.audio_dir)
                try:
                    yield staged
                except Exception:
                    raise
                staged._rebase_audio_paths(staged.audio_dir, self.audio_dir)
                backup = self.root.with_name(
                    f".{self.root.name}.backup-{uuid.uuid4().hex}"
                )
                had_original = self.root.exists()
                if had_original:
                    self.root.replace(backup)
                try:
                    staged.root.replace(self.root)
                except Exception:
                    if had_original and backup.exists():
                        backup.replace(self.root)
                    raise
                else:
                    if backup.exists():
                        shutil.rmtree(backup, ignore_errors=True)

    @staticmethod
    def inspect_bundle(source: str | Path) -> dict[str, Any]:
        archive_path = Path(source).expanduser().resolve()
        if not archive_path.is_file():
            raise FileNotFoundError(f"音色包不存在：{archive_path}")
        try:
            with zipfile.ZipFile(archive_path) as archive:
                return _validated_bundle_manifest(
                    archive,
                    bundle_name=archive_path.name,
                )
        except zipfile.BadZipFile as exc:
            raise ValueError("音色包不是有效的 ZIP 文件。") from exc

    @_synchronized
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

    @_synchronized
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

    @_synchronized
    def get(self, name_or_id: str) -> VoiceProfile:
        needle = str(name_or_id).strip().casefold()
        for profile in self.list():
            if (
                profile.profile_id.casefold() == needle
                or profile.name.casefold() == needle
            ):
                return profile
        raise KeyError(f"角色音色不存在：{name_or_id}")

    @_synchronized
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
        profile_id = (
            replace_id
            or name_match_id
            or hashlib.sha256(clean_name.casefold().encode("utf-8")).hexdigest()[:16]
        )
        safe_stem = safe_voice_file_stem(clean_name, profile_id)[:48]
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        target = self.audio_dir / f"{safe_stem}-{profile_id}.wav"
        if source != target:
            _write_portable_wav(source, target)
        emotion_target = ""
        if emotion_source is not None:
            emotion_target_path = (
                self.audio_dir / f"{safe_stem}-{profile_id}-emotion.wav"
            )
            if emotion_source != emotion_target_path:
                _write_portable_wav(emotion_source, emotion_target_path)
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

    @_synchronized
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

    @_synchronized
    def set_favorite(self, name_or_id: str, favorite: bool) -> VoiceProfile:
        return self.update_metadata(name_or_id, favorite=favorite)

    @_synchronized
    def export_bundle(
        self,
        destination: str | Path,
        names: Iterable[str] | None = None,
    ) -> Path:
        selected = list(self.list())
        if names is not None:
            needles = {
                str(item).strip().casefold() for item in names if str(item).strip()
            }
            selected = [
                item
                for item in selected
                if item.name.casefold() in needles
                or item.profile_id.casefold() in needles
            ]
        if not selected:
            raise ValueError("没有可导出的角色音色。")
        target = Path(destination).expanduser().resolve()
        if not str(target).lower().endswith(VOICE_BUNDLE_SUFFIX):
            target = target.with_name(target.name + VOICE_BUNDLE_SUFFIX)
        target.parent.mkdir(parents=True, exist_ok=True)
        manifest_profiles: list[dict[str, Any]] = []
        temporary_target = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            with zipfile.ZipFile(
                temporary_target, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
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
                        archive_name = f"audio/{profile.profile_id}-{suffix}-{_safe_archive_name(source.name)}"
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
            temporary_target.replace(target)
        finally:
            temporary_target.unlink(missing_ok=True)
        return target

    def import_bundle(
        self,
        source: str | Path,
        *,
        conflict: str = "rename",
    ) -> list[VoiceProfile]:
        with self.transaction() as staged:
            imported = staged._import_bundle_in_place(source, conflict=conflict)
            imported_ids = [item.profile_id for item in imported]
        return [self.get(profile_id) for profile_id in imported_ids]

    def _import_bundle_in_place(
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
        try:
            archive_context = zipfile.ZipFile(archive_path)
        except zipfile.BadZipFile as exc:
            raise ValueError("音色包不是有效的 ZIP 文件。") from exc
        with archive_context as archive:
            manifest = _validated_bundle_manifest(
                archive,
                bundle_name=archive_path.name,
            )
            profiles = manifest["profiles"]
            extracted_size = sum(
                max(0, int(item.file_size)) for item in archive.infolist()
            )
            self.root.parent.mkdir(parents=True, exist_ok=True)
            if (
                shutil.disk_usage(self.root.parent).free
                < extracted_size + VOICE_DISK_RESERVE_BYTES
            ):
                raise ValueError("音色包解压空间不足，已保留至少 512 MiB 安全空间。")
            with tempfile.TemporaryDirectory(
                prefix="t8_voice_import_", dir=self.root.parent
            ) as temporary:
                temporary_root = Path(temporary).resolve()
                referenced = {
                    str(raw.get(field_name) or "")
                    for raw in profiles
                    for field_name in ("audio_path", "emotion_audio_path")
                    if str(raw.get(field_name) or "")
                }
                for member_name in referenced:
                    archive.extract(
                        _safe_member(member_name).as_posix(), temporary_root
                    )
                for raw in profiles:
                    name = str(raw.get("name") or "").strip()
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
                        if (
                            temporary_root not in resolved.parents
                            or not resolved.is_file()
                        ):
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
                            pronunciation_dictionary=raw.get(
                                "pronunciation_dictionary", ""
                            ),
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

    @_synchronized
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
    "safe_voice_file_stem",
]
