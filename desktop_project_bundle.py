"""Portable dialogue-project archives for the desktop integration."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import time
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from desktop_tasks import load_task, save_task
from desktop_voice_library import VOICE_BUNDLE_SUFFIX, VoiceLibrary


PROJECT_SCHEMA_VERSION = 1
PROJECT_SUFFIX = ".indextts-project.zip"
MAX_PROJECT_BYTES = 50 * 1024**3
MAX_PROJECT_FILES = 20_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member(value: str) -> PurePosixPath:
    candidate = PurePosixPath(str(value).replace("\\", "/"))
    if (
        not value
        or candidate.is_absolute()
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError(f"工程包包含不安全路径：{value}")
    return candidate


def _task_roles(task: dict[str, Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    def add(value) -> None:
        role = str(value or "").strip()
        key = role.casefold()
        if role and key not in seen:
            seen.add(key)
            result.append(role)

    settings = task.get("settings") or {}
    for row in settings.get("timeline_rows") or []:
        if isinstance(row, (list, tuple)) and len(row) > 1:
            add(row[1])
    for value in (task.get("lines") or {}).values():
        if isinstance(value, dict):
            add((value.get("report") or {}).get("role"))
    script = str(task.get("script") or "")
    for line in script.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "|" in stripped:
            add(stripped.split("|", 1)[0])
        match = re.match(r"^\[([^\]|]+)(?:\|[^\]]+)?\]", stripped)
        if match:
            add(match.group(1))
    add(settings.get("default_role"))
    return result


def _relative_task(task: dict[str, Any], session_dir: Path) -> dict[str, Any]:
    payload = json.loads(json.dumps(task, ensure_ascii=False))
    for value in (payload.get("lines") or {}).values():
        if not isinstance(value, dict):
            continue
        old_file = str(value.get("file") or "")
        if old_file:
            value["file"] = f"task/{Path(old_file).name}"
        report = value.get("report")
        if isinstance(report, dict) and report.get("file"):
            report["file"] = f"task/{Path(str(report['file'])).name}"
    for field_name in (
        "combined_file",
        "archive_file",
        "report_file",
        "rewritten_srt_file",
    ):
        value = str(payload.get(field_name) or "")
        if not value:
            continue
        path = Path(value)
        if session_dir in path.resolve().parents:
            payload[field_name] = f"task/{path.name}"
        elif field_name == "combined_file" and path.is_file():
            payload[field_name] = "outputs/combined.wav"
        else:
            payload[field_name] = ""
    return payload


def export_project(
    output_dir: str | Path,
    task_id: str,
    voice_library: VoiceLibrary,
    destination: str | Path,
) -> Path:
    output_root = Path(output_dir).expanduser().resolve()
    task = load_task(output_root, str(task_id))
    session_dir = (output_root / str(task_id)).resolve()
    if output_root not in session_dir.parents or not session_dir.is_dir():
        raise ValueError("任务目录不存在，无法导出工程。")
    target = Path(destination).expanduser().resolve()
    if not str(target).lower().endswith(PROJECT_SUFFIX):
        target = target.with_name(target.name + PROJECT_SUFFIX)
    target.parent.mkdir(parents=True, exist_ok=True)

    assets: dict[str, Path] = {}
    for path in session_dir.rglob("*"):
        if path.is_file() and not path.name.endswith(".tmp"):
            assets[f"task/{path.relative_to(session_dir).as_posix()}"] = path
    combined = Path(str(task.get("combined_file") or ""))
    if combined.is_file() and session_dir not in combined.resolve().parents:
        assets["outputs/combined.wav"] = combined.resolve()

    with tempfile.TemporaryDirectory(prefix="t8_project_export_") as temporary:
        voice_bundle: Path | None = None
        roles = _task_roles(task)
        matched_roles: list[str] = []
        for role in roles:
            try:
                voice_library.get(role)
            except KeyError:
                continue
            matched_roles.append(role)
        if matched_roles:
            voice_bundle = Path(temporary) / ("project-voices" + VOICE_BUNDLE_SUFFIX)
            voice_library.export_bundle(voice_bundle, matched_roles)
            assets["voices/project-voices.t8voice.zip"] = voice_bundle

        file_manifest = {
            archive_name: {
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for archive_name, path in assets.items()
        }
        manifest = {
            "schemaVersion": PROJECT_SCHEMA_VERSION,
            "format": "T8star-Aix IndexTTS 2.5 Project",
            "createdAt": time.strftime("%Y-%m-%d %H:%M:%S"),
            "originalTaskId": str(task_id),
            "roles": roles,
            "includedVoices": matched_roles,
            "task": _relative_task(task, session_dir),
            "files": file_manifest,
        }
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "project.json",
                json.dumps(manifest, ensure_ascii=False, indent=2),
            )
            for archive_name, path in assets.items():
                archive.write(path, archive_name)
    return target


def _new_task_id(output_root: Path) -> str:
    while True:
        candidate = f"dialogue_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        if not (output_root / candidate).exists():
            return candidate


def _rewrite_report_paths(value: Any, session_dir: Path) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                str(session_dir / Path(str(nested)).name)
                if key == "file" and isinstance(nested, str) and nested
                else _rewrite_report_paths(nested, session_dir)
            )
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [_rewrite_report_paths(item, session_dir) for item in value]
    return value


def _rewrite_task_roles(task: dict[str, Any], mapping: dict[str, str]) -> None:
    normalized = {str(key).casefold(): str(value) for key, value in mapping.items()}

    def role_name(value: Any) -> str:
        text = str(value or "")
        return normalized.get(text.casefold(), text)

    settings = task.get("settings")
    if isinstance(settings, dict):
        settings["default_role"] = role_name(settings.get("default_role"))
        for row in settings.get("timeline_rows") or []:
            if isinstance(row, list) and len(row) > 1:
                row[1] = role_name(row[1])
    for value in (task.get("lines") or {}).values():
        if isinstance(value, dict) and isinstance(value.get("report"), dict):
            value["report"]["role"] = role_name(value["report"].get("role"))

    script = str(task.get("script") or "")
    if not script or not normalized:
        return
    try:
        json_script = json.loads(script)
    except json.JSONDecodeError:
        json_script = None
    if isinstance(json_script, list):
        for item in json_script:
            if isinstance(item, dict) and "role" in item:
                item["role"] = role_name(item.get("role"))
        task["script"] = json.dumps(json_script, ensure_ascii=False, indent=2)
        return

    rewritten_lines: list[str] = []
    for line in script.splitlines():
        bracket = re.match(r"^(\s*)\[([^\]|]+)(\|[^\]]+)?\](.*)$", line)
        if bracket:
            line = (
                f"{bracket.group(1)}[{role_name(bracket.group(2))}"
                f"{bracket.group(3) or ''}]{bracket.group(4)}"
            )
        elif "|" in line:
            prefix, remainder = line.split("|", 1)
            if prefix.strip().casefold() in normalized:
                leading = prefix[: len(prefix) - len(prefix.lstrip())]
                trailing = prefix[len(prefix.rstrip()) :]
                line = f"{leading}{role_name(prefix.strip())}{trailing}|{remainder}"
        else:
            colon = re.match(r"^(\s*)([^：:]+)([：:])(.*)$", line)
            if colon and colon.group(2).strip().casefold() in normalized:
                line = (
                    f"{colon.group(1)}{role_name(colon.group(2).strip())}"
                    f"{colon.group(3)}{colon.group(4)}"
                )
        rewritten_lines.append(line)
    task["script"] = "\n".join(rewritten_lines)


def import_project(
    source: str | Path,
    output_dir: str | Path,
    voice_library: VoiceLibrary,
    *,
    voice_conflict: str = "rename",
) -> dict[str, Any]:
    archive_path = Path(source).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"工程包不存在：{archive_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) > MAX_PROJECT_FILES:
            raise ValueError("工程包文件数量超过安全限制。")
        total_size = sum(max(0, int(item.file_size)) for item in members)
        if total_size > MAX_PROJECT_BYTES:
            raise ValueError("工程包解压后超过 50GB 安全限制。")
        for item in members:
            _safe_member(item.filename)
        try:
            manifest = json.loads(archive.read("project.json").decode("utf-8"))
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("工程包 project.json 缺失或损坏。") from exc
        if manifest.get("schemaVersion") != PROJECT_SCHEMA_VERSION:
            raise ValueError("不支持的工程包版本。")
        file_manifest = manifest.get("files")
        task = manifest.get("task")
        if not isinstance(file_manifest, dict) or not isinstance(task, dict):
            raise ValueError("工程包清单内容无效。")
        for archive_name, metadata in file_manifest.items():
            _safe_member(archive_name)
            try:
                info = archive.getinfo(archive_name)
            except KeyError as exc:
                raise ValueError(f"工程包缺少文件：{archive_name}") from exc
            if info.file_size != int(metadata.get("size", -1)):
                raise ValueError(f"工程包文件大小不匹配：{archive_name}")

        with tempfile.TemporaryDirectory(prefix="t8_project_import_") as temporary:
            temporary_root = Path(temporary).resolve()
            archive.extractall(temporary_root)
            for archive_name, metadata in file_manifest.items():
                path = (temporary_root / archive_name).resolve()
                if temporary_root not in path.parents or not path.is_file():
                    raise ValueError(f"工程包文件路径无效：{archive_name}")
                if _sha256(path) != str(metadata.get("sha256") or "").lower():
                    raise ValueError(f"工程包文件校验失败：{archive_name}")

            new_task_id = _new_task_id(output_root)
            session_dir = (output_root / new_task_id).resolve()
            session_dir.mkdir(parents=True)
            task_source = temporary_root / "task"
            if task_source.is_dir():
                for path in task_source.rglob("*"):
                    if not path.is_file() or path.name == "task.json":
                        continue
                    destination = (session_dir / path.relative_to(task_source)).resolve()
                    if session_dir not in destination.parents:
                        raise ValueError("工程任务文件路径越界。")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, destination)

            combined_source = temporary_root / "outputs" / "combined.wav"
            combined_target = output_root / f"{new_task_id}.wav"
            if combined_source.is_file():
                shutil.copy2(combined_source, combined_target)

            imported_voices = []
            voice_role_mapping: dict[str, str] = {}
            voice_bundle = temporary_root / "voices" / "project-voices.t8voice.zip"
            if voice_bundle.is_file():
                with zipfile.ZipFile(voice_bundle) as voice_archive:
                    voice_manifest = json.loads(
                        voice_archive.read("manifest.json").decode("utf-8")
                    )
                source_voice_names = [
                    str(item.get("name") or "")
                    for item in voice_manifest.get("profiles") or []
                    if isinstance(item, dict) and str(item.get("name") or "").strip()
                ]
                imported_voices = voice_library.import_bundle(
                    voice_bundle, conflict=voice_conflict
                )
                if voice_conflict == "rename":
                    voice_role_mapping = {
                        source: imported.name
                        for source, imported in zip(source_voice_names, imported_voices)
                    }
                else:
                    voice_role_mapping = {source: source for source in source_voice_names}

            restored = json.loads(json.dumps(task, ensure_ascii=False))
            _rewrite_task_roles(restored, voice_role_mapping)
            restored["task_id"] = new_task_id
            restored["imported_from"] = str(archive_path)
            restored["imported_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            for line in (restored.get("lines") or {}).values():
                if not isinstance(line, dict):
                    continue
                relative = str(line.get("file") or "")
                line["file"] = str(session_dir / Path(relative).name) if relative else ""
                if isinstance(line.get("report"), dict):
                    line["report"] = _rewrite_report_paths(line["report"], session_dir)
            report_path = session_dir / "report.json"
            if report_path.is_file():
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
                    report_path.write_text(
                        json.dumps(
                            _rewrite_report_paths(report, session_dir),
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                except (OSError, json.JSONDecodeError):
                    pass
            restored["combined_file"] = str(combined_target) if combined_target.is_file() else ""
            restored["archive_file"] = ""
            restored["report_file"] = str(report_path) if report_path.is_file() else ""
            rewritten_candidates = [
                session_dir / "rewritten_edited.srt",
                session_dir / "rewritten.srt",
            ]
            rewritten = next((item for item in rewritten_candidates if item.is_file()), None)
            restored["rewritten_srt_file"] = str(rewritten) if rewritten else ""
            save_task(output_root, restored)

    return {
        "task_id": new_task_id,
        "imported_voices": [item.name for item in imported_voices],
        "original_task_id": manifest.get("originalTaskId"),
        "source": str(archive_path),
        "status": restored.get("status", "unknown"),
    }


__all__ = [
    "PROJECT_SCHEMA_VERSION",
    "PROJECT_SUFFIX",
    "export_project",
    "import_project",
]
