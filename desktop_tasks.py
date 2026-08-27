"""Crash-safe persistent dialogue task manifests for the desktop bundle."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any


TASK_SCHEMA_VERSION = 1
_TASK_ID = re.compile(r"^dialogue_[0-9]{8}_[0-9]{6}_[0-9a-f]{8}$")


def _manifest_path(output_dir: Path, task_id: str) -> Path:
    task_id = str(task_id)
    if not _TASK_ID.fullmatch(task_id):
        raise ValueError("任务编号格式无效。")
    root = Path(output_dir).resolve()
    path = (root / task_id / "task.json").resolve()
    if root not in path.parents:
        raise ValueError("任务目录越界。")
    return path


def save_task(output_dir: Path, task: dict[str, Any]) -> dict[str, Any]:
    task = dict(task)
    task["schema_version"] = TASK_SCHEMA_VERSION
    task["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path = _manifest_path(output_dir, task.get("task_id", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return task


def create_task(
    output_dir: Path,
    task_id: str,
    *,
    script_type: str,
    script: str,
    settings: dict[str, Any],
    line_count: int,
) -> dict[str, Any]:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    return save_task(
        output_dir,
        {
            "task_id": task_id,
            "status": "running",
            "created_at": now,
            "script_type": script_type,
            "script": script,
            "settings": settings,
            "line_count": int(line_count),
            "lines": {},
            "last_error": "",
        },
    )


def load_task(output_dir: Path, task_id: str) -> dict[str, Any]:
    path = _manifest_path(output_dir, task_id)
    try:
        task = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError("所选任务不存在。") from exc
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError("任务清单损坏，无法恢复。") from exc
    if not isinstance(task, dict) or task.get("task_id") != task_id:
        raise ValueError("任务清单内容无效。")
    task.setdefault("lines", {})
    return task


def update_task_line(
    output_dir: Path,
    task: dict[str, Any],
    line_number: int,
    *,
    status: str,
    file: str = "",
    report: dict[str, Any] | None = None,
    error: str = "",
) -> dict[str, Any]:
    lines = dict(task.get("lines") or {})
    lines[str(int(line_number))] = {
        "status": str(status),
        "file": str(file),
        "report": report or {},
        "error": str(error),
    }
    task["lines"] = lines
    return save_task(output_dir, task)


def set_task_status(
    output_dir: Path,
    task: dict[str, Any],
    status: str,
    *,
    error: str = "",
) -> dict[str, Any]:
    task["status"] = str(status)
    task["last_error"] = str(error)
    return save_task(output_dir, task)


def task_choices(output_dir: Path) -> list[tuple[str, str]]:
    choices: list[tuple[str, str, str]] = []
    root = Path(output_dir)
    if not root.is_dir():
        return []
    for path in root.glob("dialogue_*/task.json"):
        try:
            task = json.loads(path.read_text(encoding="utf-8"))
            task_id = str(task["task_id"])
            if not _TASK_ID.fullmatch(task_id):
                continue
            status = str(task.get("status", "unknown"))
            updated = str(task.get("updated_at", ""))
            completed = sum(
                item.get("status") == "completed"
                for item in (task.get("lines") or {}).values()
                if isinstance(item, dict)
            )
            total = int(task.get("line_count", 0))
            label = f"{task_id} · {status} · {completed}/{total}"
            choices.append((updated, label, task_id))
        except (OSError, ValueError, TypeError, KeyError):
            continue
    choices.sort(reverse=True)
    return [(label, task_id) for _updated, label, task_id in choices]


__all__ = [
    "TASK_SCHEMA_VERSION",
    "create_task",
    "load_task",
    "save_task",
    "set_task_status",
    "task_choices",
    "update_task_line",
]
