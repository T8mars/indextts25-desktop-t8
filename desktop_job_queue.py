"""Crash-safe persistent queue shared by desktop single and dialogue jobs."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any


QUEUE_SCHEMA_VERSION = 1
JOB_KINDS = frozenset({"single", "dialogue", "srt"})
JOB_STATUSES = frozenset({"pending", "running", "completed", "failed", "cancelled"})


class DesktopJobQueue:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._guard = threading.RLock()
        with self._guard:
            payload = self._load_locked()
            recovered = False
            for job in payload["jobs"]:
                if job.get("status") == "running":
                    job["status"] = "pending"
                    job["error"] = "上次桌面程序退出时任务仍在运行，已恢复为等待执行。"
                    recovered = True
            if recovered:
                self._save_locked(payload)

    def enqueue(self, kind: str, payload: dict[str, Any], summary: str) -> dict[str, Any]:
        kind = str(kind).strip().lower()
        if kind not in JOB_KINDS:
            raise ValueError(f"不支持的任务类型：{kind}")
        serialized = json.dumps(payload, ensure_ascii=False)
        if len(serialized.encode("utf-8")) > 2 * 1024 * 1024:
            raise ValueError("单个任务配置超过 2 MiB，请缩短脚本或分批加入队列。")
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        job = {
            "job_id": f"job_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
            "kind": kind,
            "status": "pending",
            "summary": str(summary or "").strip()[:160],
            "payload": json.loads(serialized),
            "result": "",
            "error": "",
            "created_at": now,
            "updated_at": now,
        }
        with self._guard:
            state = self._load_locked()
            state["jobs"].append(job)
            state["jobs"] = state["jobs"][-1000:]
            self._save_locked(state)
        return dict(job)

    def jobs(self) -> list[dict[str, Any]]:
        with self._guard:
            return [dict(job) for job in self._load_locked()["jobs"]]

    def get(self, job_id: str) -> dict[str, Any]:
        for job in self.jobs():
            if job.get("job_id") == str(job_id):
                return job
        raise ValueError("所选队列任务不存在。")

    def pending(self) -> list[dict[str, Any]]:
        return [job for job in self.jobs() if job.get("status") == "pending"]

    def update(
        self,
        job_id: str,
        status: str,
        *,
        result: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        if status not in JOB_STATUSES:
            raise ValueError(f"无效任务状态：{status}")
        with self._guard:
            state = self._load_locked()
            for job in state["jobs"]:
                if job.get("job_id") != str(job_id):
                    continue
                job["status"] = status
                job["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                if result is not None:
                    job["result"] = str(result)
                if error is not None:
                    job["error"] = str(error)
                self._save_locked(state)
                return dict(job)
        raise ValueError("所选队列任务不存在。")

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job.get("status") not in {"pending", "running"}:
            raise ValueError("只有等待中或运行中的任务可以取消。")
        return self.update(job_id, "cancelled", error="用户已取消。")

    def retry(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job.get("status") not in {"failed", "cancelled"}:
            raise ValueError("只有失败或已取消的任务可以重新排队。")
        return self.update(job_id, "pending", result="", error="")

    def rows(self) -> list[list[str]]:
        labels = {"single": "单句", "dialogue": "多角色", "srt": "SRT"}
        return [
            [
                str(job.get("job_id", "")),
                labels.get(str(job.get("kind", "")), str(job.get("kind", ""))),
                str(job.get("status", "")),
                str(job.get("summary", "")),
                str(job.get("result", "")),
                str(job.get("error", "")),
                str(job.get("updated_at", "")),
            ]
            for job in reversed(self.jobs())
        ]

    def choices(self) -> list[tuple[str, str]]:
        return [
            (
                f"{job.get('status')} · {job.get('kind')} · {job.get('summary')}",
                str(job.get("job_id")),
            )
            for job in reversed(self.jobs())
        ]

    def _load_locked(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"schema_version": QUEUE_SCHEMA_VERSION, "jobs": []}
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            raise ValueError("持久任务队列文件损坏，请先备份后删除 task_queue.json。") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("jobs"), list):
            raise ValueError("持久任务队列文件格式无效。")
        return payload

    def _save_locked(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(payload)
        payload["schema_version"] = QUEUE_SCHEMA_VERSION
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)


__all__ = ["DesktopJobQueue", "JOB_KINDS", "JOB_STATUSES", "QUEUE_SCHEMA_VERSION"]
