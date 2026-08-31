"""Persistent human reviews and favorites for desktop A/B audio candidates."""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any


class CandidateWorkspace:
    def __init__(self, data_dir: str | Path, output_dir: str | Path) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.review_path = self.data_dir / "candidate_reviews.json"
        self.favorite_dir = self.data_dir / "candidate_favorites"

    def normalize_files(self, values: Any) -> list[str]:
        if values is None:
            return []
        if isinstance(values, (str, Path, dict)):
            values = [values]
        result: list[str] = []
        for value in values:
            if isinstance(value, dict):
                value = value.get("path") or value.get("name")
            elif hasattr(value, "path"):
                value = value.path
            if not value:
                continue
            path = self._allowed_file(value)
            rendered = str(path)
            if rendered not in result:
                result.append(rendered)
        return result

    def choices(self, values: Any) -> list[tuple[str, str]]:
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return [
            (f"候选 {labels[index] if index < len(labels) else index + 1} · {Path(path).name}", path)
            for index, path in enumerate(self.normalize_files(values))
        ]

    def save_review(
        self,
        candidate_file: str | Path,
        rating: int | float,
        note: str = "",
        *,
        favorite: bool = False,
    ) -> dict[str, Any]:
        source = self._allowed_file(candidate_file)
        try:
            score = int(float(rating))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("候选评分必须是 1–5 星。") from exc
        if not 1 <= score <= 5:
            raise ValueError("候选评分必须是 1–5 星。")
        favorite_path = ""
        if favorite:
            self.favorite_dir.mkdir(parents=True, exist_ok=True)
            target = self.favorite_dir / (
                f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}_{source.name}"
            )
            shutil.copy2(source, target)
            favorite_path = str(target)
        review = {
            "candidate_file": str(source),
            "rating": score,
            "note": str(note or "").strip()[:1000],
            "favorite_file": favorite_path,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        payload = self._load_reviews()
        records = [
            item
            for item in payload["reviews"]
            if str(item.get("candidate_file")) != str(source)
        ]
        records.append(review)
        payload["reviews"] = records[-2000:]
        self.review_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.review_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.review_path)
        return review

    def review_for(self, candidate_file: str | Path) -> dict[str, Any] | None:
        source = str(self._allowed_file(candidate_file))
        return next(
            (item for item in reversed(self._load_reviews()["reviews"]) if item.get("candidate_file") == source),
            None,
        )

    def _allowed_file(self, value: str | Path) -> Path:
        path = Path(value).resolve()
        allowed_roots = (self.data_dir, self.output_dir)
        if not path.is_file() or not any(path == root or root in path.parents for root in allowed_roots):
            raise ValueError("候选音频不存在或不在桌面整合包的数据目录中。")
        if path.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
            raise ValueError("候选文件不是支持的音频格式。")
        return path

    def _load_reviews(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.review_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"schema_version": 1, "reviews": []}
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            raise ValueError("候选评分记录损坏，请备份后删除 candidate_reviews.json。") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("reviews"), list):
            raise ValueError("候选评分记录格式无效。")
        payload["schema_version"] = 1
        return payload


__all__ = ["CandidateWorkspace"]
