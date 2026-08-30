"""Persist, render, and rebuild auditable internal speech-rate segments."""

from __future__ import annotations

import html
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


SEGMENT_RATE_HEADERS = [
    "段号",
    "语音块",
    "语言",
    "时长(秒)",
    "语速(单位/秒)",
    "基线",
    "相对基线",
    "判定",
    "重试",
    "采用",
    "文本",
]


def _number(value: Any, digits: int = 3) -> float:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return 0.0


def segment_rate_rows(reports: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for item in reports:
        ratio = _number(item.get("rate_ratio"), 3)
        suspect = bool(item.get("suspect"))
        retried = bool(item.get("retried"))
        accepted = bool(item.get("accepted"))
        decision = "异常偏慢" if suspect else "稳定" if item.get("eligible") else "样本不足"
        rows.append(
            [
                int(item.get("index") or item.get("position", 0) + 1),
                int(item.get("speech_block") or 0),
                str(item.get("language") or ""),
                _number(item.get("duration_seconds"), 3),
                _number(item.get("units_per_second"), 3),
                _number(item.get("baseline_units_per_second"), 3),
                ratio,
                decision,
                "是" if retried else "否",
                "重试结果" if accepted else "原始结果",
                str(item.get("text") or ""),
            ]
        )
    return rows


def render_segment_rate_html(reports: Sequence[Mapping[str, Any]]) -> str:
    if not reports:
        return (
            '<div class="t8-rate-empty">生成后显示内部文本分段语速；'
            "至少两个稳定长段后才建立基线。</div>"
        )
    eligible_rates = [
        float(item.get("units_per_second") or 0.0)
        for item in reports
        if item.get("eligible") and float(item.get("units_per_second") or 0.0) > 0
    ]
    ceiling = max(eligible_rates or [1.0])
    cards: list[str] = []
    for item in reports:
        rate = float(item.get("units_per_second") or 0.0)
        baseline = float(item.get("baseline_units_per_second") or 0.0)
        width = max(2.0, min(100.0, rate / ceiling * 100.0)) if rate > 0 else 2.0
        suspect = bool(item.get("suspect"))
        accepted = bool(item.get("accepted"))
        css = "suspect" if suspect else "stable" if item.get("eligible") else "short"
        badge = "异常偏慢"
        if accepted:
            badge = "已采用重试"
        elif not suspect and item.get("eligible"):
            badge = "稳定"
        elif not suspect:
            badge = "不参与基线"
        retry = ""
        if item.get("retried"):
            retry = (
                f" · 重试 {_number(item.get('retry_units_per_second'))} 单位/秒"
                + ("（采用）" if accepted else "（保留原始）")
            )
        text = html.escape(str(item.get("text") or ""))
        cards.append(
            '<div class="t8-rate-card">'
            f'<div class="t8-rate-title"><strong>第 {int(item.get("index") or 0)} 段</strong>'
            f'<span class="t8-rate-badge {css}">{html.escape(badge)}</span></div>'
            '<div class="t8-rate-track">'
            f'<span class="t8-rate-fill {css}" style="width:{width:.2f}%"></span>'
            + (
                f'<i class="t8-rate-baseline" style="left:{min(100.0, baseline / ceiling * 100.0):.2f}%"></i>'
                if baseline > 0
                else ""
            )
            + "</div>"
            f'<div class="t8-rate-meta">{_number(rate)} 单位/秒 · 基线 {_number(baseline)}'
            f" · 比例 {_number(item.get('rate_ratio'))}{retry}</div>"
            f'<div class="t8-rate-text">{text}</div></div>'
        )
    return (
        '<div class="t8-rate-chart"><div class="t8-rate-legend">'
        "柱长=实际语速，竖线=前序稳定中位基线；红色只表示强异常，不把普通情绪放慢判错。"
        "</div>"
        + "".join(cards)
        + "</div>"
    )


def segment_choices(manifest: Mapping[str, Any]) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for item in manifest.get("segments", []):
        if not isinstance(item, Mapping):
            continue
        index = int(item.get("index") or 0)
        report = item.get("report") or {}
        state = "异常" if report.get("suspect") else "稳定" if report.get("eligible") else "短段"
        text = str(item.get("text") or "").replace("\n", " ")[:38]
        choices.append((f"第 {index} 段 · {state} · {text}", str(index)))
    return choices


def _safe_relative(workspace: Path, value: str | Path) -> str:
    target = (workspace / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        relative = target.relative_to(workspace.resolve())
    except ValueError as exc:
        raise ValueError("segment artifact must stay inside its workspace") from exc
    return relative.as_posix()


def write_segment_workspace(
    workspace: Path,
    *,
    block_records: Sequence[Sequence[Mapping[str, Any]]],
    reports: Sequence[Mapping[str, Any]],
    block_pauses: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
    output_path: str | Path,
    save_audio: Callable[[Path, Any, int], None],
) -> Path:
    """Write original/retry/selected artifacts and a path-confined manifest."""

    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    report_by_position = {
        int(item.get("position") or 0): dict(item) for item in reports
    }
    segments: list[dict[str, Any]] = []
    position = 0
    for records in block_records:
        for record in records:
            index = int(record.get("index") or position + 1)
            sample_rate = int(record.get("sample_rate") or 22050)
            original_name = f"segment_{index:03d}_original.wav"
            save_audio(workspace / original_name, record.get("original_waveform", record["waveform"]), sample_rate)
            retry_name = ""
            if record.get("retry_waveform") is not None:
                retry_name = f"segment_{index:03d}_auto_retry.wav"
                save_audio(workspace / retry_name, record["retry_waveform"], sample_rate)
            selected_source = str(record.get("selected_source") or "original")
            selected_audio = retry_name if selected_source == "auto_retry" and retry_name else original_name
            segments.append(
                {
                    "index": index,
                    "position": position,
                    "speech_block": int(record.get("speech_block") or 1),
                    "language": str(record.get("language") or ""),
                    "text": str(record.get("text") or ""),
                    "sample_rate": sample_rate,
                    "original_audio": original_name,
                    "retry_audio": retry_name,
                    "selected_audio": selected_audio,
                    "selected_source": selected_source,
                    "redo_count": 0,
                    "report": report_by_position.get(position, {}),
                }
            )
            position += 1
    manifest = {
        "schema_version": 1,
        "workspace": str(workspace),
        "output_path": str(Path(output_path).resolve()),
        "segments": segments,
        "blocks": [dict(item) for item in block_pauses],
        "settings": dict(settings),
    }
    manifest_path = workspace / "segment-workspace.json"
    temp_path = manifest_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, manifest_path)
    return manifest_path


def load_segment_workspace(manifest_path: str | Path) -> dict[str, Any]:
    manifest_file = Path(manifest_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version") or 0) != 1:
        raise ValueError("unsupported segment workspace schema")
    workspace = Path(manifest.get("workspace") or manifest_file.parent).resolve()
    if manifest_file.parent != workspace:
        raise ValueError("segment workspace location does not match the manifest")
    for segment in manifest.get("segments", []):
        for key in ("original_audio", "retry_audio", "selected_audio"):
            value = str(segment.get(key) or "")
            if value:
                _safe_relative(workspace, value)
    return manifest


def selected_segment_artifacts(
    manifest: Mapping[str, Any], segment_index: int | str
) -> dict[str, Any]:
    workspace = Path(str(manifest["workspace"])).resolve()
    wanted = int(segment_index)
    for segment in manifest.get("segments", []):
        if int(segment.get("index") or 0) != wanted:
            continue
        def resolved(key: str) -> str | None:
            value = str(segment.get(key) or "")
            return str((workspace / value).resolve()) if value else None

        return {
            "segment": segment,
            "original_audio": resolved("original_audio"),
            "retry_audio": resolved("retry_audio"),
            "selected_audio": resolved("selected_audio"),
        }
    raise ValueError(f"segment {wanted} does not exist")


def select_replacement_audio(
    manifest_path: str | Path,
    segment_index: int | str,
    audio_path: str | Path,
    *,
    source: str,
    report_update: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).resolve()
    manifest = load_segment_workspace(manifest_file)
    workspace = Path(manifest["workspace"])
    relative = _safe_relative(workspace, audio_path)
    wanted = int(segment_index)
    for segment in manifest["segments"]:
        if int(segment.get("index") or 0) == wanted:
            segment["selected_audio"] = relative
            segment["selected_source"] = str(source)
            segment["redo_count"] = int(segment.get("redo_count") or 0) + 1
            if report_update:
                segment.setdefault("report", {}).update(dict(report_update))
            break
    else:
        raise ValueError(f"segment {wanted} does not exist")
    temp_path = manifest_file.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, manifest_file)
    return manifest


def compose_segment_workspace(
    manifest: Mapping[str, Any],
    load_audio: Callable[[Path], tuple[Any, int]],
) -> tuple[Any, int]:
    """Rebuild the raw waveform from currently selected internal segments."""

    import torch

    workspace = Path(str(manifest["workspace"])).resolve()
    segments = {
        int(item["index"]): item for item in manifest.get("segments", [])
    }
    blocks: list[Any] = []
    sample_rate: int | None = None
    settings = manifest.get("settings") or {}
    internal_silence_ms = int(settings.get("segment_silence_ms") or 0)
    for block in manifest.get("blocks", []):
        parts: list[Any] = []
        block_indices = [int(value) for value in block.get("segment_indices", [])]
        for offset, index in enumerate(block_indices):
            segment = segments[index]
            relative = _safe_relative(workspace, str(segment["selected_audio"]))
            waveform, rate = load_audio(workspace / relative)
            tensor = torch.as_tensor(waveform).detach().cpu().float()
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim != 2:
                tensor = tensor.reshape(1, -1)
            if sample_rate is None:
                sample_rate = int(rate)
            elif sample_rate != int(rate):
                raise ValueError("segment workspace contains mixed sample rates")
            parts.append(tensor)
            if offset < len(block_indices) - 1 and internal_silence_ms > 0:
                parts.append(
                    torch.zeros(
                        (tensor.shape[0], round(sample_rate * internal_silence_ms / 1000)),
                        dtype=tensor.dtype,
                    )
                )
        if not parts:
            continue
        blocks.append(torch.cat(parts, dim=-1))
    if sample_rate is None or not blocks:
        raise ValueError("segment workspace contains no playable segments")
    combined: list[Any] = []
    for block_index, waveform in enumerate(blocks):
        block = manifest["blocks"][block_index]
        if block_index == 0 and int(block.get("pause_before_ms") or 0) > 0:
            combined.append(
                torch.zeros(
                    (waveform.shape[0], round(sample_rate * int(block["pause_before_ms"]) / 1000)),
                    dtype=waveform.dtype,
                )
            )
        combined.append(waveform)
        if int(block.get("pause_after_ms") or 0) > 0:
            combined.append(
                torch.zeros(
                    (waveform.shape[0], round(sample_rate * int(block["pause_after_ms"]) / 1000)),
                    dtype=waveform.dtype,
                )
            )
    return torch.cat(combined, dim=-1), sample_rate


__all__ = [
    "SEGMENT_RATE_HEADERS",
    "compose_segment_workspace",
    "load_segment_workspace",
    "render_segment_rate_html",
    "segment_choices",
    "segment_rate_rows",
    "select_replacement_audio",
    "selected_segment_artifacts",
    "write_segment_workspace",
]
