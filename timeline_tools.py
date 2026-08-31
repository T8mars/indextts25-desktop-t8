"""Editable timeline, visualization, and SRT rewrite helpers."""

from __future__ import annotations

import html
import json
from dataclasses import replace
from typing import Any, Sequence

from dialogue_runtime import DialogueLine, format_emotion_override, parse_emotion_override


TIMELINE_HEADERS = [
    "序号",
    "角色",
    "语言",
    "开始(ms)",
    "结束(ms)",
    "时长系数",
    "台词",
    "逐句情感（留空继承角色）",
]


def timeline_rows(lines: Sequence[DialogueLine]) -> list[list[Any]]:
    return [
        [
            line.index,
            line.role,
            line.language,
            "" if line.start_ms is None else line.start_ms,
            "" if line.end_ms is None else line.end_ms,
            line.duration_factor,
            line.text,
            format_emotion_override(line),
        ]
        for line in lines
    ]


def _table_data(rows) -> list:
    if rows is None:
        return []
    if isinstance(rows, dict):
        return list(rows.get("data") or [])
    if hasattr(rows, "values") and hasattr(rows.values, "tolist"):
        return rows.values.tolist()
    return list(rows)


def move_timeline_row(
    rows,
    selected_index: int | float | str,
    direction: int,
) -> tuple[list[list[Any]], int, bool]:
    """Move one dialogue row while keeping authored timeline slots in place."""

    data = _table_data(rows)
    if not data:
        raise ValueError("请先解析台词，再选择需要调整的行。")
    if direction not in {-1, 1}:
        raise ValueError("台词只能上移或下移一行。")
    try:
        wanted = int(float(selected_index))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("请先选择需要调整顺序的台词行。") from exc

    normalized: list[list[Any]] = []
    current_position: int | None = None
    for position, row in enumerate(data):
        if isinstance(row, dict):
            values = [row.get(name) for name in TIMELINE_HEADERS]
        else:
            values = list(row)
        if len(values) < len(TIMELINE_HEADERS):
            values.extend([""] * (len(TIMELINE_HEADERS) - len(values)))
        try:
            row_index = int(float(values[0]))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"时间轴第 {position + 1} 行序号无效。") from exc
        if row_index == wanted:
            current_position = position
        normalized.append(values[: len(TIMELINE_HEADERS)])
    if current_position is None:
        raise ValueError(f"时间轴中找不到第 {wanted} 条台词，请重新选择。")

    target_position = current_position + direction
    if target_position < 0 or target_position >= len(normalized):
        return normalized, wanted, False

    # Subtitle times belong to display positions. Capture them before moving
    # the content so an SRT keeps its chronological cue order.
    time_slots = [(row[3], row[4]) for row in normalized]
    normalized[current_position], normalized[target_position] = (
        normalized[target_position],
        normalized[current_position],
    )
    for position, row in enumerate(normalized, 1):
        row[0] = position
        row[3], row[4] = time_slots[position - 1]
    return normalized, target_position + 1, True


def _optional_ms(value, label: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        result = int(float(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}必须是整数毫秒。") from exc
    if result < 0 or result > 86_400_000:
        raise ValueError(f"{label}必须在 0–86400000 毫秒之间。")
    return result


def apply_timeline_edits(
    original_lines: Sequence[DialogueLine],
    rows,
) -> list[DialogueLine]:
    data = _table_data(rows)
    if not data:
        return list(original_lines)
    if len(data) != len(original_lines):
        raise ValueError("时间轴编辑行数必须与已解析台词数量一致。")
    expected_indexes = {line.index for line in original_lines}
    result: list[DialogueLine] = []
    seen: set[int] = set()
    for position, row in enumerate(data, 1):
        if isinstance(row, dict):
            values = [
                row.get(name)
                for name in TIMELINE_HEADERS
            ]
        else:
            values = list(row)
        if len(values) < 7:
            raise ValueError(f"时间轴第 {position} 行缺少列。")
        try:
            index = int(float(values[0]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"时间轴第 {position} 行序号无效。") from exc
        if index not in expected_indexes or index in seen:
            raise ValueError(f"时间轴第 {position} 行序号重复或不存在：{index}")
        seen.add(index)
        role = str(values[1] or "").strip()
        language = str(values[2] or "").strip().upper()
        text = str(values[6] or "").strip()
        if not role or not text:
            raise ValueError(f"时间轴第 {position} 行角色和台词不能为空。")
        if language not in {"ZH", "EN", "JA", "ES", "AR"}:
            raise ValueError(f"时间轴第 {position} 行语言无效：{language}")
        original = next(line for line in original_lines if line.index == index)
        start_ms = _optional_ms(values[3], f"时间轴第 {position} 行开始时间")
        end_ms = _optional_ms(values[4], f"时间轴第 {position} 行结束时间")
        # Gradio's numeric Dataframe has historically round-tripped an empty
        # pair as 0/0 after a queued function completes.  A batch line without
        # authored timing must stay untimed; otherwise the next click is
        # rejected as a zero-length slot and the UI appears to erase every
        # timestamp.  Real SRT timing is never 0/0, so this recovery is narrow.
        if (
            original.start_ms is None
            and original.end_ms is None
            and start_ms == 0
            and end_ms == 0
        ):
            start_ms = None
            end_ms = None
        if (start_ms is None) != (end_ms is None):
            raise ValueError(f"时间轴第 {position} 行开始和结束时间必须同时填写或同时留空。")
        if start_ms is not None and end_ms <= start_ms:
            raise ValueError(f"时间轴第 {position} 行结束时间必须晚于开始时间。")
        try:
            duration_factor = float(values[5])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"时间轴第 {position} 行时长系数无效。") from exc
        if not 0.5 <= duration_factor <= 2.0:
            raise ValueError(f"时间轴第 {position} 行时长系数必须在 0.5–2.0。")
        if len(values) >= 8:
            emotion = parse_emotion_override(values[7], index=index)
        else:
            emotion = (
                original.emotion_mode,
                original.emotion_text,
                original.emotion_vector,
                original.emotion_strength,
                original.emotion_use_random,
            )
        result.append(
            replace(
                original,
                role=role,
                language=language,
                start_ms=start_ms,
                end_ms=end_ms,
                duration_factor=duration_factor,
                text=text,
                emotion_mode=emotion[0],
                emotion_text=emotion[1],
                emotion_vector=emotion[2],
                emotion_strength=emotion[3],
                emotion_use_random=emotion[4],
            )
        )
    return result


def apply_timeline_drag_payload(
    lines: Sequence[DialogueLine],
    payload: str | dict[str, Any],
) -> tuple[list[DialogueLine], dict[str, Any]]:
    """Apply one browser drag/resize edit with the same validation as the table."""

    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("拖拽时间轴数据不是有效 JSON。") from exc
    elif isinstance(payload, dict):
        data = dict(payload)
    else:
        raise ValueError("拖拽时间轴数据无效。")
    try:
        line_index = int(data["index"])
        start_ms = int(round(float(data["start_ms"])))
        end_ms = int(round(float(data["end_ms"])))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("拖拽时间轴缺少有效的台词序号或毫秒范围。") from exc
    if start_ms < 0 or end_ms <= start_ms:
        raise ValueError("拖拽后的结束时间必须晚于非负开始时间。")
    if end_ms > 86_400_000:
        raise ValueError("拖拽后的时间不能超过 24 小时。")
    mode = str(data.get("mode") or "move")
    if mode not in {"move", "resize_start", "resize_end", "select"}:
        raise ValueError(f"未知时间轴拖拽模式：{mode}")
    updated = []
    matched = False
    for line in lines:
        if line.index == line_index:
            updated.append(replace(line, start_ms=start_ms, end_ms=end_ms))
            matched = True
        else:
            updated.append(line)
    if not matched:
        raise ValueError(f"拖拽时间轴中不存在第 {line_index} 条台词。")
    normalized = {
        "index": line_index,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "mode": mode,
        "snapped_to_ms": data.get("snapped_to_ms"),
    }
    return updated, normalized


def format_srt_timestamp(milliseconds: int) -> str:
    value = max(0, int(milliseconds))
    hours, remainder = divmod(value, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _report_map(line_reports: Sequence[dict] | None) -> dict[int, dict]:
    return {
        int(item.get("index", position)): item
        for position, item in enumerate(line_reports or (), 1)
        if isinstance(item, dict)
    }


def rewrite_srt(
    lines: Sequence[DialogueLine],
    line_reports: Sequence[dict] | None = None,
    *,
    timing_mode: str = "actual",
    text_mode: str = "asr_passed",
    include_role: bool = True,
) -> tuple[str, dict[str, Any]]:
    if timing_mode not in {"original", "actual"}:
        raise ValueError("字幕时间模式只能是 original 或 actual。")
    if text_mode not in {"original", "asr_passed", "asr_all"}:
        raise ValueError("字幕文本模式只能是 original、asr_passed 或 asr_all。")
    reports = _report_map(line_reports)
    cursor = 0
    blocks = []
    rows = []
    for output_index, line in enumerate(lines, 1):
        report = reports.get(line.index, {})
        timeline = report.get("timeline") or {}
        if timing_mode == "actual" and timeline:
            start_ms = int(timeline.get("actual_start_ms", cursor))
            end_ms = int(timeline.get("actual_end_ms", start_ms + 1))
        elif line.start_ms is not None and line.end_ms is not None:
            start_ms, end_ms = int(line.start_ms), int(line.end_ms)
        else:
            start_ms = cursor
            end_ms = start_ms + max(1, int(report.get("actual_duration_ms", 1000)))
        end_ms = max(start_ms + 1, end_ms)
        cursor = max(cursor, end_ms)
        asr = report.get("asr") or {}
        recognized = str(asr.get("recognized_text") or "").strip()
        use_asr = bool(
            recognized
            and (text_mode == "asr_all" or (text_mode == "asr_passed" and asr.get("passed")))
        )
        text = recognized if use_asr else line.text
        if include_role:
            emotion_tag = format_emotion_override(line)
            role_tag = f"{line.role}|emotion={emotion_tag}" if emotion_tag else line.role
            rendered_text = f"[{role_tag}] {text}"
        else:
            rendered_text = text
        blocks.append(
            f"{output_index}\n{format_srt_timestamp(start_ms)} --> {format_srt_timestamp(end_ms)}\n{rendered_text}"
        )
        rows.append(
            {
                "index": line.index,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "source": "asr" if use_asr else "original",
                "text": text,
            }
        )
    return "\n\n".join(blocks) + ("\n" if blocks else ""), {
        "timing_mode": timing_mode,
        "text_mode": text_mode,
        "include_role": bool(include_role),
        "lines": rows,
    }


def render_timeline_html(
    lines: Sequence[DialogueLine],
    line_reports: Sequence[dict] | None = None,
) -> str:
    if not lines:
        return '<div class="t8-timeline-empty">请先解析台词。</div>'
    reports = _report_map(line_reports)
    entries = []
    cursor = 0
    for line in lines:
        report = reports.get(line.index, {})
        timeline = report.get("timeline") or {}
        start = int(timeline.get("actual_start_ms", line.start_ms if line.start_ms is not None else cursor))
        duration = max(1, int(report.get("actual_duration_ms", line.slot_ms or 1000)))
        end = int(timeline.get("actual_end_ms", line.end_ms if line.end_ms is not None else start + duration))
        end = max(start + 1, end)
        cursor = max(cursor, end)
        entries.append((line, start, end, report.get("asr") or {}))
    total = max(end for _line, _start, end, _asr in entries)
    canvas_total = max(1000, total + max(500, round(total * 0.08)))
    tracks = []
    for line, start, end, asr in entries:
        left = 100 * start / canvas_total
        width = max(0.8, 100 * (end - start) / canvas_total)
        hue = abs(hash(line.role)) % 360
        score = asr.get("similarity")
        score_text = "" if score is None else f" · ASR {float(score):.0%}"
        title = html.escape(
            f"#{line.index} {line.role} | {start}–{end}ms{score_text} | {line.text}",
            quote=True,
        )
        label = html.escape(f"#{line.index} {line.role} · {line.text}")
        word_markers = []
        line_duration_seconds = max((end - start) / 1000.0, 0.001)
        for word in asr.get("word_timestamps") or ():
            word_start = max(0.0, float(word.get("start") or 0.0))
            marker_left = min(100.0, 100.0 * word_start / line_duration_seconds)
            word_label = str(word.get("word") or "")
            word_end = max(word_start, float(word.get("end") or word_start))
            marker_title = html.escape(
                f"{word_label} · {word_start:.3f}–{word_end:.3f}s",
                quote=True,
            )
            word_markers.append(
                f'<span class="t8-timeline-word" title="{marker_title}" '
                f'data-snap-ms="{start + round(word_start * 1000)}" '
                f'data-snap-end-ms="{start + round(word_end * 1000)}" '
                f'style="position:absolute;left:{marker_left:.3f}%;top:0;bottom:0;'
                'width:2px;background:#38bdf8"></span>'
            )
        tracks.append(
            f'<div class="t8-timeline-track" data-index="{line.index}">'
            f'<div class="t8-timeline-bar" title="{title}" '
            f'data-index="{line.index}" data-start-ms="{start}" data-end-ms="{end}" '
            f'data-snap-start-ms="{start}" data-snap-end-ms="{end}" '
            f'style="left:{left:.3f}%;width:{width:.3f}%;background:hsl({hue} 72% 62%)">'
            '<span class="t8-timeline-handle t8-timeline-handle-start" title="拖动左边界"></span>'
            f'<span class="t8-timeline-bar-label">{label}</span>{"".join(word_markers)}'
            '<span class="t8-timeline-handle t8-timeline-handle-end" title="拖动右边界"></span>'
            '</div>'
            "</div>"
        )
    seconds = total / 1000
    canvas_seconds = canvas_total / 1000
    return (
        f'<div class="t8-timeline" data-total-ms="{canvas_total}" data-snap-threshold-px="12">'
        f'<div class="t8-timeline-scale">0s <span>成品总时长 {seconds:.2f}s</span> 可编辑至 {canvas_seconds:.2f}s</div>'
        + "".join(tracks)
        + '<div class="t8-timeline-hint">拖动音频块可平移；拖左右手柄可改边界；靠近其他边界或蓝色 ASR 逐字点会自动吸附，按住 Alt 可临时关闭吸附。拖完会同步上方表格；点击音频块可选中并单独重做。</div>'
        + "</div>"
    )


def timeline_json(lines: Sequence[DialogueLine], line_reports: Sequence[dict] | None = None) -> str:
    return json.dumps(
        {
            "lines": [line.to_dict() for line in lines],
            "reports": list(line_reports or ()),
        },
        ensure_ascii=False,
        indent=2,
    )


__all__ = [
    "TIMELINE_HEADERS",
    "apply_timeline_edits",
    "apply_timeline_drag_payload",
    "format_srt_timestamp",
    "move_timeline_row",
    "render_timeline_html",
    "rewrite_srt",
    "timeline_json",
    "timeline_rows",
]
