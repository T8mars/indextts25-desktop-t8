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
            line.start_ms,
            line.end_ms,
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
        start_ms = _optional_ms(values[3], f"时间轴第 {position} 行开始时间")
        end_ms = _optional_ms(values[4], f"时间轴第 {position} 行结束时间")
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
        original = next(line for line in original_lines if line.index == index)
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
    tracks = []
    for line, start, end, asr in entries:
        left = 100 * start / total
        width = max(0.8, 100 * (end - start) / total)
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
                f'style="position:absolute;left:{marker_left:.3f}%;top:0;bottom:0;'
                'width:2px;background:#38bdf8"></span>'
            )
        tracks.append(
            '<div class="t8-timeline-track">'
            f'<div class="t8-timeline-bar" title="{title}" '
            f'style="left:{left:.3f}%;width:{width:.3f}%;background:hsl({hue} 72% 62%)">'
            f'{label}{"".join(word_markers)}</div>'
            "</div>"
        )
    seconds = total / 1000
    return (
        '<div class="t8-timeline">'
        f'<div class="t8-timeline-scale">0s <span>总时长 {seconds:.2f}s</span> {seconds:.2f}s</div>'
        + "".join(tracks)
        + '<div class="t8-timeline-hint">提交上方表格单元格后轨道自动刷新；点击一行可单独重做并合入。仅修改时间可直接重新混音。蓝线为 ASR 逐字时间点。</div>'
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
    "format_srt_timestamp",
    "render_timeline_html",
    "rewrite_srt",
    "timeline_json",
    "timeline_rows",
]
