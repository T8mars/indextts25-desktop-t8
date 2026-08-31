"""Portable dialogue, SRT and timeline helpers for IndexTTS 2.5 frontends."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Sequence

import torch


LANGUAGES = {"ZH", "EN", "JA", "ES", "AR"}
EMOTION_OVERRIDE_MODES = {"inherit", "speaker", "vector", "text"}
_TIMESTAMP = re.compile(
    r"^(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})[,.](?P<ms>\d{3})$"
)
_SRT_RANGE = re.compile(r"^\s*(\S+)\s*-->\s*(\S+)(?:\s+.*)?$")
_BRACKET_ROLE = re.compile(r"^\s*[\[【](?P<role>[^\]】]+)[\]】]\s*(?P<text>[\s\S]*)$")
_COLON_ROLE = re.compile(r"^\s*(?P<role>[^:：|]{1,40})\s*[:：]\s*(?P<text>[\s\S]+)$")


@dataclass(frozen=True, slots=True)
class DialogueLine:
    index: int
    role: str
    text: str
    language: str = "ZH"
    start_ms: int | None = None
    end_ms: int | None = None
    duration_factor: float = 1.0
    emotion_mode: str = "inherit"
    emotion_text: str = ""
    emotion_vector: tuple[float, ...] | None = None
    emotion_strength: float = 1.0
    emotion_use_random: bool = False

    @property
    def slot_ms(self) -> int | None:
        if self.start_ms is None or self.end_ms is None:
            return None
        return max(0, self.end_ms - self.start_ms)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TimelinePlacement:
    index: int
    requested_start_ms: int
    actual_start_ms: int
    actual_end_ms: int
    overlap_ms: int
    overrun_ms: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def parse_timestamp(value: str) -> int:
    match = _TIMESTAMP.match(str(value).strip())
    if not match:
        raise ValueError(f"无效的 SRT 时间：{value}")
    hours, minutes, seconds, millis = (int(match.group(key)) for key in ("h", "m", "s", "ms"))
    if minutes > 59 or seconds > 59:
        raise ValueError(f"无效的 SRT 时间：{value}")
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def parse_emotion_override(
    value: Any,
    *,
    index: int | None = None,
) -> tuple[str, str, tuple[float, ...] | None, float, bool]:
    """Parse one optional per-line emotion override.

    Accepted compact values are ``inherit``, ``speaker``, ``text:description`` and
    ``vector:v1,...,v8``. JSON mappings/lists are also accepted so timeline and
    batch JSON users can set strength and random-vector sampling explicitly.
    """

    label = f"第 {index} 条台词" if index is not None else "逐句情感"
    if value is None:
        value = ""
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("{") or raw.startswith("["):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{label} JSON 格式错误：{exc.msg}") from exc
        else:
            for prefix in ("emotion=", "emotion:", "情感=", "情感："):
                if raw.lower().startswith(prefix.lower()):
                    raw = raw[len(prefix) :].strip()
                    break
            raw, compact_options = _split_compact_emotion_options(raw)
            lowered = raw.lower()
            if lowered in {"", "inherit", "default", "继承", "角色默认"}:
                value = {"mode": "inherit"}
            elif lowered in {"speaker", "voice", "跟随音色", "音色"}:
                value = {"mode": "speaker"}
            elif lowered.startswith(("vector:", "vector：", "向量:", "向量：")):
                value = {"mode": "vector", "vector": re.split(r"[:：]", raw, maxsplit=1)[1]}
            elif lowered.startswith(("text:", "text：", "文本:", "文本：")):
                value = {"mode": "text", "text": re.split(r"[:：]", raw, maxsplit=1)[1]}
            else:
                value = {"mode": "text", "text": raw}
            value.update(compact_options)
    elif isinstance(value, (list, tuple)):
        value = {"mode": "vector", "vector": value}
    if not isinstance(value, dict):
        raise ValueError(f"{label}必须是文本、八维数组或 JSON 对象。")

    mode = str(value.get("mode") or value.get("emotion_mode") or "").strip().lower()
    vector_value = value.get("vector", value.get("emotion_vector"))
    text = str(value.get("text", value.get("emotion_text", "")) or "").strip()
    if not mode:
        mode = "vector" if vector_value is not None else ("text" if text else "inherit")
    mode_aliases = {
        "default": "inherit",
        "role": "inherit",
        "voice": "speaker",
        "description": "text",
    }
    mode = mode_aliases.get(mode, mode)
    if mode not in EMOTION_OVERRIDE_MODES:
        raise ValueError(f"{label}模式无效：{mode}；可用 inherit、speaker、text、vector。")

    try:
        strength = float(value.get("strength", value.get("emotion_strength", 1.0)))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}强度必须是 0–1 的数值。") from exc
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"{label}强度必须在 0–1。")
    use_random_value = value.get("use_random", value.get("emotion_use_random", False))
    if isinstance(use_random_value, bool):
        use_random = use_random_value
    elif use_random_value is None:
        use_random = False
    elif isinstance(use_random_value, (int, float)) and use_random_value in {0, 1}:
        use_random = bool(use_random_value)
    elif isinstance(use_random_value, str) and use_random_value.strip().lower() in {
        "true",
        "false",
        "1",
        "0",
        "是",
        "否",
        "开",
        "关",
    }:
        use_random = use_random_value.strip().lower() in {"true", "1", "是", "开"}
    else:
        raise ValueError(f"{label}的 random/use_random 必须是 true 或 false。")

    vector: tuple[float, ...] | None = None
    if mode == "vector":
        if isinstance(vector_value, str):
            vector_value = [item for item in re.split(r"[,，\s]+", vector_value.strip()) if item]
        try:
            vector = tuple(float(item) for item in (vector_value or ()))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label}八维向量必须全部是数值。") from exc
        if len(vector) != 8:
            raise ValueError(f"{label}八维向量必须正好包含 8 个数值。")
        if any(not 0.0 <= item <= 1.0 for item in vector):
            raise ValueError(f"{label}八维向量的每个数值必须在 0–1。")
        total = sum(vector)
        if total > 0.8:
            vector = tuple(item * 0.8 / total for item in vector)
    return mode, text, vector, strength, use_random


def _split_compact_emotion_options(raw: str) -> tuple[str, dict[str, Any]]:
    """Peel recognized ``;key=value`` options from compact emotion text."""

    remaining = str(raw)
    options: dict[str, Any] = {}
    while True:
        match = re.search(
            r"[;；]\s*(strength|强度|random|随机|use_random)\s*[=:：]\s*([^;；]*?)\s*$",
            remaining,
            flags=re.IGNORECASE,
        )
        if not match:
            break
        key = match.group(1).lower()
        option_key = "strength" if key in {"strength", "强度"} else "use_random"
        options[option_key] = match.group(2)
        remaining = remaining[: match.start()].rstrip()
    return remaining.strip(), options


def format_emotion_override(line: DialogueLine) -> str:
    """Return a compact, editable representation for tables and SRT tags."""

    if line.emotion_mode == "inherit":
        base = ""
    elif line.emotion_mode == "speaker":
        base = "speaker"
    elif line.emotion_mode == "text":
        base = f"text:{line.emotion_text}"
    elif line.emotion_mode == "vector":
        base = "vector:" + ",".join(f"{item:g}" for item in line.emotion_vector or ())
    else:
        base = line.emotion_mode
    options = []
    if line.emotion_strength != 1.0:
        options.append(f"strength={line.emotion_strength:g}")
    if line.emotion_use_random:
        options.append("random=true")
    if not base and options:
        base = "inherit"
    return ";".join([part for part in (base, *options) if part])


def split_role_text_emotion(text: str, default_role: str) -> tuple[str, str, Any]:
    normalized = str(text).strip()
    for pattern in (_BRACKET_ROLE, _COLON_ROLE):
        match = pattern.match(normalized)
        if match:
            role = match.group("role").strip()
            body = match.group("text").strip()
            if role and body:
                emotion: Any = None
                if pattern is _BRACKET_ROLE and "|" in role:
                    role, metadata = (part.strip() for part in role.split("|", 1))
                    emotion = metadata
                return role, body, emotion
    return str(default_role).strip(), normalized, None


def split_role_text(text: str, default_role: str) -> tuple[str, str]:
    role, body, _emotion = split_role_text_emotion(text, default_role)
    return role, body


def _normalize_language(value: Any, default: str = "ZH") -> str:
    language = str(value or default).strip().upper()
    if language not in LANGUAGES:
        raise ValueError(f"不支持的语言：{language}")
    return language


def parse_srt(content: str, default_role: str = "旁白", default_language: str = "ZH") -> list[DialogueLine]:
    """Parse ordinary SRT, including BOM, CRLF, multiline text and dot milliseconds."""
    raw = str(content).lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        raise ValueError("SRT 内容不能为空。")
    blocks = re.split(r"\n\s*\n", raw)
    result: list[DialogueLine] = []
    for block_number, block in enumerate(blocks, 1):
        rows = [row.rstrip() for row in block.split("\n") if row.strip()]
        if not rows:
            continue
        range_index = next((i for i, row in enumerate(rows[:2]) if "-->" in row), -1)
        if range_index < 0:
            raise ValueError(f"SRT 第 {block_number} 段缺少时间范围。")
        match = _SRT_RANGE.match(rows[range_index])
        if not match:
            raise ValueError(f"SRT 第 {block_number} 段时间格式错误：{rows[range_index]}")
        start_ms, end_ms = parse_timestamp(match.group(1)), parse_timestamp(match.group(2))
        if end_ms <= start_ms:
            raise ValueError(f"SRT 第 {block_number} 段结束时间必须晚于开始时间。")
        text = "\n".join(rows[range_index + 1 :]).strip()
        if not text:
            raise ValueError(f"SRT 第 {block_number} 段没有字幕文本。")
        role, text, emotion = split_role_text_emotion(text, default_role)
        emotion_mode, emotion_text, emotion_vector, emotion_strength, emotion_use_random = (
            parse_emotion_override(emotion, index=len(result) + 1)
        )
        result.append(
            DialogueLine(
                index=len(result) + 1,
                role=role,
                text=text,
                language=_normalize_language(default_language),
                start_ms=start_ms,
                end_ms=end_ms,
                emotion_mode=emotion_mode,
                emotion_text=emotion_text,
                emotion_vector=emotion_vector,
                emotion_strength=emotion_strength,
                emotion_use_random=emotion_use_random,
            )
        )
    if not result:
        raise ValueError("SRT 中没有可生成的字幕。")
    return result


def _line_from_mapping(value: dict[str, Any], index: int, default_role: str, default_language: str) -> DialogueLine:
    role = str(value.get("role") or default_role).strip()
    text = str(value.get("text") or "").strip()
    if not role or not text:
        raise ValueError(f"第 {index} 条台词必须包含 role 和 text。")
    try:
        factor = float(value.get("duration_factor", 1.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"第 {index} 条台词最后一列必须是时长系数，例如：角色|台词|ZH|1.0。"
        ) from exc
    if not 0.5 <= factor <= 2.0:
        raise ValueError(
            f"第 {index} 条台词最后一列是时长系数（无单位倍率），必须在 0.5–2.0；"
            "它不是秒数，也不限制台词长度。正确示例：角色|台词|ZH|1.0。"
        )
    start = value.get("start_ms")
    end = value.get("end_ms")
    emotion_value = value.get("emotion")
    if emotion_value is None and any(
        key in value
        for key in (
            "emotion_mode",
            "emotion_text",
            "emotion_vector",
            "emotion_strength",
            "emotion_use_random",
        )
    ):
        emotion_value = value
    emotion_mode, emotion_text, emotion_vector, emotion_strength, emotion_use_random = (
        parse_emotion_override(emotion_value, index=index)
    )
    return DialogueLine(
        index=index,
        role=role,
        text=text,
        language=_normalize_language(value.get("language"), default_language),
        start_ms=None if start is None else int(start),
        end_ms=None if end is None else int(end),
        duration_factor=factor,
        emotion_mode=emotion_mode,
        emotion_text=emotion_text,
        emotion_vector=emotion_vector,
        emotion_strength=emotion_strength,
        emotion_use_random=emotion_use_random,
    )


def _split_batch_fields(line: str) -> list[str]:
    """Split five batch fields without breaking ``<word|reading>`` annotations."""

    fields: list[str] = []
    current: list[str] = []
    annotation_depth = 0
    escaped = False
    for character in line:
        if escaped:
            escaped_value = {"n": "\n", "r": "\r", "|": "|", "\\": "\\"}.get(
                character, "\\" + character
            )
            current.append(escaped_value)
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == "<":
            annotation_depth += 1
        elif character == ">" and annotation_depth:
            annotation_depth -= 1
        if character == "|" and annotation_depth == 0 and len(fields) < 4:
            fields.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    if escaped:
        current.append("\\")
    fields.append("".join(current).strip())
    return fields


def _escape_batch_field(value: Any) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "\\n")
        .replace("|", "\\|")
    )


def format_batch_script(lines: Sequence[DialogueLine]) -> str:
    """Serialize dialogue as editable, one-line-per-sentence batch text."""

    rendered = []
    for line in lines:
        rendered.append(
            "|".join(
                _escape_batch_field(value)
                for value in (
                    line.role,
                    line.text,
                    line.language,
                    f"{line.duration_factor:g}",
                    format_emotion_override(line),
                )
            )
        )
    return "\n".join(rendered)


def parse_batch_script(content: str, default_role: str = "旁白", default_language: str = "ZH") -> list[DialogueLine]:
    """Parse JSON or ``角色|台词|语言|时长系数|逐句情感`` text scripts."""
    raw = str(content).lstrip("\ufeff").strip()
    if not raw:
        raise ValueError("批量台词不能为空。")
    if raw.startswith("["):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON 台词格式错误：{exc.msg}（第 {exc.lineno} 行）") from exc
        if not isinstance(payload, list):
            raise ValueError("JSON 台词必须是数组。")
        invalid = next((index for index, item in enumerate(payload, 1) if not isinstance(item, dict)), None)
        if invalid is not None:
            raise ValueError(f"JSON 第 {invalid} 条台词必须是对象。")
        result = [
            _line_from_mapping(item, index, default_role, default_language)
            for index, item in enumerate(payload, 1)
        ]
        if not result:
            raise ValueError("JSON 台词数组不能为空。")
        return result

    result: list[DialogueLine] = []
    for source_line, raw_line in enumerate(raw.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = _split_batch_fields(line)
        if len(parts) < 2:
            role, text = split_role_text(line, default_role)
            if role == default_role and text == line:
                raise ValueError(f"第 {source_line} 行应为：角色|台词|语言|时长系数|逐句情感（可选）")
            parts = [role, text]
        role, text = parts[0], parts[1]
        language = parts[2] if len(parts) >= 3 and parts[2] else default_language
        factor = parts[3] if len(parts) >= 4 and parts[3] else 1.0
        emotion = parts[4] if len(parts) >= 5 and parts[4] else None
        result.append(
            _line_from_mapping(
                {
                    "role": role,
                    "text": text,
                    "language": language,
                    "duration_factor": factor,
                    "emotion": emotion,
                },
                len(result) + 1,
                default_role,
                default_language,
            )
        )
    if not result:
        raise ValueError("批量台词中没有可生成内容。")
    return result


def assign_sequential_slots(lines: Sequence[DialogueLine], gap_ms: int = 200) -> list[DialogueLine]:
    cursor = 0
    result: list[DialogueLine] = []
    for line in lines:
        start = cursor if line.start_ms is None else line.start_ms
        end = start if line.end_ms is None else line.end_ms
        result.append(replace(line, start_ms=start, end_ms=end))
        cursor = max(cursor, end) + max(0, int(gap_ms))
    return result


def missing_roles(lines: Iterable[DialogueLine], roles: Iterable[str]) -> list[str]:
    known = {str(role).strip() for role in roles}
    return sorted({line.role for line in lines if line.role not in known})


def fit_duration_factor(current_factor: float, actual_ms: float, target_ms: float) -> float:
    if actual_ms <= 0 or target_ms <= 0:
        return max(0.5, min(2.0, float(current_factor)))
    return max(0.5, min(2.0, float(current_factor) * float(target_ms) / float(actual_ms)))


def compose_timeline(
    clips: Sequence[torch.Tensor],
    lines: Sequence[DialogueLine],
    sample_rate: int,
    policy: str = "shift",
    gap_ms: int = 0,
) -> tuple[torch.Tensor, list[TimelinePlacement]]:
    """Compose BCT/CT/T clips using `shift` (no overlap) or `overlay` (preserve starts)."""
    if len(clips) != len(lines):
        raise ValueError("音频片段数量与台词数量不一致。")
    if policy not in {"shift", "overlay"}:
        raise ValueError("时间轴策略只能是 shift 或 overlay。")
    if not clips:
        return torch.zeros((1, 1, 0), dtype=torch.float32), []

    normalized: list[torch.Tensor] = []
    channels = 1
    for clip in clips:
        tensor = torch.as_tensor(clip).detach().to(dtype=torch.float32, device="cpu")
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 3:
            raise ValueError("音频张量必须为 T、CT 或 BCT。")
        if tensor.shape[0] != 1:
            tensor = tensor[:1]
        channels = max(channels, int(tensor.shape[1]))
        normalized.append(tensor)

    starts: list[int] = []
    reports: list[TimelinePlacement] = []
    cursor = 0
    for line, clip in zip(lines, normalized):
        requested = max(0, int(line.start_ms or 0) * sample_rate // 1000)
        start = max(requested, cursor) if policy == "shift" else requested
        end = start + int(clip.shape[-1])
        overlap = max(0, cursor - requested) if policy == "overlay" else max(0, start - requested)
        slot_end = None if line.end_ms is None else int(line.end_ms) * sample_rate // 1000
        overrun = 0 if slot_end is None else max(0, end - slot_end)
        reports.append(
            TimelinePlacement(
                index=line.index,
                requested_start_ms=round(requested * 1000 / sample_rate),
                actual_start_ms=round(start * 1000 / sample_rate),
                actual_end_ms=round(end * 1000 / sample_rate),
                overlap_ms=round(overlap * 1000 / sample_rate),
                overrun_ms=round(overrun * 1000 / sample_rate),
            )
        )
        starts.append(start)
        cursor = max(cursor, end + max(0, int(gap_ms)) * sample_rate // 1000)

    output = torch.zeros((1, channels, max(start + clip.shape[-1] for start, clip in zip(starts, normalized))))
    active = torch.zeros(output.shape[-1])
    for start, clip in zip(starts, normalized):
        if clip.shape[1] == 1 and channels > 1:
            clip = clip.repeat(1, channels, 1)
        elif clip.shape[1] != channels:
            clip = clip[:, :1, :].repeat(1, channels, 1)
        end = start + clip.shape[-1]
        output[..., start:end] += clip
        active[start:end] += 1
    if policy == "overlay":
        output /= active.clamp_min(1).view(1, 1, -1)
    return output.clamp(-1.0, 1.0), reports


def script_report(lines: Sequence[DialogueLine]) -> str:
    return json.dumps([line.to_dict() for line in lines], ensure_ascii=False, indent=2)


__all__ = [
    "DialogueLine",
    "EMOTION_OVERRIDE_MODES",
    "TimelinePlacement",
    "assign_sequential_slots",
    "compose_timeline",
    "fit_duration_factor",
    "format_batch_script",
    "format_emotion_override",
    "missing_roles",
    "parse_batch_script",
    "parse_emotion_override",
    "parse_srt",
    "parse_timestamp",
    "script_report",
    "split_role_text",
]
