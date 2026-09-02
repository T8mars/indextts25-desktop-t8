"""Desktop generation controls shared by the Gradio UI and its tests."""

from __future__ import annotations

import re
import warnings
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

import torch
import torch.nn.functional as F

from indextts.utils.common import fade_out_pcm_tail


LANGUAGE_AUTO_LIMITS = {"ZH": 120, "EN": 60, "JA": 100, "ES": 60, "AR": 80}
LATIN_LONG_TEXT_LANGUAGES = frozenset({"EN", "ES"})
LONG_TEXT_RETRY_MIN_TOKENS = 24
PAUSE_PRESETS = {
    "off": (0, 0, 0),
    "natural": (0, 260, 500),
    "narration": (120, 360, 700),
    "dialogue": (80, 250, 450),
}
POSTPROCESS_PRESETS = ("off", "voice_clarity", "clear_narration", "deharsh", "warm", "normalize")
DURATION_MODES = ("off", "native", "natural", "pad", "exact")

_EXPLICIT_PAUSE = re.compile(
    r"(?:<|\[)\s*pause\s*(?:=|:)\s*(\d+(?:\.\d+)?)\s*(ms|s)?\s*(?:>|\])",
    re.IGNORECASE,
)
_PROTECTED = re.compile(r"<[^>\n]+>")
_BOUNDARY = re.compile(
    r"(?:<|\[)\s*pause\s*(?:=|:)\s*\d+(?:\.\d+)?\s*(?:ms|s)?\s*(?:>|\])"
    r"|<[^>\n]+>|\r?\n(?:[ \t]*\r?\n)*|[。！？!?；;：:]|(?<!\d)\.(?!\d)|[，、]|(?<!\d),(?!\d)",
    re.IGNORECASE,
)
_TERMINAL_PUNCTUATION = frozenset("。！？!?；;：:….")
_TRAILING_CLOSERS = frozenset("'\"’”）)]}】》〉」』")
_REPEATED_SPOKEN_CHARACTER = re.compile(r"([\u3400-\u9fff\u3040-\u30ff0-9])\1{2,}")


@dataclass(frozen=True, slots=True)
class DesktopSpeechChunk:
    text: str
    pause_after_ms: int = 0
    pause_before_ms: int = 0


@dataclass(frozen=True, slots=True)
class DesktopGenerationPlan:
    language: str
    max_tokens: int
    chunks: tuple[DesktopSpeechChunk, ...]
    segments: tuple[dict[str, Any], ...]
    pause_preset: str

    @property
    def total_pause_ms(self) -> int:
        return sum(chunk.pause_before_ms + chunk.pause_after_ms for chunk in self.chunks)

    @property
    def max_segment_tokens(self) -> int:
        return max((int(item["token_count"]) for item in self.segments), default=0)

    @property
    def gpt_accel_risk(self) -> bool:
        # Kept for report/workflow compatibility. The synthetic-prompt cache
        # markers are now reset inside AccelInferenceEngine before prefill.
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "max_tokens": self.max_tokens,
            "pause_preset": self.pause_preset,
            "speech_blocks": len(self.chunks),
            "segment_count": len(self.segments),
            "total_pause_ms": self.total_pause_ms,
            "gpt_accel_risk": self.gpt_accel_risk,
            "gpt_accel_cache_fix": True,
            "chunks": [asdict(item) for item in self.chunks],
            "segments": list(self.segments),
        }


def latin_word_count(text: str) -> int:
    """Count English/Spanish words without treating punctuation as speech."""

    return len(re.findall(r"[^\W_]+(?:['’\-][^\W_]+)*", str(text or ""), re.UNICODE))


def ensure_terminal_punctuation(text: str, language: str) -> str:
    """Give an unterminated retry an explicit sentence boundary.

    IndexTTS occasionally merges the final spoken token with EOS. This helper
    is intentionally used only by a verified retry, so the normal first pass
    keeps the user's exact prosody.
    """

    value = str(text or "")
    stripped = value.rstrip()
    if not stripped:
        return value
    whitespace = value[len(stripped):]
    boundary = len(stripped)
    while boundary and stripped[boundary - 1] in _TRAILING_CLOSERS:
        boundary -= 1
    if boundary and stripped[boundary - 1] in _TERMINAL_PUNCTUATION:
        return value
    mark = "。" if str(language or "").strip().upper() in {"ZH", "ZHEN", "JA"} else "."
    return stripped[:boundary] + mark + stripped[boundary:] + whitespace


def separate_repeated_characters(text: str, language: str) -> str:
    """Add soft token boundaries to repeated CJK characters/digits on retries."""

    value = str(text or "")
    if str(language or "").strip().upper() not in {"ZH", "ZHEN", "JA"}:
        return value
    return _REPEATED_SPOKEN_CHARACTER.sub(
        lambda match: " ".join(match.group(0)),
        value,
    )


def long_text_retry_limit(language: str, current_limit: int) -> int:
    """Return a materially safer token limit for one guarded retry."""

    current = max(1, int(current_limit))
    if str(language).strip().upper() not in LATIN_LONG_TEXT_LANGUAGES:
        return current
    return max(
        LONG_TEXT_RETRY_MIN_TOKENS,
        min(current - 1, int(round(current * 2 / 3))),
    )


def assess_long_text_result(
    text: str,
    language: str,
    token_count: int,
    duration_seconds: float,
    duration_factor: float = 1.0,
    warning_messages: tuple[str, ...] | list[str] = (),
) -> list[str]:
    """Detect strong long-Latin collapse signals while avoiding normal speech variance."""

    normalized = str(language).strip().upper()
    if normalized not in LATIN_LONG_TEXT_LANGUAGES or int(token_count) < 32:
        return []
    reasons: list[str] = []
    lowered_warnings = "\n".join(str(item).lower() for item in warning_messages)
    if "max_mel_tokens" in lowered_warnings and (
        "exceed" in lowered_warnings or "stopped" in lowered_warnings
    ):
        reasons.append("max_mel_tokens_reached")
    seconds = float(duration_seconds)
    if not isfinite(seconds) or seconds <= 0:
        reasons.append("invalid_audio_duration")
        return reasons
    words = latin_word_count(text)
    if words < 24:
        return reasons
    factor = max(0.5, min(2.0, float(duration_factor)))
    # Even unusually fast narration rarely exceeds 7.5 words/s. The generous
    # upper bound catches runaway decodes without rejecting dramatic pauses.
    minimum_seconds = max(1.5, words * factor / 7.5)
    maximum_seconds = max(20.0, words * factor / 0.65 + 6.0)
    if seconds < minimum_seconds:
        reasons.append("suspiciously_short_for_latin_text")
    elif seconds > maximum_seconds:
        reasons.append("suspiciously_long_for_latin_text")
    return reasons


def run_with_long_text_guard(
    generate,
    duration_reader,
    *,
    text: str,
    language: str,
    token_count: int,
    max_tokens: int,
    duration_factor: float = 1.0,
    check_duration: bool = True,
):
    """Generate once and retry a suspicious long EN/ES result with smaller segments."""

    def invoke(limit: int):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            result = generate(int(limit))
        messages = tuple(str(item.message) for item in caught)
        duration = float(duration_reader(result))
        reasons = assess_long_text_result(
            text,
            language,
            token_count,
            duration if check_duration else max(duration, 10_000.0),
            duration_factor,
            messages,
        )
        if not check_duration:
            reasons = [item for item in reasons if item == "max_mel_tokens_reached"]
        return result, duration, messages, reasons

    requested_limit = int(max_tokens)
    first, first_duration, first_warnings, first_reasons = invoke(requested_limit)
    retry_limit = long_text_retry_limit(language, requested_limit)
    report = {
        "enabled": (
            str(language).strip().upper() in LATIN_LONG_TEXT_LANGUAGES
            and int(token_count) >= 32
        ),
        "requested_limit": requested_limit,
        "used_limit": requested_limit,
        "retried": False,
        "first_duration_seconds": round(first_duration, 4),
        "first_reasons": first_reasons,
        "first_warnings": list(first_warnings),
    }
    if not first_reasons or retry_limit >= requested_limit:
        return first, report

    second, second_duration, second_warnings, second_reasons = invoke(retry_limit)
    report.update(
        retried=True,
        used_limit=retry_limit,
        retry_duration_seconds=round(second_duration, 4),
        retry_reasons=second_reasons,
        retry_warnings=list(second_warnings),
        recovered=not second_reasons,
    )
    return second, report


def effective_segment_limit(language: str, mode: str, custom_limit: int) -> int:
    language = str(language).upper()
    if language not in LANGUAGE_AUTO_LIMITS:
        raise ValueError(f"不支持的语言：{language}")
    if mode == "auto":
        return LANGUAGE_AUTO_LIMITS[language]
    if mode != "custom" or not 20 <= int(custom_limit) <= 300:
        raise ValueError("自定义每段 Token 必须在 20–300。")
    return int(custom_limit)


def _pause_values(preset: str, comma: int, sentence: int, paragraph: int) -> tuple[int, int, int]:
    if preset == "custom":
        values = (int(comma), int(sentence), int(paragraph))
    elif preset in PAUSE_PRESETS:
        values = PAUSE_PRESETS[preset]
    else:
        raise ValueError("未知停顿预设。")
    if any(value < 0 or value > 5000 for value in values):
        raise ValueError("标点停顿必须在 0–5000 毫秒。")
    return values


def split_speech_chunks(
    text: str,
    preset: str,
    comma_pause_ms: int,
    sentence_pause_ms: int,
    paragraph_pause_ms: int,
) -> tuple[DesktopSpeechChunk, ...]:
    source = str(text or "").strip()
    if not source:
        raise ValueError("待合成文本不能为空。")
    comma, sentence, paragraph = _pause_values(
        preset, comma_pause_ms, sentence_pause_ms, paragraph_pause_ms
    )
    chunks: list[DesktopSpeechChunk] = []
    buffer: list[str] = []
    pending_leading_pause = 0

    def flush(pause: int = 0) -> None:
        nonlocal pending_leading_pause
        content = "".join(buffer).strip()
        buffer.clear()
        if content:
            chunks.append(DesktopSpeechChunk(content, int(pause), pending_leading_pause))
            pending_leading_pause = 0
        elif chunks and pause:
            previous = chunks[-1]
            chunks[-1] = DesktopSpeechChunk(
                previous.text,
                max(previous.pause_after_ms, int(pause)),
                previous.pause_before_ms,
            )
        elif pause:
            pending_leading_pause = max(pending_leading_pause, int(pause))

    position = 0
    for match in _BOUNDARY.finditer(source):
        buffer.append(source[position : match.start()])
        token = match.group(0)
        explicit = _EXPLICIT_PAUSE.fullmatch(token)
        if explicit:
            value = float(explicit.group(1))
            millis = round(value if (explicit.group(2) or "s").lower() == "ms" else value * 1000)
            if not 0 <= millis <= 30_000:
                raise ValueError("显式停顿必须在 0–30 秒。")
            flush(millis)
        elif _PROTECTED.fullmatch(token):
            buffer.append(token)
        elif "\n" in token or "\r" in token:
            flush(paragraph)
        elif token in "，、,":
            buffer.append(token)
            if comma:
                flush(comma)
        else:
            buffer.append(token)
            if sentence:
                flush(sentence)
        position = match.end()
    buffer.append(source[position:])
    flush()
    if not chunks:
        raise ValueError("文本只包含停顿标记，没有可合成内容。")
    return tuple(chunks)


def build_desktop_plan(
    tts: Any,
    text: str,
    language: str,
    segmentation_mode: str,
    max_text_tokens: int,
    pause_preset: str,
    comma_pause_ms: int,
    sentence_pause_ms: int,
    paragraph_pause_ms: int,
) -> DesktopGenerationPlan:
    language = str(language).upper()
    limit = effective_segment_limit(language, segmentation_mode, max_text_tokens)
    chunks = split_speech_chunks(
        text, pause_preset, comma_pause_ms, sentence_pause_ms, paragraph_pause_ms
    )
    prefix = f"<|{language.lower()}|> "
    segments: list[dict[str, Any]] = []
    for block_index, chunk in enumerate(chunks, 1):
        parts = tts.split_text_by_tokens(chunk.text, limit, prefix)
        for part_index, part in enumerate(parts):
            segments.append({
                "index": len(segments) + 1,
                "speech_block": block_index,
                "token_count": len(tts.tokenizer.encode(prefix + part, allowed_special="all")),
                "text": part,
                "pause_after_ms": chunk.pause_after_ms if part_index == len(parts) - 1 else 0,
                "pause_before_ms": chunk.pause_before_ms if part_index == 0 else 0,
            })
    return DesktopGenerationPlan(language, limit, chunks, tuple(segments), pause_preset)


def normalize_preflight_text(tts: Any, text: str, language: str, enabled: bool = True) -> str:
    """Mirror the model's target-text normalization for an inspectable preview."""

    source = str(text or "").strip()
    if not source or not enabled:
        return source
    language_code = str(language or "ZH").strip().upper()
    processor = getattr(tts, "text_process", None)
    try:
        if processor is not None and hasattr(processor, "clean_pattern"):
            source = processor.clean_pattern.sub(
                lambda match: processor.char_rep_map[match.group()],
                source,
            )
        if language_code in {"ZH", "ZHEN", "EN"} and processor is not None:
            normalized = processor.normalize(source)
            if str(normalized or "").strip():
                source = str(normalized).strip()
        elif language_code in {"JA", "ES"}:
            from indextts.utils.nemo_tn import normalize_text as nemo_text_normalize

            normalized = nemo_text_normalize(source, language_code.lower())
            if str(normalized or "").strip():
                source = str(normalized).strip()
    except Exception as exc:
        warnings.warn(f"长文本预检无法执行文本归一化，已保留原文：{exc}", RuntimeWarning)
    if language_code in {"JA", "ZH", "ZHEN", "EN"}:
        source = source.lower()
    elif language_code == "ES":
        source = source.upper()
    return source


def estimate_segment_seconds(text: str, language: str) -> float:
    """Estimate speaking time conservatively for preflight UX, not synthesis timing."""

    source = str(text or "").strip()
    language_code = str(language or "ZH").strip().upper()
    if language_code in {"EN", "ES", "AR"}:
        units = max(1, latin_word_count(source))
        rate = 2.55 if language_code == "AR" else 2.75
    else:
        units = max(1, len(re.sub(r"\s|[，。！？、,.!?;；:：'\"“”‘’]", "", source)))
        rate = 4.6 if language_code == "ZH" else 5.0
    return max(0.35, units / rate)


def preflight_plan_rows(plan: DesktopGenerationPlan) -> list[list[Any]]:
    """Create human-readable duration and risk rows for a generation plan."""

    rows: list[list[Any]] = []
    for item in plan.segments:
        token_count = int(item["token_count"])
        ratio = token_count / max(1, int(plan.max_tokens))
        estimated = estimate_segment_seconds(item["text"], plan.language)
        risks: list[str] = []
        if ratio >= 0.9:
            risks.append("接近 Token 上限")
        elif ratio >= 0.75:
            risks.append("Token 较高")
        if estimated >= 22:
            risks.append("预计偏长")
        elif estimated >= 14:
            risks.append("建议留意时长")
        if ratio >= 0.9 or estimated >= 22:
            risk = "高：" + "、".join(risks)
        elif risks:
            risk = "中：" + "、".join(risks)
        else:
            risk = "低"
        rows.append([
            item["index"],
            item["speech_block"],
            token_count,
            round(estimated, 1),
            risk,
            item["pause_before_ms"],
            item["pause_after_ms"],
            item["text"],
        ])
    return rows


def fit_duration_factor(current: float, actual_ms: float, target_ms: float) -> float:
    if actual_ms <= 0 or target_ms <= 0:
        return max(0.5, min(2.0, float(current)))
    return max(0.5, min(2.0, float(current) * target_ms / actual_ms))


def allocate_native_chunk_durations(
    plan: DesktopGenerationPlan, target_seconds: float
) -> tuple[float, ...]:
    """Allocate one total native duration across externally paused speech blocks."""
    target_seconds = float(target_seconds)
    if not 0.1 <= target_seconds <= 3600:
        raise ValueError("目标时长必须在 0.1–3600 秒。")
    pause_ms = sum(chunk.pause_after_ms for chunk in plan.chunks)
    if plan.chunks:
        pause_ms += plan.chunks[0].pause_before_ms
    speech_seconds = target_seconds - pause_ms / 1000.0
    if speech_seconds <= 0:
        raise ValueError(
            f"目标时长 {target_seconds:.3f} 秒不足以容纳已配置的 {pause_ms / 1000:.3f} 秒停顿。"
        )
    weights = [max(1, len(chunk.text)) for chunk in plan.chunks]
    total_weight = sum(weights)
    return tuple(speech_seconds * weight / total_weight for weight in weights)


def apply_duration_policy(
    waveform: torch.Tensor, sample_rate: int, target_seconds: float, mode: str
) -> tuple[torch.Tensor, dict[str, Any]]:
    original = waveform.shape[-1]
    if mode == "off" or float(target_seconds or 0) <= 0:
        return waveform, {"mode": "off", "action": "unchanged"}
    if mode not in DURATION_MODES or not 0.1 <= float(target_seconds) <= 3600:
        raise ValueError("目标时长必须在 0.1–3600 秒。")
    target = max(1, round(float(target_seconds) * sample_rate))
    result, action = waveform, "unchanged"
    if original < target and mode in {"pad", "exact"}:
        result, action = F.pad(waveform, (0, target - original)), "padded"
    elif original > target and mode == "exact":
        result, action = waveform[..., :target].clone(), "trimmed"
        fade = min(round(sample_rate * 0.02), target)
        if fade > 1:
            result[..., -fade:] *= torch.linspace(1.0, 0.0, fade, dtype=result.dtype)
    elif original > target and mode == "pad":
        action = "overrun_preserved"
    return result.contiguous(), {
        "mode": mode,
        "target_ms": round(target * 1000 / sample_rate),
        "original_ms": round(original * 1000 / sample_rate),
        "final_ms": round(result.shape[-1] * 1000 / sample_rate),
        "action": action,
    }


def concatenate_with_pauses(
    waveforms: list[torch.Tensor],
    sample_rate: int,
    pause_after_ms: list[int],
    leading_pause_ms: int = 0,
) -> torch.Tensor:
    if not waveforms or len(waveforms) != len(pause_after_ms):
        raise ValueError("音频块与停顿数量不一致。")
    channels = max(item.shape[-2] if item.ndim >= 2 else 1 for item in waveforms)
    output: list[torch.Tensor] = []
    leading_samples = round(max(0, int(leading_pause_ms)) * sample_rate / 1000)
    if leading_samples:
        output.append(torch.zeros((channels, leading_samples), dtype=waveforms[0].dtype))
    for item, millis in zip(waveforms, pause_after_ms):
        tensor = item if item.ndim == 2 else item.unsqueeze(0)
        if tensor.shape[0] != channels:
            tensor = tensor[:1].repeat(channels, 1)
        output.append(tensor)
        samples = round(max(0, int(millis)) * sample_rate / 1000)
        if samples:
            output.append(torch.zeros((channels, samples), dtype=tensor.dtype))
    return torch.cat(output, dim=-1)


def _peak_normalize(waveform: torch.Tensor, target_db: float) -> torch.Tensor:
    peak = waveform.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    return waveform * ((10 ** (target_db / 20.0)) / peak).clamp(max=20.0)


def postprocess_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    preset: str,
    strength: float,
    target_peak_db: float = -1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if preset not in POSTPROCESS_PRESETS or not 0 <= float(strength) <= 1:
        raise ValueError("无效的音频后处理设置。")
    source = waveform.detach().float()
    before = float(source.abs().max())
    if preset == "off" or float(strength) == 0:
        return source, {"preset": "off", "peak_before": before, "peak_after": before}
    import torchaudio.functional as AF

    processed = source
    if preset in {"voice_clarity", "clear_narration", "warm"}:
        processed = AF.highpass_biquad(processed, sample_rate, 70.0)
    if preset == "voice_clarity":
        processed = AF.equalizer_biquad(processed, sample_rate, 3200.0, gain=3.0, Q=0.7)
    elif preset == "clear_narration":
        processed = AF.equalizer_biquad(processed, sample_rate, 2800.0, gain=2.5, Q=0.8)
        threshold = 10 ** (-20 / 20)
        magnitude = processed.abs().clamp_min(1e-8)
        compressed = torch.where(magnitude > threshold, threshold * torch.pow(magnitude / threshold, 1 / 3), magnitude)
        processed = processed.sign() * compressed
    elif preset == "deharsh":
        processed = AF.equalizer_biquad(processed, sample_rate, 4800.0, gain=-4.0, Q=1.1)
        processed = AF.lowpass_biquad(processed, sample_rate, min(10_500.0, sample_rate * 0.45))
    elif preset == "warm":
        processed = AF.equalizer_biquad(processed, sample_rate, 220.0, gain=3.0, Q=0.8)
        processed = AF.equalizer_biquad(processed, sample_rate, 5200.0, gain=-1.5, Q=0.9)
    mixed = source * (1 - float(strength)) + processed * float(strength)
    mixed = _peak_normalize(mixed, float(target_peak_db)).clamp(-1, 1).contiguous()
    mixed = fade_out_pcm_tail(mixed, sample_rate)
    return mixed, {
        "preset": preset,
        "strength": float(strength),
        "target_peak_db": float(target_peak_db),
        "peak_before": before,
        "peak_after": float(mixed.abs().max()),
    }


__all__ = [
    "DURATION_MODES",
    "LANGUAGE_AUTO_LIMITS",
    "LATIN_LONG_TEXT_LANGUAGES",
    "LONG_TEXT_RETRY_MIN_TOKENS",
    "POSTPROCESS_PRESETS",
    "assess_long_text_result",
    "apply_duration_policy",
    "allocate_native_chunk_durations",
    "build_desktop_plan",
    "concatenate_with_pauses",
    "estimate_segment_seconds",
    "effective_segment_limit",
    "ensure_terminal_punctuation",
    "fit_duration_factor",
    "latin_word_count",
    "long_text_retry_limit",
    "normalize_preflight_text",
    "postprocess_waveform",
    "preflight_plan_rows",
    "run_with_long_text_guard",
    "separate_repeated_characters",
    "split_speech_chunks",
]
