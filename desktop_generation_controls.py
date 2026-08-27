"""Desktop generation controls shared by the Gradio UI and its tests."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from indextts.utils.common import fade_out_pcm_tail


LANGUAGE_AUTO_LIMITS = {"ZH": 120, "EN": 60, "JA": 100, "ES": 60, "AR": 80}
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
    "POSTPROCESS_PRESETS",
    "apply_duration_policy",
    "allocate_native_chunk_durations",
    "build_desktop_plan",
    "concatenate_with_pauses",
    "effective_segment_limit",
    "fit_duration_factor",
    "postprocess_waveform",
    "split_speech_chunks",
]
