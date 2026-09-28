"""Stable local HTTP API for the T8star-Aix IndexTTS 2.5 desktop bundle.

The API intentionally sits beside Gradio instead of exposing Gradio's generated
event API.  This keeps third-party integrations stable and lets every caller
share one model instance and one inference lock.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import hmac
import io
import secrets
import threading
import time
import uuid
import wave
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import quote

import av
import torch
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from desktop_generation_controls import (
    build_desktop_plan,
    concatenate_with_pauses,
    postprocess_waveform,
    run_with_long_text_guard,
)
from desktop_model_lifecycle import DesktopModelLifecycle
from desktop_voice_library import VoiceLibrary, VoiceProfile
from indextts.pronunciation import PronunciationEntry, process_pronunciation_text
from indextts.utils.audio_io import save_audio_file
from generation_cancellation import generation_cancellation


SUPPORTED_LANGUAGES = frozenset({"ZH", "EN", "JA", "ES", "AR"})
SUPPORTED_RESPONSE_FORMATS = frozenset({"wav", "mp3", "pcm"})
DEFAULT_MODEL_ID = "indextts-2.5"
MAX_API_TEXT_CHARS = 20_000


class InferenceCoordinator:
    """Serialize all UI and API calls that touch one IndexTTS model."""

    def __init__(self) -> None:
        self._execution_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._condition = threading.Condition(self._state_lock)
        self._source = contextvars.ContextVar("t8_inference_source", default="desktop")
        self._waiting = 0
        self._normal_claims = 0
        self._lease_waiters = 0
        self._active = ""
        self._lease_source = ""
        self._lease_token = ""
        self._completed = 0
        self._failed = 0

    @contextmanager
    def source(self, name: str):
        token = self._source.set(str(name or "unknown"))
        try:
            yield
        finally:
            self._source.reset(token)

    def run(self, callback: Callable[[], Any]) -> Any:
        source = self._source.get()
        with self._condition:
            self._waiting += 1
            while self._lease_token or self._lease_waiters:
                self._condition.wait()
            self._normal_claims += 1
        try:
            with self._execution_lock:
                with self._state_lock:
                    self._waiting -= 1
                    self._active = source
                try:
                    result = callback()
                except Exception:
                    with self._state_lock:
                        self._failed += 1
                    raise
                else:
                    with self._state_lock:
                        self._completed += 1
                    return result
                finally:
                    with self._state_lock:
                        self._active = ""
        finally:
            with self._condition:
                self._normal_claims -= 1
                self._condition.notify_all()

    def acquire_lease(self, source: str) -> str:
        """Reserve the GPU across a long-lived sidecar session.

        Unlike an RLock, this token may be released by the WebSocket cleanup
        task rather than the request thread that created the session.
        """

        token = secrets.token_urlsafe(24)
        with self._condition:
            self._waiting += 1
            self._lease_waiters += 1
            try:
                while self._lease_token or self._normal_claims:
                    self._condition.wait()
                self._lease_token = token
                self._lease_source = str(source or "sidecar")
            finally:
                self._lease_waiters -= 1
                self._waiting -= 1
        return token

    def release_lease(self, token: str) -> bool:
        with self._condition:
            if not self._lease_token or not secrets.compare_digest(self._lease_token, str(token or "")):
                return False
            self._lease_token = ""
            self._lease_source = ""
            self._condition.notify_all()
            return True

    def guard_model(self, model: Any) -> Any:
        """Wrap a model's public infer method once without changing callers."""

        with self._state_lock:
            if getattr(model, "_t8_inference_coordinator", None) is self:
                return model
            original = model.infer

            def coordinated_infer(*args, **kwargs):
                return self.run(lambda: original(*args, **kwargs))

            model.infer = coordinated_infer
            model._t8_inference_coordinator = self
            model._t8_original_infer = original
        return model

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "active": bool(self._active or self._lease_token),
                "active_source": self._lease_source or self._active,
                "lease_active": bool(self._lease_token),
                "waiting": self._waiting,
                "completed": self._completed,
                "failed": self._failed,
            }


class OpenAISpeechRequest(BaseModel):
    """OpenAI-compatible subset used by generic local TTS clients."""

    model_config = ConfigDict(extra="ignore")

    model: str = DEFAULT_MODEL_ID
    input: str = Field(min_length=1, max_length=MAX_API_TEXT_CHARS)
    voice: str = ""
    response_format: Literal["wav", "mp3", "pcm"] = "mp3"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    instructions: str = Field(default="", max_length=1_000)
    language: str | None = None

    @field_validator("input")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not str(value).strip():
            raise ValueError("input cannot be blank")
        return str(value).strip()


class NativeSpeechRequest(OpenAISpeechRequest):
    """IndexTTS-specific controls layered on the compatible request."""

    response_format: Literal["wav", "mp3", "pcm"] = "wav"
    duration_factor: float | None = Field(default=None, ge=0.5, le=2.0)
    emotion_mode: Literal["inherit", "speaker", "reference_audio", "vector", "text"] = "inherit"
    emotion_text: str = Field(default="", max_length=1_000)
    emotion_vector: list[float] | None = None
    emotion_strength: float = Field(default=0.65, ge=0.0, le=1.0)
    random_emotion: bool = False
    seed: int = Field(default=0, ge=0, le=0xFFFFFFFF)
    do_sample: bool = True
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)
    top_k: int = Field(default=30, ge=0, le=100)
    num_beams: int = Field(default=3, ge=1, le=10)
    repetition_penalty: float = Field(default=10.0, ge=0.1, le=20.0)
    length_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    max_mel_tokens: int = Field(default=1500, ge=50, le=10_000)
    max_text_tokens: int = Field(default=120, ge=20, le=1_000)
    segment_silence_ms: int = Field(default=200, ge=0, le=3_000)
    text_normalization: bool = True
    diffusion_steps: int = Field(default=25, ge=5, le=100)
    inference_cfg_rate: float = Field(default=0.7, ge=0.0, le=1.5)
    cfm_temperature: float = Field(default=1.0, ge=0.1, le=1.5)
    postprocess_preset: Literal[
        "off", "voice_clarity", "clear_narration", "deharsh", "warm", "normalize"
    ] = "off"
    postprocess_strength: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("emotion_vector")
    @classmethod
    def validate_emotion_vector(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        if len(value) != 8:
            raise ValueError("emotion_vector must contain exactly 8 values")
        return [max(0.0, min(1.0, float(item))) for item in value]


@dataclass(slots=True)
class SpeechOptions:
    text: str
    voice: str
    language: str | None = None
    response_format: str = "wav"
    speed: float = 1.0
    duration_factor: float | None = None
    emotion_mode: str = "inherit"
    emotion_text: str = ""
    emotion_vector: list[float] | None = None
    emotion_strength: float = 0.65
    random_emotion: bool = False
    seed: int = 0
    do_sample: bool = True
    temperature: float = 0.8
    top_p: float = 0.8
    top_k: int = 30
    num_beams: int = 3
    repetition_penalty: float = 10.0
    length_penalty: float = 0.0
    max_mel_tokens: int = 1500
    max_text_tokens: int = 120
    segment_silence_ms: int = 200
    text_normalization: bool = True
    diffusion_steps: int = 25
    inference_cfg_rate: float = 0.7
    cfm_temperature: float = 1.0
    postprocess_preset: str = "off"
    postprocess_strength: float = 1.0


@dataclass(slots=True)
class GeneratedSpeech:
    request_id: str
    file_path: Path
    media_type: str
    payload: bytes
    voice: str
    language: str
    duration_seconds: float
    resolved_text: str
    warnings: list[str] = field(default_factory=list)


def _profile_pronunciation_entries(raw: str, language: str) -> list[PronunciationEntry]:
    entries: list[PronunciationEntry] = []
    for row in str(raw or "").splitlines():
        value = row.strip()
        if not value or value.startswith("#"):
            continue
        parts = [part.strip() for part in value.split("|")]
        if len(parts) >= 2:
            entries.append(
                PronunciationEntry(
                    parts[0], parts[1], parts[2] if len(parts) >= 3 else language
                )
            )
    return entries


def _result_to_waveform(result: Any) -> tuple[torch.Tensor, int]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("IndexTTS returned an unsupported audio result")
    sample_rate, raw = result
    tensor = torch.as_tensor(raw).detach().cpu()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2 and tensor.shape[-1] == 1:
        tensor = tensor.transpose(0, 1)
    elif tensor.ndim != 2:
        tensor = tensor.reshape(1, -1)
    if tensor.dtype.is_floating_point:
        tensor = tensor.float()
        if tensor.numel() and float(tensor.abs().max()) > 2.0:
            tensor = tensor / 32768.0
    else:
        tensor = tensor.float() / 32768.0
    return tensor.clamp(-1, 1).contiguous(), int(sample_rate)


def encode_audio_bytes(waveform: torch.Tensor, sample_rate: int, response_format: str) -> tuple[bytes, str]:
    """Encode generated mono audio without requiring a system FFmpeg install."""

    output_format = str(response_format or "wav").lower()
    if output_format not in SUPPORTED_RESPONSE_FORMATS:
        raise ValueError(f"Unsupported response_format: {output_format}")
    tensor = torch.as_tensor(waveform).detach().cpu().float()
    if tensor.ndim == 2:
        tensor = tensor.mean(dim=0)
    elif tensor.ndim != 1:
        tensor = tensor.reshape(-1)
    pcm = (tensor.clamp(-1, 1).numpy() * 32767.0).round().astype("<i2")
    if output_format == "pcm":
        return pcm.tobytes(), "audio/pcm"
    if output_format == "wav":
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(int(sample_rate))
            output.writeframes(pcm.tobytes())
        return buffer.getvalue(), "audio/wav"

    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp3") as container:
        stream = container.add_stream("mp3", rate=int(sample_rate))
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = int(sample_rate)
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return buffer.getvalue(), "audio/mpeg"


def _emotion_kwargs(model: Any, profile: VoiceProfile, options: SpeechOptions) -> dict[str, Any]:
    mode = options.emotion_mode
    if mode == "inherit":
        mode = profile.emotion_mode
        emotion_text = profile.emotion_text
        emotion_vector = list(profile.emotion_vector)
        emotion_strength = profile.emotion_strength
        random_emotion = profile.emotion_use_random
    else:
        emotion_text = options.emotion_text
        emotion_vector = options.emotion_vector
        emotion_strength = options.emotion_strength
        random_emotion = options.random_emotion
    if mode == "text" and getattr(model, "qwen_emo", None) is None:
        raise ValueError("Current low-VRAM mode does not load text emotion analysis")
    normalized_vector = None
    if mode == "vector":
        vector = emotion_vector or [0.0] * 8
        normalized_vector = model.normalize_emo_vec(vector, apply_bias=True)
    if mode == "reference_audio" and not profile.emotion_audio_path:
        raise ValueError("Selected voice does not contain an emotion reference audio file")
    return {
        "emo_audio_prompt": profile.emotion_audio_path if mode == "reference_audio" else None,
        "emo_alpha": float(emotion_strength),
        "emo_vector": normalized_vector,
        "use_emo_text": mode == "text",
        "emo_text": str(emotion_text or "").strip() or None,
        "use_random": bool(random_emotion),
        "emotion_mode": mode,
    }


class DesktopTTSService:
    """Reusable synthesis service shared by FastAPI and the desktop UI."""

    def __init__(
        self,
        lifecycle: DesktopModelLifecycle,
        voice_library: VoiceLibrary,
        output_dir: str | Path,
        coordinator: InferenceCoordinator,
        *,
        default_voice: str = "",
        verbose: bool = False,
        save_history: bool = True,
        history_writer: Callable[[dict[str, Any]], str] | None = None,
        max_text_chars: int = MAX_API_TEXT_CHARS,
    ) -> None:
        self.lifecycle = lifecycle
        self.voice_library = voice_library
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.coordinator = coordinator
        self.default_voice = str(default_voice or "").strip()
        self.verbose = bool(verbose)
        self.save_history = bool(save_history)
        self.history_writer = history_writer
        self.max_text_chars = max(1, min(MAX_API_TEXT_CHARS, int(max_text_chars)))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def voices(self) -> list[dict[str, Any]]:
        return [
            {
                "id": item.profile_id,
                "name": item.name,
                "language": item.language,
                "emotion_mode": item.emotion_mode,
                "favorite": item.favorite,
                "tags": list(item.tags),
            }
            for item in self.voice_library.list()
        ]

    def resolve_voice(self, requested: str) -> VoiceProfile:
        candidate = str(requested or self.default_voice).strip()
        voices = self.voice_library.list()
        if not candidate:
            if len(voices) == 1:
                return voices[0]
            raise ValueError("voice is required; add/select a saved voice in the desktop voice library")
        try:
            return self.voice_library.get(candidate)
        except KeyError as exc:
            # A number of OpenAI-compatible clients hard-code one of the
            # standard OpenAI voice labels.  When the user configured a local
            # default voice, treat those labels as an alias without masking a
            # real saved profile that happens to use the same name.
            openai_aliases = {
                "alloy", "ash", "ballad", "coral", "echo", "fable",
                "nova", "onyx", "sage", "shimmer", "verse",
            }
            if self.default_voice and candidate.casefold() in openai_aliases:
                try:
                    return self.voice_library.get(self.default_voice)
                except KeyError:
                    pass
            names = ", ".join(item.name for item in voices[:20]) or "none"
            raise ValueError(f"Unknown voice '{candidate}'. Available voices: {names}") from exc

    def generate(
        self,
        options: SpeechOptions,
        *,
        cancel_event: threading.Event | None = None,
        request_id: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> GeneratedSpeech:
        generation_cancellation.begin(cancel_event)
        text = str(options.text or "").strip()
        if not text:
            raise ValueError("input cannot be blank")
        if len(text) > self.max_text_chars:
            raise ValueError(f"input exceeds the configured {self.max_text_chars} character limit")
        profile = self.resolve_voice(options.voice)
        language = str(options.language or profile.language or "ZH").upper()
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")
        response_format = str(options.response_format or "wav").lower()
        if response_format not in SUPPORTED_RESPONSE_FORMATS:
            raise ValueError(f"Unsupported response_format: {response_format}")
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("request cancelled")

        # The lifecycle can reload the model after an idle/manual release.  Guard
        # every obtained instance so UI and API requests always share one GPU
        # critical section, including after a reload.
        model = self.coordinator.guard_model(self.lifecycle.get())
        entries = _profile_pronunciation_entries(profile.pronunciation_dictionary, language)
        pronunciation = process_pronunciation_text(
            text,
            language,
            entries,
            strict=True,
        )
        resolved_text = pronunciation.text
        factor = (
            float(options.duration_factor)
            if options.duration_factor is not None
            else max(0.5, min(2.0, 1.0 / float(options.speed)))
        )
        emotion = _emotion_kwargs(model, profile, options)
        plan = build_desktop_plan(
            model,
            resolved_text,
            language,
            "auto",
            int(options.max_text_tokens),
            "off",
            100,
            300,
            600,
        )
        waveforms: list[torch.Tensor] = []
        sample_rate: int | None = None
        warnings: list[str] = []
        total_blocks = len(plan.chunks)
        if progress_callback:
            progress_callback(0, total_blocks)
        with self.coordinator.source("api"):
            for block_index, chunk in enumerate(plan.chunks):
                if cancel_event and cancel_event.is_set():
                    raise InterruptedError("request cancelled")

                def infer_with_limit(limit: int):
                    return model.infer(
                        spk_audio_prompt=profile.audio_path,
                        text=chunk.text,
                        output_path=None,
                        lang=language,
                        emo_audio_prompt=emotion["emo_audio_prompt"],
                        emo_alpha=emotion["emo_alpha"],
                        emo_vector=emotion["emo_vector"],
                        use_emo_text=emotion["use_emo_text"],
                        emo_text=emotion["emo_text"],
                        use_random=emotion["use_random"],
                        verbose=self.verbose,
                        do_sample=bool(options.do_sample),
                        temperature=float(options.temperature),
                        top_p=float(options.top_p),
                        top_k=int(options.top_k) if int(options.top_k) > 0 else None,
                        num_beams=int(options.num_beams),
                        repetition_penalty=float(options.repetition_penalty),
                        length_penalty=float(options.length_penalty),
                        max_mel_tokens=int(options.max_mel_tokens),
                        max_text_tokens_per_segment=int(limit),
                        interval_silence=int(options.segment_silence_ms),
                        text_normalization=bool(options.text_normalization),
                        duration_factor=factor,
                        seed=int(options.seed) + block_index,
                        diffusion_steps=int(options.diffusion_steps),
                        inference_cfg_rate=float(options.inference_cfg_rate),
                        cfm_temperature=float(options.cfm_temperature),
                    )

                token_count = len(
                    model.tokenizer.encode(
                        f"<|{language.lower()}|> {chunk.text}", allowed_special="all"
                    )
                )
                result, guard = run_with_long_text_guard(
                    infer_with_limit,
                    lambda item: (
                        lambda converted: converted[0].shape[-1] / converted[1]
                    )(_result_to_waveform(item)),
                    text=chunk.text,
                    language=language,
                    token_count=token_count,
                    max_tokens=plan.max_tokens,
                    duration_factor=factor,
                )
                if guard.get("retried"):
                    warnings.append(f"speech block {block_index + 1} used the long-text retry guard")
                waveform, block_rate = _result_to_waveform(result)
                if sample_rate is None:
                    sample_rate = block_rate
                elif sample_rate != block_rate:
                    raise RuntimeError("Generated speech blocks use different sample rates")
                waveforms.append(waveform)
                if progress_callback:
                    progress_callback(block_index + 1, total_blocks)

        if sample_rate is None or not waveforms:
            raise RuntimeError("IndexTTS did not return audio")
        waveform = concatenate_with_pauses(
            waveforms,
            sample_rate,
            [chunk.pause_after_ms for chunk in plan.chunks],
            plan.chunks[0].pause_before_ms,
        )
        waveform, _postprocess = postprocess_waveform(
            waveform,
            sample_rate,
            options.postprocess_preset,
            float(options.postprocess_strength),
        )
        resolved_request_id = request_id or uuid.uuid4().hex
        target = self.output_dir / (
            f"api_{time.strftime('%Y%m%d_%H%M%S')}_{resolved_request_id[:8]}.wav"
        )
        save_audio_file(target, waveform, sample_rate)
        payload, media_type = encode_audio_bytes(waveform, sample_rate, response_format)
        duration_seconds = waveform.shape[-1] / sample_rate
        if self.save_history and self.history_writer:
            warning = self.history_writer(
                {
                    "schema_version": 2,
                    "id": resolved_request_id,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "kind": "single",
                    "source": "api",
                    "title": "[API] " + " ".join(text.split())[:58],
                    "language": language,
                    "duration_factor": factor,
                    "emotion_mode": emotion["emotion_mode"],
                    "text": text,
                    "resolved_text": resolved_text,
                    "voice": profile.name,
                    "file": str(target),
                    "duration_ms": round(duration_seconds * 1000),
                }
            )
            if warning:
                warnings.append(warning)
        return GeneratedSpeech(
            request_id=resolved_request_id,
            file_path=target,
            media_type=media_type,
            payload=payload,
            voice=profile.name,
            language=language,
            duration_seconds=duration_seconds,
            resolved_text=resolved_text,
            warnings=warnings,
        )


def openai_options(request: OpenAISpeechRequest) -> SpeechOptions:
    return SpeechOptions(
        text=request.input,
        voice=request.voice,
        language=request.language,
        response_format=request.response_format,
        speed=request.speed,
        emotion_mode="text" if request.instructions.strip() else "inherit",
        emotion_text=request.instructions,
    )


def native_options(request: NativeSpeechRequest) -> SpeechOptions:
    payload = request.model_dump()
    payload.pop("model", None)
    payload["text"] = payload.pop("input")
    payload.pop("instructions", None)
    return SpeechOptions(**payload)


class ApiJobManager:
    def __init__(self, service: DesktopTTSService, *, maximum_jobs: int = 200) -> None:
        self.service = service
        self.maximum_jobs = max(10, int(maximum_jobs))
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="t8-api")
        self._guard = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def submit(self, options: SpeechOptions) -> dict[str, Any]:
        job_id = "api_" + uuid.uuid4().hex
        record = {
            "job_id": job_id,
            "status": "queued",
            "created_at": time.time(),
            "updated_at": time.time(),
            "voice": options.voice,
            "language": options.language,
            "file_path": "",
            "error": "",
            "progress": 0.0,
            "completed_blocks": 0,
            "total_blocks": 0,
            "cancel_event": threading.Event(),
            "future": None,
        }
        with self._guard:
            self._jobs[job_id] = record
            self._trim_locked()
            future = self._executor.submit(self._run, job_id, options)
            record["future"] = future
        return self.public(job_id)

    def _run(self, job_id: str, options: SpeechOptions) -> None:
        with self._guard:
            record = self._jobs[job_id]
            if record["cancel_event"].is_set():
                record.update(status="cancelled", updated_at=time.time())
                return
            record.update(status="running", updated_at=time.time())
            cancel_event = record["cancel_event"]
        try:
            result = self.service.generate(
                options,
                cancel_event=cancel_event,
                request_id=job_id.removeprefix("api_"),
                progress_callback=lambda current, total: self._progress(job_id, current, total),
            )
        except InterruptedError as exc:
            with self._guard:
                record.update(status="cancelled", error=str(exc), updated_at=time.time())
        except Exception as exc:
            with self._guard:
                record.update(
                    status="failed",
                    error=str(exc).strip() or type(exc).__name__,
                    updated_at=time.time(),
                )
        else:
            with self._guard:
                record.update(
                    status="completed",
                    progress=1.0,
                    file_path=str(result.file_path),
                    voice=result.voice,
                    language=result.language,
                    duration_seconds=round(result.duration_seconds, 4),
                    updated_at=time.time(),
                )

    def _progress(self, job_id: str, current: int, total: int) -> None:
        with self._guard:
            record = self._jobs.get(job_id)
            if record is None:
                return
            total = max(0, int(total))
            current = max(0, min(total, int(current))) if total else 0
            record.update(
                completed_blocks=current,
                total_blocks=total,
                progress=(current / total) if total else 0.0,
                updated_at=time.time(),
            )

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._guard:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(job_id)
            if record["status"] in {"completed", "failed", "cancelled"}:
                return self._public_locked(record)
            record["cancel_event"].set()
            future: Future | None = record.get("future")
            if future and future.cancel():
                record["status"] = "cancelled"
            else:
                record["status"] = "cancelling"
            record["updated_at"] = time.time()
            return self._public_locked(record)

    def public(self, job_id: str) -> dict[str, Any]:
        with self._guard:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(job_id)
            return self._public_locked(record)

    def result_path(self, job_id: str) -> Path:
        record = self.public(job_id)
        if record["status"] != "completed" or not record["file_path"]:
            raise ValueError("job has not completed")
        path = Path(record["file_path"]).resolve()
        if not path.is_relative_to(self.service.output_dir) or not path.is_file():
            raise ValueError("job output is unavailable")
        return path

    def summary(self) -> dict[str, int]:
        with self._guard:
            counts = {name: 0 for name in ("queued", "running", "cancelling", "completed", "failed", "cancelled")}
            for record in self._jobs.values():
                status = str(record.get("status"))
                counts[status] = counts.get(status, 0) + 1
            return counts

    @staticmethod
    def _public_locked(record: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in record.items()
            if key not in {"future", "cancel_event"}
        }

    def _trim_locked(self) -> None:
        if len(self._jobs) <= self.maximum_jobs:
            return
        removable = sorted(
            (
                record
                for record in self._jobs.values()
                if record["status"] in {"completed", "failed", "cancelled"}
            ),
            key=lambda item: item["updated_at"],
        )
        for record in removable[: max(0, len(self._jobs) - self.maximum_jobs)]:
            self._jobs.pop(record["job_id"], None)


class ShutdownController:
    def __init__(self) -> None:
        self._callback: Callable[[], None] | None = None

    def bind(self, callback: Callable[[], None]) -> None:
        self._callback = callback

    def request(self) -> None:
        if self._callback:
            self._callback()


def _error(message: str, status_code: int = 400, code: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": code, "code": code}},
    )


def _generation_error(exc: Exception) -> JSONResponse:
    message = str(exc).strip() or type(exc).__name__
    if isinstance(exc, (ValueError, FileNotFoundError)):
        return _error(message, 400, "invalid_request_error")
    if isinstance(exc, InterruptedError):
        return _error(message, 409, "request_cancelled")
    return _error(message, 500, "generation_error")


def _header_text(value: str) -> str:
    """Keep response headers ASCII-safe while preserving Unicode losslessly."""

    return quote(str(value or ""), safe="-._~")


def create_api_app(
    service: DesktopTTSService,
    *,
    api_key: str,
    desktop_version: str,
    coordinator: InferenceCoordinator,
    shutdown: ShutdownController | None = None,
    allow_origins: list[str] | None = None,
) -> FastAPI:
    """Create the stable API application. Gradio may be mounted on it later."""

    secret = str(api_key or "").strip()
    if len(secret) < 16:
        raise ValueError("API key must contain at least 16 characters")
    app = FastAPI(
        title="T8star-Aix IndexTTS 2.5 Local API",
        version=desktop_version,
        description="OpenAI-compatible and native local TTS endpoints backed by one IndexTTS 2.5 model.",
    )
    origins = [str(item).strip() for item in (allow_origins or []) if str(item).strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type", "X-API-Key"],
        )
    jobs = ApiJobManager(service)
    api_key_fingerprint = hashlib.sha256(secret.encode("utf-8")).hexdigest()[:16]

    def require_key(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> None:
        bearer = ""
        if authorization and authorization.lower().startswith("bearer "):
            bearer = authorization[7:].strip()
        provided = bearer or str(x_api_key or "").strip()
        if not hmac.compare_digest(provided, secret):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException):
        # The same-origin Confucius browser gateway has its own structured
        # ``detail`` contract consumed by the live-caption client.  Keep that
        # contract intact while the public TTS API continues to use the stable
        # OpenAI-style error envelope below.
        if request.url.path.startswith("/api/confucius/"):
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        return _error(
            str(exc.detail),
            status_code=exc.status_code,
            code="authentication_error" if exc.status_code == 401 else "invalid_request_error",
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_request: Request, exc: RequestValidationError):
        details = "; ".join(
            f"{'.'.join(str(part) for part in item.get('loc', ()) if part != 'body')}: {item.get('msg', 'invalid value')}"
            for item in exc.errors()
        )
        return _error(details or "Invalid request body", 422, "validation_error")

    @app.get("/health", tags=["service"])
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "t8star-indextts-2.5",
            "version": desktop_version,
            "api_key_fingerprint": api_key_fingerprint,
            "model": service.lifecycle.status(),
            "inference": coordinator.status(),
            "jobs": jobs.summary(),
            "voice_count": len(service.voices()),
        }

    @app.get("/v1/models", dependencies=[Depends(require_key)], tags=["openai-compatible"])
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": DEFAULT_MODEL_ID, "object": "model", "owned_by": "T8star-Aix"}],
        }

    @app.get("/v1/audio/voices", dependencies=[Depends(require_key)], tags=["voices"])
    def voices() -> dict[str, Any]:
        return {"object": "list", "data": service.voices(), "default": service.default_voice}

    @app.post("/v1/audio/speech", dependencies=[Depends(require_key)], tags=["openai-compatible"])
    async def openai_speech(request: OpenAISpeechRequest):
        if request.model not in {DEFAULT_MODEL_ID, "tts-1", "tts-1-hd"}:
            return _error(f"Unsupported model: {request.model}")
        try:
            result = await asyncio.to_thread(service.generate, openai_options(request))
        except Exception as exc:
            return _generation_error(exc)
        extension = request.response_format
        return Response(
            content=result.payload,
            media_type=result.media_type,
            headers={
                "Content-Disposition": f'inline; filename="speech.{extension}"',
                "X-T8-Request-ID": result.request_id,
                "X-T8-Voice": _header_text(result.voice),
                "X-T8-Header-Encoding": "percent",
                "X-T8-Duration-Seconds": f"{result.duration_seconds:.4f}",
            },
        )

    @app.post("/api/v1/speech", dependencies=[Depends(require_key)], tags=["native"])
    async def native_speech(request: NativeSpeechRequest):
        try:
            result = await asyncio.to_thread(service.generate, native_options(request))
        except Exception as exc:
            return _generation_error(exc)
        return Response(
            content=result.payload,
            media_type=result.media_type,
            headers={
                "Content-Disposition": f'inline; filename="speech.{request.response_format}"',
                "X-T8-Request-ID": result.request_id,
                "X-T8-Voice": _header_text(result.voice),
                "X-T8-Header-Encoding": "percent",
                "X-T8-Duration-Seconds": f"{result.duration_seconds:.4f}",
            },
        )

    @app.post("/api/v1/jobs", dependencies=[Depends(require_key)], tags=["jobs"])
    def create_job(request: NativeSpeechRequest):
        try:
            return jobs.submit(native_options(request))
        except Exception as exc:
            return _error(str(exc).strip() or type(exc).__name__)

    @app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_key)], tags=["jobs"])
    def get_job(job_id: str):
        try:
            return jobs.public(job_id)
        except KeyError:
            return _error("Unknown job", 404, "not_found")

    @app.delete("/api/v1/jobs/{job_id}", dependencies=[Depends(require_key)], tags=["jobs"])
    def cancel_job(job_id: str):
        try:
            return jobs.cancel(job_id)
        except KeyError:
            return _error("Unknown job", 404, "not_found")

    @app.get("/api/v1/jobs/{job_id}/audio", dependencies=[Depends(require_key)], tags=["jobs"])
    def job_audio(job_id: str):
        try:
            path = jobs.result_path(job_id)
        except KeyError:
            return _error("Unknown job", 404, "not_found")
        except ValueError as exc:
            return _error(str(exc), 409, "job_not_ready")
        return FileResponse(path, media_type="audio/wav", filename=path.name)

    @app.post("/api/v1/admin/shutdown", dependencies=[Depends(require_key)], tags=["service"])
    def shutdown_service(background_tasks: BackgroundTasks):
        if shutdown is None:
            return _error("Shutdown control is unavailable", 409, "unsupported_operation")
        background_tasks.add_task(shutdown.request)
        return {"status": "stopping"}

    app.state.t8_jobs = jobs
    app.state.t8_service = service
    app.state.t8_api_key_sha256 = hashlib.sha256(secret.encode("utf-8")).hexdigest()
    return app


__all__ = [
    "ApiJobManager",
    "DEFAULT_MODEL_ID",
    "DesktopTTSService",
    "GeneratedSpeech",
    "InferenceCoordinator",
    "NativeSpeechRequest",
    "OpenAISpeechRequest",
    "ShutdownController",
    "SpeechOptions",
    "create_api_app",
    "encode_audio_bytes",
    "native_options",
    "openai_options",
]
