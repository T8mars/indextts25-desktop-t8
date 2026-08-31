from __future__ import annotations

import asyncio
import io
import wave

import av
import gradio as gr
import numpy as np

from desktop_streaming_audio import BundledStreamingAudio, wav_bytes_to_adts


def _wav_bytes(sample_rate: int = 22050, duration: float = 0.2) -> bytes:
    samples = np.arange(round(sample_rate * duration), dtype=np.float32)
    waveform = (0.1 * np.sin(2 * np.pi * 440 * samples / sample_rate) * 32767).astype(
        np.int16
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(waveform.tobytes())
    return buffer.getvalue()


def test_streaming_encoder_does_not_require_system_ffmpeg(monkeypatch):
    monkeypatch.setenv("PATH", "")
    encoded, duration = wav_bytes_to_adts(_wav_bytes())

    assert encoded
    assert 0.18 <= duration <= 0.25
    with av.open(io.BytesIO(encoded), mode="r", format="adts") as source:
        frames = list(source.decode(audio=0))
    assert frames and sum(frame.samples for frame in frames) > 0


def test_streaming_component_keeps_core_gradio_frontend_identity(monkeypatch):
    monkeypatch.setenv("PATH", "")
    with gr.Blocks() as demo:
        component = BundledStreamingAudio(
            label="流式试听",
            streaming=True,
            autoplay=True,
            type="numpy",
            format="wav",
        )

    config = demo.get_config_file()
    item = next(entry for entry in config["components"] if entry["id"] == component._id)
    assert type(component) is gr.Audio
    assert item["type"] == "audio"
    assert item["component_class_id"] == gr.Audio.get_component_class_id()
    assert "bundledstreamingaudio" not in str(config).lower()

    encoded, duration = asyncio.run(component.covert_to_adts(_wav_bytes()))
    assert encoded
    assert 0.18 <= duration <= 0.25
