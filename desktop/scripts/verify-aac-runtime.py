"""Verify that the packaged runtime decodes AAC without system FFmpeg."""

import math
import io
import tempfile
import wave
from pathlib import Path

import av
import numpy as np

from indextts.infer_v2_5 import IndexTTS2
from desktop_streaming_audio import wav_bytes_to_adts


def write_aac(path: Path) -> None:
    sample_rate = 16000
    duration = 0.5
    samples = np.arange(int(sample_rate * duration), dtype=np.float32)
    waveform = (0.1 * np.sin(2 * math.pi * 440 * samples / sample_rate)).reshape(1, -1)

    with av.open(str(path), mode="w", format="adts") as container:
        stream = container.add_stream("aac", rate=sample_rate)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(waveform, format="fltp", layout="mono")
        frame.sample_rate = sample_rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


with tempfile.TemporaryDirectory(prefix="t8star-aac-runtime-") as temporary:
    source = Path(temporary) / "emotion-reference.aac"
    write_aac(source)
    loader = IndexTTS2.__new__(IndexTTS2)
    audio, sampling_rate = loader._load_and_cut_audio(source, 15, sr=16000)
    assert sampling_rate == 16000
    assert audio.ndim == 2 and audio.shape[0] == 1 and audio.shape[1] > 0
    assert float(audio.abs().max()) > 0

print("Bundled AAC decoding OK")

wav_buffer = io.BytesIO()
waveform = (0.1 * np.sin(2 * math.pi * 440 * np.arange(4410) / 22050) * 32767).astype(
    np.int16
)
with wave.open(wav_buffer, "wb") as target:
    target.setnchannels(1)
    target.setsampwidth(2)
    target.setframerate(22050)
    target.writeframes(waveform.tobytes())
encoded, duration = wav_bytes_to_adts(wav_buffer.getvalue())
assert encoded and 0.18 <= duration <= 0.25
with av.open(io.BytesIO(encoded), mode="r", format="adts") as source:
    assert sum(frame.samples for frame in source.decode(audio=0)) > 0

print("Bundled Gradio streaming encode OK")
