"""FFmpeg-free Gradio audio streaming for the portable Windows bundle."""

from __future__ import annotations

import io

import av
from gradio import processing_utils
from gradio.components.audio import Audio
from gradio.data_classes import FileData


def _frames(value):
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return value
    return (value,)


def wav_bytes_to_adts(data: bytes) -> tuple[bytes, float]:
    """Encode a Gradio WAV chunk as browser-playable AAC/ADTS using bundled PyAV."""

    source = av.open(io.BytesIO(data), mode="r")
    try:
        source_stream = next((item for item in source.streams if item.type == "audio"), None)
        if source_stream is None:
            raise ValueError("流式音频块中没有音频轨道。")
        sample_rate = int(source_stream.codec_context.sample_rate or source_stream.rate or 22050)
        channels = int(source_stream.codec_context.channels or 1)
        layout = "mono" if channels == 1 else "stereo"
        buffer = io.BytesIO()
        output = av.open(buffer, mode="w", format="adts")
        sample_count = 0
        try:
            encoder = output.add_stream("aac", rate=sample_rate)
            encoder.layout = layout
            resampler = av.AudioResampler(format="fltp", layout=layout, rate=sample_rate)
            for frame in source.decode(audio=0):
                for converted in _frames(resampler.resample(frame)):
                    converted.pts = None
                    sample_count += int(converted.samples)
                    for packet in encoder.encode(converted):
                        output.mux(packet)
            for converted in _frames(resampler.resample(None)):
                converted.pts = None
                sample_count += int(converted.samples)
                for packet in encoder.encode(converted):
                    output.mux(packet)
            for packet in encoder.encode(None):
                output.mux(packet)
        finally:
            output.close()
        encoded = buffer.getvalue()
        if not encoded:
            raise RuntimeError("PyAV 没有生成 AAC 流式音频。")
        return encoded, sample_count / sample_rate
    finally:
        source.close()


class BundledStreamingAudio(Audio):
    """Gradio Audio output that never shells out to ffmpeg/ffprobe."""

    @staticmethod
    def _convert_to_adts(data: bytes):
        return wav_bytes_to_adts(data)

    async def combine_stream(
        self,
        stream: list[bytes],
        desired_output_format: str | None = None,
        only_file: bool = False,
    ) -> FileData:
        del desired_output_format, only_file
        path = processing_utils.save_bytes_to_cache(
            b"".join(stream), "audio-stream.aac", cache_dir=self.GRADIO_CACHE
        )
        return FileData(
            path=path,
            is_stream=False,
            orig_name="audio-stream.aac",
        )


__all__ = ["BundledStreamingAudio", "wav_bytes_to_adts"]
