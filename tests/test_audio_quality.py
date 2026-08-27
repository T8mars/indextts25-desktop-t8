from __future__ import annotations

import torch

from audio_quality import (
    analyze_reference_audio,
    prepare_reference_audio,
    render_waveform_image,
    waveform_html,
)


def _clean_reference(sample_rate: int = 16000):
    leading = torch.zeros(sample_rate)
    time = torch.arange(sample_rate * 4) / sample_rate
    voice = 0.2 * torch.sin(2 * torch.pi * 220 * time)
    trailing = torch.zeros(sample_rate)
    return torch.cat([leading, voice, trailing]).unsqueeze(0), sample_rate


def test_reference_quality_reports_and_trims_silence():
    audio, rate = _clean_reference()
    report = analyze_reference_audio(audio, rate)
    prepared, preparation = prepare_reference_audio(audio, rate)
    assert report["duration_seconds"] == 6.0
    assert report["leading_silence_seconds"] >= 0.9
    assert report["trailing_silence_seconds"] >= 0.9
    assert preparation["trimmed"] is True
    assert 4.0 <= prepared.shape[-1] / rate <= 4.4
    assert preparation["prepared"]["score"] > report["score"]


def test_reference_quality_rejects_short_clipped_audio():
    report = analyze_reference_audio(torch.ones(1, 8000), 16000)
    assert report["usable"] is False
    assert report["score"] < 65
    assert report["clipped_ratio"] == 1.0


def test_waveform_renderers_include_word_markers_and_escape_labels():
    audio, rate = _clean_reference()
    words = [{"word": "<测试>", "start": 1.2, "end": 1.8}]
    image = render_waveform_image(audio, rate, words, width=400, height=160)
    markup = waveform_html(audio, rate, words, width=400, height=160)
    assert image.shape == (1, 160, 400, 3)
    assert "class=\"word\"" in markup
    assert "&lt;测试&gt;" in markup
