from __future__ import annotations

import numpy as np
import pytest
import torch

import speech_review


def test_transcript_review_normalizes_punctuation_and_reports_similarity():
    exact = speech_review.review_transcript("重庆银行，欢迎你！", "重庆银行欢迎你", "ZH", 0.9)
    assert exact["passed"] is True
    assert exact["similarity"] == 1.0

    mismatch = speech_review.review_transcript("第一条台词", "完全不同", "ZH", 0.8)
    assert mismatch["passed"] is False
    assert mismatch["cer"] > 0

    with pytest.raises(ValueError, match="阈值"):
        speech_review.review_transcript("a", "a", threshold=1.2)


def test_review_normalizes_traditional_chinese_numbers_and_reports_differences():
    equivalent = speech_review.review_transcript("第二十五條臺詞", "第25条台词", "ZH", 0.99)
    assert equivalent["passed"] is True
    assert equivalent["cer"] == 0
    assert equivalent["metric"] == "cer"

    changed = speech_review.review_transcript("重庆银行", "重庆银航", "ZH", 0.9)
    assert changed["differences"] == [{"operation": "replace", "expected": "行", "recognized": "航"}]


def test_english_review_uses_wer():
    review = speech_review.review_transcript("Hello brave new world", "hello new world", "EN", 0.7)
    assert review["metric"] == "wer"
    assert review["word_edit_distance"] == 1
    assert review["wer"] == 0.25
    assert review["passed"] is True


def test_arabic_review_normalizes_diacritics_tatweel_alef_and_ya():
    review = speech_review.review_transcript("إِلَى ٱلْمَدِينَةِ", "الـي المدينة", "AR", 0.99)
    assert review["metric"] == "wer"
    assert review["wer"] == 0.0
    assert "arabic_diacritics_removed" in review["normalization"]


def test_transcribe_waveform_resamples_without_external_ffmpeg(monkeypatch):
    captured = {}

    class FakeWhisper:
        def transcribe(self, audio, **kwargs):
            captured["audio"] = audio
            captured["samples"] = len(audio)
            captured.update(kwargs)
            return {"text": "hello world", "language": "en", "segments": [{"words": [{"word": "hello", "start": 0.1, "end": 0.4, "probability": 0.95}]}, {}]}

    monkeypatch.setattr(speech_review, "load_asr_model", lambda *args, **kwargs: (FakeWhisper(), "cpu"))
    result = speech_review.transcribe_waveform(
        torch.zeros(2, 22050),
        22050,
        language="EN",
        model_name="tiny",
    )
    assert captured["samples"] == 16000
    assert captured["language"] == "en"
    assert captured["fp16"] is False
    assert isinstance(captured["audio"], np.ndarray)
    assert result["text"] == "hello world"
    assert result["segments"] == 2
    assert result["backend"] == "openai_whisper"
    assert result["word_timestamps"][0]["word"] == "hello"
    assert captured["word_timestamps"] is True


def test_faster_whisper_backend_collects_word_timestamps(monkeypatch):
    class Word:
        word, start, end, probability = "fast", 0.2, 0.6, 0.91

    class Segment:
        text, words = "fast path", [Word()]

    class FakeFaster:
        def transcribe(self, audio, **kwargs):
            return iter([Segment()]), type("Info", (), {"language": "en"})()

    monkeypatch.setattr(speech_review, "resolve_asr_backend", lambda backend: "faster_whisper")
    monkeypatch.setattr(speech_review, "load_asr_model", lambda *args, **kwargs: (FakeFaster(), "cpu"))
    result = speech_review.transcribe_waveform(torch.zeros(16000), 16000, language="EN", backend="faster_whisper")
    assert result["backend"] == "faster_whisper"
    assert result["text"] == "fast path"
    assert result["word_timestamps"][0] == {"word": "fast", "start": 0.2, "end": 0.6, "probability": 0.91, "segment": 0}
