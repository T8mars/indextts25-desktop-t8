from __future__ import annotations

from pathlib import Path

import torch

from indextts.utils.reference_condition_cache import ReferenceConditionCache


def test_reference_condition_cache_roundtrip(tmp_path: Path) -> None:
    audio = tmp_path / "reference.wav"
    audio.write_bytes(b"audio-content")
    cache = ReferenceConditionCache(tmp_path / "cache", "model-a")
    expected = {"spk_cond": torch.arange(6).reshape(1, 2, 3).float()}
    cache.save("speaker", audio, expected)
    loaded = cache.load("speaker", audio, "cpu")
    assert loaded is not None
    assert torch.equal(loaded["spk_cond"], expected["spk_cond"])
    assert cache.status()["entries"] == 1


def test_cache_key_changes_with_audio_and_namespace(tmp_path: Path) -> None:
    audio = tmp_path / "reference.wav"
    audio.write_bytes(b"first")
    first = ReferenceConditionCache(tmp_path / "cache", "model-a")
    second = ReferenceConditionCache(tmp_path / "cache", "model-b")
    first.save("emotion", audio, {"emo_cond": torch.ones(1)})
    assert second.load("emotion", audio, "cpu") is None
    audio.write_bytes(b"second")
    assert first.load("emotion", audio, "cpu") is None


def test_cache_reports_hits_misses_size_and_safe_clear(tmp_path: Path) -> None:
    audio = tmp_path / "reference.wav"
    audio.write_bytes(b"audio-content")
    cache = ReferenceConditionCache(tmp_path / "cache", "model-a")
    assert cache.load("speaker", audio, "cpu") is None
    cache.save("speaker", audio, {"spk_cond": torch.ones(1, 2, 3)})
    assert cache.load("speaker", audio, "cpu") is not None
    unrelated = tmp_path / "cache" / "keep.txt"
    unrelated.write_text("keep", encoding="utf-8")

    status = cache.status()
    assert status["entries"] == 1
    assert status["bytes"] > 0
    assert status["hits"] == 1
    assert status["misses"] == 1
    assert status["hit_rate"] == 0.5
    assert status["writes"] == 1

    assert cache.clear() == 1
    assert cache.status()["entries"] == 0
    assert unrelated.read_text(encoding="utf-8") == "keep"
