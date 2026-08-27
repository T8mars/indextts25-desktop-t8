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
