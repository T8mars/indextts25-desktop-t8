import json
import io
import math
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest

from desktop_voice_library import VoiceLibrary, safe_voice_file_stem


def _write_test_wav(path: Path, *, sample: int = 100, frames: int = 240) -> Path:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(int(sample).to_bytes(2, "little", signed=True) * frames)
    return path


def _wav_payload(path: str | Path) -> bytes:
    with wave.open(str(path), "rb") as audio:
        return audio.readframes(audio.getnframes())


def _wav_bytes(*, sample: int = 100, frames: int = 240) -> bytes:
    payload = io.BytesIO()
    with wave.open(payload, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(int(sample).to_bytes(2, "little", signed=True) * frames)
    return payload.getvalue()


def test_voice_library_copies_and_removes_audio(tmp_path: Path):
    source = _write_test_wav(tmp_path / "voice.wav")
    library = VoiceLibrary(tmp_path / "data")
    saved = library.save("小明", source, "ZH", emotion_text="开心")
    assert _wav_payload(saved.audio_path) == _wav_payload(source)
    assert library.get("小明").emotion_text == "开心"
    assert library.list()[0].profile_id == saved.profile_id
    removed = library.delete(saved.profile_id)
    assert removed.name == "小明"
    assert not Path(saved.audio_path).exists()


def test_voice_library_validates_source(tmp_path: Path):
    library = VoiceLibrary(tmp_path)
    with pytest.raises(FileNotFoundError):
        library.save("A", tmp_path / "missing.wav")


def test_voice_library_rejects_fake_wav(tmp_path: Path):
    source = tmp_path / "fake.wav"
    source.write_bytes(b"RIFF-not-a-real-wave")

    with pytest.raises(ValueError, match="无法把参考音频转换成便携 WAV"):
        VoiceLibrary(tmp_path / "data").save("A", source)


def test_voice_library_can_load_edit_and_rename_existing_profile(tmp_path: Path):
    first = _write_test_wav(tmp_path / "first.wav", sample=100)
    second = _write_test_wav(tmp_path / "second.wav", sample=200)
    library = VoiceLibrary(tmp_path / "data")
    original = library.save("角色A", first)
    renamed = library.save(
        "主角",
        second,
        "EN",
        emotion_text="calm",
        replace_name_or_id=original.profile_id,
    )
    assert renamed.profile_id == original.profile_id
    assert renamed.name == "主角"
    assert renamed.language == "EN"
    assert _wav_payload(renamed.audio_path) == _wav_payload(second)
    assert not Path(original.audio_path).exists()
    with pytest.raises(KeyError):
        library.get("角色A")


def test_voice_library_persists_all_role_emotion_fields_and_copies_reference(
    tmp_path: Path,
):
    voice = _write_test_wav(tmp_path / "voice.wav", sample=100)
    emotion = _write_test_wav(tmp_path / "emotion.wav", sample=200)
    library = VoiceLibrary(tmp_path / "data")

    audio_profile = library.save(
        "角色A",
        voice,
        emotion_mode="reference_audio",
        emotion_audio=emotion,
        emotion_strength=0.75,
    )
    assert _wav_payload(audio_profile.emotion_audio_path) == _wav_payload(emotion)
    assert (
        Path(audio_profile.emotion_audio_path).parent
        == Path(audio_profile.audio_path).parent
    )

    vector_profile = library.save(
        "角色A",
        voice,
        emotion_mode="vector",
        emotion_vector=[1.0] * 8,
        emotion_strength=0.9,
        emotion_use_random=True,
        replace_name_or_id=audio_profile.profile_id,
    )
    assert vector_profile.emotion_mode == "vector"
    assert isinstance(vector_profile.emotion_vector, tuple)
    assert sum(vector_profile.emotion_vector) == pytest.approx(0.8)
    assert vector_profile.emotion_use_random is True
    assert not Path(audio_profile.emotion_audio_path).exists()
    assert library.get("角色A").emotion_vector == vector_profile.emotion_vector


def test_voice_library_migrates_legacy_profiles_with_emotion_defaults(tmp_path: Path):
    library = VoiceLibrary(tmp_path / "data")
    library.audio_dir.mkdir(parents=True)
    voice = library.audio_dir / "legacy.wav"
    voice.write_bytes(b"legacy")
    library.manifest_path.write_text(
        '{"legacy":{"profile_id":"legacy","name":"旧角色","audio_path":"'
        + str(voice).replace("\\", "\\\\")
        + '","language":"ZH","emotion_mode":"text","emotion_text":"温柔",'
        '"emotion_strength":0.6,"pronunciation_dictionary":""}}',
        encoding="utf-8",
    )

    profile = library.get("旧角色")
    assert profile.emotion_audio_path == ""
    assert profile.emotion_vector == (0.0,) * 8
    assert profile.emotion_use_random is False


def test_voice_library_requires_reference_audio_for_that_mode(tmp_path: Path):
    voice = _write_test_wav(tmp_path / "voice.wav")
    with pytest.raises(ValueError, match="需要提供情感参考音频"):
        VoiceLibrary(tmp_path / "data").save(
            "角色A", voice, emotion_mode="reference_audio"
        )


def test_voice_library_v2_search_favorite_quality_and_notes(tmp_path: Path):
    voice = _write_test_wav(tmp_path / "voice.wav")
    library = VoiceLibrary(tmp_path / "data")
    saved = library.save(
        "温柔旁白",
        voice,
        tags="女声，旁白, 女声",
        favorite=True,
        notes="适合纪录片",
        quality={"score": 91, "grade": "优秀"},
    )

    assert saved.tags == ("女声", "旁白")
    assert saved.favorite is True
    assert library.search("纪录片")[0].name == "温柔旁白"
    assert library.search(tags="旁白", favorites_only=True)[0].quality["score"] == 91
    assert library.search(tags="男声") == []

    updated = library.set_favorite(saved.profile_id, False)
    assert updated.favorite is False
    assert library.search(favorites_only=True) == []


def test_voice_bundle_round_trip_and_conflict_modes(tmp_path: Path):
    voice = _write_test_wav(tmp_path / "voice.wav", sample=100)
    emotion = _write_test_wav(tmp_path / "emotion.wav", sample=200)
    source = VoiceLibrary(tmp_path / "source")
    source.save(
        "角色A",
        voice,
        "JA",
        emotion_mode="reference_audio",
        emotion_audio=emotion,
        tags=["主角", "日语"],
        favorite=True,
        notes="第一版",
        quality={"score": 88},
    )
    bundle = source.export_bundle(tmp_path / "voices")
    assert bundle.name.endswith(".t8voice.zip")

    target = VoiceLibrary(tmp_path / "target")
    imported = target.import_bundle(bundle)
    assert [item.name for item in imported] == ["角色A"]
    restored = target.get("角色A")
    assert _wav_payload(restored.audio_path) == _wav_payload(voice)
    assert _wav_payload(restored.emotion_audio_path) == _wav_payload(emotion)
    assert restored.tags == ("主角", "日语")
    assert restored.favorite is True
    assert restored.quality["score"] == 88

    renamed = target.import_bundle(bundle, conflict="rename")
    assert renamed[0].name == "角色A（导入 2）"
    assert target.import_bundle(bundle, conflict="skip") == []


def test_voice_bundle_rejects_unsafe_members(tmp_path: Path):
    bundle = tmp_path / "unsafe.t8voice.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("../escape.wav", b"bad")
        archive.writestr(
            "manifest.json",
            '{"schemaVersion":1,"profiles":[{"name":"A","audio_path":"../escape.wav"}]}',
        )
    with pytest.raises(ValueError, match="不安全路径"):
        VoiceLibrary(tmp_path / "data").import_bundle(bundle)


def test_voice_library_normalizes_aac_to_portable_pcm_wav(tmp_path: Path):
    av = pytest.importorskip("av")
    source = tmp_path / "voice.aac"
    sample_rate = 16000
    samples = np.arange(sample_rate // 5, dtype=np.float32)
    waveform = (0.1 * np.sin(2 * math.pi * 440 * samples / sample_rate)).reshape(1, -1)
    with av.open(str(source), mode="w", format="adts") as container:
        stream = container.add_stream("aac", rate=sample_rate)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(waveform, format="fltp", layout="mono")
        frame.sample_rate = sample_rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)

    saved = VoiceLibrary(tmp_path / "data").save("AAC 角色", source)

    assert Path(saved.audio_path).suffix == ".wav"
    with wave.open(saved.audio_path, "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getframerate() == 24000
        assert audio.getnframes() > 0


def test_voice_bundle_import_is_atomic_when_later_profile_is_invalid(tmp_path: Path):
    bundle = tmp_path / "atomic.t8voice.zip"
    manifest = {
        "schemaVersion": 1,
        "profiles": [
            {"profile_id": "first", "name": "第一位", "audio_path": "audio/first.wav"},
            {
                "profile_id": "second",
                "name": "第二位",
                "audio_path": "audio/second.wav",
                "language": "XX",
            },
        ],
    }
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False))
        archive.writestr("audio/first.wav", _wav_bytes(sample=100))
        archive.writestr("audio/second.wav", _wav_bytes(sample=200))
    library = VoiceLibrary(tmp_path / "data")

    with pytest.raises(ValueError, match="不支持的语言"):
        library.import_bundle(bundle)

    assert library.list() == []
    assert not library.root.exists()


def test_voice_bundle_rejects_unlisted_members(tmp_path: Path):
    bundle = tmp_path / "extra.t8voice.zip"
    manifest = {
        "schemaVersion": 1,
        "profiles": [{"profile_id": "a", "name": "A", "audio_path": "audio/a.wav"}],
    }
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("audio/a.wav", b"voice")
        archive.writestr("audio/unlisted.wav", b"unexpected")

    with pytest.raises(ValueError, match="未列入清单"):
        VoiceLibrary(tmp_path / "data").import_bundle(bundle)


def test_voice_bundle_rejects_windows_case_aliases(tmp_path: Path):
    bundle = tmp_path / "case-alias.t8voice.zip"
    manifest = {
        "schemaVersion": 1,
        "profiles": [{"profile_id": "a", "name": "A", "audio_path": "audio/A.wav"}],
    }
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("audio/A.wav", _wav_bytes(sample=100))
        archive.writestr("audio/a.wav", _wav_bytes(sample=200))

    with pytest.raises(ValueError, match="Windows"):
        VoiceLibrary(tmp_path / "data").import_bundle(bundle)


def test_voice_bundle_export_preserves_existing_target_on_failure(
    tmp_path: Path, monkeypatch
):
    source = _write_test_wav(tmp_path / "voice.wav")
    library = VoiceLibrary(tmp_path / "data")
    library.save("角色A", source)
    target = tmp_path / "voices.t8voice.zip"
    target.write_bytes(b"existing-bundle")

    def fail_write(self, *args, **kwargs):
        raise OSError("simulated export failure")

    monkeypatch.setattr(zipfile.ZipFile, "write", fail_write)

    with pytest.raises(OSError, match="simulated export failure"):
        library.export_bundle(target)

    assert target.read_bytes() == b"existing-bundle"
    assert not list(tmp_path.glob(".voices.t8voice.zip.*.tmp"))


def test_voice_output_stem_removes_path_components():
    stem = safe_voice_file_stem(r"角色/../../不安全\\名称")
    assert "/" not in stem and "\\" not in stem and ".." not in stem
