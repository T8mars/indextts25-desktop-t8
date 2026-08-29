from pathlib import Path

import pytest

from desktop_voice_library import VoiceLibrary


def test_voice_library_copies_and_removes_audio(tmp_path: Path):
    source = tmp_path / "voice.wav"
    source.write_bytes(b"RIFF-test")
    library = VoiceLibrary(tmp_path / "data")
    saved = library.save("小明", source, "ZH", emotion_text="开心")
    assert Path(saved.audio_path).read_bytes() == b"RIFF-test"
    assert library.get("小明").emotion_text == "开心"
    assert library.list()[0].profile_id == saved.profile_id
    removed = library.delete(saved.profile_id)
    assert removed.name == "小明"
    assert not Path(saved.audio_path).exists()


def test_voice_library_validates_source(tmp_path: Path):
    library = VoiceLibrary(tmp_path)
    with pytest.raises(FileNotFoundError):
        library.save("A", tmp_path / "missing.wav")


def test_voice_library_can_load_edit_and_rename_existing_profile(tmp_path: Path):
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
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
    assert Path(renamed.audio_path).read_bytes() == b"second"
    assert not Path(original.audio_path).exists()
    with pytest.raises(KeyError):
        library.get("角色A")


def test_voice_library_persists_all_role_emotion_fields_and_copies_reference(tmp_path: Path):
    voice = tmp_path / "voice.wav"
    emotion = tmp_path / "emotion.wav"
    voice.write_bytes(b"voice")
    emotion.write_bytes(b"emotion")
    library = VoiceLibrary(tmp_path / "data")

    audio_profile = library.save(
        "角色A",
        voice,
        emotion_mode="reference_audio",
        emotion_audio=emotion,
        emotion_strength=0.75,
    )
    assert Path(audio_profile.emotion_audio_path).read_bytes() == b"emotion"
    assert Path(audio_profile.emotion_audio_path).parent == Path(audio_profile.audio_path).parent

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
    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"voice")
    with pytest.raises(ValueError, match="需要提供情感参考音频"):
        VoiceLibrary(tmp_path / "data").save(
            "角色A", voice, emotion_mode="reference_audio"
        )


def test_voice_library_v2_search_favorite_quality_and_notes(tmp_path: Path):
    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"voice")
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
    voice = tmp_path / "voice.wav"
    emotion = tmp_path / "emotion.wav"
    voice.write_bytes(b"voice-data")
    emotion.write_bytes(b"emotion-data")
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
    assert Path(restored.audio_path).read_bytes() == b"voice-data"
    assert Path(restored.emotion_audio_path).read_bytes() == b"emotion-data"
    assert restored.tags == ("主角", "日语")
    assert restored.favorite is True
    assert restored.quality["score"] == 88

    renamed = target.import_bundle(bundle, conflict="rename")
    assert renamed[0].name == "角色A（导入 2）"
    assert target.import_bundle(bundle, conflict="skip") == []


def test_voice_bundle_rejects_unsafe_members(tmp_path: Path):
    import zipfile

    bundle = tmp_path / "unsafe.t8voice.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("../escape.wav", b"bad")
        archive.writestr(
            "manifest.json",
            '{"schemaVersion":1,"profiles":[{"name":"A","audio_path":"../escape.wav"}]}',
        )
    with pytest.raises(ValueError, match="不安全路径"):
        VoiceLibrary(tmp_path / "data").import_bundle(bundle)
