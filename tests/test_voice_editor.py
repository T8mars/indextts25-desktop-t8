"""Role-editor workflows must not overwrite a voice when adding another one."""

import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

import desktop_webui
from desktop_voice_library import VoiceLibrary


def _write_voice(path, sample):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(sample.to_bytes(2, "little", signed=True) * 240)
    return str(path)


@pytest.fixture
def editor(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    data_dir.mkdir()
    output_dir.mkdir()
    first = _write_voice(tmp_path / "first.wav", 100)
    second = _write_voice(tmp_path / "second.wav", 200)
    library = VoiceLibrary(data_dir)
    original = library.save("角色A", first, emotion_text="平静", tags="旁白")
    tts = SimpleNamespace(
        cfg=SimpleNamespace(gpt=SimpleNamespace(max_mel_tokens=4096, max_text_tokens=300))
    )
    demo = desktop_webui.build_app(tts, output_dir, data_dir, verbose=False)
    events = {
        getattr(block.fn, "__name__", ""): block
        for block in demo.fns.values()
        if block.fn is not None
    }
    yield SimpleNamespace(
        demo=demo, events=events, library=library, original=original, second=second
    )
    demo.close()


def _loaded_form(editor):
    load = editor.events["load_voice_event"]
    result = load.fn("角色A")
    form = {component._id: value for component, value in zip(load.outputs, result)}
    selector = next(
        component
        for component in editor.events["save_voice_event"].inputs
        if component.label == "选择已有角色"
    )
    form[selector._id] = "角色A"
    return form


def _save(editor, form, **changes):
    event = editor.events["save_voice_event"]
    values = [
        changes.get(component.label, form.get(component._id, component.value))
        for component in event.inputs
    ]
    return event.fn(*values)


def test_loading_then_saving_a_second_role_preserves_the_first(editor):
    form = _loaded_form(editor)
    result = _save(editor, form, 角色名称="角色B", 角色音色参考=editor.second)

    assert [item.name for item in editor.library.list()] == ["角色A", "角色B"]
    assert editor.library.get("角色A") == editor.original
    assert result[6] is False


def test_new_role_with_duplicate_name_does_not_change_saved_files(editor):
    manifest_before = editor.library.manifest_path.read_bytes()
    audio_before = Path(editor.original.audio_path).read_bytes()
    form = _loaded_form(editor)
    with pytest.raises(desktop_webui.gr.Error, match="角色名称已存在"):
        _save(
            editor,
            form,
            **{
                "角色名称": "角色a",
                "角色音色参考": editor.second,
                "更新所选角色（允许改名）": False,
            },
        )

    assert editor.library.manifest_path.read_bytes() == manifest_before
    assert Path(editor.original.audio_path).read_bytes() == audio_before


def test_explicit_update_still_renames_the_selected_role(editor):
    result = _save(
        editor,
        _loaded_form(editor),
        **{
            "角色名称": "主角",
            "角色音色参考": editor.second,
            "更新所选角色（允许改名）": True,
        },
    )

    assert [item.name for item in editor.library.list()] == ["主角"]
    assert editor.library.get("主角").profile_id == editor.original.profile_id
    assert result[6] is False


def test_starting_a_new_role_clears_only_the_editor_and_can_save_another_role(editor):
    reset = editor.events["new_voice_event"]
    result = reset.fn()
    form = {component._id: value for component, value in zip(reset.outputs, result)}
    by_label = {
        component.label: value
        for component, value in zip(reset.outputs, result)
        if hasattr(component, "label")
    }
    assert by_label["角色名称"] == ""
    assert by_label["角色音色参考"] is None
    assert by_label["该角色默认情感模式"] == "跟随音色参考"
    assert by_label["更新所选角色（允许改名）"] is False
    assert by_label["选择已有角色"]["value"] is None
    assert editor.library.get("角色A") == editor.original

    # A browser applies gr.update values before submitting the next save event.
    for component_id, value in list(form.items()):
        if isinstance(value, dict) and value.get("__type__") == "update":
            form[component_id] = value.get("value")
    _save(editor, form, 角色名称="角色B", 角色音色参考=editor.second)
    assert [item.name for item in editor.library.list()] == ["角色A", "角色B"]
