from __future__ import annotations

import json
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torchaudio

import desktop_webui
from dialogue_runtime import parse_emotion_override
from desktop_tasks import create_task, update_task_line
from desktop_voice_library import VoiceLibrary, VoiceProfile
from desktop_webui import (
    build_app,
    delete_pronunciation_entry,
    describe_dialogue_timing_settings,
    load_history,
    line_emotion_kwargs,
    profile_emotion_kwargs,
    pronunciation_dictionary_path,
    pronunciation_entry_choices,
    upsert_pronunciation_entry,
)
from indextts.pronunciation import PronunciationValidationError


def _write_test_wav(path: Path, *, sample: int = 100, frames: int = 240) -> Path:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(int(sample).to_bytes(2, "little", signed=True) * frames)
    return path


class _FakeTokenizer:
    @staticmethod
    def encode(text, allowed_special="all"):
        return list(text)


class _FakeTTS:
    cfg = SimpleNamespace(gpt=SimpleNamespace(max_mel_tokens=4096, max_text_tokens=300))
    tokenizer = _FakeTokenizer()

    @staticmethod
    def split_text_by_tokens(text, limit, prefix):
        return [text]

    @staticmethod
    def normalize_emo_vec(values, apply_bias=True):
        return values

    @staticmethod
    def infer(*_args, stream_return=False, **_kwargs):
        if stream_return:
            return iter((torch.ones(1, 220) * 0.1, torch.ones(1, 330) * 0.2))
        return 22050, torch.ones(1, 550) * 0.1


class _CapturingTTS(_FakeTTS):
    def __init__(self):
        self.calls = []

    def infer(self, *_args, stream_return=False, **kwargs):
        self.calls.append(kwargs)
        return super().infer(stream_return=stream_return, **kwargs)


class _FailingStreamAccelerationTTS(_FakeTTS):
    @staticmethod
    def infer(*_args, stream_return=False, **_kwargs):
        if not stream_return:
            raise AssertionError("Failed streaming acceleration must not be retried before fallback.")

        def fail_during_iteration():
            raise RuntimeError(
                "Keyword argument waves_per_eu was specified but unrecognised"
            )
            yield  # pragma: no cover - keeps this function a generator

        return fail_during_iteration()


class _FakeQwenEmotion:
    @staticmethod
    def inference(prompt):
        assert str(prompt).strip()
        return {
            "happy": 0.0,
            "angry": 0.9,
            "sad": 0.0,
            "afraid": 0.0,
            "disgusted": 0.0,
            "melancholic": 0.0,
            "surprised": 0.3,
            "calm": 0.0,
        }


class _EmotionSuggestTTS(_CapturingTTS):
    def __init__(self):
        super().__init__()
        self.qwen_emo = None

    def ensure_qwen_emotion(self):
        self.qwen_emo = _FakeQwenEmotion()


class _SegmentedTTS(_FakeTTS):
    segments = (
        "第一段包含足够多的文字用于建立稳定语速基线。",
        "第二段同样保持自然清楚并形成可靠的比较基线。",
        "第三段故意返回异常缓慢的音频以触发自动保护。",
    )

    @classmethod
    def split_text_by_tokens(cls, text, limit, prefix):
        if text == " ".join(cls.segments):
            return list(cls.segments)
        return [text]

    @classmethod
    def infer(cls, *_args, stream_return=False, **kwargs):
        if stream_return:
            return super().infer(stream_return=True, **kwargs)
        sample_rate = 1000
        text = str(kwargs.get("text") or "")
        collector = kwargs.get("segment_collector")
        if collector is not None and text == " ".join(cls.segments):
            durations = (2.0, 2.0, 8.0)
            parts = []
            for index, (segment, duration) in enumerate(zip(cls.segments, durations), 1):
                waveform = torch.ones(1, round(sample_rate * duration)) * (0.05 * index)
                collector.append(
                    {
                        "index": index,
                        "text": segment,
                        "language": "ZH",
                        "sample_rate": sample_rate,
                        "duration_seconds": duration,
                        "waveform": waveform,
                    }
                )
                parts.append(waveform)
            return sample_rate, torch.cat(parts, dim=-1)
        return sample_rate, torch.ones(1, 2000) * 0.25


def test_desktop_builds_complete_pronunciation_workspace(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs"
    data_dir = tmp_path / "user-data"
    output_dir.mkdir()
    data_dir.mkdir()

    demo = build_app(_FakeTTS(), output_dir, data_dir, verbose=False)
    config = demo.get_config_file()
    config_text = json.dumps(config, ensure_ascii=False)
    labels = {
        component.get("props", {}).get("label") for component in config["components"]
    }
    values = {
        component.get("props", {}).get("value")
        for component in config["components"]
        if isinstance(component.get("props", {}).get("value"), str)
    }
    pronunciation_accordion = next(
        component
        for component in config["components"]
        if component.get("type") == "accordion"
        and component.get("props", {}).get("label")
        == "发音与数字处理 · 数字归一化开启，发音词典按需使用"
    )
    stream_audio = next(
        component
        for component in config["components"]
        if component.get("props", {}).get("label") == "流式试听"
    )
    assert stream_audio["type"] == "audio"
    assert "bundledstreamingaudio" not in config_text.lower()
    dictionary_language = next(
        component
        for component in config["components"]
        if component.get("props", {}).get("label") == "语言（下拉选择）"
    )
    dictionary_preview = next(
        component
        for component in config["components"]
        if component.get("props", {}).get("label")
        == "持久发音词典预览（请使用上方编辑器增删改）"
    )
    assert {choice[1] for choice in dictionary_language["props"]["choices"]} == {
        "ZH",
        "EN",
        "JA",
        "ES",
        "AR",
    }
    assert dictionary_preview["props"]["interactive"] is False
    assert "实际送入模型的文本" in labels
    assert "严格校验" in labels
    assert "Beam 数量" in labels
    assert "长度惩罚" in labels
    assert "段间静音（毫秒）" in labels
    assert "文本归一化（数字/日期）" in labels
    assert "预设名称" in labels
    assert "已保存预设" in labels
    assert "使用已保存音色库（免重复上传）" in labels
    assert "音色参考音频（音色库自动载入，或手动上传/录音）" in labels
    assert "角色音色参考" in labels
    assert "该角色默认情感模式" in labels
    assert "该角色情感参考音频" in labels
    assert "角色喜" in labels
    assert "该角色默认情感描述" in labels
    assert "该角色情感强度" in labels
    assert "该角色使用随机情感原型" in labels
    assert "更新所选角色（允许改名）" in labels
    assert "批量台词或 SRT 内容" in labels
    assert "官方时长适配倍率（无单位；小于 1 更短，大于 1 更长）" in labels
    assert "按 SRT 开始/结束时间匹配语音" in labels
    assert "时间冲突策略（上一句超时怎么办）" in labels
    assert "SRT 时长处理方式" in labels
    assert "触发二次适配的误差（毫秒）" in labels
    assert "普通批量台词句间静音（毫秒）" in labels
    assert "下载合并音频 WAV" in labels
    assert "CFM 扩散步数" in labels
    assert "CFM 引导强度" in labels
    assert "CFM 温度" in labels
    assert "随机种子" in labels
    assert "边生成边试听" in labels
    assert "内部文本分段语速报告" in labels
    assert "选择内部段" in labels
    assert "原始段试听" in labels
    assert "自动重试候选试听（未触发时为空）" in labels
    assert "当前采用段试听" in labels
    assert "已保存任务" in labels
    assert "要重做的台词序号" in labels
    assert "生成后逐句自动 ASR 校对" in labels
    assert "回写字幕时间" in labels
    assert "回写字幕文本" in labels
    assert (
        "可编辑时间轴（表格与下方可拖拽轨道双向同步；最后一列可逐句改情感）" in labels
    )
    assert "导出格式" in labels
    assert "时间轴文件下载" in labels
    assert "导入可编辑时间轴 JSON / CSV" in labels
    assert "参考缓存条目、容量与命中统计" in labels
    assert "__t8TimelineEditorInstalled" in config_text
    assert "t8-timeline-drag-payload" in config_text
    assert "每侧上下文台词数" in labels
    assert "覆盖已有逐句情感" in labels
    assert "上下文情感建议报告 JSON" in labels
    assert "完整加速能力报告" in labels
    assert "本次启动环境与加速诊断" not in labels
    assert "生成语音" in values
    assert "停止语音任务" in values
    assert "停止多角色任务" in values
    assert "返回启动配置（停止模型）" in values
    assert "打开输出目录" in values
    assert "打开日志目录" in values
    assert "打开用户数据目录" in values
    assert "当前运行状态正常" in config_text
    assert "技术诊断 JSON（排错时展开或复制）" in config_text
    assert "window.desktopApi.showLauncher" in config_text
    button_ids = {
        component.get("props", {}).get("value"): component["id"]
        for component in config["components"]
        if component.get("type") == "button"
    }
    generation_dependencies = [
        dependency
        for dependency in config["dependencies"]
        for target_id, _event_name in dependency.get("targets", [])
        if target_id == button_ids["生成语音"]
        and dependency.get("backend_fn")
    ]
    assert len(generation_dependencies) == 1
    assert generation_dependencies[0]["inputs"]
    assert generation_dependencies[0]["outputs"]
    assert pronunciation_accordion["props"]["open"] is False
    assert "一键填入中文示例" in values
    assert "多音字怎么用" in config_text
    assert "中文数字/日期归一化已就绪" in config_text
    assert "1939年" in config_text
    assert "<行长|HANG2 ZHANG3>" in config_text
    assert "M IH1 . N AH0 T" in config_text
    assert "じょうず" in config_text
    assert "添加/更新到表格" in config_text
    assert "载入 / 试听 / 编辑" in config_text
    assert "刷新音色库" in config_text
    assert "生成全部台词" in config_text
    assert "可选语言" in config_text
    assert "无单位；小于 1 更短，大于 1 更长" in config_text
    assert "This is a real English example." in config_text
    assert "载入批量真实示例" in config_text
    assert "载入 SRT 真实示例" in config_text
    assert "时间与 SRT 适配 · 普通批量默认顺延、间隔 200ms" in config_text
    assert "模型第一次生成了 **2.3 秒**" in config_text
    assert "上下文情感建议 · 默认关闭，确认后才生成" in config_text
    assert "分析上下文并填入建议" in config_text
    accordion_states = {
        component.get("props", {}).get("label"): component.get("props", {}).get("open")
        for component in config["components"]
        if component.get("type") == "accordion"
    }
    for collapsed_label in [
        "参考音频检测与裁剪 · 默认使用安全设置",
        "情感与声音表现 · 默认跟随音色",
        "高级生成参数 · 默认设置可直接使用",
        "声音 A/B 候选试听、评分与收藏 · 生成后按需展开",
        "跨段语速审计与内部单段重做 · 生成后按需展开",
        "格式说明与真实示例 · 新手需要时展开",
        "时间与 SRT 适配 · 普通批量默认顺延、间隔 200ms",
        "ASR 校对与字幕回写 · 默认关闭",
        "上下文情感建议 · 默认关闭，确认后才生成",
        "可拖拽时间轴 · 解析后按需展开",
        "字幕、逐句文件与生成报告 · 生成后展开",
        "任务恢复、单句重试与工程管理 · 按需展开",
        "模型与显存生命周期 · 默认空闲 10 分钟自动释放",
        "一键安装 / 更新可选组件 · 按需展开",
        "队列使用说明 · 按需展开",
    ]:
        assert accordion_states[collapsed_label] is False
    cancellation_dependencies = [
        item for item in config["dependencies"] if item.get("cancels")
    ]
    assert len(cancellation_dependencies) == 3
    single_stop_id = button_ids["停止语音任务"]
    dialogue_stop_id = button_ids["停止多角色任务"]
    queue_stop_id = button_ids["停止队列执行"]
    stop_dependencies = {
        target_id: dependency
        for dependency in cancellation_dependencies
        for target_id, _event_name in dependency.get("targets", [])
        if target_id in {single_stop_id, dialogue_stop_id, queue_stop_id}
    }
    assert len(stop_dependencies[single_stop_id]["cancels"]) == 1
    assert len(stop_dependencies[dialogue_stop_id]["cancels"]) == 4
    assert len(stop_dependencies[queue_stop_id]["cancels"]) == 1
    assert (
        pronunciation_dictionary_path(data_dir)
        == data_dir / "pronunciation_dictionary.yaml"
    )

    batch_help = describe_dialogue_timing_settings(
        "batch", "overlay", False, "native", 180, 300
    )
    assert "普通批量台词重叠" in batch_help
    assert "多句可能从 0 秒同时播放" in batch_help
    srt_help = describe_dialogue_timing_settings(
        "srt", "shift", True, "exact", 180, 200
    )
    assert "强制精确" in srt_help
    assert "可能丢失句尾" in srt_help

    preview = next(
        block.fn
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "preview_dialogue_event"
    )
    rows, timeline_html, status = preview(
        "batch", "旁白|这是预览测试|ZH|1.0", "旁白", "ZH"
    )
    assert rows[0][1:3] == ["旁白", "ZH"]
    assert "总时长" in timeline_html
    assert "已解析 1 条台词" in status

    generate_block = next(
        block
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "generate"
    )
    inputs = [getattr(component, "value", None) for component in generate_block.inputs]
    inputs[0] = str(_write_test_wav(tmp_path / "queue-prompt.wav"))
    inputs[1] = "流式生成回归测试"
    inputs[4] = 0
    inputs[9] = []
    outputs = list(generate_block.fn(*inputs))
    assert len(outputs) == 3
    assert outputs[0][0][0] == 22050
    assert outputs[0][0][1].shape == (220,)
    assert Path(outputs[-1][1]).is_file()
    assert "RTF" in outputs[-1][4]

    monkeypatch.setattr(
        desktop_webui,
        "transcribe_audio_file",
        lambda *args, **kwargs: {
            "text": "流式生成回归测试",
            "model": "tiny",
            "device": "cpu",
            "backend": "openai_whisper",
            "word_timestamps": [],
        },
    )
    asr_fn = next(
        block.fn
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "asr_proofread_event"
    )
    recognized, verdict, differences, report, alignment = asr_fn(
        outputs[-1][1], "流式生成回归测试", "ZH", "auto", "tiny", "cpu", 0.8
    )
    assert recognized == "流式生成回归测试"
    assert "通过" in verdict
    assert "未发现" in differences
    assert json.loads(report)["passed"] is True
    assert "t8-waveform" in alignment


def test_single_generation_can_reuse_saved_voice_without_uploading(tmp_path):
    output_dir = tmp_path / "outputs"
    data_dir = tmp_path / "user-data"
    output_dir.mkdir()
    data_dir.mkdir()
    source = _write_test_wav(tmp_path / "narrator.wav")
    saved = VoiceLibrary(data_dir).save("旁白", source)

    demo = build_app(_FakeTTS(), output_dir, data_dir, verbose=False)
    load_saved_voice = next(
        block.fn
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "load_single_voice_event"
    )
    refresh_saved_voices = next(
        block.fn
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "refresh_single_voice_event"
    )

    audio_path, status = load_saved_voice("旁白")
    assert Path(audio_path) == Path(saved.audio_path)
    assert "可直接生成" in status
    assert "不会覆盖本页当前的语言、情感或生成参数" in status

    selector_update, refreshed_audio, refresh_status = refresh_saved_voices("旁白")
    assert selector_update["value"] == "旁白"
    assert Path(refreshed_audio) == Path(saved.audio_path)
    assert "共 1 个角色" in refresh_status

    save_saved_voice = next(
        block.fn
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "save_voice_event"
    )
    second_source = _write_test_wav(tmp_path / "role-b.wav", sample=200)
    save_result = save_saved_voice(
        "角色B",
        str(second_source),
        "ZH",
        0,
        None,
        "",
        0.65,
        False,
        *([0.0] * 8),
        "",
        None,
        False,
    )
    assert len(save_result) == 8
    assert save_result[3]["value"] == "角色B"
    assert Path(save_result[4]).is_file()
    assert "已同步“角色B”" in save_result[5]

    delete_saved_voice = next(
        block.fn
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "delete_voice_event"
    )
    delete_result = delete_saved_voice("角色B", "角色B")
    assert len(delete_result) == 8
    assert delete_result[3]["value"] is None
    assert delete_result[4] is None
    assert "请重新选择或上传参考音频" in delete_result[5]


def test_cross_segment_workspace_can_audition_redo_and_merge_one_internal_segment(tmp_path):
    output_dir = tmp_path / "outputs"
    data_dir = tmp_path / "user-data"
    output_dir.mkdir()
    data_dir.mkdir()
    demo = build_app(_SegmentedTTS(), output_dir, data_dir, verbose=False)
    generate_block = next(
        block
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "generate"
    )
    inputs = [getattr(component, "value", None) for component in generate_block.inputs]
    by_label = {
        getattr(component, "label", None): index
        for index, component in enumerate(generate_block.inputs)
    }
    inputs[0] = "fake-prompt.wav"
    inputs[1] = " ".join(_SegmentedTTS.segments)
    inputs[4] = 0
    inputs[9] = []
    inputs[by_label["边生成边试听"]] = False
    outputs = list(generate_block.fn(*inputs))[-1]

    assert len(outputs) == 10
    manifest_path = Path(outputs[9])
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["segments"]) == 3
    assert manifest["segments"][2]["selected_source"] == "auto_retry"
    assert "已采用重试" in outputs[7]

    load_artifacts = next(
        block.fn
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "load_segment_artifacts_event"
    )
    original, retry, selected, status = load_artifacts(str(manifest_path), "3")
    assert Path(original).is_file()
    assert Path(retry).is_file()
    assert Path(selected) == Path(retry)
    assert "当前采用" in status

    redo = next(
        block.fn
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "redo_internal_segment_event"
    )
    redo_result = redo(
        str(manifest_path),
        "3",
        progress=lambda *_args, **_kwargs: None,
    )
    assert Path(redo_result[0]).is_file()
    assert Path(redo_result[6]).name.startswith("segment_003_manual_retry_")
    assert "单独重做并合入" in redo_result[7]
    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated["segments"][2]["selected_source"] == "manual_retry"


def test_desktop_context_emotion_suggestions_fill_timeline_without_synthesis(tmp_path):
    output_dir = tmp_path / "outputs"
    data_dir = tmp_path / "user-data"
    output_dir.mkdir()
    data_dir.mkdir()
    tts = _EmotionSuggestTTS()
    demo = build_app(tts, output_dir, data_dir, verbose=False)
    suggest = next(
        block.fn
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "suggest_dialogue_emotions_event"
    )
    script = "角色A|先等等。|ZH|1.0\n角色B|你竟然骗了我！|ZH|1.0"
    rows, timeline_html, status, report_json = suggest(
        "batch",
        script,
        "角色A",
        "ZH",
        [],
        1,
        False,
        progress=lambda *_args, **_kwargs: None,
    )

    assert len(rows) == 2
    first_override = parse_emotion_override(rows[0][7])
    assert first_override[0] == "vector"
    assert first_override[2][1] == pytest.approx(0.6)
    assert first_override[2][6] == pytest.approx(0.2)
    assert "尚未生成音频" in status
    assert "总时长" in timeline_html
    report = json.loads(report_json)
    assert report["classified_count"] == 2
    assert report["requires_user_confirmation"] is True
    assert report["started_synthesis"] is False
    assert report["temporary_qwen_released"] is True
    assert tts.qwen_emo is None
    assert tts.calls == []


def test_saved_role_emotion_modes_resolve_to_isolated_inference_arguments(tmp_path):
    voice = tmp_path / "voice.wav"
    emotion = tmp_path / "emotion.wav"
    voice.write_bytes(b"voice")
    emotion.write_bytes(b"emotion")

    speaker = VoiceProfile("speaker", "旁白", str(voice))
    assert profile_emotion_kwargs(_FakeTTS(), speaker, True)["emo_vector"] is None

    audio_profile = VoiceProfile(
        "audio",
        "角色A",
        str(voice),
        emotion_mode="reference_audio",
        emotion_audio_path=str(emotion),
        emotion_strength=0.7,
    )
    audio_kwargs = profile_emotion_kwargs(_FakeTTS(), audio_profile, True)
    assert audio_kwargs["emo_audio_prompt"] == str(emotion)
    assert audio_kwargs["emo_alpha"] == pytest.approx(0.7)

    vector_profile = VoiceProfile(
        "vector",
        "角色B",
        str(voice),
        emotion_mode="vector",
        emotion_vector=(0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2),
        emotion_use_random=True,
    )
    vector_kwargs = profile_emotion_kwargs(_FakeTTS(), vector_profile, True)
    assert vector_kwargs["emo_vector"] == list(vector_profile.emotion_vector)
    assert vector_kwargs["use_random"] is True
    assert vector_kwargs["emo_audio_prompt"] is None

    text_profile = VoiceProfile(
        "text",
        "角色C",
        str(voice),
        emotion_mode="text",
        emotion_text="克制但紧张",
    )
    text_kwargs = profile_emotion_kwargs(_FakeTTS(), text_profile, True)
    assert text_kwargs["use_emo_text"] is True
    assert text_kwargs["emo_text"] == "克制但紧张"
    with pytest.raises(ValueError, match="未加载 QwenEmotion"):
        profile_emotion_kwargs(_FakeTTS(), text_profile, False)


def test_line_emotion_override_wins_without_mutating_role_default(tmp_path):
    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"voice")
    profile = VoiceProfile(
        "role",
        "旁白",
        str(voice),
        emotion_mode="text",
        emotion_text="角色默认平静",
    )
    line = desktop_webui.DialogueLine(
        1,
        "旁白",
        "突然生气。",
        emotion_mode="vector",
        emotion_vector=(0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        emotion_strength=0.7,
    )
    kwargs, source = line_emotion_kwargs(_FakeTTS(), profile, line, True)
    assert source == "line_override"
    assert kwargs["emo_vector"][1] == pytest.approx(0.8)
    assert kwargs["emo_alpha"] == pytest.approx(0.7)
    assert profile.emotion_text == "角色默认平静"


def test_desktop_single_generation_retries_failed_asr_candidate(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs"
    data_dir = tmp_path / "user-data"
    output_dir.mkdir()
    data_dir.mkdir()
    tts = _CapturingTTS()
    demo = build_app(tts, output_dir, data_dir, verbose=False)
    transcripts = iter(("错误结果", "自动质检文本", "自动质检文版"))
    monkeypatch.setattr(desktop_webui, "asr_available", lambda *_args: True)
    monkeypatch.setattr(
        desktop_webui,
        "transcribe_audio_file",
        lambda *_args, **_kwargs: {
            "text": next(transcripts),
            "model": "tiny",
            "device": "cpu",
            "backend": "openai_whisper",
            "word_timestamps": [],
        },
    )
    generate_block = next(
        block
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "generate"
    )
    inputs = [getattr(component, "value", None) for component in generate_block.inputs]
    by_label = {
        getattr(component, "label", None): index
        for index, component in enumerate(generate_block.inputs)
    }
    inputs[0] = "fake-prompt.wav"
    inputs[1] = "自动质检文本"
    inputs[4] = 0
    inputs[9] = []
    inputs[by_label["追加候选数量"]] = 2
    inputs[by_label["质检 ASR 模型"]] = "tiny"
    inputs[by_label["质检 ASR 设备"]] = "cpu"

    outputs = list(generate_block.fn(*inputs))

    assert len(tts.calls) == 3
    assert Path(outputs[-1][1]).is_file()
    assert len(outputs[-1][2]) == 3
    assert '"attempt_count": 3' in outputs[-1][4]
    assert "RTF" in outputs[-1][5]


def test_desktop_streaming_acceleration_failure_reloads_normal_mode(tmp_path):
    output_dir = tmp_path / "outputs"
    data_dir = tmp_path / "user-data"
    output_dir.mkdir()
    data_dir.mkdir()
    normal = _CapturingTTS()
    demo = build_app(
        _FailingStreamAccelerationTTS(),
        output_dir,
        data_dir,
        verbose=False,
        fallback_factory=lambda: normal,
    )
    generate_block = next(
        block
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "generate"
    )
    inputs = [getattr(component, "value", None) for component in generate_block.inputs]
    by_label = {
        getattr(component, "label", None): index
        for index, component in enumerate(generate_block.inputs)
    }
    inputs[0] = "fake-prompt.wav"
    inputs[1] = "流式加速失败后应自动回退普通模式。"
    inputs[4] = 0
    inputs[9] = []
    inputs[by_label["边生成边试听"]] = True

    outputs = list(generate_block.fn(*inputs))

    assert len(normal.calls) == 1
    assert Path(outputs[-1][1]).is_file()
    final_text = "\n".join(str(item) for item in outputs[-1])
    assert "waves_per_eu" in final_text
    assert "已释放加速模型、重载普通模式并自动重试" in final_text


def test_desktop_dialogue_routes_saved_emotions_per_role(tmp_path):
    output_dir = tmp_path / "outputs"
    data_dir = tmp_path / "user-data"
    output_dir.mkdir()
    data_dir.mkdir()
    voice_a = _write_test_wav(tmp_path / "a.wav", sample=100)
    voice_b = _write_test_wav(tmp_path / "b.wav", sample=200)
    emotion_b = _write_test_wav(tmp_path / "b-emotion.wav", sample=300)
    library = VoiceLibrary(data_dir)
    library.save(
        "角色A",
        voice_a,
        emotion_mode="vector",
        emotion_vector=(0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2),
    )
    saved_b = library.save(
        "角色B",
        voice_b,
        emotion_mode="reference_audio",
        emotion_audio=emotion_b,
        emotion_strength=0.7,
    )
    tts = _CapturingTTS()
    demo = build_app(tts, output_dir, data_dir, verbose=False)
    generate_dialogue = next(
        block
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "generate_dialogue_event"
    )
    inputs = [
        getattr(component, "value", None) for component in generate_dialogue.inputs
    ]
    inputs[0] = "batch"
    inputs[1] = (
        "角色A|高兴地出发。|ZH|1.0\n"
        "角色B|担心地回应。|ZH|1.0\n"
        "角色A|同一个角色突然生气。|ZH|1.0|text:愤怒、激动"
    )
    inputs[2] = "角色A"
    inputs[3] = "ZH"
    inputs[4] = []
    result = list(generate_dialogue.fn(*inputs))[-1]

    assert len(tts.calls) == 3
    first, second, third = tts.calls
    assert first["emo_vector"] is not None
    assert first["emo_audio_prompt"] is None
    assert second["emo_vector"] is None
    assert second["emo_audio_prompt"] == saved_b.emotion_audio_path
    assert second["emo_alpha"] == pytest.approx(0.7)
    assert third["use_emo_text"] is True
    assert third["emo_text"] == "愤怒、激动"
    assert third["emo_vector"] is None
    report = json.loads(result[3])
    assert [line["emotion_mode"] for line in report["lines"]] == [
        "vector",
        "reference_audio",
        "text",
    ]
    assert [line["emotion_source"] for line in report["lines"]] == [
        "role_default",
        "role_default",
        "line_override",
    ]
    assert report["performance"]["elapsed_seconds"] >= 0
    assert report["performance"]["rtf"] >= 0
    assert "RTF" in result[6]


def test_saved_dialogue_can_be_retimed_and_rewritten_without_tts(tmp_path):
    output_dir = tmp_path / "outputs"
    data_dir = tmp_path / "user-data"
    output_dir.mkdir()
    data_dir.mkdir()
    demo = build_app(_FakeTTS(), output_dir, data_dir, verbose=False)

    task_id = "dialogue_20260825_120000_abcdef12"
    script = "1\n00:00:00,000 --> 00:00:01,000\n[旁白] 时间轴测试。"
    task = create_task(
        output_dir,
        task_id,
        script_type="srt",
        script=script,
        settings={
            "default_role": "旁白",
            "default_language": "ZH",
            "postprocess_preset": "off",
        },
        line_count=1,
    )
    clip = output_dir / task_id / "0001_旁白.wav"
    torchaudio.save(str(clip), torch.ones(1, 22050) * 0.05, 22050)
    update_task_line(
        output_dir,
        task,
        1,
        status="completed",
        file=str(clip),
        report={"index": 1, "actual_duration_ms": 1000},
    )
    rebuild = next(
        block.fn
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "rebuild_dialogue_timeline_event"
    )
    result = rebuild(
        task_id,
        [[1, "旁白", "ZH", 200, 1200, 1.0, "时间轴测试。"]],
        "overlay",
        0,
        "actual",
        "original",
        True,
    )
    assert Path(result[0]).is_file()
    assert Path(result[1]).is_file()
    assert Path(result[2]).read_text(encoding="utf-8-sig").startswith("1\n00:00:00,200")
    assert "未重新执行 TTS" in result[6]
    assert "总时长 1.20s" in result[7]

    with pytest.raises(Exception, match="旧音频已不匹配"):
        rebuild(
            task_id,
            [[1, "旁白", "ZH", 200, 1200, 1.0, "已经换成另一句。"]],
            "overlay",
            0,
            "actual",
            "original",
            True,
        )


def test_memory_policy_recovers_from_bad_values_and_persists_updates(tmp_path):
    output_dir = tmp_path / "outputs"
    data_dir = tmp_path / "user-data"
    output_dir.mkdir()
    data_dir.mkdir()
    (data_dir / "memory_policy.json").write_text(
        json.dumps({"idle_seconds": "not-a-number", "recycle_after_generations": None}),
        encoding="utf-8",
    )
    demo = build_app(_FakeTTS(), output_dir, data_dir, verbose=False)
    update_policy = next(
        block.fn
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "update_memory_policy_event"
    )
    report = json.loads(update_policy(True, 120, 3))

    assert report["policy"] == {
        "release_after_generation": True,
        "idle_seconds": 120.0,
        "recycle_after_generations": 3,
    }
    saved = json.loads((data_dir / "memory_policy.json").read_text(encoding="utf-8"))
    assert saved == report["policy"]


def test_persistent_queue_replays_single_generation_after_enqueue(tmp_path):
    output_dir = tmp_path / "outputs"
    data_dir = tmp_path / "user-data"
    output_dir.mkdir()
    data_dir.mkdir()
    demo = build_app(_FakeTTS(), output_dir, data_dir, verbose=False)
    generate_block = next(
        block
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "generate"
    )
    enqueue = next(
        block.fn
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "enqueue_single_event"
    )
    run_queue = next(
        block.fn
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "run_job_queue_event"
    )
    inputs = [getattr(component, "value", None) for component in generate_block.inputs]
    by_label = {
        getattr(component, "label", None): index
        for index, component in enumerate(generate_block.inputs)
    }
    inputs[0] = str(_write_test_wav(tmp_path / "queue-prompt.wav"))
    inputs[1] = "持久队列真实回放。"
    inputs[4] = 0
    inputs[9] = []
    inputs[by_label["边生成边试听"]] = False
    queued_rows, _selector, status = enqueue(*inputs)
    assert queued_rows[0][2] == "pending"
    assert "已加入单句任务" in status

    queue_outputs = list(run_queue())
    final_rows = queue_outputs[-1][0]
    assert final_rows[0][2] == "completed"
    assert Path(final_rows[0][4]).is_file()
    saved_queue = json.loads((data_dir / "task_queue.json").read_text(encoding="utf-8"))
    assert saved_queue["jobs"][0]["status"] == "completed"
    queued_prompt = Path(saved_queue["jobs"][0]["payload"]["inputs"][0])
    assert data_dir / "queue_assets" in queued_prompt.parents
    assert queued_prompt.is_file()


def test_preflight_and_candidate_ab_events_are_wired_to_collapsed_workspaces(tmp_path):
    output_dir = tmp_path / "outputs"
    data_dir = tmp_path / "user-data"
    output_dir.mkdir()
    data_dir.mkdir()
    demo = build_app(_FakeTTS(), output_dir, data_dir, verbose=False)

    preview = next(
        block.fn
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "preview_segments_event"
    )
    rows, normalized, status = preview(
        "1939年，长文本预检。",
        "ZH",
        [],
        True,
        "auto",
        120,
        "natural",
        100,
        300,
        600,
    )
    assert normalized
    assert len(rows[0]) == 8
    assert rows[0][3] > 0
    assert "预计总时长" in status

    candidate_a = data_dir / "quality_candidates" / "test" / "a.wav"
    candidate_b = data_dir / "quality_candidates" / "test" / "b.wav"
    candidate_a.parent.mkdir(parents=True)
    candidate_a.write_bytes(b"RIFF-a")
    candidate_b.write_bytes(b"RIFF-b")
    refresh_candidates = next(
        block.fn
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "refresh_candidate_workspace_event"
    )
    selector, selected_audio, rating, note, candidate_status = refresh_candidates(
        [str(candidate_a), str(candidate_b)]
    )
    assert len(selector["choices"]) == 2
    assert selected_audio == str(candidate_a)
    assert rating == 3
    assert note == ""
    assert "2 个候选" in candidate_status


def test_edited_timeline_row_can_be_regenerated_alone_and_merged(tmp_path):
    output_dir = tmp_path / "outputs"
    data_dir = tmp_path / "user-data"
    output_dir.mkdir()
    data_dir.mkdir()
    voice = _write_test_wav(tmp_path / "voice.wav")
    VoiceLibrary(data_dir).save("贞贞", voice)
    tts = _CapturingTTS()
    demo = build_app(tts, output_dir, data_dir, verbose=False)

    generate_block = next(
        block
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "generate_dialogue_event"
    )
    preview = next(
        block.fn
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "preview_dialogue_event"
    )
    script = "贞贞|第一句。|ZH|1.0\n贞贞|我是你们的贞贞。|ZH|1.0"
    rows, _visual, _status = preview("batch", script, "贞贞", "ZH")
    inputs = [getattr(component, "value", None) for component in generate_block.inputs]
    inputs[0] = "batch"
    inputs[1] = script
    inputs[2] = "贞贞"
    inputs[3] = "ZH"
    inputs[4] = rows
    list(generate_block.fn(*inputs))
    assert len(tts.calls) == 2

    task_id = next(output_dir.glob("dialogue_*/task.json")).parent.name
    edited_rows = [list(row) for row in rows]
    edited_rows[1][6] = "我是你们的贞贞啊啊。"
    retry_inputs = list(inputs)
    retry_inputs[4] = edited_rows
    retry_inputs[-2] = task_id
    retry_inputs[-1] = 2
    retry_result = list(generate_block.fn(*retry_inputs))[-1]

    assert len(tts.calls) == 3
    assert tts.calls[-1]["text"] == "我是你们的贞贞啊啊。"
    assert Path(retry_result[0]).is_file()
    assert retry_result[-2] == retry_result[0]
    assert "100%" in retry_result[-1]
    report = json.loads(retry_result[3])
    assert report["lines"][0]["restored_from_task"] is True
    assert report["lines"][1]["text"] == "我是你们的贞贞啊啊。"

    load_editor = next(
        block.fn
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "load_dialogue_task_editor_event"
    )
    loaded = load_editor(task_id)
    assert loaded[4][1][6] == "我是你们的贞贞啊啊。"
    assert "单独重做并合入" in loaded[7]

    select_row = next(
        block.fn
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "select_timeline_row_event"
    )
    selected_number, selected_status = select_row(
        edited_rows,
        SimpleNamespace(index=(1, 6)),
    )
    assert selected_number == 2
    assert "第 2 条" in selected_status
    move_up = next(
        block.fn
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "move_dialogue_line_up_event"
    )
    moved_rows, moved_visual, moved_status, moved_number = move_up(
        "batch",
        script,
        "贞贞",
        "ZH",
        edited_rows,
        2,
        retry_result[3],
    )
    assert moved_number == 1
    assert moved_rows[0][6] == "我是你们的贞贞啊啊。"
    assert moved_rows[1][6] == "第一句。"
    assert [row[0] for row in moved_rows] == [1, 2]
    assert "原有 SRT 时间槽保持原位" in moved_status
    assert "我是你们的贞贞啊啊" in moved_visual
    refresh_bindings = [
        block
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "refresh_timeline_event"
    ]
    assert len(refresh_bindings) == 2


def test_editable_timeline_ui_exports_and_imports_without_full_project(tmp_path):
    output_dir = tmp_path / "outputs"
    data_dir = tmp_path / "user-data"
    output_dir.mkdir()
    data_dir.mkdir()
    demo = build_app(_FakeTTS(), output_dir, data_dir, verbose=False)
    preview = next(
        block.fn
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "preview_dialogue_event"
    )
    export_timeline = next(
        block.fn
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "export_editable_timeline_event"
    )
    import_timeline = next(
        block.fn
        for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "import_editable_timeline_event"
    )
    script = (
        "旁白|第一句。|ZH|1.0|text:平静;strength=0.75\n"
        "角色A|第二句。|ZH|1.0|vector:0,0.8,0,0,0,0,0,0"
    )
    rows, _visual, _status = preview("batch", script, "旁白", "ZH")
    rows[0][3:5] = [100, 900]

    exported, status = export_timeline("batch", script, "旁白", "ZH", rows, "json")
    assert Path(exported).is_file()
    assert "2 条" in status
    imported_type, imported_script, imported_rows, visual, status, selected = import_timeline(
        exported, "旁白", "ZH"
    )
    assert imported_type == "batch"
    assert "strength=0.75" in imported_script
    assert imported_rows[0][3:5] == [100, 900]
    assert "第一句" in visual
    assert "用于下一次生成" in status
    assert selected == 1


def test_dropdown_dictionary_editor_adds_updates_and_deletes_entries():
    rows, selected, warnings = upsert_pronunciation_entry(
        [], None, "银行", "ZH", "yin2 hang2", True, True
    )
    assert warnings == ()
    assert selected == "0"
    assert rows == [["银行", "ZH", "YIN2 HANG2", True, True]]
    assert pronunciation_entry_choices(rows)[0][1] == "0"

    rows, selected, _warnings = upsert_pronunciation_entry(
        rows, None, "minute", "EN", "m ih1 . n ah0 t", True, False
    )
    assert selected == "1"
    assert rows[1] == ["minute", "EN", "M IH1 . N AH0 T", True, False]

    with pytest.raises(PronunciationValidationError, match="已存在"):
        upsert_pronunciation_entry(
            rows, None, "Minute", "EN", "M IH1 N AH0 T", True, False
        )
    with pytest.raises(PronunciationValidationError, match="声调"):
        upsert_pronunciation_entry(rows, None, "错误", "ZH", "BAD9", True, True)

    assert delete_pronunciation_entry(rows, "0") == [
        ["minute", "EN", "M IH1 . N AH0 T", True, False]
    ]


def test_history_keeps_original_and_resolved_pronunciation_text(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "history.jsonl").write_text(
        '{"created_at":"now","language":"ZH","text":"银行","resolved_text":"<银行|YIN2 HANG2>","file":"x.wav"}\n',
        encoding="utf-8",
    )
    row = load_history(output_dir)[0]
    assert row[4:6] == ["银行", "<银行|YIN2 HANG2>"]
