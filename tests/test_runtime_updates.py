from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

import desktop_model_download
import desktop_webui
import runtime_acceleration
from desktop_presets import delete_preset, list_presets, load_preset, save_preset
from indextts.infer_v2_5 import (
    IndexTTS2,
    QwenEmotion,
    _console_text,
    select_gpt_inference_dtype,
)
from indextts.utils import common
from indextts.utils import model_download
from indextts.gpt import model_v2


def test_desktop_preset_round_trip_copies_audio_and_deletes_cleanly(tmp_path):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"RIFF-test")
    data_dir = tmp_path / "data"
    settings = {"language": "ZH", "advanced": {"num_beams": 4}}

    save_preset(data_dir, "温柔旁白", settings, prompt_audio=str(audio))
    assert list_presets(data_dir) == ["温柔旁白"]
    loaded = load_preset(data_dir, "温柔旁白")
    assert loaded["settings"] == settings
    assert loaded["audio"]["prompt"] != str(audio)
    assert open(loaded["audio"]["prompt"], "rb").read() == b"RIFF-test"

    assert delete_preset(data_dir, "温柔旁白") is True
    assert list_presets(data_dir) == []


def test_runtime_policy_uses_low_vram_mode_and_gates_qwen(monkeypatch):
    cuda = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        get_device_properties=lambda index: SimpleNamespace(total_memory=8 * 1024**3),
        is_bf16_supported=lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(desktop_webui.torch, "cuda", cuda)
    policy = desktop_webui.select_runtime_policy()
    assert policy["low_vram"] is True
    assert policy["use_bf16"] is True
    assert policy["use_fp16"] is False
    assert policy["reference_device"] == "cpu"
    assert policy["use_qwen_emo"] is False
    assert desktop_webui.select_runtime_policy(force_qwen_emo=True)["use_qwen_emo"] is True


def test_qwen_emotion_accepts_label_style_outputs():
    emotion = QwenEmotion.__new__(QwenEmotion)
    emotion.cn_key_to_en = {
        "高兴": "happy",
        "愤怒": "angry",
        "悲伤": "sad",
        "恐惧": "afraid",
        "反感": "disgusted",
        "低落": "melancholic",
        "惊讶": "surprised",
        "自然": "calm",
    }
    emotion.desired_vector_order = list(emotion.cn_key_to_en)
    emotion.max_score = 1.2
    emotion.min_score = 0.0
    assert emotion.convert({"emotion": "自然"})["calm"] == 1.0
    redirected = emotion.convert({"高兴": "自然"})
    assert redirected["happy"] == 0.0
    assert redirected["calm"] == 1.0


def test_console_logging_cannot_crash_on_spanish_under_legacy_windows_encoding(monkeypatch):
    monkeypatch.setattr("indextts.infer_v2_5.sys.stdout", SimpleNamespace(encoding="ascii"))
    assert _console_text("pequeño") == r"peque\xf1o"


def test_qwen_emotion_can_be_loaded_lazily_once(monkeypatch, tmp_path):
    created = []

    class FakeQwenEmotion:
        def __init__(self, model_dir):
            created.append(model_dir)

    monkeypatch.setattr("indextts.infer_v2_5.QwenEmotion", FakeQwenEmotion)
    model = IndexTTS2.__new__(IndexTTS2)
    model.model_dir = str(tmp_path)
    model.cfg = SimpleNamespace(qwen_emo_path="qwen")
    model.qwen_emo = None

    first = model.ensure_qwen_emotion()
    second = model.ensure_qwen_emotion()

    assert first is second
    assert created == [str(tmp_path / "qwen")]


def test_pcm_save_normalizes_before_torchaudio(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        common.torchaudio,
        "save",
        lambda path, wav, rate, **kwargs: captured.update(wav=wav.clone(), kwargs=kwargs),
    )
    pcm = torch.tensor([[0.0, 16383.5, -32767.0]], dtype=torch.float32)
    common.save_pcm_wav(tmp_path / "test.wav", pcm, 22050)
    assert captured["wav"].dtype == torch.float32
    assert float(captured["wav"].abs().max()) <= 1.0
    assert torch.allclose(captured["wav"], pcm / 32767.0)


def test_pcm_tail_fade_is_non_destructive_and_ends_at_zero():
    source = torch.full((1, 100), 32767.0)
    faded = common.fade_out_pcm_tail(source, 1000, duration_ms=20)
    assert torch.equal(source, torch.full((1, 100), 32767.0))
    assert torch.equal(faded[..., :-20], source[..., :-20])
    assert faded[0, -20] == source[0, -20]
    assert faded[0, -1] == 0
    assert torch.all(faded[..., -20:-1].diff() <= 0)


def test_config_download_uses_the_requested_official_model_version(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        model_download,
        "_download_single_file",
        lambda repo, filename, target: calls.append((repo, filename, target)),
    )
    model_download.ensure_config_available(str(tmp_path), version="2.5")
    assert calls[0][:2] == ("IndexTeam/IndexTTS-2.5", "config.yaml")


def test_desktop_manifest_uses_the_complete_huggingface_mirror_with_modelscope_fallback():
    metadata = desktop_model_download.MODEL_FILES["bpe.model"]
    assert metadata == {
        "size": 475997,
        "sha256": "b2a5ce8090d32da3642cc4f81fdc996376bc6dd3f4cd5e3d165f71120d9f2bc8",
        "modelScopeRepository": "IndexTeam/IndexTTS-2",
    }
    assert desktop_model_download.REPO_ID == "t8star/IndexTTS-2.5-Comfy"
    assert desktop_model_download._file_source("bpe.model", "huggingface") == (
        "t8star/IndexTTS-2.5-Comfy",
        desktop_model_download.MODEL_REVISION,
    )
    assert desktop_model_download._file_source("bpe.model", "modelscope") == (
        "IndexTeam/IndexTTS-2",
        desktop_model_download.MODELSCOPE_REVISION,
    )


def test_acceleration_preflight_reports_versions_without_loading_a_model():
    report = runtime_acceleration.probe_acceleration("cpu")
    assert report["cuda"] is False
    assert report["versions"]["torch"] == str(torch.__version__)
    assert set(report["versions"]) == {
        "torch",
        "cuda_runtime",
        "deepspeed",
        "flash_attn",
        "triton",
        "ninja",
    }


def test_deepspeed_receives_bfloat16_instead_of_treating_bf16_as_fp16(monkeypatch):
    class FakeInference:
        def eval(self):
            return self

    captured = {}

    def init_inference(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(module=FakeInference())

    monkeypatch.setattr(model_v2, "GPT2Config", lambda **kwargs: object())
    monkeypatch.setattr(model_v2, "GPT2InferenceModel", lambda *args, **kwargs: FakeInference())
    monkeypatch.setattr(model_v2.torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "deepspeed", SimpleNamespace(init_inference=init_inference))

    voice = model_v2.UnifiedVoice.__new__(model_v2.UnifiedVoice)
    voice.max_mel_tokens = 10
    voice.max_text_tokens = 10
    voice.number_mel_codes = 20
    voice.model_dim = 8
    voice.layers = 1
    voice.heads = 1
    voice.use_accel = False
    voice.gpt = SimpleNamespace(wte=None)
    voice.mel_pos_embedding = object()
    voice.mel_embedding = object()
    voice.final_norm = object()
    voice.mel_head = object()
    voice.post_init_gpt2_config(
        use_deepspeed=True,
        kv_cache=True,
        half=True,
        deepspeed_dtype=torch.bfloat16,
    )
    assert captured["dtype"] is torch.bfloat16
    assert captured["replace_with_kernel_inject"] is True


def test_desktop_deepspeed_uses_windows_kernel_compatible_fp16():
    assert select_gpt_inference_dtype(True, True, "cuda:0") is torch.float16
    assert select_gpt_inference_dtype(True, False, "cuda:0") is torch.bfloat16
    assert select_gpt_inference_dtype(False, False, "cpu") is None
    assert select_gpt_inference_dtype(False, False, "cuda:0", True) is torch.float16


def test_desktop_auto_precision_falls_back_to_fp16_without_native_bf16(monkeypatch):
    cuda = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        get_device_properties=lambda index: SimpleNamespace(total_memory=8 * 1024**3),
        is_bf16_supported=lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(desktop_webui.torch, "cuda", cuda)
    policy = desktop_webui.select_runtime_policy()
    assert policy["precision"] == "float16"
    assert policy["use_fp16"] is True
    assert policy["use_bf16"] is False
