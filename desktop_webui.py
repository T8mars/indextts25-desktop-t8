"""IndexTTS 2.5-only desktop WebUI for the T8star-Aix Electron package."""

from __future__ import annotations

import argparse
import gc
import json
import threading
import time
import traceback
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path

import gradio as gr
import torch
import torchaudio

from audio_quality import analyze_reference_audio, prepare_reference_audio, waveform_html
from candidate_quality import combined_candidate_score, select_best_candidate, technical_audio_review
from context_emotion import suggest_context_emotions
from audiocpp_backend import (
    AUDIOCPP_MODEL_REPOSITORY,
    probe_audiocpp,
    run_audiocpp,
)
from audiocpp_component_manager import (
    component_status as audiocpp_component_status,
    install_model as install_audiocpp_model,
    install_runtime as install_audiocpp_runtime,
)
from desktop_project_bundle import export_project, import_project
from desktop_voice_library import VoiceLibrary, VoiceProfile, safe_voice_file_stem
from desktop_presets import delete_preset, list_presets, load_preset, save_preset
from desktop_tasks import (
    create_task,
    load_task,
    set_task_status,
    task_choices,
    update_task_line,
)
from desktop_generation_controls import (
    allocate_native_chunk_durations,
    apply_duration_policy,
    build_desktop_plan,
    concatenate_with_pauses,
    postprocess_waveform,
    run_with_long_text_guard,
)
from desktop_model_lifecycle import DesktopModelLifecycle
from desktop_streaming_audio import BundledStreamingAudio
from dialogue_runtime import (
    DialogueLine,
    compose_timeline,
    fit_duration_factor,
    missing_roles,
    parse_batch_script,
    parse_srt,
)
from speech_review import ASR_BACKENDS, ASR_MODELS, asr_available, review_transcript, transcribe_audio_file
from timeline_tools import (
    TIMELINE_HEADERS,
    apply_timeline_drag_payload,
    apply_timeline_edits,
    render_timeline_html,
    rewrite_srt,
    timeline_rows,
)
from indextts.infer_v2_5 import IndexTTS2
from indextts.speech_rate_guard import (
    assess_segment_speech_rates,
    retry_candidate_improves_rate,
)
from indextts.utils.reference_condition_cache import ReferenceConditionCache
from indextts.pronunciation import (
    ANNOTATION_PATTERN,
    PronunciationEntry,
    PronunciationValidationError,
    entries_from_rows,
    entries_to_rows,
    format_pronunciation_report,
    load_dictionary,
    make_annotation,
    process_pronunciation_text,
    save_dictionary,
    validate_reading,
)
from runtime_acceleration import MODES, format_acceleration_report, probe_acceleration, resolve_acceleration
from runtime_metrics import (
    finish_runtime_measurement,
    format_runtime_metrics,
    start_runtime_measurement,
)


APP_TITLE = "T8star-Aix · IndexTTS 2.5"
DESKTOP_VERSION = "0.22.0"
MODEL_MANIFEST = json.loads(
    (Path(__file__).resolve().parent / "desktop_model_manifest.json").read_text(encoding="utf-8")
)
OFFICIAL_MODEL_REVISION = MODEL_MANIFEST["modelRevision"]
OFFICIAL_MODEL_SIZES = {
    relative_path: metadata["size"]
    for relative_path, metadata in MODEL_MANIFEST["files"].items()
}
EMOTION_MODES = ["跟随音色参考", "情感参考音频", "八维情感向量", "情感描述文本"]
EMOTION_LABELS = ["喜", "怒", "哀", "惧", "厌恶", "低落", "惊喜", "平静"]
PROFILE_EMOTION_MODE_IDS = ("speaker", "reference_audio", "vector", "text")
HISTORY_LOCK = threading.Lock()
DICTIONARY_FILENAME = "pronunciation_dictionary.yaml"
DICTIONARY_HEADERS = ["文字/词语", "语言", "读音", "启用", "区分大小写"]
LANGUAGE_CHOICES = [
    ("中文（ZH）", "ZH"),
    ("英语（EN）", "EN"),
    ("日语（JA）", "JA"),
    ("西班牙语（ES）", "ES"),
    ("阿拉伯语（AR）", "AR"),
]
DICTIONARY_LANGUAGES = [value for _label, value in LANGUAGE_CHOICES]
LOW_VRAM_THRESHOLD_GB = 10.0
SAMPLE_PRONUNCIATION_ENTRIES = [
    PronunciationEntry("要求", "YAO4 QIU2", "ZH"),
    PronunciationEntry("银行", "YIN2 HANG2", "ZH"),
    PronunciationEntry("行长", "HANG2 ZHANG3", "ZH"),
    PronunciationEntry("重庆", "CHONG2 QING4", "ZH"),
    PronunciationEntry(
        "Bilibili", "B IY1 . L IY1 . B IY1 . L IY1", "EN", True, False
    ),
    PronunciationEntry("上手", "じょうず", "JA"),
]
SAMPLE_PRONUNCIATION_TEXT = (
    "小明<要求|YAO4 QIU2>这个题的答案是多少。今天的<行程|XING2 CHENG2>顺利，"
    "<银行|YIN2 HANG2>的<行长|HANG2 ZHANG3>去了<重庆|CHONG2 QING4>。"
)
SAMPLE_BATCH_SCRIPT = (
    "旁白|欢迎使用逐句情感控制。|ZH|1.0|text:平静、从容地介绍\n"
    "旁白|等等，这件事太令人震惊了！|ZH|1.0|vector:0,0,0,0,0,0,0.8,0\n"
    "旁白|现在恢复角色音色库里的默认情感。|ZH|1.0\n"
    "旁白|This is a real English example.|EN|1.0|text:平静、自然"
)
SAMPLE_SRT_SCRIPT = """1
00:00:00,000 --> 00:00:03,000
[旁白|emotion=text:平静、从容地介绍] 欢迎使用 IndexTTS 2.5。

2
00:00:03,200 --> 00:00:06,000
[旁白|emotion=vector:0,0.8,0,0,0,0,0,0] 这一次，同一个角色变得生气。"""


def describe_dialogue_timing_settings(
    script_type: str,
    timeline_policy: str,
    fit_slots: bool,
    slot_mode: str,
    tolerance_ms: int,
    batch_gap_ms: int,
) -> str:
    """Explain the effective timing behavior in user-facing language."""

    script_type = str(script_type or "batch")
    timeline_policy = str(timeline_policy or "shift")
    slot_mode = str(slot_mode or "pad")
    tolerance_ms = max(0, int(tolerance_ms or 0))
    batch_gap_ms = max(0, int(batch_gap_ms or 0))

    if script_type == "batch":
        conflict = (
            "当前选择了“保留起点并混音”。普通批量台词没有 SRT 起点，"
            "多句可能从 0 秒同时播放；请改成“顺延，避免重叠”。"
            if timeline_policy == "overlay"
            else f"台词按顺序排列，每句结束后保留 {batch_gap_ms} 毫秒静音。"
        )
        warning = "\n\n> **注意：当前配置会导致普通批量台词重叠。**" if timeline_policy == "overlay" else ""
        return (
            "### 当前设置解释：普通批量台词\n\n"
            f"- {conflict}\n"
            "- “适配 SRT 字幕槽位”、收尾模式和允许误差在批量模式下不生效。\n"
            f"- 推荐：**顺延，避免重叠 + 句间静音 {batch_gap_ms or 200} 毫秒**。"
            f"{warning}"
        )

    conflict = (
        "保留每条 SRT 的原始开始时间；前一句超时会与下一句重叠混音。"
        if timeline_policy == "overlay"
        else "前一句超时会把后一句顺延，保证两句不重叠，但成品可能比原 SRT 更长。"
    )
    if not fit_slots:
        slot_description = (
            "未启用时长适配：模型按自然长度生成；下方收尾模式和允许误差不会生效。"
        )
    else:
        mode_descriptions = {
            "native": (
                "原生单次适配：把 SRT 目标时长直接交给 IndexTTS 2.5，"
                "最后补齐或裁到槽位长度；允许误差不参与该模式。"
            ),
            "natural": (
                f"自然适配：与槽位相差超过 {tolerance_ms} 毫秒时调整语速再生成一次，"
                "不补静音、不裁剪，优先保证台词完整。"
            ),
            "pad": (
                f"不丢字模式：与槽位相差超过 {tolerance_ms} 毫秒时先调整语速；"
                "不足补静音，超长完整保留。"
            ),
            "exact": (
                f"强制精确：与槽位相差超过 {tolerance_ms} 毫秒时先调整语速；"
                "不足补静音，超长直接裁剪，槽位太短时可能丢失句尾。"
            ),
        }
        slot_description = mode_descriptions.get(slot_mode, mode_descriptions["pad"])
    return (
        "### 当前设置解释：SRT 字幕配音\n\n"
        f"- **时间冲突：** {conflict}\n"
        f"- **时长处理：** {slot_description}\n"
        "- “普通批量句间静音”对 SRT 不生效；SRT 的停顿来自时间码。"
    )


def profile_emotion_kwargs(
    tts: IndexTTS2,
    profile: VoiceProfile,
    qwen_emotion_available: bool,
) -> dict:
    """Resolve one saved role's emotion without leaking it to another role."""

    mode = profile.emotion_mode
    result = {
        "emo_audio_prompt": None,
        "emo_alpha": float(profile.emotion_strength),
        "emo_vector": None,
        "use_emo_text": False,
        "emo_text": None,
        "use_random": False,
    }
    if mode == "speaker":
        return result
    if mode == "reference_audio":
        emotion_audio = Path(profile.emotion_audio_path)
        if not emotion_audio.is_file():
            raise ValueError(f"角色“{profile.name}”的情感参考音频不存在：{emotion_audio}")
        result["emo_audio_prompt"] = str(emotion_audio)
        return result
    if mode == "vector":
        result["emo_vector"] = tts.normalize_emo_vec(
            list(profile.emotion_vector), apply_bias=True
        )
        result["use_random"] = bool(profile.emotion_use_random)
        return result
    if mode == "text":
        if not qwen_emotion_available:
            raise ValueError(
                f"角色“{profile.name}”使用文本情感，但当前低显存模式未加载 QwenEmotion。"
            )
        result["use_emo_text"] = True
        result["emo_text"] = profile.emotion_text.strip() or None
        return result
    raise ValueError(f"角色“{profile.name}”使用了未知情感模式：{mode}")


def line_emotion_kwargs(
    tts: IndexTTS2,
    profile: VoiceProfile,
    line: DialogueLine,
    qwen_emotion_available: bool,
) -> tuple[dict, str]:
    """Apply a line override without changing the saved role default."""

    mode = str(line.emotion_mode or "inherit")
    if mode == "inherit":
        return profile_emotion_kwargs(tts, profile, qwen_emotion_available), "role_default"
    result = {
        "emo_audio_prompt": None,
        "emo_alpha": float(line.emotion_strength),
        "emo_vector": None,
        "use_emo_text": False,
        "emo_text": None,
        "use_random": False,
    }
    if mode == "speaker":
        return result, "line_override"
    if mode == "vector":
        if line.emotion_vector is None or len(line.emotion_vector) != 8:
            raise ValueError(f"第 {line.index} 条台词的八维情感向量无效。")
        result["emo_vector"] = tts.normalize_emo_vec(
            list(line.emotion_vector), apply_bias=True
        )
        result["use_random"] = bool(line.emotion_use_random)
        return result, "line_override"
    if mode == "text":
        if not qwen_emotion_available:
            raise ValueError(
                f"第 {line.index} 条台词使用文本情感，但当前低显存模式未加载 QwenEmotion。"
            )
        result["use_emo_text"] = True
        result["emo_text"] = line.emotion_text.strip() or line.text
        return result, "line_override"
    raise ValueError(f"第 {line.index} 条台词使用了未知情感模式：{mode}")

CSS = """
.gradio-container {
  width: min(100%, 1480px) !important;
  max-width: 1480px !important;
  margin: 0 auto !important;
}
.gradio-container main.fillable {
  width: 100% !important;
  max-width: none !important;
  padding: clamp(18px, 2.2vw, 34px) !important;
}
.gradio-container main.fillable > .wrap,
.gradio-container main.fillable .contain,
.gradio-container #component-0 {
  width: 100% !important;
  max-width: none !important;
}
.t8-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 32px;
  padding: clamp(18px, 2vw, 26px) clamp(20px, 2.2vw, 30px);
  margin-bottom: 14px;
  border: 1px solid rgba(251, 114, 153, .28);
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(251,114,153,.13), rgba(89,125,255,.08));
}
.t8-header-title { min-width: 0; }
.t8-eyebrow { color: #fb7299; font-size: 11px; font-weight: 750; letter-spacing: .11em; }
.t8-header-copy { flex: 0 0 auto; text-align: right; }
.t8-header h1 { margin: 7px 0 0 !important; font-size: clamp(27px, 2.6vw, 38px) !important; line-height: 1.08 !important; }
.t8-header p { margin: 3px 0 !important; opacity: .8; }
.t8-credit { color: #fb7299; font-weight: 700; }
.t8-primary-grid {
  display: grid !important;
  grid-template-columns: minmax(320px, .78fr) minmax(0, 1.5fr);
  align-items: stretch !important;
  gap: 16px !important;
}
.t8-prompt-audio { align-self: stretch !important; }
.t8-prompt-audio > div { height: 100% !important; }
.t8-meta-row { align-items: end !important; }
.t8-section { border-radius: 14px !important; }
.t8-pronunciation-tip {
  align-items: center !important;
  gap: 18px !important;
  margin: 14px 0 8px !important;
  padding: 14px 16px !important;
  border: 1px solid rgba(251, 114, 153, .38) !important;
  border-radius: 14px !important;
  background: linear-gradient(120deg, rgba(251,114,153,.11), rgba(255,255,255,.72)) !important;
}
.t8-pronunciation-tip .prose { margin: 0 !important; }
.t8-pronunciation-tip p { margin: 2px 0 !important; }
.t8-pronunciation-tip code { color: #c52f68; font-weight: 700; }
.t8-pronunciation-accordion {
  border: 1px solid rgba(251, 114, 153, .3) !important;
  box-shadow: 0 8px 24px rgba(31, 41, 55, .045) !important;
}
.t8-pronunciation-accordion > button { font-weight: 750 !important; color: #d63372 !important; }
.t8-timing-summary {
  margin: 10px 0 6px !important;
  padding: 14px 16px !important;
  border: 1px solid rgba(89,125,255,.26) !important;
  border-radius: 14px !important;
  background: linear-gradient(120deg, rgba(89,125,255,.08), rgba(251,114,153,.08)) !important;
}
.t8-timing-summary .prose { margin: 0 !important; }
.t8-timing-summary h3 { margin: 0 0 7px !important; font-size: 16px !important; }
.t8-timing-summary p,.t8-timing-summary li { font-size: 13px !important; line-height: 1.55 !important; }
.t8-timing-guide { border: 1px solid rgba(89,125,255,.22) !important; }
.t8-timing-guide > button { font-weight: 750 !important; color: #3156c8 !important; }
.t8-actions { gap: 14px !important; }
.t8-generate { min-height: 48px; }
.t8-timeline { padding: 14px; border: 1px solid rgba(251,114,153,.28); border-radius: 14px; background: rgba(255,255,255,.68); overflow: hidden; }
.t8-timeline-scale { display: flex; justify-content: space-between; font-size: 11px; opacity: .7; margin-bottom: 8px; }
.t8-timeline-track { position: relative; height: 38px; margin: 6px 0; border-radius: 8px; background: rgba(148,163,184,.13); overflow: hidden; touch-action: none; }
.t8-timeline-bar { position: absolute; top: 4px; bottom: 4px; min-width: 8px; padding: 5px 12px; border-radius: 6px; color: #152238; font-size: 11px; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; cursor: grab; user-select: none; touch-action: none; box-shadow: 0 2px 8px rgba(15,23,42,.12); }
.t8-timeline-bar.t8-dragging { cursor: grabbing; z-index: 4; box-shadow: 0 5px 16px rgba(15,23,42,.28); }
.t8-timeline-bar-label { position: relative; z-index: 2; pointer-events: none; }
.t8-timeline-handle { position: absolute; z-index: 5; top: 0; bottom: 0; width: 9px; background: rgba(255,255,255,.42); cursor: ew-resize; }
.t8-timeline-handle:hover { background: rgba(255,255,255,.78); }
.t8-timeline-handle-start { left: 0; border-radius: 6px 0 0 6px; }
.t8-timeline-handle-end { right: 0; border-radius: 0 6px 6px 0; }
.t8-timeline-word { pointer-events: none; z-index: 3; }
#t8-timeline-drag-payload { display: none !important; }
.t8-timeline-hint,.t8-timeline-empty { margin-top: 9px; font-size: 12px; opacity: .68; }
.t8-footer { text-align: center; opacity: .58; font-size: 12px; padding: 14px; }
@media (max-width: 900px) {
  .gradio-container main.fillable { padding: 16px !important; }
  .t8-header { align-items: flex-start; flex-direction: column; gap: 10px; }
  .t8-header-copy { text-align: left; }
  .t8-primary-grid { grid-template-columns: 1fr; }
  .t8-prompt-audio { min-height: 280px; }
}
@media (max-width: 560px) {
  .gradio-container main.fillable { padding: 10px !important; }
  .t8-header { padding: 17px; border-radius: 14px; }
  .t8-header h1 { font-size: 26px !important; }
  .t8-actions { flex-direction: column !important; }
}
"""

TIMELINE_EDITOR_JS = r"""() => {
  if (window.__t8TimelineEditorInstalled) return [];
  window.__t8TimelineEditorInstalled = true;

  const setPayload = (payload) => {
    const host = document.querySelector('#t8-timeline-drag-payload');
    const input = host?.querySelector('textarea, input');
    if (!input) return;
    const proto = input.tagName === 'TEXTAREA'
      ? window.HTMLTextAreaElement.prototype
      : window.HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
    if (setter) setter.call(input, JSON.stringify(payload));
    else input.value = JSON.stringify(payload);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  };

  const valuesFrom = (element, attributes) => {
    const values = [];
    for (const attribute of attributes) {
      const value = Number(element.getAttribute(attribute));
      if (Number.isFinite(value)) values.push(value);
    }
    return values;
  };

  document.addEventListener('pointerdown', (event) => {
    const bar = event.target.closest?.('.t8-timeline-bar');
    if (!bar || event.button !== 0) return;
    const timeline = bar.closest('.t8-timeline');
    const track = bar.closest('.t8-timeline-track');
    if (!timeline || !track) return;
    event.preventDefault();

    const totalMs = Math.max(1, Number(timeline.dataset.totalMs) || 1);
    const rect = track.getBoundingClientRect();
    const originalStart = Number(bar.dataset.startMs) || 0;
    const originalEnd = Number(bar.dataset.endMs) || originalStart + 1;
    const duration = Math.max(1, originalEnd - originalStart);
    const pointerStart = event.clientX;
    const requestedMode = event.target.closest('.t8-timeline-handle-start')
      ? 'resize_start'
      : event.target.closest('.t8-timeline-handle-end')
        ? 'resize_end'
        : 'move';
    const snapValues = [];
    timeline.querySelectorAll('.t8-timeline-bar, .t8-timeline-word').forEach((element) => {
      if (element === bar || (requestedMode === 'move' && bar.contains(element))) return;
      snapValues.push(...valuesFrom(element, ['data-snap-ms', 'data-snap-start-ms', 'data-snap-end-ms']));
    });
    const thresholdMs = totalMs * (Number(timeline.dataset.snapThresholdPx) || 12) / Math.max(1, rect.width);
    let currentStart = originalStart;
    let currentEnd = originalEnd;
    let snappedTo = null;
    let moved = false;

    const nearest = (candidate) => {
      let best = null;
      for (const value of snapValues) {
        const distance = Math.abs(value - candidate);
        if (!best || distance < best.distance) best = { value, distance };
      }
      return best && best.distance <= thresholdMs ? best : null;
    };

    const draw = () => {
      bar.style.left = `${100 * currentStart / totalMs}%`;
      bar.style.width = `${Math.max(0.4, 100 * (currentEnd - currentStart) / totalMs)}%`;
      bar.dataset.startMs = String(Math.round(currentStart));
      bar.dataset.endMs = String(Math.round(currentEnd));
      bar.title = `#${bar.dataset.index} · ${Math.round(currentStart)}–${Math.round(currentEnd)}ms` +
        (snappedTo === null ? '' : ` · 已吸附 ${Math.round(snappedTo)}ms`);
    };

    const onMove = (moveEvent) => {
      const deltaPx = moveEvent.clientX - pointerStart;
      if (Math.abs(deltaPx) >= 2) moved = true;
      const deltaMs = deltaPx * totalMs / Math.max(1, rect.width);
      snappedTo = null;
      if (requestedMode === 'resize_start') {
        currentStart = Math.max(0, Math.min(originalEnd - 50, originalStart + deltaMs));
        const snap = moveEvent.altKey ? null : nearest(currentStart);
        if (snap) { currentStart = snap.value; snappedTo = snap.value; }
      } else if (requestedMode === 'resize_end') {
        currentEnd = Math.min(totalMs, Math.max(originalStart + 50, originalEnd + deltaMs));
        const snap = moveEvent.altKey ? null : nearest(currentEnd);
        if (snap) { currentEnd = snap.value; snappedTo = snap.value; }
      } else {
        currentStart = Math.max(0, Math.min(totalMs - duration, originalStart + deltaMs));
        currentEnd = currentStart + duration;
        if (!moveEvent.altKey) {
          const startSnap = nearest(currentStart);
          const endSnap = nearest(currentEnd);
          const snap = !startSnap ? endSnap : !endSnap ? startSnap
            : startSnap.distance <= endSnap.distance ? startSnap : endSnap;
          if (snap) {
            const shift = snap === startSnap ? snap.value - currentStart : snap.value - currentEnd;
            currentStart += shift;
            currentEnd += shift;
            snappedTo = snap.value;
          }
        }
      }
      currentStart = Math.max(0, Math.min(currentStart, currentEnd - 50));
      currentEnd = Math.min(totalMs, Math.max(currentEnd, currentStart + 50));
      draw();
    };

    const onUp = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
      bar.classList.remove('t8-dragging');
      setPayload({
        index: Number(bar.dataset.index),
        start_ms: Math.round(currentStart),
        end_ms: Math.round(currentEnd),
        mode: moved ? requestedMode : 'select',
        snapped_to_ms: snappedTo === null ? null : Math.round(snappedTo),
      });
    };

    bar.classList.add('t8-dragging');
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp, { once: true });
    window.addEventListener('pointercancel', onUp, { once: true });
  });
  return [];
}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T8star-Aix IndexTTS 2.5 Desktop WebUI")
    parser.add_argument("--model_dir", required=True, help="External IndexTTS 2.5 model directory")
    parser.add_argument("--output_dir", required=True, help="User-writable output directory")
    parser.add_argument("--data_dir", help="Persistent Electron user-data directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 inference on supported devices")
    parser.add_argument(
        "--precision",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="UnifiedVoice/GPT precision; auto prefers native BF16 then FP16",
    )
    parser.add_argument(
        "--reference-device",
        choices=["auto", "same", "cpu"],
        default="auto",
        help="Reference encoder placement; auto uses CPU on low-VRAM CUDA devices",
    )
    parser.add_argument(
        "--reuse-spk-cond-for-emo",
        action="store_true",
        help="Reuse speaker conditioning only for implicit default emotion",
    )
    parser.add_argument("--qwen_emo", action="store_true", help="Force QwenEmotion on low-VRAM GPUs")
    parser.add_argument("--acceleration", choices=MODES, default="off", help="Optional acceleration mode")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def select_runtime_policy(
    force_bf16: bool = False,
    force_qwen_emo: bool = False,
    precision: str = "auto",
    reference_device: str = "auto",
) -> dict:
    """Choose precision and QwenEmotion loading from the available CUDA VRAM."""
    if not torch.cuda.is_available():
        return {
            "vram_gb": None,
            "low_vram": False,
            "use_bf16": False,
            "use_fp16": False,
            "precision": "float32",
            "reference_device": "cpu",
            "use_qwen_emo": True,
            "bf16_supported": False,
        }
    index = torch.cuda.current_device()
    vram_gb = torch.cuda.get_device_properties(index).total_memory / (1024 ** 3)
    low_vram = vram_gb < LOW_VRAM_THRESHOLD_GB
    try:
        if hasattr(torch.cuda, "device"):
            with torch.cuda.device(index):
                bf16_supported = bool(
                    torch.cuda.is_bf16_supported(including_emulation=False)
                )
        else:
            bf16_supported = bool(
                torch.cuda.is_bf16_supported(including_emulation=False)
            )
    except TypeError:
        try:
            bf16_supported = bool(torch.cuda.is_bf16_supported(index))
        except TypeError:
            bf16_supported = bool(torch.cuda.is_bf16_supported())
    requested_precision = str(precision or "auto").strip().lower()
    if force_bf16:
        requested_precision = "bfloat16"
    if requested_precision == "auto":
        selected_precision = "bfloat16" if bf16_supported else "float16"
    elif requested_precision == "bfloat16" and not bf16_supported:
        selected_precision = "float16"
    else:
        selected_precision = requested_precision
    requested_reference = str(reference_device or "auto").strip().lower()
    selected_reference = (
        "cpu"
        if requested_reference == "cpu"
        or (requested_reference == "auto" and low_vram)
        else f"cuda:{index}"
    )
    return {
        "vram_gb": vram_gb,
        "low_vram": low_vram,
        "use_bf16": selected_precision == "bfloat16",
        "use_fp16": selected_precision == "float16",
        "precision": selected_precision,
        "reference_device": selected_reference,
        "use_qwen_emo": bool(force_qwen_emo or not low_vram),
        "bf16_supported": bf16_supported,
    }


def validate_model_dir(model_dir: Path) -> None:
    required = list(OFFICIAL_MODEL_SIZES)
    missing = [item for item in required if not (model_dir / item).exists()]
    if missing:
        raise FileNotFoundError("Incomplete IndexTTS 2.5 model directory; missing: " + ", ".join(missing))
    mismatched = [
        item
        for item, expected_size in OFFICIAL_MODEL_SIZES.items()
        if (model_dir / item).stat().st_size != expected_size
    ]
    if mismatched:
        raise RuntimeError(
            "IndexTTS 2.5 model files are outdated or corrupt: " + ", ".join(mismatched)
        )


def history_path(output_dir: Path) -> Path:
    return output_dir / "history.jsonl"


def load_history(output_dir: Path, limit: int = 50) -> list[list[str]]:
    path = history_path(output_dir)
    if not path.exists():
        return []
    rows: list[list[str]] = []
    with HISTORY_LOCK:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            rows.append([
                item.get("created_at", ""),
                item.get("language", ""),
                item.get("duration_factor", 1.0),
                item.get("emotion_mode", ""),
                item.get("text", ""),
                item.get("resolved_text", item.get("text", "")),
                item.get("file", ""),
            ])
    return list(reversed(rows[-limit:]))


def append_history(output_dir: Path, item: dict) -> None:
    with HISTORY_LOCK:
        with history_path(output_dir).open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(item, ensure_ascii=False) + "\n")


def pronunciation_dictionary_path(data_dir: Path) -> Path:
    return data_dir / DICTIONARY_FILENAME


def load_pronunciation_rows(data_dir: Path) -> list[list[object]]:
    try:
        return entries_to_rows(load_dictionary(pronunciation_dictionary_path(data_dir)))
    except (OSError, ValueError):
        traceback.print_exc()
        return []


def pronunciation_entry_choices(rows) -> list[tuple[str, str]]:
    """Build stable dropdown choices for the read-only dictionary preview."""

    return [
        (f"{index + 1}. {entry.term} · {entry.language} · {entry.reading}", str(index))
        for index, entry in enumerate(entries_from_rows(rows))
    ]


def upsert_pronunciation_entry(
    rows,
    selected_index,
    term: str,
    language: str,
    reading: str,
    enabled: bool,
    case_sensitive: bool,
    *,
    pinyin_vocab_path: str | Path | None = None,
) -> tuple[list[list[object]], str, tuple[str, ...]]:
    """Validate and add/update one dictionary entry from dropdown-based controls."""

    entries = entries_from_rows(rows)
    term = str(term or "").strip()
    language = str(language or "ZH").strip().upper()
    if not term:
        raise PronunciationValidationError("文字/词语不能为空。")
    if language not in DICTIONARY_LANGUAGES:
        raise PronunciationValidationError(f"不支持的词典语言：{language}")
    normalized, warnings, errors = validate_reading(
        reading,
        language,
        pinyin_vocab_path=pinyin_vocab_path,
    )
    if errors:
        raise PronunciationValidationError("；".join(errors))

    index: int | None = None
    if selected_index not in (None, ""):
        try:
            index = int(selected_index)
        except (TypeError, ValueError) as exc:
            raise ValueError("所选词条序号无效，请重新选择。") from exc
        if index < 0 or index >= len(entries):
            raise ValueError("所选词条已不存在，请重新选择。")

    for current_index, current in enumerate(entries):
        if current_index == index:
            continue
        if current.language == language and current.term.casefold() == term.casefold():
            raise PronunciationValidationError(f"词典中已存在 {term}（{language}）。")

    entry = PronunciationEntry(
        term=term,
        reading=normalized,
        language=language,
        enabled=bool(enabled),
        case_sensitive=bool(case_sensitive),
    )
    if index is None:
        entries.append(entry)
        index = len(entries) - 1
    else:
        entries[index] = entry
    return entries_to_rows(entries), str(index), warnings


def delete_pronunciation_entry(rows, selected_index) -> list[list[object]]:
    entries = entries_from_rows(rows)
    if selected_index in (None, ""):
        raise ValueError("请先从“已有词条”列表选择要删除的记录。")
    try:
        index = int(selected_index)
    except (TypeError, ValueError) as exc:
        raise ValueError("所选词条序号无效，请重新选择。") from exc
    if index < 0 or index >= len(entries):
        raise ValueError("所选词条已不存在，请重新选择。")
    del entries[index]
    return entries_to_rows(entries)


def build_app(
    tts: IndexTTS2,
    output_dir: Path,
    data_dir: Path,
    verbose: bool,
    acceleration_report: str = "",
    fallback_factory=None,
    model_factory=None,
) -> gr.Blocks:
    dictionary_file = pronunciation_dictionary_path(data_dir)
    qwen_emotion_available = getattr(tts, "qwen_emo", object()) is not None
    exact_vocab_candidates = (
        Path(__file__).resolve().parent / "checkpoints" / "pinyin.vocab",
        Path(__file__).resolve().parent / "indextts" / "pinyin.vocab",
    )
    exact_vocab_path = next((path for path in exact_vocab_candidates if path.is_file()), None)
    voice_library = VoiceLibrary(data_dir)
    audiocpp_initial_status = audiocpp_component_status(data_dir)
    runtime_fallback_used = False
    runtime_fallback_note = ""
    initial_tts = tts
    lifecycle = DesktopModelLifecycle(initial_tts, model_factory or (lambda: initial_tts))
    memory_policy = {
        "release_after_generation": False,
        "idle_seconds": 600.0,
        "recycle_after_generations": 0,
    }

    def ensure_model():
        nonlocal tts
        tts = lifecycle.get()
        return tts

    def apply_memory_policy() -> dict:
        nonlocal tts
        report = lifecycle.after_generation(**memory_policy)
        if not report.get("loaded"):
            tts = None
        return report

    def update_memory_policy_event(release_after, idle_seconds, recycle_after):
        memory_policy.update(
            release_after_generation=bool(release_after),
            idle_seconds=max(0.0, float(idle_seconds)),
            recycle_after_generations=max(0, int(recycle_after)),
        )
        return json.dumps(
            {"policy": memory_policy, "runtime": lifecycle.status()},
            ensure_ascii=False,
            indent=2,
        )

    def release_model_event():
        nonlocal tts
        report = lifecycle.release("manual")
        tts = None
        return json.dumps(report, ensure_ascii=False, indent=2)

    def refresh_model_status_event():
        return json.dumps(
            {"policy": memory_policy, "runtime": lifecycle.status()},
            ensure_ascii=False,
            indent=2,
        )

    reference_cache_root = data_dir / "reference_condition_cache"

    def _reference_cache():
        active = getattr(tts, "reference_condition_cache", None) if tts is not None else None
        return active or ReferenceConditionCache(reference_cache_root)

    def refresh_reference_cache_event():
        return json.dumps(
            {
                "cache": _reference_cache().status(),
                "explanation": (
                    "hits/misses 是本次已加载模型的会话统计；entries/bytes 是磁盘实时状态。"
                    "清理只删除本整合包 reference_condition_cache 下的 safetensors。"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )

    def clear_reference_cache_event():
        cache = _reference_cache()
        before = cache.status()
        removed = cache.clear()
        return json.dumps(
            {
                "action": "clear_reference_condition_cache",
                "removed_entries": removed,
                "before": before,
                "after": cache.status(),
            },
            ensure_ascii=False,
            indent=2,
        )

    def execute_with_runtime_fallback(callback):
        """Retry once with a freshly loaded normal model after optional acceleration fails."""

        nonlocal tts, runtime_fallback_used, runtime_fallback_note
        ensure_model()
        try:
            result = callback()
        except Exception as exc:
            if fallback_factory is None or runtime_fallback_used:
                raise
            runtime_fallback_note = (
                f"可选加速运行失败：{type(exc).__name__}: {exc}；"
                "已释放加速模型、重载普通模式并自动重试。"
            )
            print(">> " + runtime_fallback_note, flush=True)
            traceback.print_exc()
            lifecycle.release("acceleration_fallback")
            tts = None
            try:
                tts = fallback_factory()
            except Exception as fallback_exc:
                raise RuntimeError(
                    runtime_fallback_note
                    + f" 普通模式重载也失败：{type(fallback_exc).__name__}: {fallback_exc}"
                ) from fallback_exc
            lifecycle.replace(tts)
            runtime_fallback_used = True
            result = callback()
        return result, runtime_fallback_note

    @contextmanager
    def safe_gpt_acceleration(sampling_values: dict, plan=None):
        ensure_model()
        engine = getattr(getattr(tts, "gpt", None), "accel_engine", None)
        compatible = bool(
            sampling_values.get("do_sample")
            and float(sampling_values.get("top_p", 0)) == 1.0
            and int(sampling_values.get("top_k") or 0) == 0
            and int(sampling_values.get("num_beams", 1)) == 1
            and float(sampling_values.get("repetition_penalty", 1)) == 1.0
            and float(sampling_values.get("length_penalty", 0)) == 0.0
        )
        guarded = False
        disabled = engine is not None and not compatible
        if disabled:
            tts.gpt.accel_engine = None
        try:
            yield disabled, guarded
        finally:
            if disabled:
                tts.gpt.accel_engine = engine

    def parse_rows(rows, language: str = "ZH") -> list[PronunciationEntry]:
        return entries_from_rows(rows, language)

    def validate_entries(entries: list[PronunciationEntry]) -> tuple[list[str], list[str]]:
        warnings: list[str] = []
        errors: list[str] = []
        for entry in entries:
            if not entry.term:
                errors.append("词典包含空文字项。")
                continue
            _reading, item_warnings, item_errors = validate_reading(
                entry.reading,
                entry.language,
                pinyin_vocab_path=exact_vocab_path,
            )
            warnings.extend(f"{entry.term}：{message}" for message in item_warnings)
            errors.extend(f"{entry.term}：{message}" for message in item_errors)
        return list(dict.fromkeys(warnings)), list(dict.fromkeys(errors))

    def dictionary_status(entries: list[PronunciationEntry], prefix: str = "") -> str:
        warnings, errors = validate_entries(entries)
        lines = [f"{prefix}当前共 {len(entries)} 条发音词典记录。".strip()]
        if exact_vocab_path is None:
            lines.append("中文拼音按合法音节结构校验；未发现可选的官方精确词表。")
        if warnings:
            lines.append("警告：")
            lines.extend(f"- {message}" for message in warnings)
        if errors:
            lines.append("需要修正：")
            lines.extend(f"- {message}" for message in errors)
        return "\n".join(lines)

    def save_dictionary_event(rows):
        entries = parse_rows(rows)
        save_dictionary(dictionary_file, entries)
        return dictionary_status(entries, "已保存。")

    def dictionary_editor_values(entry: PronunciationEntry | None = None):
        if entry is None:
            return "", "ZH", "", True, True
        return (
            entry.term,
            entry.language,
            entry.reading,
            entry.enabled,
            entry.case_sensitive,
        )

    def dictionary_table_and_editor(entries: list[PronunciationEntry], message: str):
        rows = entries_to_rows(entries)
        return (
            rows,
            gr.update(choices=pronunciation_entry_choices(rows), value=None),
            *dictionary_editor_values(),
            dictionary_status(entries, message),
        )

    def import_dictionary_event(uploaded_path: str | None):
        if not uploaded_path:
            raise gr.Error("请先选择 YAML 或 JSON 发音词典。")
        try:
            entries = load_dictionary(uploaded_path)
        except Exception as exc:
            raise gr.Error(f"词典导入失败：{exc}") from exc
        save_dictionary(dictionary_file, entries)
        return dictionary_table_and_editor(entries, "已导入并保存。")

    def export_dictionary_event(rows):
        entries = parse_rows(rows)
        save_dictionary(dictionary_file, entries)
        export_path = data_dir / "exports" / "T8star-Aix-IndexTTS25-pronunciation.yaml"
        save_dictionary(export_path, entries)
        return str(export_path), dictionary_status(entries, "已保存并导出。")

    def load_examples_event():
        return dictionary_table_and_editor(
            SAMPLE_PRONUNCIATION_ENTRIES,
            "已载入中英日示例，点击“保存词典”后持久化。",
        )

    def select_dictionary_entry_event(rows, selected_index):
        entries = parse_rows(rows)
        if selected_index in (None, ""):
            return (*dictionary_editor_values(), "请选择已有词条，或直接填写新词条。")
        try:
            entry = entries[int(selected_index)]
        except (ValueError, TypeError, IndexError) as exc:
            raise gr.Error("所选词条已不存在，请重新选择。") from exc
        return (
            *dictionary_editor_values(entry),
            f"正在编辑：{entry.term}（{entry.language}）。修改后点击“添加/更新到表格”，再保存词典。",
        )

    def upsert_dictionary_entry_event(
        rows,
        selected_index,
        term,
        entry_language,
        reading,
        enabled,
        case_sensitive,
    ):
        try:
            updated_rows, updated_index, warnings = upsert_pronunciation_entry(
                rows,
                selected_index,
                term,
                entry_language,
                reading,
                enabled,
                case_sensitive,
                pinyin_vocab_path=exact_vocab_path,
            )
        except (PronunciationValidationError, ValueError) as exc:
            raise gr.Error(f"词条无法加入：{exc}") from exc
        entry = parse_rows(updated_rows)[int(updated_index)]
        status = "词条已加入表格，尚未持久保存。"
        if warnings:
            status += " 警告：" + "；".join(warnings)
        return (
            updated_rows,
            gr.update(
                choices=pronunciation_entry_choices(updated_rows),
                value=updated_index,
            ),
            *dictionary_editor_values(entry),
            status,
        )

    def delete_dictionary_entry_event(rows, selected_index):
        try:
            updated_rows = delete_pronunciation_entry(rows, selected_index)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        entries = parse_rows(updated_rows)
        return (
            updated_rows,
            gr.update(choices=pronunciation_entry_choices(updated_rows), value=None),
            *dictionary_editor_values(),
            f"已从表格删除词条；当前剩余 {len(entries)} 条，点击“保存词典”后持久化。",
        )

    def clear_dictionary_editor_event():
        return (
            gr.update(value=None),
            *dictionary_editor_values(),
            "已切换为新增词条模式。填写后点击“添加/更新到表格”。",
        )

    def fill_chinese_pronunciation_example_event():
        return (
            SAMPLE_PRONUNCIATION_TEXT,
            "ZH",
            "已填入中文多音字示例。上传参考音频后可直接生成；也可点击“预览最终发音文本”检查。",
        )

    def search_dictionary_event(rows, query: str):
        entries = parse_rows(rows)
        query = str(query or "").strip().casefold()
        if query:
            entries = [
                entry
                for entry in entries
                if query in entry.term.casefold() or query in entry.reading.casefold()
            ]
        if not entries:
            return "未找到匹配词条。"
        lines = ["| 文字/词语 | 语言 | 读音 | 启用 |", "|---|---|---|---|"]
        for entry in entries[:50]:
            term = entry.term.replace("|", "\\|")
            reading = entry.reading.replace("|", "\\|")
            lines.append(f"| {term} | {entry.language} | {reading} | {'是' if entry.enabled else '否'} |")
        if len(entries) > 50:
            lines.append(f"\n仅显示前 50 条，共 {len(entries)} 条匹配。")
        return "\n".join(lines)

    def insert_annotation_event(text: str, term: str, reading: str, annotation_lang: str):
        term = str(term or "").strip()
        try:
            annotation, warnings = make_annotation(
                term,
                reading,
                annotation_lang,
                pinyin_vocab_path=exact_vocab_path,
            )
        except PronunciationValidationError as exc:
            raise gr.Error(str(exc)) from exc
        source = str(text or "")
        cursor = 0
        updated = None
        for match in ANNOTATION_PATTERN.finditer(source):
            position = source.find(term, cursor, match.start())
            if position >= 0:
                updated = source[:position] + annotation + source[position + len(term) :]
                break
            cursor = match.end()
        if updated is None:
            position = source.find(term, cursor)
            if position >= 0:
                updated = source[:position] + annotation + source[position + len(term) :]
            else:
                updated = f"{source.rstrip()}\n{annotation}".lstrip()
        message = f"已插入 {annotation}。"
        if warnings:
            message += "\n" + "\n".join(f"- {item}" for item in warnings)
        return updated, message

    def preview_pronunciation_event(text: str, language: str, rows):
        result = process_pronunciation_text(
            text,
            language,
            parse_rows(rows, language),
            strict=False,
            pinyin_vocab_path=exact_vocab_path,
        )
        return result.text, format_pronunciation_report(result)

    def preview_segments_event(
        text: str,
        language: str,
        segmentation_mode: str,
        max_tokens: int,
        pause_preset: str,
        comma_pause_ms: int,
        sentence_pause_ms: int,
        paragraph_pause_ms: int,
    ):
        ensure_model()
        source = str(text or "").strip()
        if not source:
            raise gr.Error("请先输入需要预览的文本。")
        plan = build_desktop_plan(
            tts,
            source,
            language,
            segmentation_mode,
            int(max_tokens),
            pause_preset,
            int(comma_pause_ms),
            int(sentence_pause_ms),
            int(paragraph_pause_ms),
        )
        rows = [
            [
                item["index"],
                item["speech_block"],
                item["token_count"],
                item["pause_before_ms"],
                item["pause_after_ms"],
                item["text"],
            ]
            for item in plan.segments
        ]
        return rows, (
            f"共 {len(plan.segments)} 个 Token 分段 / {len(plan.chunks)} 个停顿语音块；"
            f"有效上限 {plan.max_tokens} Token，外加停顿 {plan.total_pause_ms}ms；"
            "GPT 合成提示 KV Cache 修复已启用。"
        )

    def change_emotion_mode(mode: int):
        return (
            gr.update(visible=mode == 1),
            gr.update(visible=mode == 2),
            gr.update(visible=mode == 3),
            gr.update(visible=mode in (1, 2, 3)),
        )

    def generate(
        prompt_audio: str,
        text: str,
        language: str,
        duration_factor: float,
        emotion_mode: int,
        emotion_audio: str,
        emotion_weight: float,
        emotion_text: str,
        random_emotion: bool,
        pronunciation_rows,
        pronunciation_strict: bool,
        quality_retry_count,
        quality_asr_backend,
        quality_asr_model,
        quality_asr_device,
        quality_threshold,
        *values,
        progress=gr.Progress(),
    ):
        ensure_model()
        if not prompt_audio:
            raise gr.Error("请先从已保存音色库选择角色，或上传/录制音色参考音频。")
        text = (text or "").strip()
        if not text:
            raise gr.Error("请输入需要合成的文本。")
        duration_factor = float(duration_factor)
        if not 0.5 <= duration_factor <= 2.0:
            raise gr.Error("时长系数必须在 0.5 到 2.0 之间。")

        try:
            pronunciation_entries = parse_rows(pronunciation_rows, language)
            pronunciation_result = process_pronunciation_text(
                text,
                language,
                pronunciation_entries,
                strict=bool(pronunciation_strict),
                pinyin_vocab_path=exact_vocab_path,
            )
        except PronunciationValidationError as exc:
            raise gr.Error(f"发音标注校验失败：{exc}") from exc
        save_dictionary(dictionary_file, pronunciation_entries)
        resolved_text = pronunciation_result.text

        vector_values = list(values[:8])
        advanced_values = values[8:]
        (
            do_sample,
            temperature,
            top_p,
            top_k,
            num_beams,
            repetition_penalty,
            length_penalty,
            max_mel_tokens,
            seed,
            diffusion_steps,
            inference_cfg_rate,
            cfm_temperature,
            stream_preview,
            segmentation_mode,
            max_text_tokens,
            segment_silence_ms,
            pause_preset,
            comma_pause_ms,
            sentence_pause_ms,
            paragraph_pause_ms,
            text_normalization,
            target_duration_mode,
            target_duration_seconds,
            postprocess_preset,
            postprocess_strength,
        ) = advanced_values

        seed = int(seed)
        retry_count = max(0, int(quality_retry_count or 0))
        diffusion_steps = int(diffusion_steps)
        inference_cfg_rate = float(inference_cfg_rate)
        cfm_temperature = float(cfm_temperature)
        if not 0 <= seed <= 0xFFFFFFFF:
            raise gr.Error("随机种子必须在 0 到 4294967295 之间。")
        if not 5 <= diffusion_steps <= 100:
            raise gr.Error("CFM 扩散步数必须在 5 到 100 之间。")
        if not 0 <= inference_cfg_rate <= 1.5:
            raise gr.Error("CFM 引导强度必须在 0 到 1.5 之间。")
        if not 0.1 <= cfm_temperature <= 1.5:
            raise gr.Error("CFM 温度必须在 0.1 到 1.5 之间。")

        if emotion_mode == 1 and not emotion_audio:
            raise gr.Error("情感参考音频模式需要上传情感参考音频。")

        emo_vector = None
        if emotion_mode == 2:
            emo_vector = tts.normalize_emo_vec(vector_values, apply_bias=True)

        target = output_dir / f"tts_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.wav"
        sampling_values = {
            "do_sample": bool(do_sample),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "num_beams": int(num_beams),
            "repetition_penalty": float(repetition_penalty),
            "length_penalty": float(length_penalty),
        }
        plan = build_desktop_plan(
            tts,
            resolved_text,
            language,
            str(segmentation_mode),
            int(max_text_tokens),
            str(pause_preset),
            int(comma_pause_ms),
            int(sentence_pause_ms),
            int(paragraph_pause_ms),
        )
        target_duration_seconds = float(target_duration_seconds or 0)
        if target_duration_mode != "off" and not 0.1 <= target_duration_seconds <= 3600:
            raise gr.Error("启用目标时长时，目标时长必须在 0.1–3600 秒。")
        performance_measurement = start_runtime_measurement()
        accel_disabled = False
        cache_risk_guarded = False
        long_text_guard_reports: list[dict] = []
        segment_rate_guard_reports: list[dict] = []

        def result_to_waveform(result):
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError("IndexTTS 返回了无法识别的音频结果。")
            sample_rate, raw = result
            tensor = torch.as_tensor(raw).detach().cpu()
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim == 2 and tensor.shape[-1] == 1:
                tensor = tensor.transpose(0, 1)
            elif tensor.ndim != 2:
                tensor = tensor.reshape(1, -1)
            if not tensor.dtype.is_floating_point:
                tensor = tensor.float() / 32768.0
            else:
                tensor = tensor.float()
                if tensor.numel() and float(tensor.abs().max()) > 2.0:
                    tensor = tensor / 32768.0
            return tensor.clamp(-1, 1).contiguous(), int(sample_rate)

        def result_duration_seconds(result):
            tensor, sample_rate = result_to_waveform(result)
            return tensor.shape[-1] / sample_rate

        def stream_piece_to_waveform(piece):
            if isinstance(piece, tuple) and len(piece) == 2:
                return result_to_waveform(piece)
            tensor = torch.as_tensor(piece).detach().cpu()
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim != 2:
                tensor = tensor.reshape(1, -1)
            tensor = tensor.float()
            if tensor.numel() and float(tensor.abs().max()) > 2.0:
                tensor = tensor / 32768.0
            return tensor.clamp(-1, 1).contiguous(), 22050

        def infer_once(
            factor: float,
            native_target_seconds: float | None = None,
            seed_offset: int = 0,
        ):
            tts.gr_progress = progress
            waveforms = []
            block_segment_records: list[list[dict]] = []
            sample_rate = None
            native_chunk_durations = (
                allocate_native_chunk_durations(plan, native_target_seconds)
                if native_target_seconds is not None
                else (None,) * len(plan.chunks)
            )
            with safe_gpt_acceleration(sampling_values, plan) as (disabled, guarded):
                for block_index, chunk in enumerate(plan.chunks):
                    latest_segment_records: list[dict] = []

                    def generate_with_limit(limit: int):
                        nonlocal latest_segment_records
                        latest_segment_records = []
                        return tts.infer(
                            spk_audio_prompt=prompt_audio,
                            text=chunk.text,
                            output_path=None,
                            lang=language,
                            emo_audio_prompt=emotion_audio if emotion_mode == 1 else None,
                            emo_alpha=float(emotion_weight),
                            emo_vector=emo_vector,
                            use_emo_text=emotion_mode == 3,
                            emo_text=(emotion_text or "").strip() or None,
                            use_random=bool(random_emotion),
                            verbose=verbose,
                            do_sample=bool(do_sample),
                            temperature=float(temperature),
                            top_p=float(top_p),
                            top_k=int(top_k) if int(top_k) > 0 else None,
                            num_beams=int(num_beams),
                            repetition_penalty=float(repetition_penalty),
                            length_penalty=float(length_penalty),
                            max_mel_tokens=int(max_mel_tokens),
                            max_text_tokens_per_segment=int(limit),
                            interval_silence=int(segment_silence_ms),
                            text_normalization=bool(text_normalization),
                            duration_factor=float(factor),
                            target_duration=native_chunk_durations[block_index],
                            seed=seed + int(seed_offset) + block_index,
                            diffusion_steps=diffusion_steps,
                            inference_cfg_rate=inference_cfg_rate,
                            cfm_temperature=cfm_temperature,
                            segment_collector=latest_segment_records,
                        )

                    block_token_count = len(
                        tts.tokenizer.encode(
                            f"<|{str(language).lower()}|> {chunk.text}",
                            allowed_special="all",
                        )
                    )
                    inference_result, guard_report = run_with_long_text_guard(
                        generate_with_limit,
                        result_duration_seconds,
                        text=chunk.text,
                        language=language,
                        token_count=block_token_count,
                        max_tokens=plan.max_tokens,
                        duration_factor=factor,
                        check_duration=native_chunk_durations[block_index] is None,
                    )
                    guard_report.update(
                        speech_block=block_index + 1,
                        seed_offset=int(seed_offset),
                    )
                    long_text_guard_reports.append(guard_report)
                    waveform, block_rate = result_to_waveform(inference_result)
                    if sample_rate is None:
                        sample_rate = block_rate
                    elif sample_rate != block_rate:
                        raise RuntimeError("停顿语音块的采样率不一致。")
                    waveforms.append(waveform)
                    normalized_records: list[dict] = []
                    for raw_record in latest_segment_records:
                        record_rate = int(raw_record.get("sample_rate") or block_rate)
                        if record_rate != block_rate:
                            raise RuntimeError("内部文本分段的采样率不一致。")
                        record_waveform = torch.as_tensor(
                            raw_record.get("waveform")
                        ).detach().cpu()
                        if record_waveform.ndim == 1:
                            record_waveform = record_waveform.unsqueeze(0)
                        elif record_waveform.ndim != 2:
                            record_waveform = record_waveform.reshape(1, -1)
                        record_waveform = record_waveform.float()
                        if (
                            record_waveform.numel()
                            and float(record_waveform.abs().max()) > 2.0
                        ):
                            record_waveform = record_waveform / 32768.0
                        normalized_records.append(
                            {
                                **raw_record,
                                "index": len(
                                    [
                                        item
                                        for records in block_segment_records
                                        for item in records
                                    ]
                                )
                                + len(normalized_records)
                                + 1,
                                "speech_block": block_index + 1,
                                "language": language,
                                "sample_rate": block_rate,
                                "duration_seconds": (
                                    record_waveform.shape[-1] / block_rate
                                ),
                                "waveform": record_waveform.clamp(-1, 1).contiguous(),
                            }
                        )
                    if not normalized_records:
                        normalized_records.append(
                            {
                                "index": sum(len(item) for item in block_segment_records)
                                + 1,
                                "speech_block": block_index + 1,
                                "text": chunk.text,
                                "language": language,
                                "sample_rate": block_rate,
                                "duration_seconds": waveform.shape[-1] / block_rate,
                                "waveform": waveform,
                            }
                        )
                    block_segment_records.append(normalized_records)

                flat_records = [
                    record for records in block_segment_records for record in records
                ]
                rate_reports = assess_segment_speech_rates(flat_records)
                if native_target_seconds is None and len(flat_records) >= 3:
                    retry_limit = max(
                        20,
                        min(
                            max(20, int(plan.max_tokens) - 1),
                            int(round(int(plan.max_tokens) * 2 / 3)),
                        ),
                    )
                    for rate_report in rate_reports:
                        if not rate_report.get("suspect"):
                            continue
                        rate_report["retried"] = False
                        if not bool(do_sample):
                            rate_report["retry_skipped"] = "deterministic_sampling"
                            continue
                        position = int(rate_report["position"])
                        record = flat_records[position]
                        try:
                            retry_result = tts.infer(
                                spk_audio_prompt=prompt_audio,
                                text=str(record["text"]),
                                output_path=None,
                                lang=language,
                                emo_audio_prompt=(
                                    emotion_audio if emotion_mode == 1 else None
                                ),
                                emo_alpha=float(emotion_weight),
                                emo_vector=emo_vector,
                                use_emo_text=emotion_mode == 3,
                                emo_text=(emotion_text or "").strip() or None,
                                use_random=bool(random_emotion),
                                verbose=verbose,
                                do_sample=True,
                                temperature=float(temperature),
                                top_p=float(top_p),
                                top_k=int(top_k) if int(top_k) > 0 else None,
                                num_beams=int(num_beams),
                                repetition_penalty=float(repetition_penalty),
                                length_penalty=float(length_penalty),
                                max_mel_tokens=int(max_mel_tokens),
                                max_text_tokens_per_segment=retry_limit,
                                interval_silence=int(segment_silence_ms),
                                text_normalization=False,
                                duration_factor=float(factor),
                                seed=(
                                    seed
                                    + int(seed_offset)
                                    + 100_003
                                    + position
                                ),
                                diffusion_steps=diffusion_steps,
                                inference_cfg_rate=inference_cfg_rate,
                                cfm_temperature=cfm_temperature,
                            )
                            retry_waveform, retry_rate = result_to_waveform(retry_result)
                            if retry_rate != sample_rate:
                                raise RuntimeError("语速异常段重试采样率不一致。")
                            retry_units_per_second = (
                                float(rate_report["speech_units"])
                                / max(
                                    1e-6,
                                    retry_waveform.shape[-1] / retry_rate,
                                )
                            )
                            accepted = retry_candidate_improves_rate(
                                float(rate_report["units_per_second"]),
                                retry_units_per_second,
                                float(rate_report["baseline_units_per_second"]),
                            )
                            rate_report.update(
                                retried=True,
                                retry_limit=retry_limit,
                                retry_duration_seconds=round(
                                    retry_waveform.shape[-1] / retry_rate, 4
                                ),
                                retry_units_per_second=round(
                                    retry_units_per_second, 4
                                ),
                                accepted=accepted,
                            )
                            if accepted:
                                record["waveform"] = retry_waveform
                                record["duration_seconds"] = (
                                    retry_waveform.shape[-1] / retry_rate
                                )
                        except Exception as retry_error:
                            rate_report.update(
                                retried=True,
                                accepted=False,
                                retry_error=(
                                    str(retry_error).strip()
                                    or type(retry_error).__name__
                                ),
                            )
                    if any(item.get("accepted") for item in rate_reports):
                        silence_samples = round(
                            sample_rate * int(segment_silence_ms) / 1000
                        )
                        for block_index, records in enumerate(block_segment_records):
                            rebuilt: list[torch.Tensor] = []
                            for record_index, record in enumerate(records):
                                rebuilt.append(record["waveform"])
                                if (
                                    record_index < len(records) - 1
                                    and silence_samples > 0
                                ):
                                    rebuilt.append(
                                        torch.zeros(
                                            (record["waveform"].shape[0], silence_samples),
                                            dtype=record["waveform"].dtype,
                                        )
                                    )
                            waveforms[block_index] = torch.cat(rebuilt, dim=-1)
                for rate_report in rate_reports:
                    rate_report["seed_offset"] = int(seed_offset)
                segment_rate_guard_reports.extend(rate_reports)
            assert sample_rate is not None
            combined = concatenate_with_pauses(
                waveforms,
                sample_rate,
                [chunk.pause_after_ms for chunk in plan.chunks],
                plan.chunks[0].pause_before_ms,
            )
            return combined, sample_rate, disabled, guarded

        def stream_infer_once(factor: float, native_target_seconds: float | None = None):
            """Yield playable chunks and return the same final tensor as non-streaming inference."""
            tts.gr_progress = progress
            block_waveforms = []
            sample_rate = 22050
            native_chunk_durations = (
                allocate_native_chunk_durations(plan, native_target_seconds)
                if native_target_seconds is not None
                else (None,) * len(plan.chunks)
            )
            with safe_gpt_acceleration(sampling_values, plan) as (disabled, guarded):
                leading_ms = plan.chunks[0].pause_before_ms
                if leading_ms:
                    leading = torch.zeros(1, round(sample_rate * leading_ms / 1000))
                    yield leading, f"正在播放前置停顿 {leading_ms}ms"
                for block_index, chunk in enumerate(plan.chunks):
                    stream = tts.infer(
                        spk_audio_prompt=prompt_audio,
                        text=chunk.text,
                        output_path=None,
                        lang=language,
                        emo_audio_prompt=emotion_audio if emotion_mode == 1 else None,
                        emo_alpha=float(emotion_weight),
                        emo_vector=emo_vector,
                        use_emo_text=emotion_mode == 3,
                        emo_text=(emotion_text or "").strip() or None,
                        use_random=bool(random_emotion),
                        verbose=verbose,
                        do_sample=bool(do_sample),
                        temperature=float(temperature),
                        top_p=float(top_p),
                        top_k=int(top_k) if int(top_k) > 0 else None,
                        num_beams=int(num_beams),
                        repetition_penalty=float(repetition_penalty),
                        length_penalty=float(length_penalty),
                        max_mel_tokens=int(max_mel_tokens),
                        max_text_tokens_per_segment=int(plan.max_tokens),
                        interval_silence=int(segment_silence_ms),
                        text_normalization=bool(text_normalization),
                        duration_factor=float(factor),
                        target_duration=native_chunk_durations[block_index],
                        seed=seed + block_index,
                        diffusion_steps=diffusion_steps,
                        inference_cfg_rate=inference_cfg_rate,
                        cfm_temperature=cfm_temperature,
                        stream_return=True,
                    )
                    pieces = []
                    for piece_index, piece in enumerate(stream, 1):
                        waveform_piece, piece_rate = stream_piece_to_waveform(piece)
                        if piece_rate != sample_rate:
                            raise RuntimeError("流式音频块的采样率不一致。")
                        pieces.append(waveform_piece)
                        yield waveform_piece, (
                            f"正在流式生成语音块 {block_index + 1}/{len(plan.chunks)}，"
                            f"已返回 {piece_index} 个音频片段"
                        )
                    if not pieces:
                        raise RuntimeError("IndexTTS 没有返回流式音频片段。")
                    block_waveforms.append(torch.cat(pieces, dim=-1))
                    pause_ms = chunk.pause_after_ms
                    if pause_ms:
                        pause = torch.zeros(1, round(sample_rate * pause_ms / 1000))
                        yield pause, f"正在播放语音块后的停顿 {pause_ms}ms"
            combined = concatenate_with_pauses(
                block_waveforms,
                sample_rate,
                [chunk.pause_after_ms for chunk in plan.chunks],
                plan.chunks[0].pause_before_ms,
            )
            return combined, sample_rate, disabled, guarded

        try:
            native_requested = target_duration_mode == "native" and target_duration_seconds > 0
            long_latin_guard_required = (
                str(language).upper() in {"EN", "ES"}
                and any(int(segment["token_count"]) >= 32 for segment in plan.segments)
            )
            segment_rate_guard_required = (
                not native_requested
                and bool(do_sample)
                and len(plan.segments) >= 3
            )
            stream_effective = (
                bool(stream_preview)
                and target_duration_mode in {"off", "native"}
                and retry_count == 0
                and not long_latin_guard_required
                and not segment_rate_guard_required
            )
            stream_note = ""
            if bool(stream_preview) and not stream_effective:
                if segment_rate_guard_required:
                    stream_note = (
                        "多段长文本需要先完成跨段语速异常检测与异常段单独重做；"
                        "本次自动关闭流式试听，仅输出校验后的最终音频。"
                    )
                elif long_latin_guard_required:
                    stream_note = (
                        "长英文/西语需要在返回前完成异常检测和自动缩短分段重试；"
                        "本次自动关闭流式试听，仅输出校验后的最终音频。"
                    )
                else:
                    stream_note = (
                        "ASR 自动质检或所选兼容目标时长模式需要完成全部候选后再播放；"
                        "本次仅输出最终音频。"
                    )
            if stream_effective:
                stream_generator = stream_infer_once(
                    duration_factor,
                    target_duration_seconds if native_requested else None,
                )
                while True:
                    try:
                        preview_waveform, preview_status = next(stream_generator)
                    except StopIteration as finished:
                        waveform, sample_rate, accel_disabled, cache_risk_guarded = finished.value
                        runtime_note = ""
                        break
                    yield (
                        (sample_rate if 'sample_rate' in locals() else 22050, preview_waveform.squeeze(0).numpy()),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        format_pronunciation_report(pronunciation_result) + "\n\n" + preview_status,
                        "正在生成；完成后显示真实耗时、RTF 与 CUDA 峰值显存。",
                    )
            else:
                (waveform, sample_rate, accel_disabled, cache_risk_guarded), runtime_note = execute_with_runtime_fallback(
                    lambda: infer_once(
                        duration_factor,
                        target_duration_seconds if native_requested else None,
                    )
                )
            used_factor = duration_factor
            duration_report = {"mode": "off", "action": "unchanged"}
            if native_requested:
                waveform, duration_report = apply_duration_policy(
                    waveform, sample_rate, target_duration_seconds, "exact"
                )
                duration_report["mode"] = "native"
                duration_report["engine"] = "length_regulator"
            elif target_duration_mode != "off" and target_duration_seconds > 0:
                actual_ms = waveform.shape[-1] * 1000 / sample_rate
                fitted = fit_duration_factor(
                    used_factor, actual_ms, target_duration_seconds * 1000
                )
                if abs(fitted - used_factor) >= 0.02:
                    (waveform, sample_rate, disabled_again, guarded_again), second_note = execute_with_runtime_fallback(
                        lambda: infer_once(fitted)
                    )
                    accel_disabled = accel_disabled or disabled_again
                    cache_risk_guarded = cache_risk_guarded or guarded_again
                    runtime_note = "；".join(item for item in (runtime_note, second_note) if item)
                    used_factor = fitted
                if target_duration_mode in {"pad", "exact"}:
                    waveform, duration_report = apply_duration_policy(
                        waveform, sample_rate, target_duration_seconds, target_duration_mode
                    )
                else:
                    duration_report = {
                        "mode": "natural",
                        "target_ms": round(target_duration_seconds * 1000),
                        "final_ms": round(waveform.shape[-1] * 1000 / sample_rate),
                        "used_duration_factor": round(used_factor, 4),
                        "action": "regenerated" if used_factor != duration_factor else "unchanged",
                    }
            waveform, postprocess_report = postprocess_waveform(
                waveform,
                sample_rate,
                str(postprocess_preset),
                float(postprocess_strength),
            )
            quality_report = {"enabled": False, "additional_candidates": 0}
            candidate_paths: list[str] = []
            if retry_count:
                quality_asr_enabled = asr_available(str(quality_asr_backend))
                candidate_dir = data_dir / "quality_candidates" / target.stem
                candidate_dir.mkdir(parents=True, exist_ok=True)

                def review_candidate(candidate, candidate_rate, attempt_index):
                    candidate_path = candidate_dir / f"candidate_{attempt_index + 1:02d}_seed_{seed + attempt_index * 100_003}.wav"
                    torchaudio.save(str(candidate_path), candidate, candidate_rate)
                    candidate_paths.append(str(candidate_path))
                    technical = technical_audio_review(candidate, candidate_rate)
                    result = {
                        "expected_text": text,
                        "recognized_text": "",
                        "passed": False,
                        "similarity": None,
                        "technical": technical,
                        "attempt": attempt_index + 1,
                        "seed": seed + attempt_index * 100_003,
                    }
                    if quality_asr_enabled:
                        try:
                            transcript = transcribe_audio_file(
                                candidate_path,
                                language=language,
                                backend=str(quality_asr_backend),
                                model_name=str(quality_asr_model),
                                device=str(quality_asr_device),
                                download_root=data_dir / "asr_models",
                            )
                            result.update(transcript)
                            result.update(review_transcript(
                                text,
                                transcript["text"],
                                language,
                                float(quality_threshold),
                            ))
                        except Exception as exc:
                            result["error"] = str(exc).strip() or type(exc).__name__
                    result["combined_score"] = combined_candidate_score(
                        technical["score"], result.get("similarity")
                    )
                    return result

                candidates = [(waveform, sample_rate)]
                attempts = [review_candidate(waveform, sample_rate, 0)]
                for retry_index in range(1, retry_count + 1):
                    (candidate, candidate_rate, disabled_again, guarded_again), retry_note = (
                        execute_with_runtime_fallback(
                            lambda retry_index=retry_index: infer_once(
                                used_factor,
                                target_duration_seconds if native_requested else None,
                                retry_index * 100_003,
                            )
                        )
                    )
                    accel_disabled = accel_disabled or disabled_again
                    cache_risk_guarded = cache_risk_guarded or guarded_again
                    runtime_note = "；".join(
                        item for item in (runtime_note, retry_note) if item
                    )
                    if native_requested:
                        candidate, _candidate_duration = apply_duration_policy(
                            candidate,
                            candidate_rate,
                            target_duration_seconds,
                            "exact",
                        )
                    elif target_duration_mode in {"pad", "exact"}:
                        candidate, _candidate_duration = apply_duration_policy(
                            candidate,
                            candidate_rate,
                            target_duration_seconds,
                            target_duration_mode,
                        )
                    candidate, _candidate_post = postprocess_waveform(
                        candidate,
                        candidate_rate,
                        str(postprocess_preset),
                        float(postprocess_strength),
                    )
                    candidate_review = review_candidate(
                        candidate,
                        candidate_rate,
                        retry_index,
                    )
                    candidates.append((candidate, candidate_rate))
                    attempts.append(candidate_review)
                selected_index = select_best_candidate(attempts)
                waveform, sample_rate = candidates[selected_index]
                selected_review = attempts[selected_index]
                quality_report = {
                    "enabled": quality_asr_enabled,
                    "selection_method": "asr+technical" if quality_asr_enabled else "technical",
                    "additional_candidates": retry_count,
                    "attempt_count": len(attempts),
                    "selected_candidate": selected_index + 1,
                    "selected_seed": selected_review["seed"],
                    "passed": bool(selected_review["passed"]),
                    "similarity": selected_review.get("similarity"),
                    "recognized_text": selected_review.get("recognized_text", selected_review.get("text", "")),
                    "attempts": attempts,
                }
            torchaudio.save(str(target), waveform, sample_rate)
        except Exception as exc:
            traceback.print_exc()
            detail = str(exc).strip() or type(exc).__name__
            raise gr.Error(f"语音生成失败：{detail}") from exc
        if not target.exists():
            raise gr.Error("语音生成失败，请查看桌面启动日志。")

        waveform, sample_rate = torchaudio.load(str(target))
        audio_duration = waveform.shape[-1] / sample_rate
        performance = finish_runtime_measurement(
            performance_measurement,
            audio_duration,
        )
        metrics = format_runtime_metrics(performance)
        if accel_disabled and not cache_risk_guarded:
            metrics += "；当前采样参数与 GPT 加速不兼容，本次已使用普通 GPT 路径"
        if cache_risk_guarded:
            metrics += "；GPT 合成提示 KV Cache 保护已触发"
        metrics += (
            f"；分段 {len(plan.segments)}（上限 {plan.max_tokens} Token），"
            f"外加停顿 {plan.total_pause_ms}ms；目标时长={json.dumps(duration_report, ensure_ascii=False)}；"
            f"CFM={diffusion_steps}步/CFG {inference_cfg_rate:.2f}/温度 {cfm_temperature:.2f}/seed {seed}；"
            f"后处理={json.dumps(postprocess_report, ensure_ascii=False)}"
        )
        metrics += f"；ASR自动质检={json.dumps(quality_report, ensure_ascii=False)}"
        latin_guard_reports = [
            item for item in long_text_guard_reports if item.get("enabled")
        ]
        if latin_guard_reports:
            metrics += "；长英文/西语保护=" + json.dumps(
                latin_guard_reports, ensure_ascii=False
            )
        rate_guard_reports = [
            item
            for item in segment_rate_guard_reports
            if item.get("eligible") or item.get("suspect")
        ]
        if rate_guard_reports:
            metrics += "；跨段语速保护=" + json.dumps(
                rate_guard_reports, ensure_ascii=False
            )
        if runtime_note:
            metrics += "；" + runtime_note
        if stream_note:
            metrics += "；" + stream_note

        append_history(output_dir, {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "language": language,
            "duration_factor": duration_factor,
            "emotion_mode": EMOTION_MODES[emotion_mode],
            "text": text,
            "resolved_text": resolved_text,
            "file": str(target),
            "metrics": metrics,
            "performance": performance,
            "segment_rate_guard": rate_guard_reports,
        })
        memory_report = apply_memory_policy()
        metrics += "；模型生命周期=" + json.dumps(memory_report, ensure_ascii=False)
        yield (
            gr.skip(),
            str(target),
            candidate_paths,
            load_history(output_dir),
            format_pronunciation_report(pronunciation_result) + "\n\n" + metrics,
            metrics,
        )

    def save_preset_event(
        name,
        prompt_audio_value,
        text_value,
        language_value,
        duration_value,
        emotion_mode_value,
        emotion_audio_value,
        emotion_weight_value,
        emotion_text_value,
        random_emotion_value,
        pronunciation_strict_value,
        *values,
    ):
        vector_values = [float(value) for value in values[:8]]
        (
            do_sample_value,
            temperature_value,
            top_p_value,
            top_k_value,
            num_beams_value,
            repetition_penalty_value,
            length_penalty_value,
            max_mel_tokens_value,
            seed_value,
            diffusion_steps_value,
            inference_cfg_rate_value,
            cfm_temperature_value,
            stream_preview_value,
            segmentation_mode_value,
            max_text_tokens_value,
            segment_silence_value,
            pause_preset_value,
            comma_pause_value,
            sentence_pause_value,
            paragraph_pause_value,
            text_normalization_value,
            target_duration_mode_value,
            target_duration_seconds_value,
            postprocess_preset_value,
            postprocess_strength_value,
        ) = values[8:]
        settings = {
            "text": str(text_value or ""),
            "language": str(language_value or "ZH"),
            "duration_factor": float(duration_value),
            "emotion_mode": int(emotion_mode_value or 0),
            "emotion_weight": float(emotion_weight_value),
            "emotion_text": str(emotion_text_value or ""),
            "random_emotion": bool(random_emotion_value),
            "emotion_vector": vector_values,
            "pronunciation_strict": bool(pronunciation_strict_value),
            "advanced": {
                "do_sample": bool(do_sample_value),
                "temperature": float(temperature_value),
                "top_p": float(top_p_value),
                "top_k": int(top_k_value),
                "num_beams": int(num_beams_value),
                "repetition_penalty": float(repetition_penalty_value),
                "length_penalty": float(length_penalty_value),
                "max_mel_tokens": int(max_mel_tokens_value),
                "seed": int(seed_value),
                "diffusion_steps": int(diffusion_steps_value),
                "inference_cfg_rate": float(inference_cfg_rate_value),
                "cfm_temperature": float(cfm_temperature_value),
                "stream_preview": bool(stream_preview_value),
                "segmentation_mode": str(segmentation_mode_value),
                "max_text_tokens": int(max_text_tokens_value),
                "segment_silence_ms": int(segment_silence_value),
                "pause_preset": str(pause_preset_value),
                "comma_pause_ms": int(comma_pause_value),
                "sentence_pause_ms": int(sentence_pause_value),
                "paragraph_pause_ms": int(paragraph_pause_value),
                "text_normalization": bool(text_normalization_value),
                "target_duration_mode": str(target_duration_mode_value),
                "target_duration_seconds": float(target_duration_seconds_value),
                "postprocess_preset": str(postprocess_preset_value),
                "postprocess_strength": float(postprocess_strength_value),
            },
        }
        try:
            save_preset(
                data_dir,
                name,
                settings,
                prompt_audio=prompt_audio_value,
                emotion_audio=emotion_audio_value,
            )
        except Exception as exc:
            raise gr.Error(f"预设保存失败：{exc}") from exc
        choices = [""] + list_presets(data_dir)
        return gr.update(choices=choices, value=str(name).strip()), f"已保存预设：{str(name).strip()}"

    def load_preset_event(name):
        preset = load_preset(data_dir, name)
        if not preset:
            raise gr.Error("预设不存在或已损坏。")
        settings = preset.get("settings") or {}
        advanced = settings.get("advanced") or {}
        vector = list(settings.get("emotion_vector") or [0.0] * 8)[:8]
        vector.extend([0.0] * (8 - len(vector)))
        mode = int(settings.get("emotion_mode", 0))
        warning = ""
        if mode == 3 and not qwen_emotion_available:
            mode = 0
            warning = "；当前为低显存模式，情感描述文本已改为跟随音色参考"
        audio = preset.get("audio") or {}
        return (
            audio.get("prompt"),
            settings.get("text", ""),
            settings.get("language", "ZH"),
            settings.get("duration_factor", 1.0),
            mode,
            audio.get("emotion"),
            settings.get("emotion_weight", 0.65),
            settings.get("emotion_text", ""),
            settings.get("random_emotion", False),
            *vector,
            settings.get("pronunciation_strict", True),
            advanced.get("do_sample", True),
            advanced.get("temperature", 0.8),
            advanced.get("top_p", 0.8),
            advanced.get("top_k", 30),
            advanced.get("num_beams", 3),
            advanced.get("repetition_penalty", 10.0),
            advanced.get("length_penalty", 0.0),
            advanced.get("max_mel_tokens", 1500),
            advanced.get("seed", 0),
            advanced.get("diffusion_steps", 25),
            advanced.get("inference_cfg_rate", 0.7),
            advanced.get("cfm_temperature", 1.0),
            advanced.get("stream_preview", True),
            advanced.get("segmentation_mode", "auto"),
            advanced.get("max_text_tokens", 120),
            advanced.get("segment_silence_ms", 200),
            advanced.get("pause_preset", "off"),
            advanced.get("comma_pause_ms", 100),
            advanced.get("sentence_pause_ms", 300),
            advanced.get("paragraph_pause_ms", 600),
            advanced.get("text_normalization", True),
            advanced.get("target_duration_mode", "off"),
            advanced.get("target_duration_seconds", 0.0),
            advanced.get("postprocess_preset", "off"),
            advanced.get("postprocess_strength", 1.0),
            f"已载入预设：{name}{warning}",
        )

    def delete_preset_event(name):
        if not name:
            raise gr.Error("请先选择要删除的预设。")
        if not delete_preset(data_dir, name):
            raise gr.Error("预设不存在。")
        return gr.update(choices=[""] + list_presets(data_dir), value=""), f"已删除预设：{name}"

    def voice_rows(query="", tags="", favorites_only=False):
        def emotion_summary(item: VoiceProfile) -> str:
            if item.emotion_mode == "speaker":
                return "跟随音色参考"
            if item.emotion_mode == "reference_audio":
                return "情感参考音频 · " + Path(item.emotion_audio_path).name
            if item.emotion_mode == "text":
                return "文本 · " + (item.emotion_text or "分析每句台词")
            populated = [
                f"{label}{value:.2f}"
                for label, value in zip(EMOTION_LABELS, item.emotion_vector)
                if value > 0
            ]
            suffix = " · 随机原型" if item.emotion_use_random else ""
            return "八维向量 · " + (" / ".join(populated) or "全零") + suffix

        rows = []
        for item in voice_library.search(
            str(query or ""), tags=str(tags or ""), favorites_only=bool(favorites_only)
        ):
            quality = item.quality or {}
            score = quality.get("score")
            quality_text = (
                f"{score}/100 · {quality.get('grade', '')}" if score is not None else "未检测"
            )
            rows.append(
                [
                    "★" if item.favorite else "",
                    item.name,
                    "、".join(item.tags),
                    item.language,
                    quality_text,
                    emotion_summary(item),
                    item.notes,
                    item.audio_path,
                ]
            )
        return rows

    def voice_choices():
        return [item.name for item in voice_library.list()]

    def filter_voice_library_event(query, tags, favorites_only):
        rows = voice_rows(query, tags, favorites_only)
        return rows, f"当前显示 {len(rows)} 个音色；搜索会匹配名称、标签和备注。"

    def load_single_voice_event(name):
        """Load a persisted voice into the regular single-generation audio input."""

        if not name:
            return gr.update(), (
                "未选择已保存音色；仍可在下方上传、拖入或录制参考音频。"
            )
        try:
            profile = voice_library.get(name)
        except KeyError as exc:
            raise gr.Error(f"载入已保存音色失败：{exc}") from exc
        return (
            profile.audio_path,
            f"已载入“{profile.name}”的音色参考，可直接生成。"
            "这里只复用音色音频，不会覆盖本页当前的语言、情感或生成参数。",
        )

    def refresh_single_voice_event(selected):
        choices = voice_choices()
        current = str(selected or "").strip()
        if current and current in choices:
            profile = voice_library.get(current)
            return (
                gr.update(choices=choices, value=current),
                profile.audio_path,
                f"音色库已刷新，共 {len(choices)} 个角色；已重新载入“{profile.name}”。",
            )
        if current:
            return (
                gr.update(choices=choices, value=None),
                None,
                f"音色库已刷新，共 {len(choices)} 个角色；原选择已不存在，请重新选择。",
            )
        return (
            gr.update(choices=choices, value=None),
            gr.update(),
            f"音色库已刷新，共 {len(choices)} 个角色。请选择一个角色，或继续上传参考音频。",
        )

    def save_voice_event(
        name,
        audio,
        voice_language,
        emotion_mode_value,
        emotion_audio_value,
        emotion_text_value,
        emotion_strength_value,
        emotion_random_value,
        *values,
    ):
        if len(values) >= 14:
            tags_value, favorite_value, notes_value = values[:3]
            vector_values = values[3:11]
            dictionary_text, selected_voice, update_selected = values[11:14]
        elif len(values) >= 11:  # Desktop 0.20 and older event/test compatibility.
            tags_value, favorite_value, notes_value = "", False, ""
            vector_values = values[:8]
            dictionary_text, selected_voice, update_selected = values[8:11]
        else:
            raise gr.Error("保存角色音色参数不完整，请刷新页面后重试。")
        if not audio:
            raise gr.Error("请上传角色的音色参考音频。")
        if update_selected and not selected_voice:
            raise gr.Error("要更新或改名，请先选择并载入已有角色。")
        try:
            emotion_mode_id = PROFILE_EMOTION_MODE_IDS[int(emotion_mode_value)]
        except (IndexError, TypeError, ValueError) as exc:
            raise gr.Error("请选择有效的角色情感模式。") from exc
        try:
            try:
                waveform, sample_rate = torchaudio.load(str(audio))
                quality = analyze_reference_audio(waveform, int(sample_rate))
            except Exception:
                quality = {}
            profile = voice_library.save(
                name,
                audio,
                voice_language,
                emotion_mode=emotion_mode_id,
                emotion_text=str(emotion_text_value or "").strip(),
                emotion_strength=float(emotion_strength_value),
                emotion_audio=emotion_audio_value,
                emotion_vector=vector_values,
                emotion_use_random=bool(emotion_random_value),
                pronunciation_dictionary=str(dictionary_text or ""),
                tags=str(tags_value or ""),
                favorite=bool(favorite_value),
                notes=str(notes_value or ""),
                quality=quality,
                replace_name_or_id=selected_voice if update_selected else None,
            )
        except Exception as exc:
            raise gr.Error(f"保存角色音色失败：{exc}") from exc
        choices = voice_choices()
        return (
            voice_rows(),
            gr.update(choices=choices, value=profile.name),
            gr.update(choices=choices),
            gr.update(choices=choices, value=profile.name),
            profile.audio_path,
            f"已同步“{profile.name}”到语音生成页，可直接复用其音色参考。",
            False,
            (
                f"已保存角色音色：{profile.name}。参考音频质量 "
                f"{profile.quality.get('score', '—')}/100（{profile.quality.get('grade', '未评级')}）。"
            ),
        )

    def load_voice_event(name):
        if not name:
            raise gr.Error("请先选择要载入的角色。")
        try:
            profile = voice_library.get(name)
        except KeyError as exc:
            raise gr.Error(str(exc)) from exc
        return (
            profile.name,
            profile.audio_path,
            profile.language,
            PROFILE_EMOTION_MODE_IDS.index(profile.emotion_mode),
            profile.emotion_audio_path or None,
            profile.emotion_text,
            profile.emotion_strength,
            profile.emotion_use_random,
            *profile.emotion_vector,
            profile.pronunciation_dictionary,
            "、".join(profile.tags),
            profile.favorite,
            profile.notes,
            True,
            f"已载入：{profile.name}。可直接试听；修改后勾选“更新所选角色”即可覆盖或改名。",
        )

    def export_voice_bundle_event(selected_voice, export_all):
        names = None if bool(export_all) else [str(selected_voice or "").strip()]
        if names is not None and not names[0]:
            raise gr.Error("请选择一个角色，或勾选导出全部音色。")
        target = (
            data_dir
            / "exports"
            / f"T8star-Aix-voices-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        try:
            bundle = voice_library.export_bundle(target, names)
        except Exception as exc:
            raise gr.Error(f"导出音色包失败：{exc}") from exc
        return str(bundle), f"已导出可供桌面端和 ComfyUI 共用的音色包：{bundle.name}"

    def import_voice_bundle_event(bundle_path, conflict_mode):
        if not bundle_path:
            raise gr.Error("请先选择 .t8voice.zip 音色包。")
        try:
            imported = voice_library.import_bundle(bundle_path, conflict=str(conflict_mode))
        except Exception as exc:
            raise gr.Error(f"导入音色包失败：{exc}") from exc
        choices = voice_choices()
        names = "、".join(item.name for item in imported) or "没有新增角色"
        return (
            voice_rows(),
            gr.update(choices=choices, value=imported[0].name if imported else None),
            gr.update(choices=choices),
            gr.update(choices=choices),
            f"音色包导入完成：{names}",
        )

    def delete_voice_event(name, single_selected=None):
        if not name:
            raise gr.Error("请先选择要删除的角色。")
        try:
            removed = voice_library.delete(name)
        except KeyError as exc:
            raise gr.Error(str(exc)) from exc
        choices = voice_choices()
        deleted_single_selection = (
            str(single_selected or "").strip().casefold() == removed.name.casefold()
        )
        return (
            voice_rows(),
            gr.update(choices=choices, value=None),
            gr.update(choices=choices),
            gr.update(
                choices=choices,
                value=None if deleted_single_selection else single_selected,
            ),
            None if deleted_single_selection else gr.update(),
            (
                f"已删除当前单句音色“{removed.name}”，请重新选择或上传参考音频。"
                if deleted_single_selection
                else f"音色库已更新，共 {len(choices)} 个角色。"
            ),
            False,
            f"已删除角色音色：{removed.name}",
        )

    def import_script_event(path):
        if not path:
            raise gr.Error("请先选择 SRT、TXT 或 JSON 文件。")
        source = Path(path)
        try:
            content = source.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            content = source.read_text(encoding="gb18030")
        script_type = "srt" if source.suffix.lower() == ".srt" else "batch"
        return content, script_type

    def parse_dialogue(script_type, script, default_role, default_language):
        parser = parse_srt if script_type == "srt" else parse_batch_script
        return parser(script, default_role, default_language)

    def preview_dialogue_event(script_type, script, default_role, default_language):
        try:
            lines = parse_dialogue(script_type, script, default_role, default_language)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        missing = missing_roles(lines, voice_choices())
        rows = timeline_rows(lines)
        status = f"已解析 {len(lines)} 条台词。"
        if missing:
            status += " 缺少角色音色：" + "、".join(missing)
        else:
            status += " 所有角色均已映射。"
        return rows, render_timeline_html(lines), status

    def refresh_timeline_event(
        script_type,
        script,
        default_role,
        default_language,
        edited_rows,
        generation_report,
    ):
        try:
            parsed = parse_dialogue(script_type, script, default_role, default_language)
            lines = apply_timeline_edits(parsed, edited_rows)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        reports = timeline_reports_with_edits(lines, generation_report)
        return render_timeline_html(lines, reports), f"时间轴已自动刷新，共 {len(lines)} 条；编辑会用于下一次生成或单句重做。"

    def timeline_reports_with_edits(lines, generation_report):
        try:
            payload = json.loads(str(generation_report or ""))
            source = payload.get("lines") if isinstance(payload, dict) else []
        except (TypeError, json.JSONDecodeError):
            source = []
        report_map = {
            int(item.get("index", position)): dict(item)
            for position, item in enumerate(source or (), 1)
            if isinstance(item, dict)
        }
        reports = []
        for line in lines:
            item = report_map.get(line.index)
            if not item:
                continue
            timeline = dict(item.get("timeline") or {})
            if line.start_ms is not None and line.end_ms is not None:
                timeline.update(
                    actual_start_ms=int(line.start_ms),
                    actual_end_ms=int(line.end_ms),
                )
            item["timeline"] = timeline
            reports.append(item)
        return reports

    def apply_timeline_drag_event(
        script_type,
        script,
        default_role,
        default_language,
        edited_rows,
        drag_payload,
        generation_report,
    ):
        try:
            parsed = parse_dialogue(script_type, script, default_role, default_language)
            lines = apply_timeline_edits(parsed, edited_rows)
            lines, drag = apply_timeline_drag_payload(lines, drag_payload)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        reports = timeline_reports_with_edits(lines, generation_report)
        snapped = drag.get("snapped_to_ms")
        action = {
            "move": "已平移",
            "resize_start": "已修改左边界",
            "resize_end": "已修改右边界",
            "select": "已选择",
        }[drag["mode"]]
        status = (
            f"{action}第 {drag['index']} 条：{drag['start_ms']}–{drag['end_ms']}ms；"
            "已同步上方表格。可直接重新混音，或单独重做该句。"
        )
        if snapped is not None:
            status += f" 已吸附到 {int(round(float(snapped)))}ms。"
        return (
            timeline_rows(lines),
            render_timeline_html(lines, reports),
            status,
            drag["index"],
        )

    def suggest_dialogue_emotions_event(
        script_type,
        script,
        default_role,
        default_language,
        edited_rows,
        context_window,
        overwrite_existing,
        progress=gr.Progress(),
    ):
        """Write suggestions into the editable table without starting synthesis."""

        try:
            parsed = parse_dialogue(script_type, script, default_role, default_language)
            lines = apply_timeline_edits(parsed, edited_rows)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        ensure_model()
        had_qwen = getattr(tts, "qwen_emo", None) is not None
        released_temporary_qwen = False
        try:
            tts.ensure_qwen_emotion()
            if getattr(tts, "qwen_emo", None) is None:
                raise RuntimeError("QwenEmotion 加载后仍不可用。")

            def update_progress(position, total, line):
                if line is None:
                    progress(1.0, desc="上下文情感分析完成，等待用户确认")
                    return
                progress(
                    (position, max(1, total)),
                    desc=f"分析第 {getattr(line, 'index', position + 1)} 条情感：{getattr(line, 'role', '')}",
                )

            suggested, report = suggest_context_emotions(
                lines,
                tts.qwen_emo.inference,
                context_window=int(context_window),
                overwrite_existing=bool(overwrite_existing),
                progress=update_progress,
            )
        except Exception as exc:
            raise gr.Error(
                "上下文情感分析失败："
                f"{type(exc).__name__}: {exc}。可切换到显存更充足的模式后重试，"
                "或继续在时间轴最后一列手工填写 text:/vector:。"
            ) from exc
        finally:
            if not had_qwen and getattr(tts, "qwen_emo", None) is not None:
                tts.qwen_emo = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                released_temporary_qwen = True
        report["temporary_qwen_released"] = released_temporary_qwen
        report["instruction"] = (
            "建议已写入可编辑时间轴最后一列；请逐行确认、修改或清空，"
            "只有点击“生成全部台词”后才会开始合成。"
        )
        status = (
            f"已分析 {report['classified_count']} 条上下文情感，"
            f"保留 {report['preserved_count']} 条已有人工设置。"
            " 建议仅写入表格，尚未生成音频；请确认后再点击生成。"
        )
        return (
            timeline_rows(suggested),
            render_timeline_html(suggested),
            status,
            json.dumps(report, ensure_ascii=False, indent=2),
        )

    def load_dialogue_task_editor_event(task_id):
        task_id = str(task_id or "").strip()
        if not task_id:
            return tuple(gr.update() for _ in range(9))
        try:
            task = load_task(output_dir, task_id)
            saved = task.get("settings") or {}
            task_script_type = str(task.get("script_type") or "batch")
            task_script = str(task.get("script") or "")
            task_default_role = saved.get("default_role", "旁白")
            task_default_language = saved.get("default_language", "ZH")
            parsed = parse_dialogue(
                task_script_type,
                task_script,
                task_default_role,
                task_default_language,
            )
            lines = apply_timeline_edits(parsed, saved.get("timeline_rows"))
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        saved_lines = task.get("lines") or {}
        reports = [
            dict((saved_lines.get(str(line.index)) or {}).get("report") or line.to_dict())
            for line in lines
        ]
        report_payload = {"lines": reports}
        report_file = Path(str(task.get("report_file") or ""))
        if report_file.is_file():
            try:
                loaded_report = json.loads(report_file.read_text(encoding="utf-8-sig"))
                if isinstance(loaded_report, dict):
                    report_payload = loaded_report
            except (OSError, json.JSONDecodeError):
                pass
        return (
            task_script_type,
            task_script,
            task_default_role,
            task_default_language,
            timeline_rows(lines),
            saved.get("timeline_policy", "shift"),
            render_timeline_html(lines, reports),
            f"任务 {task_id} 已载入；编辑表格会自动刷新轨道，选中一行后可单独重做并合入。",
            json.dumps(report_payload, ensure_ascii=False, indent=2),
        )

    def export_dialogue_project_event(task_id):
        task_id = str(task_id or "").strip()
        if not task_id:
            raise gr.Error("请先选择一个已保存任务。")
        target = (
            data_dir
            / "exports"
            / f"T8star-Aix-{task_id}-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        try:
            project = export_project(output_dir, task_id, voice_library, target)
        except Exception as exc:
            raise gr.Error(f"导出完整配音工程失败：{exc}") from exc
        return str(project), f"完整工程已导出：{project.name}。其中包含任务、参数、时间轴、逐句音频、报告和所用音色。"

    def import_dialogue_project_event(project_path, voice_conflict):
        if not project_path:
            raise gr.Error("请先选择 .indextts-project.zip 工程包。")
        try:
            result = import_project(
                project_path,
                output_dir,
                voice_library,
                voice_conflict=str(voice_conflict),
            )
            loaded = list(load_dialogue_task_editor_event(result["task_id"]))
        except Exception as exc:
            raise gr.Error(f"导入完整配音工程失败：{exc}") from exc
        imported_voices = "、".join(result["imported_voices"]) or "无（使用了现有音色或工程未包含音色）"
        loaded[7] = (
            f"工程已导入为任务 {result['task_id']}；导入音色：{imported_voices}。"
            "可以继续未完成任务、单句重做或直接重新混音。"
        )
        choices = voice_choices()
        return (
            gr.update(choices=task_choices(output_dir), value=result["task_id"]),
            *loaded,
            voice_rows(),
            gr.update(choices=choices),
            gr.update(choices=choices),
        )

    def select_timeline_row_event(edited_rows, event: gr.SelectData):
        if isinstance(edited_rows, dict):
            rows = list(edited_rows.get("data") or [])
        elif hasattr(edited_rows, "values") and hasattr(edited_rows.values, "tolist"):
            rows = edited_rows.values.tolist()
        else:
            rows = list(edited_rows or [])
        index = event.index
        row_position = int(index[0] if isinstance(index, (tuple, list)) else index)
        if row_position < 0 or row_position >= len(rows):
            raise gr.Error("请选择时间轴中的有效台词行。")
        try:
            line_number = int(float(rows[row_position][0]))
        except (TypeError, ValueError, IndexError) as exc:
            raise gr.Error("所选时间轴行缺少有效台词序号。") from exc
        return line_number, f"已选择第 {line_number} 条；修改台词后点击“重做选中/指定句并重新合并”。"

    def asr_proofread_event(audio_path, expected_text, language, backend, model_name, device, threshold):
        if not audio_path:
            raise gr.Error("请先生成或上传需要校对的音频。")
        if not str(expected_text or "").strip():
            raise gr.Error("请输入用于比对的原始文本。")
        try:
            transcript = transcribe_audio_file(
                audio_path,
                language=language,
                backend=backend,
                model_name=model_name,
                device=device,
                download_root=data_dir / "asr_models",
            )
            review = review_transcript(expected_text, transcript["text"], language, threshold)
        except Exception as exc:
            raise gr.Error(f"ASR 校对失败：{str(exc).strip() or type(exc).__name__}") from exc
        payload = {**transcript, **review}
        verdict = "通过" if review["passed"] else "需复核"
        metric_name = review["metric"].upper()
        differences = review.get("differences") or []
        if differences:
            rows = ["| 类型 | 原文 | 识别 |", "|---|---|---|"]
            for item in differences:
                rows.append(f"| {item['operation']} | {item['expected'] or '∅'} | {item['recognized'] or '∅'} |")
            diff_markdown = "\n".join(rows)
        else:
            diff_markdown = "未发现归一化后的文本差异。"
        status = f"{verdict} · 相似度 {review['similarity']:.1%} · {metric_name} {review['metric_error_rate']:.3f} · {transcript['backend']}"
        waveform, sample_rate = torchaudio.load(str(audio_path))
        alignment = waveform_html(
            waveform,
            sample_rate,
            transcript.get("word_timestamps") or [],
        )
        return (
            transcript["text"],
            status,
            diff_markdown,
            json.dumps(payload, ensure_ascii=False, indent=2),
            alignment,
        )

    def inspect_reference_event(audio_path, auto_prepare, maximum_seconds, padding_ms):
        if not audio_path:
            raise gr.Error("请先从已保存音色库选择角色，或上传/录制音色参考音频。")
        try:
            waveform, sample_rate = torchaudio.load(str(audio_path))
            if bool(auto_prepare):
                prepared, report = prepare_reference_audio(
                    waveform,
                    sample_rate,
                    max_seconds=float(maximum_seconds),
                    padding_ms=int(padding_ms),
                )
                prepared_dir = data_dir / "prepared_references"
                prepared_dir.mkdir(parents=True, exist_ok=True)
                prepared_path = prepared_dir / f"reference_{uuid.uuid4().hex[:12]}.wav"
                torchaudio.save(str(prepared_path), prepared, sample_rate)
                output_path = str(prepared_path)
                display_waveform = prepared
            else:
                quality = analyze_reference_audio(waveform, sample_rate)
                report = {
                    "original": quality,
                    "prepared": quality,
                    "trimmed": False,
                    "selected_start_seconds": 0.0,
                    "selected_end_seconds": quality["duration_seconds"],
                }
                output_path = str(audio_path)
                display_waveform = waveform
        except Exception as exc:
            raise gr.Error(f"参考音频检测失败：{str(exc).strip() or type(exc).__name__}") from exc
        return (
            output_path,
            json.dumps(report, ensure_ascii=False, indent=2),
            waveform_html(display_waveform, sample_rate),
        )

    def probe_audiocpp_event(executable):
        if not str(executable or "").strip():
            executable = audiocpp_component_status(data_dir).get("executable")
        return json.dumps(probe_audiocpp(executable), ensure_ascii=False, indent=2)

    def audiocpp_status_event():
        status = audiocpp_component_status(data_dir)
        return (
            status.get("executable") or gr.update(),
            status.get("modelPath") or gr.update(),
            status.get("installedBackend") or gr.update(),
            json.dumps(status, ensure_ascii=False, indent=2),
        )

    def _audiocpp_progress(progress):
        def callback(event):
            percent = max(0.0, min(100.0, float(event.get("percent") or 0.0)))
            progress(percent / 100.0, desc=str(event.get("message") or event.get("label") or "audio.cpp"))

        return callback

    def install_audiocpp_runtime_event(backend, progress=gr.Progress()):
        try:
            result = install_audiocpp_runtime(
                data_dir, str(backend), callback=_audiocpp_progress(progress)
            )
            status = audiocpp_component_status(data_dir)
        except Exception as exc:
            raise gr.Error(f"安装 audio.cpp 运行时失败：{exc}") from exc
        return (
            status.get("executable") or result.get("executable"),
            status.get("modelPath") or gr.update(),
            status.get("installedBackend") or str(backend),
            json.dumps(status, ensure_ascii=False, indent=2),
        )

    def install_audiocpp_model_event(quantization, progress=gr.Progress()):
        try:
            result = install_audiocpp_model(
                data_dir,
                str(quantization),
                callback=_audiocpp_progress(progress),
            )
            status = audiocpp_component_status(data_dir)
        except Exception as exc:
            raise gr.Error(f"下载 audio.cpp GGUF 模型失败：{exc}") from exc
        return (
            status.get("executable") or gr.update(),
            status.get("modelPath") or result.get("modelPath"),
            status.get("installedBackend") or gr.update(),
            json.dumps(status, ensure_ascii=False, indent=2),
        )

    def install_audiocpp_all_event(backend, quantization, progress=gr.Progress()):
        callback = _audiocpp_progress(progress)
        try:
            runtime = install_audiocpp_runtime(data_dir, str(backend), callback=callback)
            model = install_audiocpp_model(
                data_dir, str(quantization), callback=callback
            )
            status = audiocpp_component_status(data_dir)
        except Exception as exc:
            raise gr.Error(f"一键安装 audio.cpp 完整组件失败：{exc}") from exc
        status["lastInstall"] = {"runtime": runtime, "model": model}
        return (
            status.get("executable"),
            status.get("modelPath"),
            status.get("installedBackend") or str(backend),
            json.dumps(status, ensure_ascii=False, indent=2),
        )

    def generate_audiocpp_event(
        executable,
        model_dir,
        speaker_audio,
        source_text,
        source_language,
        backend,
        duration,
        emotion_text,
        memory_saver,
    ):
        if not speaker_audio:
            raise gr.Error("audio.cpp 实验后端需要音色参考 WAV。")
        installed = audiocpp_component_status(data_dir, verify_hash=True)
        executable = str(executable or installed.get("executable") or "").strip()
        model_dir = str(model_dir or installed.get("modelPath") or "").strip()

        def managed_path_key(value):
            return str(Path(str(value)).expanduser().resolve()).casefold() if str(value).strip() else ""

        managed_executable_values = {
            managed_path_key(installed.get("executable")),
            managed_path_key((installed.get("runtime") or {}).get("executable")),
        }
        managed_model_values = {
            managed_path_key(installed.get("modelPath")),
            managed_path_key((installed.get("model") or {}).get("modelPath")),
        }
        executable_is_managed = not executable or managed_path_key(executable) in managed_executable_values
        model_is_managed = not model_dir or managed_path_key(model_dir) in managed_model_values
        if executable_is_managed and not installed.get("runtimeReady"):
            raise gr.Error(
                "已安装的 audio.cpp 运行时缺失或校验失败，请在组件区重新安装。"
            )
        if model_is_managed and not installed.get("modelReady"):
            raise gr.Error(
                "已安装的 audio.cpp GGUF 模型缺失或校验失败，请重新下载/校验模型。"
            )
        if executable_is_managed:
            executable = str(installed.get("executable") or executable)
        if model_is_managed:
            model_dir = str(installed.get("modelPath") or model_dir)
        installed_backend = str(installed.get("installedBackend") or "").strip()
        if executable_is_managed and installed_backend:
            backend = installed_backend
        target = output_dir / f"audiocpp_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.wav"
        try:
            report = run_audiocpp(
                executable,
                model_dir,
                speaker_audio,
                target,
                source_text,
                source_language,
                backend=str(backend),
                duration_factor=float(duration),
                memory_saver=bool(memory_saver),
                emotion_text=str(emotion_text or ""),
            )
        except Exception as exc:
            raise gr.Error(f"audio.cpp 实验推理失败：{str(exc).strip() or type(exc).__name__}") from exc
        return str(target), json.dumps(report, ensure_ascii=False, indent=2)

    def profile_pronunciation_entries(raw: str, language: str) -> list[PronunciationEntry]:
        entries: list[PronunciationEntry] = []
        for row in str(raw or "").splitlines():
            value = row.strip()
            if not value or value.startswith("#"):
                continue
            parts = [part.strip() for part in value.split("|")]
            if len(parts) >= 2:
                entries.append(PronunciationEntry(parts[0], parts[1], parts[2] if len(parts) >= 3 else language))
        return entries

    def generate_dialogue_event(
        script_type,
        script,
        default_role,
        default_language,
        edited_timeline_rows,
        timeline_policy,
        fit_slots,
        slot_duration_mode,
        fit_tolerance,
        batch_gap,
        dialogue_segmentation_mode,
        dialogue_max_text_tokens,
        dialogue_pause_preset,
        dialogue_comma_pause_ms,
        dialogue_sentence_pause_ms,
        dialogue_paragraph_pause_ms,
        dialogue_postprocess_preset,
        dialogue_postprocess_strength,
        dialogue_seed,
        dialogue_diffusion_steps,
        dialogue_inference_cfg_rate,
        dialogue_cfm_temperature,
        dialogue_asr_enabled,
        dialogue_asr_backend,
        dialogue_asr_model,
        dialogue_asr_device,
        dialogue_asr_threshold,
        dialogue_asr_retry_count,
        subtitle_timing_mode,
        subtitle_text_mode,
        subtitle_include_role,
        resume_task_id,
        force_line_number,
        progress=gr.Progress(),
    ):
        resume_task_id = str(resume_task_id or "").strip()
        force_line_number = int(force_line_number or 0)
        requested_timeline_rows = edited_timeline_rows
        task = None
        if resume_task_id:
            try:
                task = load_task(output_dir, resume_task_id)
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
            saved = task.get("settings") or {}
            script_type = task.get("script_type", script_type)
            script = task.get("script", script)
            default_role = saved.get("default_role", default_role)
            default_language = saved.get("default_language", default_language)
            if force_line_number and requested_timeline_rows is not None:
                try:
                    has_requested_rows = len(requested_timeline_rows) > 0
                except TypeError:
                    has_requested_rows = True
                edited_timeline_rows = (
                    requested_timeline_rows
                    if has_requested_rows
                    else saved.get("timeline_rows", requested_timeline_rows)
                )
            else:
                edited_timeline_rows = saved.get("timeline_rows", requested_timeline_rows)
            timeline_policy = saved.get("timeline_policy", timeline_policy)
            fit_slots = saved.get("fit_slots", fit_slots)
            slot_duration_mode = saved.get("slot_duration_mode", slot_duration_mode)
            fit_tolerance = saved.get("fit_tolerance", fit_tolerance)
            batch_gap = saved.get("batch_gap", batch_gap)
            dialogue_segmentation_mode = saved.get("segmentation_mode", dialogue_segmentation_mode)
            dialogue_max_text_tokens = saved.get("max_text_tokens", dialogue_max_text_tokens)
            dialogue_pause_preset = saved.get("pause_preset", dialogue_pause_preset)
            dialogue_comma_pause_ms = saved.get("comma_pause_ms", dialogue_comma_pause_ms)
            dialogue_sentence_pause_ms = saved.get("sentence_pause_ms", dialogue_sentence_pause_ms)
            dialogue_paragraph_pause_ms = saved.get("paragraph_pause_ms", dialogue_paragraph_pause_ms)
            dialogue_postprocess_preset = saved.get("postprocess_preset", dialogue_postprocess_preset)
            dialogue_postprocess_strength = saved.get("postprocess_strength", dialogue_postprocess_strength)
            dialogue_seed = saved.get("seed", dialogue_seed)
            dialogue_diffusion_steps = saved.get("diffusion_steps", dialogue_diffusion_steps)
            dialogue_inference_cfg_rate = saved.get("inference_cfg_rate", dialogue_inference_cfg_rate)
            dialogue_cfm_temperature = saved.get("cfm_temperature", dialogue_cfm_temperature)
            dialogue_asr_enabled = saved.get("asr_enabled", dialogue_asr_enabled)
            dialogue_asr_backend = saved.get("asr_backend", dialogue_asr_backend)
            dialogue_asr_model = saved.get("asr_model", dialogue_asr_model)
            dialogue_asr_device = saved.get("asr_device", dialogue_asr_device)
            dialogue_asr_threshold = saved.get("asr_threshold", dialogue_asr_threshold)
            dialogue_asr_retry_count = saved.get("asr_retry_count", dialogue_asr_retry_count)
            subtitle_timing_mode = saved.get("subtitle_timing_mode", subtitle_timing_mode)
            subtitle_text_mode = saved.get("subtitle_text_mode", subtitle_text_mode)
            subtitle_include_role = saved.get("subtitle_include_role", subtitle_include_role)
        try:
            lines = parse_dialogue(script_type, script, default_role, default_language)
            lines = apply_timeline_edits(lines, edited_timeline_rows)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        dialogue_asr_retry_count = max(0, int(dialogue_asr_retry_count or 0))
        dialogue_review_enabled = bool(dialogue_asr_enabled or dialogue_asr_retry_count > 0)
        if dialogue_review_enabled and not asr_available(str(dialogue_asr_backend)):
            raise gr.Error("所选 ASR 后端不可用；请安装 openai-whisper / faster-whisper，或切换后端。")
        if force_line_number and not resume_task_id:
            raise gr.Error("单句重试前请先选择一个已保存任务。")
        if force_line_number and not 1 <= force_line_number <= len(lines):
            raise gr.Error(f"台词序号必须在 1 到 {len(lines)} 之间。")
        if force_line_number:
            saved_rows = (task.get("settings") or {}).get("timeline_rows")
            saved_lines_for_comparison = apply_timeline_edits(
                parse_dialogue(script_type, script, default_role, default_language),
                saved_rows,
            )
            changed_other_lines = [
                current.index
                for current, previous in zip(lines, saved_lines_for_comparison)
                if current.index != force_line_number
                and (
                    current.role != previous.role
                    or current.language != previous.language
                    or current.text != previous.text
                    or current.duration_factor != previous.duration_factor
                )
            ]
            if changed_other_lines:
                numbers = "、".join(str(item) for item in changed_other_lines)
                raise gr.Error(
                    f"第 {numbers} 条也修改了会影响声音的内容；请分别重做这些台词，或点“生成全部台词”。"
                )
        profiles = {item.name: item for item in voice_library.list()}
        missing = missing_roles(lines, profiles)
        if missing:
            raise gr.Error("请先在“角色音色库”中保存这些角色：" + "、".join(missing))

        run_id = resume_task_id or f"dialogue_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        session_dir = output_dir / run_id
        session_dir.mkdir(parents=True, exist_ok=bool(resume_task_id))
        if force_line_number:
            incomplete = []
            saved_lines = task.get("lines") or {}
            for saved_script_line in lines:
                if saved_script_line.index == force_line_number:
                    continue
                saved_line = saved_lines.get(str(saved_script_line.index)) or {}
                saved_profile = profiles[saved_script_line.role]
                saved_file = session_dir / (
                    f"{saved_script_line.index:04d}_"
                    f"{safe_voice_file_stem(saved_profile.name, saved_profile.profile_id)}.wav"
                )
                if saved_line.get("status") != "completed" or not saved_file.is_file():
                    incomplete.append(saved_script_line.index)
            if incomplete:
                detail = "、".join(str(index) for index in incomplete[:8])
                suffix = "……" if len(incomplete) > 8 else ""
                raise gr.Error(
                    f"任务还有未完成台词（{detail}{suffix}），请先点“继续未完成任务”，完成后再单句重试。"
                )
        clips: list[torch.Tensor] = []
        clip_paths: list[Path] = []
        report_lines: list[dict] = []
        sample_rate: int | None = None
        sampling_values = {
            "do_sample": True,
            "top_p": 0.8,
            "top_k": 30,
            "num_beams": 3,
            "repetition_penalty": 10.0,
            "length_penalty": 0.0,
        }
        dialogue_seed = int(dialogue_seed)
        dialogue_diffusion_steps = int(dialogue_diffusion_steps)
        dialogue_inference_cfg_rate = float(dialogue_inference_cfg_rate)
        dialogue_cfm_temperature = float(dialogue_cfm_temperature)
        dialogue_asr_threshold = float(dialogue_asr_threshold)
        task_settings = {
            "default_role": default_role,
            "default_language": default_language,
            "timeline_policy": timeline_policy,
            "fit_slots": bool(fit_slots),
            "slot_duration_mode": slot_duration_mode,
            "fit_tolerance": int(fit_tolerance),
            "batch_gap": int(batch_gap),
            "segmentation_mode": dialogue_segmentation_mode,
            "max_text_tokens": int(dialogue_max_text_tokens),
            "pause_preset": dialogue_pause_preset,
            "comma_pause_ms": int(dialogue_comma_pause_ms),
            "sentence_pause_ms": int(dialogue_sentence_pause_ms),
            "paragraph_pause_ms": int(dialogue_paragraph_pause_ms),
            "postprocess_preset": dialogue_postprocess_preset,
            "postprocess_strength": float(dialogue_postprocess_strength),
            "seed": dialogue_seed,
            "diffusion_steps": dialogue_diffusion_steps,
            "inference_cfg_rate": dialogue_inference_cfg_rate,
            "cfm_temperature": dialogue_cfm_temperature,
            "timeline_rows": timeline_rows(lines),
            "asr_enabled": bool(dialogue_asr_enabled),
            "asr_backend": str(dialogue_asr_backend),
            "asr_model": str(dialogue_asr_model),
            "asr_device": str(dialogue_asr_device),
            "asr_threshold": dialogue_asr_threshold,
            "asr_retry_count": dialogue_asr_retry_count,
            "subtitle_timing_mode": str(subtitle_timing_mode),
            "subtitle_text_mode": str(subtitle_text_mode),
            "subtitle_include_role": bool(subtitle_include_role),
        }
        if task is None:
            task = create_task(
                output_dir,
                run_id,
                script_type=script_type,
                script=script,
                settings=task_settings,
                line_count=len(lines),
            )
        else:
            task["settings"] = task_settings
            task = set_task_status(output_dir, task, "running")
        performance_measurement = start_runtime_measurement()

        def review_line_audio(path: Path, line, language_value: str) -> dict | None:
            if not dialogue_review_enabled:
                return None
            try:
                transcript = transcribe_audio_file(
                    path,
                    language=language_value,
                    backend=str(dialogue_asr_backend),
                    model_name=str(dialogue_asr_model),
                    device=str(dialogue_asr_device),
                    download_root=data_dir / "asr_models",
                )
                review = review_transcript(
                    line.text,
                    transcript["text"],
                    language_value,
                    dialogue_asr_threshold,
                )
                return {**transcript, **review}
            except Exception as exc:
                return {
                    "expected_text": line.text,
                    "recognized_text": "",
                    "passed": False,
                    "similarity": 0.0,
                    "threshold": dialogue_asr_threshold,
                    "language": language_value,
                    "backend": str(dialogue_asr_backend),
                    "model": str(dialogue_asr_model),
                    "error": str(exc).strip() or type(exc).__name__,
                }

        for offset, line in enumerate(lines):
            progress((offset / max(len(lines), 1)), desc=f"生成第 {offset + 1}/{len(lines)} 条：{line.role}")
            profile = profiles[line.role]
            target = session_dir / (
                f"{line.index:04d}_{safe_voice_file_stem(profile.name, profile.profile_id)}.wav"
            )
            saved_line = (task.get("lines") or {}).get(str(line.index)) or {}
            should_reuse = bool(
                saved_line.get("status") == "completed"
                and target.is_file()
                and force_line_number != line.index
            )
            if should_reuse:
                cached_waveform, cached_rate = torchaudio.load(str(target))
                cached_clip = cached_waveform.unsqueeze(0)
                if sample_rate is None:
                    sample_rate = int(cached_rate)
                elif sample_rate != int(cached_rate):
                    raise gr.Error("已恢复的逐句音频采样率不一致，无法合并。")
                clips.append(cached_clip)
                clip_paths.append(target)
                cached_report = dict(saved_line.get("report") or {})
                cached_report.setdefault("file", str(target))
                cached_report["restored_from_task"] = True
                if dialogue_review_enabled and not cached_report.get("asr"):
                    progress(
                        (offset / max(len(lines), 1)),
                        desc=f"ASR 校对第 {offset + 1}/{len(lines)} 条",
                    )
                    cached_report["asr"] = review_line_audio(
                        target,
                        line,
                        line.language or profiles[line.role].language,
                    )
                    task = update_task_line(
                        output_dir,
                        task,
                        line.index,
                        status="completed",
                        file=str(target),
                        report=cached_report,
                    )
                report_lines.append(cached_report)
                continue
            ensure_model()
            task = update_task_line(
                output_dir,
                task,
                line.index,
                status="running",
                file=str(target),
            )
            language_value = line.language or profile.language
            pronunciation_result = process_pronunciation_text(
                line.text,
                language_value,
                profile_pronunciation_entries(profile.pronunciation_dictionary, language_value),
                strict=True,
                pinyin_vocab_path=exact_vocab_path,
            )
            try:
                resolved_emotion, emotion_source = line_emotion_kwargs(
                    tts, profile, line, qwen_emotion_available
                )
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
            line_plan = build_desktop_plan(
                tts,
                pronunciation_result.text,
                language_value,
                str(dialogue_segmentation_mode),
                int(dialogue_max_text_tokens),
                str(dialogue_pause_preset),
                int(dialogue_comma_pause_ms),
                int(dialogue_sentence_pause_ms),
                int(dialogue_paragraph_pause_ms),
            )
            line_long_text_guard_reports: list[dict] = []

            def infer_line(
                factor: float,
                native_target_seconds: float | None = None,
                seed_offset: int = 0,
            ):
                def infer_once():
                    tts.gr_progress = progress
                    waveforms = []
                    rate_value = None
                    native_chunk_durations = (
                        allocate_native_chunk_durations(line_plan, native_target_seconds)
                        if native_target_seconds is not None
                        else (None,) * len(line_plan.chunks)
                    )
                    with safe_gpt_acceleration(sampling_values, line_plan) as (disabled, guarded):
                        for chunk_index, chunk in enumerate(line_plan.chunks):
                            def generate_with_limit(limit: int):
                                return tts.infer(
                                    spk_audio_prompt=profile.audio_path,
                                    text=chunk.text,
                                    output_path=None,
                                    lang=language_value,
                                    **resolved_emotion,
                                    verbose=verbose,
                                    duration_factor=float(factor),
                                    do_sample=True,
                                    temperature=0.8,
                                    top_p=0.8,
                                    top_k=30,
                                    num_beams=3,
                                    repetition_penalty=10.0,
                                    length_penalty=0.0,
                                    max_mel_tokens=1500,
                                    max_text_tokens_per_segment=int(limit),
                                    interval_silence=200,
                                    text_normalization=True,
                                    target_duration=native_chunk_durations[chunk_index],
                                    seed=(
                                        dialogue_seed
                                        + offset * 1000
                                        + int(seed_offset)
                                        + chunk_index
                                    ),
                                    diffusion_steps=dialogue_diffusion_steps,
                                    inference_cfg_rate=dialogue_inference_cfg_rate,
                                    cfm_temperature=dialogue_cfm_temperature,
                                )

                            def result_duration_seconds(value):
                                if not isinstance(value, tuple) or len(value) != 2:
                                    return 0.0
                                result_rate, result_raw = value
                                result_tensor = torch.as_tensor(result_raw)
                                if result_tensor.ndim == 1:
                                    samples = result_tensor.shape[0]
                                elif result_tensor.ndim == 2 and result_tensor.shape[-1] == 1:
                                    samples = result_tensor.shape[0]
                                else:
                                    samples = result_tensor.shape[-1]
                                return samples / max(1, int(result_rate))

                            block_token_count = len(
                                tts.tokenizer.encode(
                                    f"<|{str(language_value).lower()}|> {chunk.text}",
                                    allowed_special="all",
                                )
                            )
                            result, guard_report = run_with_long_text_guard(
                                generate_with_limit,
                                result_duration_seconds,
                                text=chunk.text,
                                language=language_value,
                                token_count=block_token_count,
                                max_tokens=line_plan.max_tokens,
                                duration_factor=factor,
                                check_duration=native_chunk_durations[chunk_index] is None,
                            )
                            guard_report.update(
                                speech_block=chunk_index + 1,
                                seed_offset=int(seed_offset),
                            )
                            line_long_text_guard_reports.append(guard_report)
                            if not isinstance(result, tuple) or len(result) != 2:
                                raise RuntimeError("IndexTTS 返回了无法识别的逐句音频。")
                            block_rate, raw = result
                            tensor = torch.as_tensor(raw).detach().cpu()
                            if tensor.ndim == 1:
                                tensor = tensor.unsqueeze(0)
                            elif tensor.ndim == 2 and tensor.shape[-1] == 1:
                                tensor = tensor.transpose(0, 1)
                            elif tensor.ndim != 2:
                                tensor = tensor.reshape(1, -1)
                            tensor = tensor.float()
                            if tensor.numel() and float(tensor.abs().max()) > 2.0:
                                tensor = tensor / 32768.0
                            if rate_value is None:
                                rate_value = int(block_rate)
                            elif rate_value != int(block_rate):
                                raise RuntimeError("逐句停顿块采样率不一致。")
                            waveforms.append(tensor.clamp(-1, 1))
                    assert rate_value is not None
                    combined_line = concatenate_with_pauses(
                        waveforms,
                        rate_value,
                        [chunk.pause_after_ms for chunk in line_plan.chunks],
                        line_plan.chunks[0].pause_before_ms,
                    )
                    torchaudio.save(str(target), combined_line, rate_value)
                    return combined_line.unsqueeze(0), int(rate_value), disabled, guarded

                (waveform_value, rate_value, disabled, guarded), runtime_note = (
                    execute_with_runtime_fallback(infer_once)
                )
                return waveform_value, rate_value, disabled, guarded, runtime_note

            def recorded_infer_line(
                factor: float,
                native_target_seconds: float | None = None,
                seed_offset: int = 0,
            ):
                nonlocal task
                try:
                    return infer_line(factor, native_target_seconds, seed_offset)
                except Exception as exc:
                    detail = str(exc).strip() or type(exc).__name__
                    task = update_task_line(
                        output_dir,
                        task,
                        line.index,
                        status="failed",
                        file=str(target),
                        error=detail,
                    )
                    task = set_task_status(output_dir, task, "failed", error=detail)
                    raise

            used_factor = line.duration_factor
            native_slot = bool(
                fit_slots and line.slot_ms and slot_duration_mode == "native"
            )
            waveform, rate, accel_disabled, cache_guarded, runtime_note = recorded_infer_line(
                used_factor,
                line.slot_ms / 1000.0 if native_slot else None,
            )
            actual_ms = waveform.shape[-1] * 1000 / rate
            regenerated = False
            duration_adjustment = {"mode": "off", "action": "unchanged"}
            if native_slot:
                waveform, duration_adjustment = apply_duration_policy(
                    waveform, rate, line.slot_ms / 1000.0, "exact"
                )
                duration_adjustment["mode"] = "native"
                actual_ms = waveform.shape[-1] * 1000 / rate
                torchaudio.save(str(target), waveform[0], rate)
            elif bool(fit_slots) and line.slot_ms and abs(actual_ms - line.slot_ms) > int(fit_tolerance):
                fitted = fit_duration_factor(used_factor, actual_ms, line.slot_ms)
                if abs(fitted - used_factor) >= 0.02:
                    used_factor = fitted
                    waveform, rate, accel_disabled, cache_guarded, runtime_note = recorded_infer_line(used_factor)
                    actual_ms = waveform.shape[-1] * 1000 / rate
                    regenerated = True
            if bool(fit_slots) and line.slot_ms and slot_duration_mode in {"pad", "exact"}:
                waveform, duration_adjustment = apply_duration_policy(
                    waveform, rate, line.slot_ms / 1000.0, slot_duration_mode
                )
                actual_ms = waveform.shape[-1] * 1000 / rate
                torchaudio.save(str(target), waveform[0], rate)
            if sample_rate is None:
                sample_rate = rate
            elif sample_rate != rate:
                raise gr.Error("逐句音频采样率不一致，无法合并。")
            clips.append(waveform)
            clip_paths.append(target)
            line_report = {
                **line.to_dict(),
                "resolved_text": pronunciation_result.text,
                "emotion_mode": (
                    profile.emotion_mode
                    if line.emotion_mode == "inherit"
                    else line.emotion_mode
                ),
                "emotion_source": emotion_source,
                "line_emotion_mode": line.emotion_mode,
                "line_emotion_text": line.emotion_text,
                "line_emotion_vector": line.emotion_vector,
                "line_emotion_strength": line.emotion_strength,
                "line_emotion_use_random": line.emotion_use_random,
                "actual_duration_ms": round(actual_ms),
                "used_duration_factor": round(used_factor, 4),
                "regenerated_for_slot": regenerated,
                "native_duration": native_slot,
                "gpt_accel_fallback": bool(accel_disabled),
                "gpt_accel_cache_guard": bool(cache_guarded),
                "text_plan": line_plan.to_dict(),
                "long_text_guard": [
                    item
                    for item in line_long_text_guard_reports
                    if item.get("enabled")
                ],
                "duration_adjustment": duration_adjustment,
                "runtime_acceleration_fallback": runtime_note,
                "file": str(target),
            }
            if dialogue_review_enabled:
                progress(
                    ((offset + 0.8) / max(len(lines), 1)),
                    desc=f"ASR 校对第 {offset + 1}/{len(lines)} 条",
                )
                selected_waveform = waveform
                selected_rate = rate
                selected_review = review_line_audio(target, line, language_value) or {}
                asr_attempts = [
                    {
                        "attempt": 1,
                        "seed": dialogue_seed + offset * 1000,
                        "passed": bool(selected_review.get("passed")),
                        "similarity": float(selected_review.get("similarity", 0.0)),
                        "recognized_text": selected_review.get(
                            "recognized_text", selected_review.get("text", "")
                        ),
                    }
                ]
                selected_seed = dialogue_seed + offset * 1000
                for retry_index in range(1, dialogue_asr_retry_count + 1):
                    if selected_review.get("passed") or selected_review.get("error"):
                        break
                    candidate, candidate_rate, disabled_again, guarded_again, retry_note = (
                        recorded_infer_line(
                            used_factor,
                            line.slot_ms / 1000.0 if native_slot else None,
                            retry_index * 100_003,
                        )
                    )
                    accel_disabled = accel_disabled or disabled_again
                    cache_guarded = cache_guarded or guarded_again
                    runtime_note = "；".join(
                        item for item in (runtime_note, retry_note) if item
                    )
                    if native_slot:
                        candidate, _candidate_adjustment = apply_duration_policy(
                            candidate,
                            candidate_rate,
                            line.slot_ms / 1000.0,
                            "exact",
                        )
                    elif bool(fit_slots) and line.slot_ms and slot_duration_mode in {
                        "pad",
                        "exact",
                    }:
                        candidate, _candidate_adjustment = apply_duration_policy(
                            candidate,
                            candidate_rate,
                            line.slot_ms / 1000.0,
                            slot_duration_mode,
                        )
                    torchaudio.save(str(target), candidate[0], candidate_rate)
                    candidate_review = review_line_audio(target, line, language_value) or {}
                    candidate_seed = (
                        dialogue_seed + offset * 1000 + retry_index * 100_003
                    )
                    asr_attempts.append(
                        {
                            "attempt": retry_index + 1,
                            "seed": candidate_seed,
                            "passed": bool(candidate_review.get("passed")),
                            "similarity": float(candidate_review.get("similarity", 0.0)),
                            "recognized_text": candidate_review.get(
                                "recognized_text", candidate_review.get("text", "")
                            ),
                        }
                    )
                    if float(candidate_review.get("similarity", 0.0)) > float(
                        selected_review.get("similarity", 0.0)
                    ):
                        selected_waveform = candidate
                        selected_rate = candidate_rate
                        selected_review = candidate_review
                        selected_seed = candidate_seed
                    if candidate_review.get("passed"):
                        selected_waveform = candidate
                        selected_rate = candidate_rate
                        selected_review = candidate_review
                        selected_seed = candidate_seed
                        break
                waveform, rate = selected_waveform, selected_rate
                actual_ms = waveform.shape[-1] * 1000 / rate
                torchaudio.save(str(target), waveform[0], rate)
                line_report["actual_duration_ms"] = round(actual_ms)
                line_report["asr"] = {
                    **selected_review,
                    "selected_seed": selected_seed,
                    "attempt_count": len(asr_attempts),
                    "retry_count": max(0, len(asr_attempts) - 1),
                    "attempts": asr_attempts,
                }
                if sample_rate != rate:
                    raise gr.Error("ASR 重试输出采样率不一致，无法合并。")
                clips[-1] = waveform
            line_report["model_lifecycle"] = apply_memory_policy()
            report_lines.append(line_report)
            task = update_task_line(
                output_dir,
                task,
                line.index,
                status="completed",
                file=str(target),
                report=line_report,
            )

        assert sample_rate is not None
        mixed, placements = compose_timeline(
            clips,
            lines,
            sample_rate,
            timeline_policy,
            int(batch_gap) if script_type == "batch" else 0,
        )
        for line, item, placement, clip_path in zip(lines, report_lines, placements, clip_paths):
            effective_emotion_mode = item.get("emotion_mode", "speaker")
            item.update(line.to_dict())
            item["line_emotion_mode"] = line.emotion_mode
            item["emotion_mode"] = effective_emotion_mode
            item["timeline"] = placement.to_dict()
            task = update_task_line(
                output_dir,
                task,
                line.index,
                status="completed",
                file=str(clip_path),
                report=item,
            )
        processed_mixed, postprocess_report = postprocess_waveform(
            mixed[0],
            sample_rate,
            str(dialogue_postprocess_preset),
            float(dialogue_postprocess_strength),
        )
        mixed = processed_mixed.unsqueeze(0)
        combined = output_dir / f"{run_id}.wav"
        torchaudio.save(str(combined), mixed[0], sample_rate, encoding="PCM_S", bits_per_sample=16)
        duration_seconds = mixed.shape[-1] / sample_rate
        performance = finish_runtime_measurement(
            performance_measurement,
            duration_seconds,
        )
        elapsed = float(performance["elapsed_seconds"])
        rewritten_srt_content, subtitle_rewrite_report = rewrite_srt(
            lines,
            report_lines,
            timing_mode=str(subtitle_timing_mode),
            text_mode=str(subtitle_text_mode),
            include_role=bool(subtitle_include_role),
        )
        rewritten_srt_path = session_dir / "rewritten.srt"
        rewritten_srt_path.write_text(rewritten_srt_content, encoding="utf-8-sig")
        asr_results = [item.get("asr") for item in report_lines if item.get("asr")]
        report = {
            "version": DESKTOP_VERSION,
            "script_type": script_type,
            "timeline_policy": timeline_policy,
            "fit_srt_slots": bool(fit_slots),
            "slot_duration_mode": slot_duration_mode,
            "postprocess": postprocess_report,
            "cfm": {
                "seed": dialogue_seed,
                "diffusion_steps": dialogue_diffusion_steps,
                "inference_cfg_rate": dialogue_inference_cfg_rate,
                "temperature": dialogue_cfm_temperature,
            },
            "asr": {
                "enabled": dialogue_review_enabled,
                "backend": str(dialogue_asr_backend),
                "model": str(dialogue_asr_model),
                "device": str(dialogue_asr_device),
                "threshold": dialogue_asr_threshold,
                "maximum_retries": dialogue_asr_retry_count,
                "reviewed": len(asr_results),
                "passed": sum(bool(item.get("passed")) for item in asr_results),
                "failed": sum(not bool(item.get("passed")) for item in asr_results),
            },
            "subtitle_rewrite": subtitle_rewrite_report,
            "sample_rate": sample_rate,
            "duration_ms": round(duration_seconds * 1000),
            "elapsed_seconds": round(elapsed, 3),
            "rtf": performance["rtf"],
            "performance": performance,
            "lines": report_lines,
        }
        report_path = session_dir / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        archive = output_dir / f"{run_id}.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
            output_zip.write(combined, "combined.wav")
            output_zip.write(report_path, "report.json")
            output_zip.write(rewritten_srt_path, "rewritten.srt")
            for clip_path in clip_paths:
                output_zip.write(clip_path, f"lines/{clip_path.name}")
        append_history(output_dir, {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "language": "MULTI",
            "duration_factor": "per-line",
            "emotion_mode": f"{len(lines)} 条 / {len(set(line.role for line in lines))} 角色",
            "text": f"{script_type.upper()} 多角色配音",
            "resolved_text": format_runtime_metrics(performance),
            "file": str(combined),
        })
        task["combined_file"] = str(combined)
        task["archive_file"] = str(archive)
        task["report_file"] = str(report_path)
        task["rewritten_srt_file"] = str(rewritten_srt_path)
        task = set_task_status(output_dir, task, "completed")
        progress(1, desc="多角色配音完成")
        return (
            str(combined),
            str(archive),
            str(rewritten_srt_path),
            json.dumps(report, ensure_ascii=False, indent=2),
            load_history(output_dir),
            gr.update(choices=task_choices(output_dir), value=run_id),
            f"任务 {run_id} 已完成；{format_runtime_metrics(performance)}；"
            "已生成 ASR 校对报告、回写字幕和可编辑时间轴。",
            render_timeline_html(lines, report_lines),
            str(combined),
        )

    def rebuild_dialogue_timeline_event(
        task_id,
        edited_timeline_rows,
        timeline_policy,
        batch_gap,
        subtitle_timing_mode,
        subtitle_text_mode,
        subtitle_include_role,
    ):
        task_id = str(task_id or "").strip()
        if not task_id:
            raise gr.Error("请先选择一个已完成任务。")
        try:
            task = load_task(output_dir, task_id)
            saved = task.get("settings") or {}
            parser = parse_srt if task.get("script_type") == "srt" else parse_batch_script
            parsed = parser(
                task.get("script", ""),
                saved.get("default_role", "旁白"),
                saved.get("default_language", "ZH"),
            )
            lines = apply_timeline_edits(parsed, edited_timeline_rows or saved.get("timeline_rows"))
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

        clips, clip_paths, report_lines = [], [], []
        sample_rate = None
        for line in lines:
            saved_line = (task.get("lines") or {}).get(str(line.index)) or {}
            clip_path = Path(str(saved_line.get("file") or ""))
            if saved_line.get("status") != "completed" or not clip_path.is_file():
                raise gr.Error(f"任务第 {line.index} 条音频缺失，请先继续未完成任务。")
            waveform, rate = torchaudio.load(str(clip_path))
            if sample_rate is None:
                sample_rate = int(rate)
            elif sample_rate != int(rate):
                raise gr.Error("逐句音频采样率不一致，无法重新混音。")
            clips.append(waveform.unsqueeze(0))
            clip_paths.append(clip_path)
            report_lines.append(dict(saved_line.get("report") or {**line.to_dict()}))
        assert sample_rate is not None
        mixed, placements = compose_timeline(
            clips,
            lines,
            sample_rate,
            str(timeline_policy),
            int(batch_gap) if task.get("script_type") == "batch" else 0,
        )
        for line, report_line, placement, clip_path in zip(lines, report_lines, placements, clip_paths):
            report_line.update(line.to_dict())
            report_line["timeline"] = placement.to_dict()
            report_line["timeline_edited"] = True
            task = update_task_line(
                output_dir,
                task,
                line.index,
                status="completed",
                file=str(clip_path),
                report=report_line,
            )
        processed, postprocess_report = postprocess_waveform(
            mixed[0],
            sample_rate,
            str(saved.get("postprocess_preset", "off")),
            float(saved.get("postprocess_strength", 1.0)),
        )
        session_dir = output_dir / task_id
        combined = output_dir / f"{task_id}_edited.wav"
        torchaudio.save(str(combined), processed, sample_rate, encoding="PCM_S", bits_per_sample=16)
        srt_content, subtitle_report = rewrite_srt(
            lines,
            report_lines,
            timing_mode=str(subtitle_timing_mode),
            text_mode=str(subtitle_text_mode),
            include_role=bool(subtitle_include_role),
        )
        srt_path = session_dir / "rewritten_edited.srt"
        srt_path.write_text(srt_content, encoding="utf-8-sig")
        report = {
            "version": DESKTOP_VERSION,
            "task_id": task_id,
            "timeline_edited": True,
            "timeline_policy": str(timeline_policy),
            "postprocess": postprocess_report,
            "sample_rate": sample_rate,
            "duration_ms": round(processed.shape[-1] * 1000 / sample_rate),
            "subtitle_rewrite": subtitle_report,
            "lines": report_lines,
        }
        report_path = session_dir / "report_edited.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        archive = output_dir / f"{task_id}_edited.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
            output_zip.write(combined, "combined_edited.wav")
            output_zip.write(srt_path, "rewritten_edited.srt")
            output_zip.write(report_path, "report_edited.json")
            for clip_path in clip_paths:
                output_zip.write(clip_path, f"lines/{clip_path.name}")
        saved.update(
            {
                "timeline_rows": timeline_rows(lines),
                "timeline_policy": str(timeline_policy),
                "batch_gap": int(batch_gap),
                "subtitle_timing_mode": str(subtitle_timing_mode),
                "subtitle_text_mode": str(subtitle_text_mode),
                "subtitle_include_role": bool(subtitle_include_role),
            }
        )
        task["settings"] = saved
        task["combined_file"] = str(combined)
        task["archive_file"] = str(archive)
        task["report_file"] = str(report_path)
        task["rewritten_srt_file"] = str(srt_path)
        task = set_task_status(output_dir, task, "completed")
        return (
            str(combined),
            str(archive),
            str(srt_path),
            json.dumps(report, ensure_ascii=False, indent=2),
            load_history(output_dir),
            gr.update(choices=task_choices(output_dir), value=task_id),
            f"任务 {task_id} 已按编辑后的时间轴重新混音；未重新执行 TTS。",
            render_timeline_html(lines, report_lines),
            str(combined),
        )

    theme = gr.themes.Soft(
        primary_hue="pink",
        secondary_hue="blue",
        neutral_hue="slate",
    )
    with gr.Blocks(title=APP_TITLE, theme=theme, css=CSS, js=TIMELINE_EDITOR_JS) as demo:
        gr.HTML(
            f"""
            <section class="t8-header">
              <div class="t8-header-title">
                <span class="t8-eyebrow">INDEXTTS 2.5 · DESKTOP {DESKTOP_VERSION}</span>
                <h1>T8star-Aix · IndexTTS 2.5</h1>
              </div>
              <div class="t8-header-copy">
                <p>多语言、情感可控的本地零样本语音合成</p>
                <p class="t8-credit">整合制作：B站 @T8star-Aix</p>
              </div>
            </section>
            """
        )

        with gr.Tab("语音生成"):
            with gr.Row(elem_classes=["t8-primary-grid"]):
                with gr.Column(scale=1):
                    gr.Markdown(
                        "**已有角色不用重复上传：** 从音色库选择后会自动填入下方参考音频；"
                        "需要临时音色时仍可上传、拖入或录制。"
                    )
                    with gr.Row():
                        single_voice_select = gr.Dropdown(
                            choices=voice_choices(),
                            value=None,
                            label="使用已保存音色库（免重复上传）",
                            info="选中即载入该角色保存的音色参考；不会覆盖本页情感和生成参数。",
                            scale=3,
                        )
                        refresh_single_voice_button = gr.Button(
                            "刷新音色库",
                            min_width=120,
                            scale=0,
                        )
                    single_voice_status = gr.Markdown(
                        "尚未选择已保存音色；也可以直接使用下方上传/录音。"
                    )
                    prompt_audio = gr.Audio(
                        label="音色参考音频（音色库自动载入，或手动上传/录音）",
                        sources=["upload", "microphone"],
                        type="filepath",
                        elem_classes=["t8-prompt-audio"],
                    )
                with gr.Column(scale=2):
                    text = gr.TextArea(
                        label="目标文本",
                        placeholder="请输入需要合成的文本",
                        lines=7,
                    )
                    with gr.Row(elem_classes=["t8-meta-row"]):
                        language = gr.Dropdown(
                            choices=LANGUAGE_CHOICES,
                            value="ZH",
                            label="目标语言",
                            scale=1,
                            min_width=170,
                        )
                        duration_factor = gr.Slider(
                            0.5,
                            2.0,
                            value=1.0,
                            step=0.01,
                            label="官方时长适配倍率（无单位；小于 1 更短，大于 1 更长）",
                            info="控制目标声学长度，不等同于自然语气语速；建议 0.8–1.25，极端值可能拉长或失真",
                            scale=3,
                            min_width=320,
                        )

            with gr.Accordion("参考音频质量检测与自动裁剪", open=True):
                gr.Markdown(
                    "建议使用 **3–10 秒、单人、无背景音乐、无混响、无削波** 的清晰人声。"
                    "检测会显示静音、响度、削波、估算信噪比和波形；自动裁剪只生成副本，不覆盖原文件。"
                )
                with gr.Row():
                    reference_auto_prepare = gr.Checkbox(
                        value=True,
                        label="自动裁掉首尾静音并选取高信息片段",
                    )
                    reference_maximum_seconds = gr.Slider(
                        3,
                        30,
                        value=15,
                        step=0.5,
                        label="最长保留时长（秒）",
                    )
                    reference_padding_ms = gr.Slider(
                        0,
                        1000,
                        value=150,
                        step=10,
                        label="首尾保留（毫秒）",
                    )
                    inspect_reference_button = gr.Button("检测并应用", variant="primary")
                reference_quality_report = gr.TextArea(
                    label="参考音频质量报告 JSON",
                    lines=10,
                    interactive=False,
                )
                reference_waveform = gr.HTML(
                    '<div class="t8-timeline-empty">检测后显示参考音频波形。</div>'
                )

            with gr.Accordion("完整参数预设（含参考音频）", open=False, elem_classes=["t8-section"]):
                with gr.Row():
                    preset_name = gr.Textbox(
                        label="预设名称", placeholder="例如：温柔旁白", scale=2
                    )
                    preset_select = gr.Dropdown(
                        choices=[""] + list_presets(data_dir),
                        value="",
                        label="已保存预设",
                        scale=2,
                    )
                    save_preset_button = gr.Button("保存/覆盖", variant="primary")
                    load_preset_button = gr.Button("载入")
                    delete_preset_button = gr.Button("删除", variant="stop")
                preset_status = gr.Markdown(
                    "预设会保存当前文本、语言、情感、高级参数以及参考音频，保存在本机用户数据目录。"
                )

            with gr.Row(elem_classes=["t8-pronunciation-tip"]):
                gr.Markdown(
                    "**多音字怎么用？** 直接在目标文本中写 `＜原文字词|带声调数字的拼音＞`。"
                    "例如：`＜要求|YAO4 QIU2＞`。多音字处在词语中时应标注整个词，不要只包单字。"
                    "下面的设置区已默认展开，也可以先点右侧按钮查看完整示例。"
                )
                quick_pronunciation_example_button = gr.Button(
                    "一键填入中文示例", variant="secondary", min_width=190, scale=0
                )

            with gr.Accordion(
                "多音字使用方法与发音设置（默认展开）",
                open=True,
                elem_classes=["t8-section", "t8-pronunciation-accordion"],
            ):
                gr.Markdown(
                    """
#### 方法一：直接标注当前文本（最简单）

1. 在“目标文本”中找到需要指定读音的字或词。
2. 改写成 `＜原文字词|拼音＞`，实际输入时使用英文半角尖括号 `< >` 和竖线 `|`。
3. 每个汉字对应一个带声调数字的拼音，拼音之间用空格分开，声调用 `1–5` 表示。

**可直接复制的例子：** `小明<要求|YAO4 QIU2>这个题的答案是多少。今天的<行程|XING2 CHENG2>顺利，<银行|YIN2 HANG2>的<行长|HANG2 ZHANG3>去了<重庆|CHONG2 QING4>。`

**连续词语要整词标注：** 官方问题 #792 中，`小明<要|YAO4>求…` 可能仍被词义覆盖成一声；请写成 `小明<要求|YAO4 QIU2>…`。每个汉字必须对应一个拼音音节。

不会手写格式时，使用下面的三个输入框，再点击“把标注插入目标文本”。
                    """
                )
                with gr.Row():
                    annotation_term = gr.Textbox(label="需要标注的文字", placeholder="例如：行")
                    annotation_reading = gr.Textbox(label="指定读音", placeholder="例如：XING2")
                    annotation_language = gr.Dropdown(
                        choices=LANGUAGE_CHOICES[:3], value="ZH", label="标注语言"
                    )
                    insert_annotation_button = gr.Button("把标注插入目标文本", variant="primary")
                gr.Markdown(
                    """
#### 方法二：保存为发音词典（同一词反复使用时）

使用下方编辑器添加词条，**语言必须从下拉列表选择**。点击“添加/更新到表格”后还要点击“保存词典”，重启软件后才会继续保留。生成时会自动把匹配词语转换成精确标注；长词优先，手工 `<文字|读音>` 标注优先。

**中英日可直接照抄的格式：**

| 语言 | 文字/词语 | 读音写法 | 说明 |
|---|---|---|---|
| `ZH` | `银行` | `YIN2 HANG2` | 每个汉字对应一个拼音，必须带 `1–5` 声调数字 |
| `EN` | `minute` | `M IH1 . N AH0 T` | 使用 CMU 音素，音节之间可用英文句点分隔 |
| `JA` | `上手` | `じょうず` | 使用平假名或片假名 |

`ES / AR` 暂无专用音素校验，读音会原样传给 IndexTTS 2.5。表格是只读预览，修改时先从“已有词条”选择，再使用带语言下拉框的编辑器。
                    """
                )
                initial_dictionary_rows = load_pronunciation_rows(data_dir)
                with gr.Group(elem_classes=["t8-dictionary-editor"]):
                    dictionary_entry_select = gr.Dropdown(
                        choices=pronunciation_entry_choices(initial_dictionary_rows),
                        value=None,
                        label="已有词条（选择后载入编辑器）",
                        info="不选择时会新增；选择后会更新该词条。",
                    )
                    with gr.Row():
                        dictionary_entry_term = gr.Textbox(
                            label="文字/词语",
                            placeholder="例如：银行 / minute / 上手",
                            scale=3,
                        )
                        dictionary_entry_language = gr.Dropdown(
                            choices=LANGUAGE_CHOICES,
                            value="ZH",
                            label="语言（下拉选择）",
                            allow_custom_value=False,
                            scale=1,
                        )
                        dictionary_entry_reading = gr.Textbox(
                            label="读音",
                            placeholder="例如：YIN2 HANG2",
                            scale=4,
                        )
                    with gr.Row():
                        dictionary_entry_enabled = gr.Checkbox(value=True, label="启用词条")
                        dictionary_entry_case = gr.Checkbox(
                            value=True,
                            label="区分大小写",
                            info="主要用于英文；中文和日文通常保持开启即可。",
                        )
                        upsert_dictionary_entry_button = gr.Button(
                            "添加/更新到表格",
                            variant="primary",
                        )
                        clear_dictionary_editor_button = gr.Button("清空，新增另一条")
                        delete_dictionary_entry_button = gr.Button("删除所选词条", variant="stop")
                dictionary_table = gr.Dataframe(
                    headers=DICTIONARY_HEADERS,
                    value=initial_dictionary_rows,
                    datatype=["str", "str", "str", "bool", "bool"],
                    type="array",
                    col_count=(5, "fixed"),
                    interactive=False,
                    wrap=True,
                    column_widths=["25%", "10%", "35%", "10%", "15%"],
                    show_row_numbers=True,
                    show_search="search",
                    label="持久发音词典预览（请使用上方编辑器增删改）",
                )
                with gr.Row():
                    save_dictionary_button = gr.Button("保存词典", variant="primary")
                    load_examples_button = gr.Button("载入中英日示例词典")
                    import_dictionary_file = gr.File(
                        label="导入 YAML/JSON", file_types=[".yaml", ".yml", ".json"], type="filepath"
                    )
                    import_dictionary_button = gr.Button("导入")
                    export_dictionary_button = gr.Button("导出")
                    exported_dictionary_file = gr.File(label="导出的词典", interactive=False)
                with gr.Row():
                    dictionary_search = gr.Textbox(label="搜索词典", placeholder="输入文字或读音")
                    dictionary_search_button = gr.Button("搜索")
                    pronunciation_strict = gr.Checkbox(
                        value=True,
                        label="严格校验",
                        info="发现无效拼音、CMU 音素或日语假名时阻止生成",
                    )
                dictionary_search_result = gr.Markdown("输入关键词可搜索当前表格。")
                with gr.Row():
                    pronunciation_preview_button = gr.Button("预览最终发音文本")
                    pronunciation_preview = gr.TextArea(
                        label="实际送入模型的文本", lines=5, interactive=False
                    )
                pronunciation_report = gr.Markdown(
                    dictionary_status(parse_rows(load_pronunciation_rows(data_dir)))
                )

            with gr.Accordion("情感控制", open=True, elem_classes=["t8-section", "t8-emotion-section"]):
                if not qwen_emotion_available:
                    gr.Markdown(
                        "当前显卡显存低于 10GB：已自动使用低显存模式并暂不加载 QwenEmotion。"
                        "音色跟随、情感参考音频和八维情感向量仍可正常使用。"
                    )
                emotion_mode = gr.Radio(
                    choices=EMOTION_MODES if qwen_emotion_available else EMOTION_MODES[:3],
                    value=EMOTION_MODES[0],
                    type="index",
                    label="控制方式",
                )
                with gr.Group(visible=False) as emotion_audio_group:
                    emotion_audio = gr.Audio(label="情感参考音频", type="filepath")
                with gr.Group(visible=False) as emotion_vector_group:
                    with gr.Row():
                        vector_controls = [
                            gr.Slider(0, 1, value=0, step=0.05, label=label)
                            for label in EMOTION_LABELS
                        ]
                    random_emotion = gr.Checkbox(label="情感随机采样", value=False)
                with gr.Group(visible=False) as emotion_text_group:
                    emotion_text = gr.Textbox(
                        label="情感描述",
                        placeholder="例如：克制着激动、危险正在逼近；留空则分析目标文本",
                    )
                with gr.Group(visible=False) as emotion_weight_group:
                    emotion_weight = gr.Slider(0, 1, value=0.65, step=0.01, label="情感权重")

            with gr.Accordion("高级生成参数", open=False, elem_classes=["t8-section"]):
                with gr.Row():
                    do_sample = gr.Checkbox(value=True, label="随机采样")
                    temperature = gr.Slider(0.1, 2, value=0.8, step=0.1, label="Temperature")
                    top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P")
                with gr.Row():
                    top_k = gr.Slider(0, 100, value=30, step=1, label="Top K（0 表示关闭）")
                    num_beams = gr.Slider(1, 10, value=3, step=1, label="Beam 数量")
                    repetition_penalty = gr.Slider(0.1, 20, value=10, step=0.1, label="重复惩罚")
                    length_penalty = gr.Slider(-2, 2, value=0, step=0.05, label="长度惩罚")
                with gr.Row():
                    max_mel_tokens = gr.Slider(
                        50,
                        int(tts.cfg.gpt.max_mel_tokens),
                        value=1500,
                        step=10,
                        label="最大语音 Token",
                    )
                    seed = gr.Number(
                        value=0,
                        minimum=0,
                        maximum=4294967295,
                        precision=0,
                        label="随机种子",
                        info="固定后同一组参数可复现；每个外部分段会自动递增",
                    )
                with gr.Row():
                    diffusion_steps = gr.Slider(
                        5, 100, value=25, step=1,
                        label="CFM 扩散步数",
                        info="官方默认 25；更高通常更稳定但更慢，旁白可试 40–50",
                    )
                    inference_cfg_rate = gr.Slider(
                        0, 1.5, value=0.7, step=0.05,
                        label="CFM 引导强度",
                        info="提高后更贴近参考音色/音高，过高可能过度平滑",
                    )
                    cfm_temperature = gr.Slider(
                        0.1, 1.5, value=1.0, step=0.05,
                        label="CFM 温度",
                        info="降低可减少抖动；稳定旁白可试 0.8",
                    )
                with gr.Row():
                    segmentation_mode = gr.Dropdown(
                        choices=[("按语言自动（推荐）", "auto"), ("手动指定", "custom")],
                        value="auto",
                        label="长文本分段模式",
                        info="自动：EN/ES 60、AR 80、JA 100、ZH 120 Token",
                    )
                    max_text_tokens = gr.Slider(
                        20,
                        int(tts.cfg.gpt.max_text_tokens),
                        value=120,
                        step=10,
                        label="每段最大文本 Token",
                        info="长文本会按标点和 Token 预算自动切分后合并",
                    )
                    segment_silence_ms = gr.Slider(
                        0, 3000, value=200, step=10, label="段间静音（毫秒）"
                    )
                    text_normalization = gr.Checkbox(
                        value=True,
                        label="文本归一化",
                        info="处理数字、日期和常见符号；精确发音标注仍会保留",
                    )
                gr.Markdown(
                    "**停顿写法：** 文本中可直接插入 `<pause=0.5>`（秒）或 `<pause=500ms>`；"
                    "显式停顿在所有预设下都有效。标点预设会真实拆分语音块并插入静音。"
                )
                gr.Markdown(
                    "**跨段语速保护（自动）**：长文本至少形成 3 个有效分段后，"
                    "以前两段及后续稳定段的中位语速为基线；只有某段突然降到基线 45% 以下"
                    "才会单独重做该段，而且新结果明显更接近基线时才替换。普通情绪放慢、"
                    "短句和原生目标时长模式不会被强行拉速；详情写入生成报告。"
                )
                with gr.Row():
                    pause_preset = gr.Dropdown(
                        choices=[
                            ("关闭（仅显式停顿）", "off"),
                            ("自然", "natural"),
                            ("有声书/旁白", "narration"),
                            ("对话", "dialogue"),
                            ("自定义", "custom"),
                        ],
                        value="off",
                        label="标点停顿预设",
                    )
                    comma_pause_ms = gr.Slider(0, 5000, value=100, step=10, label="逗号停顿（毫秒）")
                    sentence_pause_ms = gr.Slider(0, 5000, value=300, step=10, label="句末停顿（毫秒）")
                    paragraph_pause_ms = gr.Slider(0, 5000, value=600, step=10, label="段落停顿（毫秒）")
                with gr.Row():
                    target_duration_mode = gr.Dropdown(
                        choices=[
                            ("关闭", "off"),
                            ("原生单次适配（推荐/实验）", "native"),
                            ("自然适配（不裁剪）", "natural"),
                            ("严格槽位（不足补静音，超长保留）", "pad"),
                            ("强制精确（补静音或裁剪）", "exact"),
                        ],
                        value="off",
                        label="目标时长模式",
                    )
                    target_duration_seconds = gr.Number(
                        value=0.0,
                        minimum=0.0,
                        maximum=3600.0,
                        step=0.1,
                        label="目标时长（秒）",
                        info="启用目标时长模式后填写 0.1–3600 秒",
                    )
                    postprocess_preset = gr.Dropdown(
                        choices=[
                            ("关闭", "off"),
                            ("人声清晰", "voice_clarity"),
                            ("清晰旁白", "clear_narration"),
                            ("去刺耳", "deharsh"),
                            ("温暖", "warm"),
                            ("仅峰值归一化", "normalize"),
                        ],
                        value="off",
                        label="可选音频后处理",
                    )
                    postprocess_strength = gr.Slider(0, 1, value=1.0, step=0.05, label="后处理强度")
                with gr.Row():
                    quality_retry_count = gr.Slider(
                        0,
                        3,
                        value=0,
                        step=1,
                        label="追加候选数量",
                        info="0=单次；1–3 会更换 seed 生成并保留候选。ASR 可用时结合台词相似度选优，否则按音频质量选优",
                    )
                    quality_asr_backend = gr.Dropdown(
                        choices=list(ASR_BACKENDS), value="auto", label="质检 ASR 后端"
                    )
                    quality_asr_model = gr.Dropdown(
                        choices=list(ASR_MODELS), value="base", label="质检 ASR 模型"
                    )
                    quality_asr_device = gr.Dropdown(
                        choices=["auto", "cuda", "cpu"], value="auto", label="质检 ASR 设备"
                    )
                    quality_threshold = gr.Slider(
                        0,
                        1,
                        value=0.82,
                        step=0.01,
                        label="质检通过阈值",
                    )
                segment_preview_button = gr.Button("预览长文本分段")
                segment_preview_table = gr.Dataframe(
                    headers=["段号", "语音块", "Token 数", "段前停顿ms", "段后停顿ms", "分段文本"],
                    datatype=["number", "number", "number", "number", "number", "str"],
                    interactive=False,
                    wrap=True,
                    label="官方 Token 分段预览",
                )
                segment_preview_status = gr.Markdown("生成前可先查看长文本会如何切分。")

            with gr.Row(elem_classes=["t8-actions"]):
                stream_preview = gr.Checkbox(
                    value=True,
                    label="边生成边试听",
                    info="关闭或原生目标时长模式可实时返回；停止按钮会取消当前任务",
                )
                generate_button = gr.Button("生成语音", variant="primary", elem_classes=["t8-generate"])
                stop_button = gr.Button("停止当前/排队任务", variant="stop")
            stream_audio = BundledStreamingAudio(
                label="流式试听",
                streaming=True,
                autoplay=True,
                type="numpy",
                format="wav",
            )
            output_audio = gr.Audio(label="最终生成结果", type="filepath")
            candidate_audio_files = gr.Files(
                label="全部候选音频（可分别试听、下载或用于单段替换）",
                file_types=["audio"],
                interactive=False,
            )
            generation_performance = gr.Markdown(
                "尚未生成。RTF=生成耗时÷音频时长；小于 1 表示生成速度快于实时。"
                "峰值显存为本次生成期间 PyTorch 实际分配峰值，缓存峰值是 CUDA 保留内存。"
            )
            with gr.Accordion("ASR 自动校对当前结果", open=False):
                gr.Markdown(
                    "使用本地 Whisper 识别最终音频；中文/日文显示 CER，英文/西语/阿语显示 WER。"
                    "校对会统一简繁体、数字和标点，并输出差异明细及词级时间戳；"
                    "音频直接以内存波形送入 ASR，不依赖系统 FFmpeg。首次使用所选模型时会下载权重，"
                    "并保存在启动器用户数据目录的 `asr_models` 文件夹。"
                )
                with gr.Row():
                    single_asr_backend = gr.Dropdown(choices=list(ASR_BACKENDS), value="auto", label="ASR 后端")
                    single_asr_model = gr.Dropdown(choices=list(ASR_MODELS), value="base", label="ASR 模型")
                    single_asr_device = gr.Dropdown(choices=["auto", "cuda", "cpu"], value="auto", label="ASR 设备")
                    single_asr_language = gr.Dropdown(
                        choices=[("自动检测", "AUTO"), *LANGUAGE_CHOICES],
                        value="AUTO",
                        label="ASR 语言",
                    )
                    single_asr_threshold = gr.Slider(0, 1, value=0.82, step=0.01, label="通过阈值")
                single_asr_button = gr.Button("校对当前生成结果", variant="primary")
                single_asr_text = gr.TextArea(label="ASR 识别文本", lines=3, interactive=False)
                single_asr_status = gr.Markdown("尚未执行 ASR 校对。")
                single_asr_diff = gr.Markdown("差异明细会显示在这里。")
                single_asr_report = gr.TextArea(label="ASR 校对报告 JSON", lines=8, interactive=False)
                single_asr_waveform = gr.HTML(
                    '<div class="t8-timeline-empty">校对后显示音频波形和逐字时间标记。</div>'
                )

        with gr.Tab("角色音色库"):
            gr.Markdown(
                "把每个角色的音色与独立情感保存一次，批量台词和 SRT 会按角色名自动匹配。"
                "音色库 2.0 支持标签、收藏、备注、保存时质量评分和便携音色包；"
                "音色和情感参考音频都会复制到 Electron 用户数据目录，原文件移动后仍可使用。"
            )
            with gr.Row():
                with gr.Column():
                    voice_name = gr.Textbox(label="角色名称", placeholder="例如：旁白、小明、店长")
                    voice_audio = gr.Audio(label="角色音色参考", sources=["upload", "microphone"], type="filepath")
                    voice_language = gr.Dropdown(choices=LANGUAGE_CHOICES, value="ZH", label="默认语言")
                    voice_emotion_mode = gr.Radio(
                        choices=EMOTION_MODES,
                        value=EMOTION_MODES[0],
                        type="index",
                        label="该角色默认情感模式",
                        info="每个角色独立保存；不会把多个角色的情感数值混成一个情绪。",
                    )
                    with gr.Group(visible=False) as voice_emotion_audio_group:
                        voice_emotion_audio = gr.Audio(
                            label="该角色情感参考音频",
                            sources=["upload", "microphone"],
                            type="filepath",
                        )
                    with gr.Group(visible=False) as voice_emotion_vector_group:
                        with gr.Row():
                            voice_vector_controls = [
                                gr.Slider(0, 1, value=0, step=0.05, label=f"角色{label}")
                                for label in EMOTION_LABELS
                            ]
                        voice_random_emotion = gr.Checkbox(
                            label="该角色使用随机情感原型",
                            value=False,
                        )
                    with gr.Group(visible=False) as voice_emotion_text_group:
                        voice_emotion_text = gr.Textbox(
                            label="该角色默认情感描述",
                            placeholder="例如：克制而紧张；留空时分析每句台词",
                        )
                    with gr.Group(visible=False) as voice_emotion_strength_group:
                        voice_emotion_strength = gr.Slider(
                            0,
                            1,
                            value=0.65,
                            step=0.01,
                            label="该角色情感强度",
                        )
                    voice_dictionary = gr.TextArea(
                        label="角色专属发音词典（可选）",
                        placeholder="每行：文字|读音|语言，例如 行长|HANG2 ZHANG3|ZH",
                        lines=4,
                    )
                    with gr.Row():
                        voice_tags = gr.Textbox(
                            label="标签（逗号分隔）",
                            placeholder="例如：女声、旁白、温柔、日语",
                        )
                        voice_favorite = gr.Checkbox(value=False, label="收藏音色")
                    voice_notes = gr.TextArea(
                        label="备注（可搜索）",
                        placeholder="例如：纪录片旁白；适合平静情绪；录制于安静环境",
                        lines=2,
                    )
                    save_voice_button = gr.Button("保存角色音色", variant="primary")
                with gr.Column():
                    delete_voice_select = gr.Dropdown(
                        choices=voice_choices(),
                        label="选择已有角色",
                        info="载入后可试听、修改或改名；不载入时同名保存会覆盖。",
                    )
                    load_voice_button = gr.Button("载入 / 试听 / 编辑")
                    voice_update_selected = gr.Checkbox(
                        value=False,
                        label="更新所选角色（允许改名）",
                    )
                    delete_voice_button = gr.Button("删除角色音色", variant="stop")
                    with gr.Row():
                        voice_search = gr.Textbox(
                            label="搜索名称/标签/备注", placeholder="输入关键词"
                        )
                        voice_tag_filter = gr.Textbox(
                            label="必须包含标签", placeholder="多个标签用逗号分隔"
                        )
                    voice_favorites_only = gr.Checkbox(
                        value=False, label="只显示收藏"
                    )
                    filter_voice_button = gr.Button("筛选音色库")
                    with gr.Accordion("音色包导入 / 导出（桌面与 ComfyUI 共用）", open=False):
                        voice_bundle_import = gr.File(
                            label="导入 .t8voice.zip",
                            file_types=[".zip"],
                            type="filepath",
                        )
                        voice_import_conflict = gr.Dropdown(
                            choices=[
                                ("同名自动改名", "rename"),
                                ("同名覆盖", "replace"),
                                ("同名跳过", "skip"),
                            ],
                            value="rename",
                            label="同名音色处理",
                        )
                        with gr.Row():
                            import_voice_bundle_button = gr.Button("导入音色包")
                            export_all_voices = gr.Checkbox(value=True, label="导出全部音色")
                            export_voice_bundle_button = gr.Button("导出音色包")
                        voice_bundle_download = gr.File(
                            label="音色包下载", interactive=False
                        )
                    voice_status = gr.Markdown("尚未操作。DeepSpeed 等可选依赖与音色库无关。")
            voice_table = gr.Dataframe(
                headers=[
                    "收藏",
                    "角色",
                    "标签",
                    "语言",
                    "参考质量",
                    "独立情感设置",
                    "备注",
                    "本地音色音频",
                ],
                value=voice_rows(),
                interactive=False,
                wrap=True,
            )

        with gr.Tab("多角色 / 批量台词 / SRT"):
            gr.Markdown(
                "**可选语言：** 中文 `ZH`、英语 `EN`、日语 `JA`、西班牙语 `ES`、阿拉伯语 `AR`。  \n"
                "**批量格式：** `角色|台词|语言|时长系数|逐句情感`，最后一列可省略；也支持 JSON 数组。"
                "时长系数是官方的 **0.5–2.0 无单位时长适配倍率**：`0.8` 更快、`1.0` 原速、`1.2` 更慢；"
                "它不是秒数，也不等同于自然的语气语速，幅度过大可能拉长或失真。  \n"
                "**同一角色逐句情感：** `text:生气、激动` 或 `vector:喜,怒,哀,惧,厌恶,低落,惊喜,平静`；"
                "留空继承角色音色库的默认情感。  \n"
                "**SRT 角色写法：** `[小明] 台词`；逐句情感写成 `[小明|emotion=text:生气、激动] 台词`。"
                "没有角色标记时使用默认角色；"
                "SRT 全文使用下方“默认语言”，混合语言请改用批量格式。"
            )
            with gr.Accordion("真实示例（可直接载入或复制）", open=True):
                gr.Markdown(
                    """
**批量台词示例**（角色名要与“角色音色库”中已保存的名称一致）：

```text
旁白|先用平静语气介绍。|ZH|1.0|text:平静、从容
旁白|同一个角色突然非常生气！|ZH|1.0|vector:0,0.8,0,0,0,0,0,0
旁白|这一句留空，恢复角色默认情感。|ZH|1.0
旁白|This is a real English example.|EN|1.0|text:平静、自然
```

**SRT 示例**（时间码本身是 `时:分:秒,毫秒`，不是时长系数）：

```srt
1
00:00:00,000 --> 00:00:03,000
[旁白|emotion=text:平静、从容] 欢迎使用 IndexTTS 2.5。

2
00:00:03,200 --> 00:00:06,000
[旁白|emotion=vector:0,0.8,0,0,0,0,0,0] 同一个角色现在变得生气。
```
                    """
                )
                with gr.Row():
                    load_batch_example_button = gr.Button("载入批量真实示例")
                    load_srt_example_button = gr.Button("载入 SRT 真实示例")
            with gr.Row():
                dialogue_file = gr.File(label="导入 SRT / TXT / JSON", file_types=[".srt", ".txt", ".json"], type="filepath")
                import_dialogue_button = gr.Button("读取文件")
                dialogue_type = gr.Dropdown(choices=["batch", "srt"], value="batch", label="脚本格式")
                dialogue_default_role = gr.Dropdown(
                    choices=voice_choices(),
                    value=voice_choices()[0] if voice_choices() else None,
                    allow_custom_value=True,
                    label="默认角色",
                )
                dialogue_default_language = gr.Dropdown(choices=LANGUAGE_CHOICES, value="ZH", label="默认语言")
            dialogue_script = gr.TextArea(
                label="批量台词或 SRT 内容",
                lines=14,
                value=SAMPLE_BATCH_SCRIPT,
            )
            with gr.Row():
                timeline_policy = gr.Radio(
                    choices=[("顺延，避免重叠", "shift"), ("保留 SRT 起点并混音", "overlay")],
                    value="shift",
                    label="时间冲突策略（上一句超时怎么办）",
                )
                fit_srt_slots = gr.Checkbox(
                    value=False,
                    label="按 SRT 开始/结束时间匹配语音",
                    info="仅 SRT 生效；普通批量台词请关闭",
                )
                slot_duration_mode = gr.Dropdown(
                    choices=[
                        ("不足补静音，超长保留（推荐，不丢字）", "pad"),
                        ("自然适配（不裁剪）", "natural"),
                        ("原生单次适配（可能裁掉句尾）", "native"),
                        ("强制精确：补静音或裁剪", "exact"),
                    ],
                    value="pad",
                    label="SRT 时长处理方式",
                )
                fit_tolerance_ms = gr.Slider(
                    0,
                    2000,
                    value=180,
                    step=10,
                    label="触发二次适配的误差（毫秒）",
                    info="原生单次适配不使用此值",
                )
                batch_gap_ms = gr.Slider(
                    0,
                    5000,
                    value=200,
                    step=10,
                    label="普通批量台词句间静音（毫秒）",
                    info="只对 batch 生效；SRT 使用自身时间码",
                )
            timeline_settings_summary = gr.Markdown(
                describe_dialogue_timing_settings("batch", "shift", False, "pad", 180, 200),
                elem_classes=["t8-timing-summary"],
            )
            with gr.Accordion(
                "时间设置看不懂？展开查看 2 秒字幕实例与推荐配置",
                open=True,
                elem_classes=["t8-timing-guide"],
            ):
                gr.Markdown(
                    """
**真实例子：** SRT 给某句的时间是 `00:00:00,000 → 00:00:02,000`，也就是必须放进 **2 秒槽位**；模型第一次生成了 **2.3 秒**。

| 处理方式 | 这句最终会怎样 |
| --- | --- |
| 不足补静音，超长保留（推荐） | 短了补到 2 秒；长了完整保留，保证不丢字 |
| 自然适配 | 超过允许误差时调语速重做，但不裁剪，可能仍略长于 2 秒 |
| 原生单次适配 | 生成时直接指定 2 秒，最后仍会补齐或裁到 2 秒，过短槽位可能丢句尾 |
| 强制精确 | 短了补静音，长了裁到 2 秒；槽位过短可能裁掉句尾 |

**不知道怎么选时：**

- 普通批量对白：关闭 SRT 时长适配，选择“顺延”，句间静音设为 `200～500 ms`。
- 普通 SRT 配音：开启 SRT 时长适配，选择“原生单次适配”，冲突策略选择“顺延”。
- 必须严格卡视频时间：选择“强制精确”，生成后检查句尾是否被裁掉。
- 只有确实需要两个人同时说话时，才选择“保留 SRT 起点并混音”。
                    """
                )
            with gr.Accordion("批量分段、停顿与后处理", open=False):
                gr.Markdown(
                    "每条台词支持 `<pause=0.5>` / `<pause=500ms>`；自动分段会针对英语和西语使用更稳妥的 60 Token 上限。"
                )
                with gr.Row():
                    dialogue_segmentation_mode = gr.Dropdown(
                        choices=[("按语言自动（推荐）", "auto"), ("手动指定", "custom")],
                        value="auto",
                        label="分段模式",
                    )
                    dialogue_max_text_tokens = gr.Slider(20, 300, value=120, step=5, label="手动每段 Token")
                    dialogue_pause_preset = gr.Dropdown(
                        choices=[("关闭", "off"), ("自然", "natural"), ("旁白", "narration"), ("对话", "dialogue"), ("自定义", "custom")],
                        value="off",
                        label="标点停顿预设",
                    )
                with gr.Row():
                    dialogue_comma_pause_ms = gr.Slider(0, 5000, value=100, step=10, label="逗号停顿ms")
                    dialogue_sentence_pause_ms = gr.Slider(0, 5000, value=300, step=10, label="句末停顿ms")
                    dialogue_paragraph_pause_ms = gr.Slider(0, 5000, value=600, step=10, label="段落停顿ms")
                with gr.Row():
                    dialogue_postprocess_preset = gr.Dropdown(
                        choices=[("关闭", "off"), ("人声清晰", "voice_clarity"), ("清晰旁白", "clear_narration"), ("去刺耳", "deharsh"), ("温暖", "warm"), ("归一化", "normalize")],
                        value="off",
                        label="合并音频后处理",
                    )
                    dialogue_postprocess_strength = gr.Slider(0, 1, value=1.0, step=0.05, label="后处理强度")
                with gr.Row():
                    dialogue_seed = gr.Number(value=0, minimum=0, maximum=4294967295, precision=0, label="起始 seed")
                    dialogue_diffusion_steps = gr.Slider(5, 100, value=25, step=1, label="CFM 扩散步数")
                    dialogue_inference_cfg_rate = gr.Slider(0, 1.5, value=0.7, step=0.05, label="CFM 引导强度")
                    dialogue_cfm_temperature = gr.Slider(0.1, 1.5, value=1.0, step=0.05, label="CFM 温度")
            with gr.Accordion("ASR 自动校对与字幕自动回写", open=True):
                gr.Markdown(
                    "启用后逐句使用本地 Whisper 校对，并把识别文本、CER/WER、差异、词级时间戳和通过状态写入任务报告。"
                    "回写字幕默认采用实际混音时间；只有通过阈值的识别文本才替换原字幕，低分结果保留原文。"
                    "ASR 权重首次使用时下载到启动器用户数据目录的 `asr_models` 文件夹。"
                )
                with gr.Row():
                    dialogue_asr_enabled = gr.Checkbox(value=False, label="生成后逐句自动 ASR 校对")
                    dialogue_asr_backend = gr.Dropdown(choices=list(ASR_BACKENDS), value="auto", label="ASR 后端")
                    dialogue_asr_model = gr.Dropdown(choices=list(ASR_MODELS), value="base", label="ASR 模型")
                    dialogue_asr_device = gr.Dropdown(choices=["auto", "cuda", "cpu"], value="auto", label="ASR 设备")
                    dialogue_asr_threshold = gr.Slider(0, 1, value=0.82, step=0.01, label="ASR 通过阈值")
                    dialogue_asr_retry_count = gr.Slider(
                        0,
                        3,
                        value=0,
                        step=1,
                        label="失败自动重试次数",
                        info="每次更换 seed，保留 ASR 相似度最高的一次；大批量会增加耗时",
                    )
                with gr.Row():
                    subtitle_timing_mode = gr.Dropdown(
                        choices=[("实际混音时间（推荐）", "actual"), ("原始字幕时间", "original")],
                        value="actual",
                        label="回写字幕时间",
                    )
                    subtitle_text_mode = gr.Dropdown(
                        choices=[
                            ("仅使用通过校对的 ASR 文本（推荐）", "asr_passed"),
                            ("始终使用 ASR 文本", "asr_all"),
                            ("始终保留原字幕", "original"),
                        ],
                        value="asr_passed",
                        label="回写字幕文本",
                    )
                    subtitle_include_role = gr.Checkbox(value=True, label="回写字幕保留 [角色] 前缀")
            with gr.Row():
                preview_dialogue_button = gr.Button("解析并检查角色")
                refresh_timeline_button = gr.Button("手动刷新可视化时间轴")
                generate_dialogue_button = gr.Button("生成全部台词", variant="primary")
                stop_dialogue_button = gr.Button("停止排队任务", variant="stop")
            dialogue_status = gr.Markdown("请先在“角色音色库”保存脚本中使用的角色。")
            dialogue_preview = gr.Dataframe(
                headers=TIMELINE_HEADERS,
                datatype=["number", "str", "str", "number", "number", "number", "str", "str"],
                type="array",
                interactive=True,
                wrap=True,
                label="可编辑时间轴（表格与下方可拖拽轨道双向同步；最后一列可逐句改情感）",
            )
            with gr.Accordion("上下文情感自动标注（先建议，确认后才生成）", open=True):
                gr.Markdown(
                    "使用本地 QwenEmotion 读取目标台词及前后文，为每句建议八维情感向量和强度。"
                    "结果只写入上方表格最后一列，**不会自动合成音频**；请逐行检查后再点击生成。"
                )
                with gr.Row():
                    dialogue_emotion_context_window = gr.Slider(
                        0,
                        5,
                        value=2,
                        step=1,
                        label="每侧上下文台词数",
                        info="2 表示参考前 2 句和后 2 句；不会混用其他角色的情感",
                    )
                    dialogue_emotion_overwrite = gr.Checkbox(
                        value=False,
                        label="覆盖已有逐句情感",
                        info="默认关闭：保留你已经手工填写的 text:/vector:",
                    )
                    suggest_dialogue_emotion_button = gr.Button(
                        "分析上下文并填入建议",
                        variant="secondary",
                    )
                dialogue_emotion_report = gr.TextArea(
                    label="上下文情感建议报告 JSON",
                    lines=8,
                    interactive=False,
                )
            dialogue_timeline_visual = gr.HTML(render_timeline_html([]))
            dialogue_timeline_drag_payload = gr.Textbox(
                value="",
                elem_id="t8-timeline-drag-payload",
                container=False,
                interactive=True,
            )
            with gr.Row(equal_height=True):
                dialogue_output = gr.Audio(label="合并音频", type="filepath", scale=8)
                dialogue_combined_download = gr.DownloadButton(
                    "下载合并音频 WAV",
                    value=None,
                    variant="primary",
                    scale=2,
                    min_width=190,
                )
            dialogue_archive = gr.File(label="逐句音频 + combined.wav + report.json + rewritten.srt", interactive=False)
            dialogue_rewritten_srt = gr.File(label="自动回写字幕 SRT", interactive=False)
            dialogue_report = gr.TextArea(label="生成报告 JSON", lines=14, interactive=False)
            gr.Markdown(
                "完成提示与 `report.json > performance` 会记录真实生成耗时、音频时长、RTF、"
                "CUDA 分配峰值、缓存峰值及相对模型常驻显存的本次生成增量。"
            )
            with gr.Accordion("任务恢复与单句重试", open=True):
                gr.Markdown(
                    "每完成一条台词就立即保存任务清单。软件意外关闭后，选择任务并继续即可跳过已完成台词；"
                    "填写台词序号后可只重做该句并重新合并。"
                )
                with gr.Row():
                    dialogue_task_select = gr.Dropdown(
                        choices=task_choices(output_dir),
                        label="已保存任务",
                    )
                    refresh_dialogue_tasks_button = gr.Button("刷新任务列表")
                    resume_dialogue_task_button = gr.Button("继续所选任务", variant="primary")
                with gr.Row():
                    retry_dialogue_line_number = gr.Number(
                        value=1,
                        minimum=1,
                        precision=0,
                        label="要重做的台词序号",
                    )
                    retry_dialogue_line_button = gr.Button("重做选中/指定句并重新合并")
                    rebuild_dialogue_timeline_button = gr.Button("按编辑时间轴重新混音（不重新生成）")
                with gr.Accordion("完整配音工程导入 / 导出", open=False):
                    gr.Markdown(
                        "工程包会保存脚本/SRT、角色与情感、全部生成参数、可编辑时间轴、"
                        "逐句 WAV、合并音频、字幕和报告，并附带当前任务使用的便携音色包。"
                    )
                    with gr.Row():
                        export_dialogue_project_button = gr.Button(
                            "导出当前任务工程", variant="secondary"
                        )
                        dialogue_project_download = gr.File(
                            label="完整工程包下载", interactive=False
                        )
                    with gr.Row():
                        dialogue_project_import = gr.File(
                            label="导入 .indextts-project.zip",
                            file_types=[".zip"],
                            type="filepath",
                        )
                        project_voice_conflict = gr.Dropdown(
                            choices=[
                                ("同名音色自动改名", "rename"),
                                ("同名音色覆盖", "replace"),
                                ("同名音色跳过", "skip"),
                            ],
                            value="rename",
                            label="工程内音色冲突处理",
                        )
                        import_dialogue_project_button = gr.Button(
                            "导入并打开工程", variant="primary"
                        )
                gr.Markdown(
                    "操作：可直接拖动下方音频块改变位置，或拖动左右手柄改变起止时间；"
                    "靠近其他台词边界和 ASR 逐字时间点会自动吸附，按住 Alt 可临时关闭吸附。"
                    "每次拖动都会自动同步上方表格并选中该句；修改台词/角色/语言/逐句情感后，"
                    "点击“重做选中/指定句并重新合并”，只重新生成这一句，其他逐句 WAV 直接复用。"
                )
                dialogue_task_status = gr.Markdown("尚未选择任务。")
                new_dialogue_task_state = gr.Textbox(value="", visible=False)
                no_force_line_state = gr.Number(value=0, precision=0, visible=False)

        with gr.Tab("环境与可选加速"):
            gr.Markdown(
                "加速模式在 Electron 启动页选择，修改后需要重启推理服务。"
                "基础模式不需要 DeepSpeed、FlashAttention 或 Triton；缺少时会自动回退，不影响普通推理。"
            )
            gr.TextArea(label="本次启动环境与加速诊断", value=acceleration_report, lines=24, interactive=False)
            gr.Markdown(
                "- `auto_safe`：只在工具链齐全时使用 BigVGAN CUDA 融合核。\n"
                "- `torch_compile`：需要可选 Triton，首次生成有编译开销。\n"
                "- `gpt_accel`：需要 FlashAttention + Triton；不兼容采样参数会自动走普通 GPT。\n"
                "- `deepspeed`：仅显式选择时使用；Windows 包固定使用兼容的 FP16 GPT 内核，初始化失败会自动回退。"
            )
            with gr.Accordion("模型与显存生命周期", open=True):
                gr.Markdown(
                    "这里只管理本整合包加载的 IndexTTS 2.5 模型，不会清理其他程序。"
                    "手动或自动释放后，下次生成会自动重新加载模型。"
                )
                with gr.Row():
                    release_after_generation = gr.Checkbox(
                        value=False,
                        label="每次生成后释放模型",
                    )
                    idle_release_seconds = gr.Number(
                        value=600,
                        minimum=0,
                        maximum=86400,
                        precision=0,
                        label="空闲自动释放（秒，0=关闭）",
                    )
                    recycle_after_generations = gr.Number(
                        value=0,
                        minimum=0,
                        maximum=1000,
                        precision=0,
                        label="连续生成多少次后重载（0=关闭）",
                    )
                with gr.Row():
                    apply_memory_policy_button = gr.Button("应用显存策略", variant="primary")
                    refresh_model_status_button = gr.Button("刷新模型状态")
                    release_model_button = gr.Button("立即释放 IndexTTS 模型", variant="stop")
                model_memory_status = gr.TextArea(
                    label="模型与 CUDA 显存状态",
                    value=refresh_model_status_event(),
                    lines=12,
                    interactive=False,
                )
            with gr.Accordion("参考条件缓存管理", open=True):
                gr.Markdown(
                    "音色和情感参考编码结果会按音频内容、模型版本、精度及参考设备隔离缓存。"
                    "重复使用同一参考音频可跳过参考编码；清理缓存不会删除原音频或模型。"
                )
                with gr.Row():
                    refresh_reference_cache_button = gr.Button("刷新参考缓存状态")
                    clear_reference_cache_button = gr.Button(
                        "清理参考条件缓存", variant="stop"
                    )
                reference_cache_status = gr.TextArea(
                    label="参考缓存条目、容量与命中统计",
                    value=refresh_reference_cache_event(),
                    lines=13,
                    interactive=False,
                )

        with gr.Tab("实验 audio.cpp 后端"):
            gr.Markdown(
                "这是与默认 Python 推理完全隔离的可选后端，不会自动替换当前模型。"
                "现在可一键安装经过 Release SHA-256 校验的 Windows 运行时，并从"
                f" [官方 GGUF 仓库](https://huggingface.co/{AUDIOCPP_MODEL_REPOSITORY}) 自动下载固定 revision 模型。"
                "Q8 模型约 3.5GB；下载支持断点续传，CUDA 安装会同时包含匹配的官方 CUDA 12.4 运行库。"
                "其文本归一化实现与官方 Python 路径并非所有边界输入都完全一致，"
                "请先对中文、英语、日语、西语和阿语做试听对比。"
            )
            with gr.Accordion("一键安装 / 更新可选组件", open=True):
                with gr.Row():
                    audiocpp_install_backend = gr.Dropdown(
                        choices=[
                            ("NVIDIA CUDA 12.4（含运行库）", "cuda"),
                            ("Vulkan（NVIDIA/AMD/Intel）", "vulkan"),
                            ("CPU", "cpu"),
                        ],
                        value="cuda",
                        label="运行时版本",
                    )
                    audiocpp_quantization = gr.Dropdown(
                        choices=[
                            ("Q8_0 · 约 3.5GB（推荐）", "q8_0"),
                            ("F16 · 约 4.5GB", "f16"),
                            ("原始精度 · 约 7.9GB", "original"),
                        ],
                        value="q8_0",
                        label="GGUF 模型精度",
                    )
                with gr.Row():
                    audiocpp_install_runtime_button = gr.Button("安装/更新运行时")
                    audiocpp_install_model_button = gr.Button("下载/校验 GGUF 模型")
                    audiocpp_install_all_button = gr.Button(
                        "一键安装完整 audio.cpp", variant="primary"
                    )
                    audiocpp_refresh_status_button = gr.Button("刷新组件状态")
            with gr.Row():
                audiocpp_executable = gr.Textbox(
                    label="audiocpp_cli.exe 绝对路径",
                    value=audiocpp_initial_status.get("executable") or "",
                    placeholder=r"D:\audio.cpp\audiocpp_cli.exe",
                )
                audiocpp_model_dir = gr.Textbox(
                    label="IndexTTS2.5-GGUF 模型目录或 GGUF 路径",
                    value=audiocpp_initial_status.get("modelPath") or "",
                    placeholder=r"D:\models\IndexTTS2.5-GGUF",
                )
                audiocpp_probe_button = gr.Button("检测 CLI")
            with gr.Row():
                audiocpp_speaker = gr.Audio(
                    label="audio.cpp 音色参考 WAV",
                    sources=["upload", "microphone"],
                    type="filepath",
                )
                audiocpp_text = gr.TextArea(
                    label="audio.cpp 待合成文本",
                    value="这是隔离的 audio.cpp IndexTTS 2.5 实验后端。",
                    lines=5,
                )
            with gr.Row():
                audiocpp_language = gr.Dropdown(
                    choices=[("自动", "AUTO"), *LANGUAGE_CHOICES],
                    value="ZH",
                    label="语言",
                )
                audiocpp_backend = gr.Dropdown(
                    choices=["cuda", "cpu", "vulkan", "hip"],
                    value=audiocpp_initial_status.get("installedBackend") or "cuda",
                    label="audio.cpp 计算后端",
                )
                audiocpp_duration = gr.Slider(
                    0.5,
                    2.0,
                    value=1.0,
                    step=0.05,
                    label="官方时长适配倍率（无单位）",
                )
                audiocpp_memory_saver = gr.Checkbox(
                    value=True,
                    label="请求阶段结束后释放临时图",
                )
            audiocpp_emotion_text = gr.Textbox(
                label="情感描述（可选）",
                placeholder="例如：平静而坚定",
            )
            audiocpp_generate_button = gr.Button("使用 audio.cpp 实验后端生成", variant="primary")
            audiocpp_output = gr.Audio(label="audio.cpp 生成结果", type="filepath")
            audiocpp_report = gr.TextArea(
                label="audio.cpp 探测/生成报告 JSON",
                value=json.dumps(audiocpp_initial_status, ensure_ascii=False, indent=2),
                lines=14,
                interactive=False,
            )

        with gr.Tab("生成历史"):
            refresh_history = gr.Button("刷新历史")
            history = gr.Dataframe(
                headers=["时间", "语言", "时长系数", "情感", "原文", "发音文本", "文件"],
                value=load_history(output_dir),
                interactive=False,
                wrap=True,
            )
        refresh_history.click(lambda: load_history(output_dir), outputs=history, queue=False)

        apply_memory_policy_button.click(
            update_memory_policy_event,
            inputs=[
                release_after_generation,
                idle_release_seconds,
                recycle_after_generations,
            ],
            outputs=model_memory_status,
            queue=False,
        )
        refresh_model_status_button.click(
            refresh_model_status_event,
            outputs=model_memory_status,
            queue=False,
        )
        release_model_button.click(
            release_model_event,
            outputs=model_memory_status,
            queue=False,
        )
        refresh_reference_cache_button.click(
            refresh_reference_cache_event,
            outputs=reference_cache_status,
            queue=False,
        )
        clear_reference_cache_button.click(
            clear_reference_cache_event,
            outputs=reference_cache_status,
            queue=False,
        )
        audiocpp_refresh_status_button.click(
            audiocpp_status_event,
            outputs=[
                audiocpp_executable,
                audiocpp_model_dir,
                audiocpp_backend,
                audiocpp_report,
            ],
            queue=False,
        )
        audiocpp_install_runtime_button.click(
            install_audiocpp_runtime_event,
            inputs=audiocpp_install_backend,
            outputs=[
                audiocpp_executable,
                audiocpp_model_dir,
                audiocpp_backend,
                audiocpp_report,
            ],
            concurrency_limit=1,
        )
        audiocpp_install_model_button.click(
            install_audiocpp_model_event,
            inputs=audiocpp_quantization,
            outputs=[
                audiocpp_executable,
                audiocpp_model_dir,
                audiocpp_backend,
                audiocpp_report,
            ],
            concurrency_limit=1,
        )
        audiocpp_install_all_button.click(
            install_audiocpp_all_event,
            inputs=[audiocpp_install_backend, audiocpp_quantization],
            outputs=[
                audiocpp_executable,
                audiocpp_model_dir,
                audiocpp_backend,
                audiocpp_report,
            ],
            concurrency_limit=1,
        )
        audiocpp_probe_button.click(
            probe_audiocpp_event,
            inputs=audiocpp_executable,
            outputs=audiocpp_report,
            queue=False,
        )
        audiocpp_generate_button.click(
            generate_audiocpp_event,
            inputs=[
                audiocpp_executable,
                audiocpp_model_dir,
                audiocpp_speaker,
                audiocpp_text,
                audiocpp_language,
                audiocpp_backend,
                audiocpp_duration,
                audiocpp_emotion_text,
                audiocpp_memory_saver,
            ],
            outputs=[audiocpp_output, audiocpp_report],
            concurrency_limit=1,
        )

        gr.HTML(
            '<div class="t8-footer">基于 Bilibili IndexTTS 2.5 · 非官方桌面整合版 · 本地离线推理</div>'
        )

        emotion_mode.change(
            change_emotion_mode,
            inputs=emotion_mode,
            outputs=[emotion_audio_group, emotion_vector_group, emotion_text_group, emotion_weight_group],
            queue=False,
        )
        voice_emotion_mode.change(
            change_emotion_mode,
            inputs=voice_emotion_mode,
            outputs=[
                voice_emotion_audio_group,
                voice_emotion_vector_group,
                voice_emotion_text_group,
                voice_emotion_strength_group,
            ],
            queue=False,
        )
        insert_annotation_button.click(
            insert_annotation_event,
            inputs=[text, annotation_term, annotation_reading, annotation_language],
            outputs=[text, pronunciation_report],
            queue=False,
        )
        quick_pronunciation_example_button.click(
            fill_chinese_pronunciation_example_event,
            outputs=[text, language, pronunciation_report],
            queue=False,
        )
        inspect_reference_button.click(
            inspect_reference_event,
            inputs=[
                prompt_audio,
                reference_auto_prepare,
                reference_maximum_seconds,
                reference_padding_ms,
            ],
            outputs=[prompt_audio, reference_quality_report, reference_waveform],
            queue=False,
        )
        single_voice_select.change(
            load_single_voice_event,
            inputs=single_voice_select,
            outputs=[prompt_audio, single_voice_status],
            queue=False,
        )
        refresh_single_voice_button.click(
            refresh_single_voice_event,
            inputs=single_voice_select,
            outputs=[single_voice_select, prompt_audio, single_voice_status],
            queue=False,
        )
        dictionary_editor_components = [
            dictionary_entry_term,
            dictionary_entry_language,
            dictionary_entry_reading,
            dictionary_entry_enabled,
            dictionary_entry_case,
        ]
        dictionary_table_editor_outputs = [
            dictionary_table,
            dictionary_entry_select,
            *dictionary_editor_components,
            pronunciation_report,
        ]
        dictionary_entry_select.input(
            select_dictionary_entry_event,
            inputs=[dictionary_table, dictionary_entry_select],
            outputs=[*dictionary_editor_components, pronunciation_report],
            queue=False,
        )
        upsert_dictionary_entry_button.click(
            upsert_dictionary_entry_event,
            inputs=[
                dictionary_table,
                dictionary_entry_select,
                *dictionary_editor_components,
            ],
            outputs=dictionary_table_editor_outputs,
            queue=False,
        )
        clear_dictionary_editor_button.click(
            clear_dictionary_editor_event,
            outputs=[
                dictionary_entry_select,
                *dictionary_editor_components,
                pronunciation_report,
            ],
            queue=False,
        )
        delete_dictionary_entry_button.click(
            delete_dictionary_entry_event,
            inputs=[dictionary_table, dictionary_entry_select],
            outputs=dictionary_table_editor_outputs,
            queue=False,
        )
        save_dictionary_button.click(
            save_dictionary_event,
            inputs=[dictionary_table],
            outputs=[pronunciation_report],
            queue=False,
        )
        load_examples_button.click(
            load_examples_event,
            outputs=dictionary_table_editor_outputs,
            queue=False,
        )
        import_dictionary_button.click(
            import_dictionary_event,
            inputs=[import_dictionary_file],
            outputs=dictionary_table_editor_outputs,
            queue=False,
        )
        export_dictionary_button.click(
            export_dictionary_event,
            inputs=[dictionary_table],
            outputs=[exported_dictionary_file, pronunciation_report],
            queue=False,
        )
        dictionary_search_button.click(
            search_dictionary_event,
            inputs=[dictionary_table, dictionary_search],
            outputs=[dictionary_search_result],
            queue=False,
        )
        pronunciation_preview_button.click(
            preview_pronunciation_event,
            inputs=[text, language, dictionary_table],
            outputs=[pronunciation_preview, pronunciation_report],
            queue=False,
        )
        segment_preview_button.click(
            preview_segments_event,
            inputs=[
                text,
                language,
                segmentation_mode,
                max_text_tokens,
                pause_preset,
                comma_pause_ms,
                sentence_pause_ms,
                paragraph_pause_ms,
            ],
            outputs=[segment_preview_table, segment_preview_status],
            queue=False,
        )
        save_voice_button.click(
            save_voice_event,
            inputs=[
                voice_name,
                voice_audio,
                voice_language,
                voice_emotion_mode,
                voice_emotion_audio,
                voice_emotion_text,
                voice_emotion_strength,
                voice_random_emotion,
                voice_tags,
                voice_favorite,
                voice_notes,
                *voice_vector_controls,
                voice_dictionary,
                delete_voice_select,
                voice_update_selected,
            ],
            outputs=[
                voice_table,
                delete_voice_select,
                dialogue_default_role,
                single_voice_select,
                prompt_audio,
                single_voice_status,
                voice_update_selected,
                voice_status,
            ],
            queue=False,
        )
        load_voice_button.click(
            load_voice_event,
            inputs=[delete_voice_select],
            outputs=[
                voice_name,
                voice_audio,
                voice_language,
                voice_emotion_mode,
                voice_emotion_audio,
                voice_emotion_text,
                voice_emotion_strength,
                voice_random_emotion,
                *voice_vector_controls,
                voice_dictionary,
                voice_tags,
                voice_favorite,
                voice_notes,
                voice_update_selected,
                voice_status,
            ],
            queue=False,
        )
        delete_voice_button.click(
            delete_voice_event,
            inputs=[delete_voice_select, single_voice_select],
            outputs=[
                voice_table,
                delete_voice_select,
                dialogue_default_role,
                single_voice_select,
                prompt_audio,
                single_voice_status,
                voice_update_selected,
                voice_status,
            ],
            queue=False,
        )
        filter_voice_button.click(
            filter_voice_library_event,
            inputs=[voice_search, voice_tag_filter, voice_favorites_only],
            outputs=[voice_table, voice_status],
            queue=False,
        )
        import_voice_bundle_button.click(
            import_voice_bundle_event,
            inputs=[voice_bundle_import, voice_import_conflict],
            outputs=[
                voice_table,
                delete_voice_select,
                dialogue_default_role,
                single_voice_select,
                voice_status,
            ],
            queue=False,
        )
        export_voice_bundle_button.click(
            export_voice_bundle_event,
            inputs=[delete_voice_select, export_all_voices],
            outputs=[voice_bundle_download, voice_status],
            queue=False,
        )
        import_dialogue_button.click(
            import_script_event,
            inputs=[dialogue_file],
            outputs=[dialogue_script, dialogue_type],
            queue=False,
        )
        load_batch_example_button.click(
            lambda: (SAMPLE_BATCH_SCRIPT, "batch"),
            outputs=[dialogue_script, dialogue_type],
            queue=False,
        )
        load_srt_example_button.click(
            lambda: (SAMPLE_SRT_SCRIPT, "srt"),
            outputs=[dialogue_script, dialogue_type],
            queue=False,
        )
        timing_help_inputs = [
            dialogue_type,
            timeline_policy,
            fit_srt_slots,
            slot_duration_mode,
            fit_tolerance_ms,
            batch_gap_ms,
        ]
        for timing_control in timing_help_inputs:
            timing_control.change(
                describe_dialogue_timing_settings,
                inputs=timing_help_inputs,
                outputs=timeline_settings_summary,
                queue=False,
            )
        preview_dialogue_button.click(
            preview_dialogue_event,
            inputs=[dialogue_type, dialogue_script, dialogue_default_role, dialogue_default_language],
            outputs=[dialogue_preview, dialogue_timeline_visual, dialogue_status],
            queue=False,
        )
        refresh_timeline_button.click(
            refresh_timeline_event,
            inputs=[
                dialogue_type,
                dialogue_script,
                dialogue_default_role,
                dialogue_default_language,
                dialogue_preview,
                dialogue_report,
            ],
            outputs=[dialogue_timeline_visual, dialogue_status],
            queue=False,
        )
        dialogue_preview.input(
            refresh_timeline_event,
            inputs=[
                dialogue_type,
                dialogue_script,
                dialogue_default_role,
                dialogue_default_language,
                dialogue_preview,
                dialogue_report,
            ],
            outputs=[dialogue_timeline_visual, dialogue_status],
            queue=False,
        )
        dialogue_timeline_drag_payload.input(
            apply_timeline_drag_event,
            inputs=[
                dialogue_type,
                dialogue_script,
                dialogue_default_role,
                dialogue_default_language,
                dialogue_preview,
                dialogue_timeline_drag_payload,
                dialogue_report,
            ],
            outputs=[
                dialogue_preview,
                dialogue_timeline_visual,
                dialogue_task_status,
                retry_dialogue_line_number,
            ],
            queue=False,
        )
        suggest_dialogue_emotion_button.click(
            suggest_dialogue_emotions_event,
            inputs=[
                dialogue_type,
                dialogue_script,
                dialogue_default_role,
                dialogue_default_language,
                dialogue_preview,
                dialogue_emotion_context_window,
                dialogue_emotion_overwrite,
            ],
            outputs=[
                dialogue_preview,
                dialogue_timeline_visual,
                dialogue_status,
                dialogue_emotion_report,
            ],
            concurrency_limit=1,
        )
        preset_select.change(
            lambda value: value or "", inputs=preset_select, outputs=preset_name, queue=False
        )
        preset_advanced_components = [
            do_sample,
            temperature,
            top_p,
            top_k,
            num_beams,
            repetition_penalty,
            length_penalty,
            max_mel_tokens,
            seed,
            diffusion_steps,
            inference_cfg_rate,
            cfm_temperature,
            stream_preview,
            segmentation_mode,
            max_text_tokens,
            segment_silence_ms,
            pause_preset,
            comma_pause_ms,
            sentence_pause_ms,
            paragraph_pause_ms,
            text_normalization,
            target_duration_mode,
            target_duration_seconds,
            postprocess_preset,
            postprocess_strength,
        ]
        preset_state_components = [
            prompt_audio,
            text,
            language,
            duration_factor,
            emotion_mode,
            emotion_audio,
            emotion_weight,
            emotion_text,
            random_emotion,
            *vector_controls,
            pronunciation_strict,
            *preset_advanced_components,
        ]
        save_preset_button.click(
            save_preset_event,
            inputs=[
                preset_name,
                prompt_audio,
                text,
                language,
                duration_factor,
                emotion_mode,
                emotion_audio,
                emotion_weight,
                emotion_text,
                random_emotion,
                pronunciation_strict,
                *vector_controls,
                *preset_advanced_components,
            ],
            outputs=[preset_select, preset_status],
            queue=False,
        )
        preset_load_event = load_preset_button.click(
            load_preset_event,
            inputs=preset_select,
            outputs=[*preset_state_components, preset_status],
            queue=False,
        )
        preset_load_event.then(
            change_emotion_mode,
            inputs=emotion_mode,
            outputs=[emotion_audio_group, emotion_vector_group, emotion_text_group, emotion_weight_group],
            queue=False,
        )
        delete_preset_button.click(
            delete_preset_event,
            inputs=preset_select,
            outputs=[preset_select, preset_status],
            queue=False,
        )
        generation_event = generate_button.click(
            generate,
            inputs=[
                prompt_audio,
                text,
                language,
                duration_factor,
                emotion_mode,
                emotion_audio,
                emotion_weight,
                emotion_text,
                random_emotion,
                dictionary_table,
                pronunciation_strict,
                quality_retry_count,
                quality_asr_backend,
                quality_asr_model,
                quality_asr_device,
                quality_threshold,
                *vector_controls,
                do_sample,
                temperature,
                top_p,
                top_k,
                num_beams,
                repetition_penalty,
                length_penalty,
                max_mel_tokens,
                seed,
                diffusion_steps,
                inference_cfg_rate,
                cfm_temperature,
                stream_preview,
                segmentation_mode,
                max_text_tokens,
                segment_silence_ms,
                pause_preset,
                comma_pause_ms,
                sentence_pause_ms,
                paragraph_pause_ms,
                text_normalization,
                target_duration_mode,
                target_duration_seconds,
                postprocess_preset,
                postprocess_strength,
            ],
            outputs=[
                stream_audio,
                output_audio,
                candidate_audio_files,
                history,
                pronunciation_report,
                generation_performance,
            ],
            concurrency_limit=1,
        )
        stop_button.click(fn=None, cancels=[generation_event], queue=False)
        single_asr_button.click(
            asr_proofread_event,
            inputs=[
                output_audio,
                text,
                single_asr_language,
                single_asr_backend,
                single_asr_model,
                single_asr_device,
                single_asr_threshold,
            ],
            outputs=[
                single_asr_text,
                single_asr_status,
                single_asr_diff,
                single_asr_report,
                single_asr_waveform,
            ],
            concurrency_limit=1,
        )
        dialogue_common_inputs = [
            dialogue_type,
            dialogue_script,
            dialogue_default_role,
            dialogue_default_language,
            dialogue_preview,
            timeline_policy,
            fit_srt_slots,
            slot_duration_mode,
            fit_tolerance_ms,
            batch_gap_ms,
            dialogue_segmentation_mode,
            dialogue_max_text_tokens,
            dialogue_pause_preset,
            dialogue_comma_pause_ms,
            dialogue_sentence_pause_ms,
            dialogue_paragraph_pause_ms,
            dialogue_postprocess_preset,
            dialogue_postprocess_strength,
            dialogue_seed,
            dialogue_diffusion_steps,
            dialogue_inference_cfg_rate,
            dialogue_cfm_temperature,
            dialogue_asr_enabled,
            dialogue_asr_backend,
            dialogue_asr_model,
            dialogue_asr_device,
            dialogue_asr_threshold,
            dialogue_asr_retry_count,
            subtitle_timing_mode,
            subtitle_text_mode,
            subtitle_include_role,
        ]
        dialogue_outputs = [
            dialogue_output,
            dialogue_archive,
            dialogue_rewritten_srt,
            dialogue_report,
            history,
            dialogue_task_select,
            dialogue_task_status,
            dialogue_timeline_visual,
            dialogue_combined_download,
        ]
        refresh_dialogue_tasks_button.click(
            lambda: gr.update(choices=task_choices(output_dir)),
            outputs=dialogue_task_select,
            queue=False,
        )
        dialogue_task_select.change(
            load_dialogue_task_editor_event,
            inputs=dialogue_task_select,
            outputs=[
                dialogue_type,
                dialogue_script,
                dialogue_default_role,
                dialogue_default_language,
                dialogue_preview,
                timeline_policy,
                dialogue_timeline_visual,
                dialogue_task_status,
                dialogue_report,
            ],
            queue=False,
        )
        export_dialogue_project_button.click(
            export_dialogue_project_event,
            inputs=dialogue_task_select,
            outputs=[dialogue_project_download, dialogue_task_status],
            queue=False,
        )
        import_dialogue_project_button.click(
            import_dialogue_project_event,
            inputs=[dialogue_project_import, project_voice_conflict],
            outputs=[
                dialogue_task_select,
                dialogue_type,
                dialogue_script,
                dialogue_default_role,
                dialogue_default_language,
                dialogue_preview,
                timeline_policy,
                dialogue_timeline_visual,
                dialogue_task_status,
                dialogue_report,
                voice_table,
                delete_voice_select,
                single_voice_select,
            ],
            queue=False,
        )
        dialogue_preview.select(
            select_timeline_row_event,
            inputs=dialogue_preview,
            outputs=[retry_dialogue_line_number, dialogue_task_status],
            queue=False,
        )
        dialogue_generation_event = generate_dialogue_button.click(
            generate_dialogue_event,
            inputs=[*dialogue_common_inputs, new_dialogue_task_state, no_force_line_state],
            outputs=dialogue_outputs,
            concurrency_limit=1,
        )
        resume_dialogue_event = resume_dialogue_task_button.click(
            generate_dialogue_event,
            inputs=[*dialogue_common_inputs, dialogue_task_select, no_force_line_state],
            outputs=dialogue_outputs,
            concurrency_limit=1,
        )
        retry_dialogue_event = retry_dialogue_line_button.click(
            generate_dialogue_event,
            inputs=[*dialogue_common_inputs, dialogue_task_select, retry_dialogue_line_number],
            outputs=dialogue_outputs,
            concurrency_limit=1,
        )
        rebuild_dialogue_event = rebuild_dialogue_timeline_button.click(
            rebuild_dialogue_timeline_event,
            inputs=[
                dialogue_task_select,
                dialogue_preview,
                timeline_policy,
                batch_gap_ms,
                subtitle_timing_mode,
                subtitle_text_mode,
                subtitle_include_role,
            ],
            outputs=dialogue_outputs,
            concurrency_limit=1,
        )
        stop_dialogue_button.click(
            fn=None,
            cancels=[dialogue_generation_event, resume_dialogue_event, retry_dialogue_event, rebuild_dialogue_event],
            queue=False,
        )

    return demo


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    data_dir = Path(args.data_dir).resolve() if args.data_dir else output_dir.parent / "data"
    validate_model_dir(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f">> T8star-Aix desktop model directory: {model_dir}", flush=True)
    print(f">> T8star-Aix desktop output directory: {output_dir}", flush=True)
    print(f">> T8star-Aix desktop data directory: {data_dir}", flush=True)
    print(f">> Official model revision: {OFFICIAL_MODEL_REVISION}", flush=True)
    policy = select_runtime_policy(
        args.bf16,
        args.qwen_emo,
        args.precision,
        args.reference_device,
    )
    if policy["vram_gb"] is not None:
        print(
            f">> CUDA VRAM: {policy['vram_gb']:.1f} GB | "
            f"low_vram={policy['low_vram']} | precision={policy['precision']} | "
            f"reference={policy['reference_device']} | qwen_emo={policy['use_qwen_emo']}",
            flush=True,
        )
    if args.bf16 and not policy["use_bf16"]:
        print(
            f">> Native BF16 is unavailable; using {policy['precision']}.",
            flush=True,
        )
    acceleration_device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    capabilities = probe_acceleration(acceleration_device)
    acceleration = resolve_acceleration(args.acceleration, acceleration_device, capabilities)
    print(
        f">> Optional acceleration: requested={acceleration.requested} "
        f"effective={acceleration.effective} | {acceleration.reason}",
        flush=True,
    )

    def load_tts(selection):
        return IndexTTS2(
            cfg_path=str(model_dir / "config.yaml"),
            model_dir=str(model_dir),
            use_bf16=policy["use_bf16"],
            use_fp16=policy["use_fp16"],
            reference_device=policy["reference_device"],
            reuse_spk_cond_for_emo=args.reuse_spk_cond_for_emo,
            reference_cache_dir=str(data_dir / "reference_condition_cache"),
            reference_cache_namespace=(
                f"{OFFICIAL_MODEL_REVISION}:{policy['precision']}:"
                f"{policy['reference_device']}"
            ),
            use_qwen_emo=policy["use_qwen_emo"],
            **selection.constructor_kwargs(),
        )

    startup_fallback = ""
    try:
        tts = load_tts(acceleration)
    except Exception as exc:
        if acceleration.effective == "off":
            raise
        traceback.print_exc()
        startup_fallback = f"可选加速初始化失败：{type(exc).__name__}: {exc}；已自动回退普通模式。"
        print(">> " + startup_fallback, flush=True)
        acceleration = resolve_acceleration("off", acceleration_device, capabilities)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        tts = load_tts(acceleration)
    diagnostic = format_acceleration_report(acceleration, capabilities)
    if startup_fallback:
        diagnostic += "\n\n" + startup_fallback
    normal_acceleration = resolve_acceleration("off", acceleration_device, capabilities)
    fallback_factory = (
        (lambda: load_tts(normal_acceleration))
        if acceleration.effective != "off"
        else None
    )
    demo = build_app(
        tts,
        output_dir,
        data_dir,
        args.verbose,
        diagnostic,
        fallback_factory=fallback_factory,
        model_factory=lambda: load_tts(acceleration),
    )
    demo.queue(max_size=20, default_concurrency_limit=1)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=False,
        show_error=True,
        quiet=True,
        allowed_paths=[str(output_dir), str(data_dir)],
    )


if __name__ == "__main__":
    main()
