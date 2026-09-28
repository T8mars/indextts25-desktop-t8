from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from desktop_webui import CONFUCIUS_LIVE_HTML, CSS


ROOT = Path(__file__).resolve().parents[1]


def test_live_caption_actions_are_aligned_and_include_safe_clear():
    assert 'id="confucius-live-copy"' in CONFUCIUS_LIVE_HTML
    assert 'id="confucius-live-download"' in CONFUCIUS_LIVE_HTML
    assert 'id="confucius-live-clear"' in CONFUCIUS_LIVE_HTML
    assert 'id="confucius-file-input"' in CONFUCIUS_LIVE_HTML
    assert 'id="confucius-file-transcribe"' in CONFUCIUS_LIVE_HTML
    assert "清理字幕" in CONFUCIUS_LIVE_HTML
    assert "不写入临时目录、不外传" in CONFUCIUS_LIVE_HTML
    assert "阿拉伯语自动安全回退本地 Whisper" in CONFUCIUS_LIVE_HTML
    assert "内容背景（可选）" in CONFUCIUS_LIVE_HTML
    assert "帮助模型理解场景，不是待转写文字" in CONFUCIUS_LIVE_HTML
    assert "重点词 / 专有名词（可选）" in CONFUCIUS_LIVE_HTML
    assert "适合人名、产品名和术语；中文、英文逗号都可以" in CONFUCIUS_LIVE_HTML
    assert ".t8-live-help" in CSS
    assert ".t8-live-actions { align-items: stretch !important; }" in CSS
    assert "height: 44px; min-height: 44px" in CSS
    assert "display: inline-flex; align-items: center" in CSS


def test_clear_caption_keeps_session_and_hides_cleared_prefix(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the live-caption browser-state test")
    harness = tmp_path / "confucius-live-client-test.cjs"
    source = (ROOT / "confucius_live_client.js").read_text(encoding="utf-8")
    harness.write_text(
        """
const vm = require("node:vm");
const elements = new Map();
for (const id of ["confucius-live-status", "confucius-live-stable", "confucius-live-preview"]) {
  elements.set(id, {id, textContent: "", dataset: {}, addEventListener() {}});
}
const context = {
  document: {
    documentElement: {},
    getElementById(id) { return elements.get(id) || null; },
    createElement() { return {}; },
  },
  MutationObserver: class { observe() {} },
  window: {addEventListener() {}},
  location: {protocol: "http:", host: "127.0.0.1:7860"},
  console,
  setTimeout,
  clearTimeout,
};
vm.createContext(context);
vm.runInContext(process.env.T8_CLIENT_SOURCE, context);
vm.runInContext(`
  active = {
    rawStable: "", rawPreview: "", clearedStable: "", clearedPreview: "",
    stopping: false,
  };
  writeSessionCaption(active, "旧字幕", "临时字幕");
  clearTranscript();
  if (document.getElementById("confucius-live-stable").textContent !== "") throw new Error("stable not cleared");
  if (document.getElementById("confucius-live-preview").textContent !== "") throw new Error("preview not cleared");
  if (!document.getElementById("confucius-live-status").textContent.includes("继续")) throw new Error("session was not preserved");
  writeSessionCaption(active, "旧字幕 新字幕", "临时字幕 新预览");
  if (document.getElementById("confucius-live-stable").textContent !== "新字幕") throw new Error("cleared stable prefix returned");
  if (document.getElementById("confucius-live-preview").textContent !== "新预览") throw new Error("cleared preview prefix returned");
`, context);
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [node, str(harness)],
        env={**os.environ, "T8_CLIENT_SOURCE": source},
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout

@pytest.mark.parametrize("input_rate", [44_100, 48_000])
def test_audio_worklet_flushes_exact_finite_16khz_pcm(tmp_path, input_rate):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the AudioWorklet resampler test")
    harness = tmp_path / f"confucius-worklet-{input_rate}.cjs"
    source = (ROOT / "confucius_live_worklet.js").read_text(encoding="utf-8")
    harness.write_text(
        """
const vm = require("node:vm");
let Processor = null;
const emitted = [];
class AudioWorkletProcessor {
  constructor() {
    this.port = {
      onmessage: null,
      postMessage(value) {
        if (value.type === "pcm") emitted.push(new Float32Array(value.samples));
      },
    };
  }
}
const context = {
  AudioWorkletProcessor,
  Float32Array,
  Math,
  sampleRate: Number(process.env.T8_INPUT_RATE),
  registerProcessor(_name, value) { Processor = value; },
};
vm.createContext(context);
vm.runInContext(process.env.T8_WORKLET_SOURCE, context);
if (!Processor) throw new Error("worklet was not registered");
const processor = new Processor();
const total = Number(process.env.T8_INPUT_RATE);
for (let offset = 0; offset < total; offset += 128) {
  const count = Math.min(128, total - offset);
  processor.process([[new Float32Array(count).fill(0.25)]]);
}
processor.port.onmessage({data: {type: "flush"}});
const samples = emitted.reduce((sum, item) => sum + item.length, 0);
if (samples !== 16000) throw new Error(`expected 16000 samples, got ${samples}`);
for (const chunk of emitted) {
  for (const value of chunk) {
    if (!Number.isFinite(value) || Math.abs(value) > 1.01) throw new Error("invalid PCM sample");
  }
}
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [node, str(harness)],
        env={
            **os.environ,
            "T8_WORKLET_SOURCE": source,
            "T8_INPUT_RATE": str(input_rate),
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout

