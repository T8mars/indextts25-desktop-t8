const fs = require("node:fs");
const path = require("node:path");
const { spawnSync } = require("node:child_process");
const asar = require("@electron/asar");

const desktopRoot = path.resolve(__dirname, "..");
const projectRoot = path.resolve(desktopRoot, "..");
const sourcePackage = JSON.parse(fs.readFileSync(path.join(desktopRoot, "package.json"), "utf8"));
const packagedRoot = path.join(
  desktopRoot,
  "out",
  `T8star-Aix-IndexTTS-2.5-v${sourcePackage.version}-win32-x64`,
  "resources"
);

if (!fs.existsSync(packagedRoot)) {
  console.error(`Packaged resources do not exist: ${packagedRoot}`);
  process.exit(1);
}

const appAsarPath = path.join(packagedRoot, "app.asar");
if (!fs.existsSync(appAsarPath)) {
  console.error("Packaged Electron app.asar is missing.");
  process.exit(1);
}
const packagedMainSource = asar.extractFile(appAsarPath, "src/main.js").toString("utf8");
const packagedHtmlSource = asar.extractFile(appAsarPath, "src/index.html").toString("utf8");
const packagedPreloadSource = asar.extractFile(appAsarPath, "src/preload.js").toString("utf8");
const packagedRendererSource = asar.extractFile(appAsarPath, "src/renderer.js").toString("utf8");
const packagedDiagnosticSource = asar.extractFile(appAsarPath, "src/diagnostic_report.js").toString("utf8");
const packagedProfileSource = asar.extractFile(appAsarPath, "src/runtime_profiles.js").toString("utf8");
const packagedUpdateSource = asar.extractFile(appAsarPath, "src/update_manager.js").toString("utf8");
const packagedPackage = JSON.parse(asar.extractFile(appAsarPath, "package.json").toString("utf8"));
if (packagedPackage.version !== sourcePackage.version) {
  console.error(`Packaged Desktop version drift: ${packagedPackage.version} != ${sourcePackage.version}`);
  process.exit(1);
}
if (packagedPackage.dependencies?.yauzl !== "3.4.0") {
  console.error("Packaged desktop updater is missing the pinned yauzl runtime dependency.");
  process.exit(1);
}
if (
  !packagedMainSource.includes("probeRuntimeHardware") ||
  !packagedMainSource.includes('ipcMain.handle("desktop:apply-runtime-profile"') ||
  !packagedMainSource.includes('ipcMain.handle("desktop:refresh-diagnostics"') ||
  !packagedMainSource.includes('ipcMain.handle("desktop:export-diagnostics"') ||
  !packagedMainSource.includes('ipcMain.handle("desktop:run-runtime-benchmark"') ||
  !packagedMainSource.includes('ipcMain.handle("desktop:check-updates"') ||
  !packagedMainSource.includes('ipcMain.handle("desktop:download-update"') ||
  !packagedMainSource.includes('ipcMain.handle("desktop:install-update"') ||
  !packagedMainSource.includes("checkForUpdates") ||
  !packagedMainSource.includes("markUpdateHealthyIfRequested") ||
  !packagedMainSource.includes("MODEL_DOWNLOAD_PROGRESS_PREFIX") ||
  !packagedMainSource.includes("attachModelDownloadOutput") ||
  !packagedMainSource.includes("Hardware probe only (model not loaded)") ||
  !packagedPreloadSource.includes("refreshDiagnostics") ||
  !packagedPreloadSource.includes("exportDiagnostics") ||
  !packagedPreloadSource.includes("runRuntimeBenchmark") ||
  !packagedPreloadSource.includes("checkUpdates") ||
  !packagedPreloadSource.includes("downloadUpdate") ||
  !packagedPreloadSource.includes("installUpdate") ||
  !packagedRendererSource.includes("renderAccelerationDiagnostics") ||
  !packagedRendererSource.includes("renderBenchmark") ||
  !packagedRendererSource.includes("renderUpdateReport") ||
  !packagedRendererSource.includes("renderModelDownload") ||
  !packagedDiagnosticSource.includes("createDiagnosticReport") ||
  !packagedDiagnosticSource.includes("aio.lib/cufile.lib") ||
  !packagedHtmlSource.includes('id="runtimeProfile"') ||
  !packagedHtmlSource.includes('id="applyRuntimeProfileButton"') ||
  !packagedHtmlSource.includes('id="hardwareSummary"') ||
  !packagedHtmlSource.includes('id="refreshDiagnosticsButton"') ||
  !packagedHtmlSource.includes('id="exportDiagnosticsButton"') ||
  !packagedHtmlSource.includes('id="diagnosticsGrid"') ||
  !packagedHtmlSource.includes('id="runBenchmarkButton"') ||
  !packagedHtmlSource.includes('id="checkUpdatesButton"') ||
  !packagedHtmlSource.includes('id="downloadUpdateButton"') ||
  !packagedHtmlSource.includes('id="installUpdateButton"') ||
  !packagedHtmlSource.includes('id="modelDownloadPanel"') ||
  !packagedHtmlSource.includes('id="modelDownloadProgress"') ||
  !packagedUpdateSource.includes("verifyManifestSignature") ||
  !packagedUpdateSource.includes("verifyPayloadFiles") ||
  !["low_vram", "balanced", "max_speed", "compatibility"].every((name) =>
    packagedProfileSource.includes(`${name}:`)
  )
) {
  console.error("Packaged launcher is missing manual-start hardware profiles.");
  process.exit(1);
}

const pythonFolder = fs.readdirSync(packagedRoot).find((entry) => {
  return entry.startsWith("cpython-") && fs.existsSync(path.join(packagedRoot, entry, "python.exe"));
});
if (!pythonFolder) {
  console.error("Bundled Python runtime was not found in packaged resources.");
  process.exit(1);
}

const pythonExe = path.join(packagedRoot, pythonFolder, "python.exe");
const sitePackages = path.join(packagedRoot, "site-packages");
const manifestPath = path.join(packagedRoot, "desktop_model_manifest.json");
if (!fs.existsSync(manifestPath)) {
  console.error("Packaged model manifest is missing.");
  process.exit(1);
}
const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
if (manifest.codeRevision !== "ee40fa7d6c6b8a2c7f06105f9f1e65775b74868c") {
  console.error(`Unexpected packaged code revision: ${manifest.codeRevision}`);
  process.exit(1);
}
if (
  manifest.modelRepository !== "t8star/IndexTTS-2.5-Comfy" ||
  manifest.modelRevision !== "14166a7401f9f87f53770a1784390e8c0e9da15a" ||
  manifest.files?.["bpe.model"]?.modelScopeRepository !== "IndexTeam/IndexTTS-2" ||
  manifest.files?.["bpe.model"]?.sha256 !==
    "b2a5ce8090d32da3642cc4f81fdc996376bc6dd3f4cd5e3d165f71120d9f2bc8"
) {
  console.error("Packaged model manifest is missing the complete pinned Hugging Face mirror or ModelScope fallback.");
  process.exit(1);
}
if (!fs.existsSync(path.join(packagedRoot, "portable-update-helper.ps1"))) {
  console.error("Packaged portable update helper is missing.");
  process.exit(1);
}
const accelerationManifestPath = path.join(packagedRoot, "desktop_acceleration_manifest.json");
if (!fs.existsSync(accelerationManifestPath)) {
  console.error("Packaged acceleration manifest is missing.");
  process.exit(1);
}

const moduleRoot = path.join(packagedRoot, "indextts");
for (const legacyFile of ["infer.py", "infer_v2.py", "cli.py", "cli_v2.py"]) {
  if (fs.existsSync(path.join(moduleRoot, legacyFile))) {
    console.error(`Legacy entrypoint should not be packaged: ${legacyFile}`);
    process.exit(1);
  }
}

const pruneReportPath = path.join(packagedRoot, "runtime-prune-report.json");
if (!fs.existsSync(pruneReportPath)) {
  console.error("Packaged runtime pruning report is missing.");
  process.exit(1);
}
const pruneReport = JSON.parse(fs.readFileSync(pruneReportPath, "utf8"));
const torchRoot = path.join(sitePackages, "torch");
const torchLib = path.join(torchRoot, "lib");
for (const filename of pruneReport.removed.map((item) => item.file)) {
  if (fs.existsSync(path.join(torchLib, filename))) {
    console.error(`Pruned PyTorch archive unexpectedly remains: ${filename}`);
    process.exit(1);
  }
}
for (const filename of pruneReport.preservedForExtensions) {
  if (!fs.existsSync(path.join(torchLib, filename))) {
    console.error(`PyTorch extension import library is missing: ${filename}`);
    process.exit(1);
  }
}
if (!fs.existsSync(path.join(torchRoot, "include", "torch", "extension.h"))) {
  console.error("PyTorch extension headers are missing.");
  process.exit(1);
}
if (Number(pruneReport.savedBytes || 0) < 2 * 1024 ** 3) {
  console.error(`Runtime pruning saved too little space: ${pruneReport.savedBytes || 0} bytes`);
  process.exit(1);
}

const inferSource = fs.readFileSync(path.join(moduleRoot, "infer_v2_5.py"), "utf8");
const speakerEmbeddingCalls = inferSource.match(
  /self\.get_emb\(input_features,\s*attention_mask\)/g
) || [];
const speakerAudioLoads = inferSource.match(
  /self\._load_and_cut_audio\(spk_audio_prompt,\s*15,\s*verbose\)/g
) || [];
if (
  !inferSource.includes("duration_factor=1.0") ||
  !inferSource.includes("target_duration=None") ||
  !inferSource.includes("use_fp16=False") ||
  !inferSource.includes("reference_device=None") ||
  !inferSource.includes("reuse_spk_cond_for_emo=False") ||
  !inferSource.includes("reference_cache_dir=None") ||
  !inferSource.includes("ReferenceConditionCache") ||
  !inferSource.includes("resolve_gpt_precision") ||
  !inferSource.includes("diffusion_steps=25") ||
  !inferSource.includes("cfm_temperature=1.0") ||
  !inferSource.includes("kv_cache=True") ||
  !inferSource.includes("save_pcm_wav") ||
  !inferSource.includes("normalize_content") ||
  inferSource.includes("self.use_gpt_latent")
) {
  console.error("Packaged IndexTTS 2.5 source is missing an expected upstream runtime update.");
  process.exit(1);
}
if (speakerEmbeddingCalls.length !== 1 || speakerAudioLoads.length !== 1) {
  console.error(
    `Packaged speaker-reference preprocessing is duplicated: embeddings=${speakerEmbeddingCalls.length}, loads=${speakerAudioLoads.length}.`
  );
  process.exit(1);
}

const accelGuardSource = fs.readFileSync(path.join(moduleRoot, "accel_cache_guard.py"), "utf8");
const accelEngineSource = fs.readFileSync(path.join(moduleRoot, "accel", "accel_engine.py"), "utf8");
if (
  !accelGuardSource.includes("sequence.num_cached_tokens = 0") ||
  !accelEngineSource.includes("reset_synthetic_prompt_cache_markers(sequences, tts_embeddings)")
) {
  console.error("Packaged GPT acceleration source is missing the synthetic-prompt KV-cache guard.");
  process.exit(1);
}

const modelDownloadSource = fs.readFileSync(path.join(moduleRoot, "utils", "model_download.py"), "utf8");
if (!modelDownloadSource.includes("_VERSION_TO_REPO") || !modelDownloadSource.includes("ensure_config_available")) {
  console.error("Packaged model download helper is not the synchronized IndexTTS 2.5 version.");
  process.exit(1);
}

const desktopSource = fs.readFileSync(path.join(packagedRoot, "desktop_webui.py"), "utf8");
const voiceLibrarySource = fs.readFileSync(path.join(packagedRoot, "desktop_voice_library.py"), "utf8");
for (const moduleName of ["desktop_presets.py", "desktop_voice_library.py", "desktop_generation_controls.py", "desktop_model_lifecycle.py", "desktop_streaming_audio.py", "desktop_tasks.py", "desktop_project_bundle.py", "desktop_runtime_benchmark.py", "audio_quality.py", "audiocpp_backend.py", "audiocpp_component_manager.py", "candidate_quality.py", "speech_review.py", "timeline_tools.py", "context_emotion.py", "dialogue_runtime.py", "runtime_acceleration.py", "runtime_benchmark.py", "runtime_metrics.py"]) {
  if (!fs.existsSync(path.join(packagedRoot, moduleName))) {
    console.error(`Packaged desktop runtime module is missing: ${moduleName}`);
    process.exit(1);
  }
}
if (
  !desktopSource.includes("--data_dir") ||
  !desktopSource.includes("多音字怎么用") ||
  !desktopSource.includes("多音字使用方法与发音设置（默认展开）") ||
  !desktopSource.includes("open=True") ||
  !desktopSource.includes("完整参数预设（含参考音频）") ||
  !desktopSource.includes("段间静音（毫秒）") ||
  !desktopSource.includes("目标时长（秒）") ||
  !desktopSource.includes("标点停顿预设") ||
  !desktopSource.includes("可选音频后处理") ||
  !desktopSource.includes("边生成边试听") ||
  !desktopSource.includes("CFM 扩散步数") ||
  !desktopSource.includes("原生单次适配") ||
  !desktopSource.includes("任务恢复与单句重试") ||
  !desktopSource.includes("ASR 自动校对当前结果") ||
  !desktopSource.includes("ASR 自动校对与字幕自动回写") ||
  !desktopSource.includes("可编辑时间轴") ||
  !desktopSource.includes("t8-timeline-drag-payload") ||
  !desktopSource.includes("按住 Alt 可临时关闭吸附") ||
  !desktopSource.includes("按编辑时间轴重新混音") ||
  !desktopSource.includes("角色音色库") ||
  !desktopSource.includes("使用已保存音色库（免重复上传）") ||
  !desktopSource.includes("load_single_voice_event") ||
  !desktopSource.includes("该角色默认情感模式") ||
  !desktopSource.includes("profile_emotion_kwargs") ||
  !desktopSource.includes("line_emotion_kwargs") ||
  !desktopSource.includes("BundledStreamingAudio") ||
  !desktopSource.includes("逐句情感") ||
  !desktopSource.includes("多角色 / 批量台词 / SRT") ||
  !desktopSource.includes("环境与可选加速") ||
  !desktopSource.includes("参考音频质量检测与自动裁剪") ||
  !desktopSource.includes("追加候选数量") ||
  !desktopSource.includes("全部候选音频") ||
  !desktopSource.includes("模型与显存生命周期") ||
  !desktopSource.includes("参考条件缓存管理") ||
  !desktopSource.includes("run_with_long_text_guard") ||
  !desktopSource.includes("实验 audio.cpp 后端") ||
  !desktopSource.includes("select_runtime_policy") ||
  !desktopSource.includes("--precision") ||
  !desktopSource.includes("--reference-device") ||
  !desktopSource.includes("--reuse-spk-cond-for-emo") ||
  !desktopSource.includes("start_runtime_measurement") ||
  !desktopSource.includes("finish_runtime_measurement") ||
  !desktopSource.includes("CUDA 分配峰值") ||
  !voiceLibrarySource.includes("emotion_audio_path") ||
  !voiceLibrarySource.includes("emotion_vector") ||
  !voiceLibrarySource.includes("emotion_use_random")
) {
  console.error("Packaged desktop WebUI is missing pronunciation, role-emotion, preset, advanced, or VRAM controls.");
  process.exit(1);
}
const packagedDesktopVersion = desktopSource.match(/^DESKTOP_VERSION\s*=\s*"([^"]+)"/m)?.[1];
if (packagedDesktopVersion !== packagedPackage.version) {
  console.error(`Packaged WebUI version drift: ${packagedDesktopVersion} != ${packagedPackage.version}`);
  process.exit(1);
}

const check = spawnSync(pythonExe, [
  "-c",
  [
    "import torch, gradio, transformers, flash_attn, triton, deepspeed",
    "from indextts.infer_v2_5 import IndexTTS2",
    "from indextts.pronunciation import PronunciationEntry, process_pronunciation_text",
    "from desktop_generation_controls import DesktopGenerationPlan, DesktopSpeechChunk, allocate_native_chunk_durations, effective_segment_limit, split_speech_chunks",
    "from desktop_model_lifecycle import DesktopModelLifecycle",
    "from audio_quality import analyze_reference_audio, waveform_html",
    "from audiocpp_backend import build_audiocpp_command",
    "from desktop_tasks import task_choices",
    "from speech_review import ASR_BACKENDS, asr_available, review_transcript",
    "from timeline_tools import rewrite_srt",
    "from context_emotion import suggest_context_emotions",
    "from dialogue_runtime import DialogueLine",
    "from runtime_metrics import finish_runtime_measurement, format_runtime_metrics, start_runtime_measurement",
    "from runtime_acceleration import probe_acceleration",
    "from runtime_benchmark import recommend_benchmark_mode",
    "from candidate_quality import combined_candidate_score",
    "result = process_pronunciation_text('银行的行长', 'ZH', [PronunciationEntry('银行', 'YIN2 HANG2', 'ZH'), PronunciationEntry('行长', 'HANG2 ZHANG3', 'ZH')], strict=True)",
    "assert result.text == '<银行|YIN2 HANG2>的<行长|HANG2 ZHANG3>'",
    "assert effective_segment_limit('EN', 'auto', 120) == 60",
    "assert split_speech_chunks('第一句<pause=0.5>第二句', 'off', 0, 0, 0)[0].pause_after_ms == 500",
    "assert recommend_benchmark_mode([{'status': 'ok', 'requested_mode': 'off', 'effective_mode': 'off', 'rtf': 1.0}])['mode'] == 'off'",
    "assert combined_candidate_score(0.8, None) == 0.8",
    "duration_plan = DesktopGenerationPlan('EN', 60, (DesktopSpeechChunk('a'), DesktopSpeechChunk('bbb')), (), 'off')",
    "assert allocate_native_chunk_durations(duration_plan, 8.0) == (2.0, 6.0)",
    "assert task_choices('不存在的任务目录') == []",
    "assert asr_available()",
    "assert ASR_BACKENDS == ('auto', 'openai_whisper', 'faster_whisper')",
    "assert review_transcript('第二十五條臺詞', '第25条台词', 'ZH', 0.99)['passed']",
    "assert review_transcript('one small test', 'one test', 'EN', 0.5)['metric'] == 'wer'",
    "srt, _ = rewrite_srt([DialogueLine(1, '旁白', '测试', 'ZH', 0, 1000)], [], timing_mode='original')",
    "assert '00:00:00,000 --> 00:00:01,000' in srt",
    "measurement = start_runtime_measurement()",
    "performance = finish_runtime_measurement(measurement, 2.0)",
    "assert 'RTF' in format_runtime_metrics(performance)",
    "preflight = probe_acceleration('cpu')",
    "assert preflight['versions']['torch'] == str(torch.__version__)",
    "assert set(preflight['versions']) == {'torch', 'cuda_runtime', 'deepspeed', 'flash_attn', 'triton', 'ninja'}",
    "assert torch.__version__ == '2.8.0+cu128'",
    "print(torch.__version__, gradio.__version__, transformers.__version__, flash_attn.__version__, triton.__version__, deepspeed.__version__, 'pronunciation=OK', 'acceleration=OK', 'generation_controls=OK', 'tasks=OK', 'asr=OK', 'timeline=OK')"
  ].join("; ")
], {
  cwd: packagedRoot,
  encoding: "utf8",
  env: {
    ...process.env,
    PYTHONUTF8: "1",
    PYTHONPATH: [packagedRoot, sitePackages].join(path.delimiter)
  }
});

if (check.status !== 0) {
  console.error(check.stdout);
  console.error(check.stderr);
  process.exit(check.status || 1);
}

console.log(`Packaged Python runtime OK: ${check.stdout.trim()}`);
console.log(`Packaged runtime pruning OK: ${pruneReport.savedGiB} GiB saved; extension libraries preserved`);
console.log(`Packaged code/model baseline: ${manifest.codeRevision.slice(0, 8)} / ${manifest.modelRevision.slice(0, 8)}`);
console.log(`Project source used for packaging: ${projectRoot}`);

const systemRoot = process.env.SystemRoot || "C:\\Windows";
const aacCheck = spawnSync(pythonExe, [path.join(__dirname, "verify-aac-runtime.py")], {
  cwd: packagedRoot,
  encoding: "utf8",
  env: {
    ...process.env,
    PATH: [path.dirname(pythonExe), sitePackages, path.join(systemRoot, "System32"), systemRoot].join(path.delimiter),
    PYTHONUTF8: "1",
    PYTHONPATH: [packagedRoot, sitePackages].join(path.delimiter)
  }
});

if (aacCheck.status !== 0) {
  console.error(aacCheck.stdout);
  console.error(aacCheck.stderr);
  process.exit(aacCheck.status || 1);
}

console.log(aacCheck.stdout.trim());

const unicodeTestRoot = path.join(desktopRoot, "out", `中文路径回归测试-${process.pid}`);
try {
  fs.mkdirSync(unicodeTestRoot, { recursive: true });
  fs.cpSync(path.join(sitePackages, "wetext"), path.join(unicodeTestRoot, "wetext"), {
    recursive: true
  });

  const unicodeCheck = spawnSync(pythonExe, [
    "-c",
    [
      "from indextts.utils.front import TextNormalizer",
      "normalizer = TextNormalizer()",
      "normalizer.load()",
      "assert normalizer.normalize('123 dollars') == 'one hundred and twenty three dollars'",
      "print('Unicode install path normalization OK')"
    ].join("; ")
  ], {
    cwd: packagedRoot,
    encoding: "utf8",
    env: {
      ...process.env,
      PYTHONUTF8: "1",
      PYTHONPATH: [unicodeTestRoot, packagedRoot, sitePackages].join(path.delimiter)
    }
  });

  if (unicodeCheck.status !== 0) {
    console.error(unicodeCheck.stdout);
    console.error(unicodeCheck.stderr);
    process.exit(unicodeCheck.status || 1);
  }

  console.log(unicodeCheck.stdout.trim());
} finally {
  fs.rmSync(unicodeTestRoot, { recursive: true, force: true });
}
