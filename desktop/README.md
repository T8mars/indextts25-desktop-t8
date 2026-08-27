# T8star-Aix · IndexTTS 2.5 Desktop

Windows Electron desktop integration for IndexTTS 2.5. The packaged application includes:

- Electron desktop shell
- CPython 3.10 runtime
- all Python runtime dependencies, including CUDA-enabled PyTorch
- IndexTTS 2.5 inference source and the dedicated desktop WebUI
- official `duration_factor` speaking-speed control and the synthetic-prompt GPT KV-cache correctness fix
- automatic low-VRAM precision/QwenEmotion adaptation
- named full-parameter presets with copied reference audio
- complete 2.5 sampling, segmentation and text-normalization controls
- reproducible seed plus CFM diffusion-step, CFG-strength, and noise-temperature controls
- language-aware automatic segmentation with token/pause preview
- punctuation presets and explicit `<pause=0.5>` / `<pause=500ms>` silence
- one-pass native target duration plus legacy natural, pad-only, and sample-exact modes
- streaming playback with cancellation while speech blocks are generated
- optional clarity, narration, deharsh, warm, and normalize post-processing
- persistent character voice library
- multi-role batch/JSON dialogue and SRT dubbing with timeline reports
- per-line emotion overrides for the same role (text description, eight-dimensional vector, speaker-following, or role-default inheritance)
- crash-safe dialogue task manifests, restart/resume, and selected-line regeneration
- local Whisper ASR proofreading with OpenAI/faster-whisper backends, CER/WER, normalized diffs, and word timestamps
- rewritten SRT export using original timing or the generated audio's actual timeline
- editable millisecond timeline table, visual track preview, and no-inference re-mixing
- bundled FlashAttention 2.8.3, Triton Windows 3.4.0.post21, and DeepSpeed 0.17.5 Windows acceleration wheels with automatic fallback
- explicit auto/BF16/FP16/FP32 selection before model loading, plus native-BF16 fallback detection
- optional CPU placement for Wav2Vec/CAMPPlus reference encoders and fast default-emotion condition reuse
- no-model acceleration preflight with per-mode availability/reason, exact bundled dependency versions, refresh, and JSON diagnostic export

The large model files are intentionally external. On first launch, select a complete IndexTTS 2.5 model directory.
Version 0.16.0 is aligned to code revision `ee40fa7d` and model revision `c39ce5ba`. The launcher keeps the no-model acceleration preflight, and now adds an explicitly triggered real benchmark that sequentially measures supported modes with identical reference audio, text, and seed, then recommends a near-fastest low-complexity mode. It also provides a manual, read-only upstream update check. Single-voice generation can create up to four retained candidates and selects the best using local ASR plus technical waveform checks, or technical checks alone when Whisper is unavailable. Extracted speaker/emotion conditions are cached across model reloads in content-addressed `safetensors` files under the Electron user-data directory, isolated by official model revision, precision, and reference device. No benchmark, update check, model load, download, or acceleration mode starts automatically.
The launcher validates official model file sizes, while the downloader performs full SHA-256 verification.
This release also supports running the portable package from Windows paths containing Chinese characters.
AAC/M4A and other compressed reference audio is decoded by the bundled PyAV runtime, without requiring a system FFmpeg installation. Streaming previews are also encoded by bundled PyAV, so Gradio no longer calls an external `ffmpeg` or `ffprobe` executable and cannot fail with `[WinError 2]` merely because those programs are absent from `PATH`.
The setup screen and Gradio workspace now adapt from compact windows to full-screen desktop layouts.
The pronunciation workspace supports inline Chinese Pinyin, English CMU phonemes, Japanese kana,
an editable persistent dictionary, preview/validation, search, and YAML/JSON import/export. The dictionary
is stored under Electron's per-user data directory and is never written into the external model directory.

The visible `角色音色库` tab copies named voice and emotion-reference audio into the same user-data directory. Each role
can independently use speaker-following, emotion-reference audio, an eight-dimensional emotion vector, or Qwen emotion
text. Existing roles can be loaded back for auditioning, editing, overwriting, or renaming without changing their library ID. The
`多角色 / 批量台词 / SRT` tab accepts `角色|台词|语言|时长系数|逐句情感`, JSON arrays, or SRT with `[角色] 台词`
and `角色：台词` markers. The optional fifth column supports `text:生气、激动` or
`vector:喜,怒,哀,惧,厌恶,低落,惊喜,平静`; an empty value inherits the saved role emotion. SRT can use
`[角色|emotion=text:平静、从容] 台词`. This lets consecutive lines keep one role and one cloned voice while changing
emotion independently. It exports a combined WAV plus a ZIP containing per-line WAV files and `report.json`.
The UI lists all five language codes and makes clear that the supported 0.5–2.0 duration factor is a unitless
duration multiplier, not a maximum number of seconds. Copyable batch and SRT examples can be loaded with one click.
SRT slot finishing now defaults to safe padding: short audio is padded, while overlong speech is preserved so dialogue
is not silently cut off. One-pass native length regulation, natural overrun, legacy retry, and hard-trim modes remain
available as explicit choices with truncation warnings.

The generation page defaults to language-aware segmentation: EN/ES 60, AR 80, JA 100, and ZH 120 tokens.
The preview table shows speech-block IDs, planned token counts, and external silence before synthesis. Punctuation pause
presets perform real chunked inference and silence insertion; explicit pause tags work even when the preset is off.
Target duration can use the native length regulator in one pass. The generation page streams completed model chunks to
an autoplay preview while separately saving the final WAV; legacy two-pass duration modes wait for the final result to
avoid playing an obsolete first pass. Audio post-processing is optional and `off` preserves the unprocessed waveform.

The dialogue workspace writes `task.json` after every completed line. Interrupted tasks can be resumed after restarting
the application, and any selected line can be regenerated without repeating the completed lines before recombining the
timeline and ZIP archive. The preview table is editable: start/end values are milliseconds, and clicking the timeline
refresh button validates and redraws the tracks. After generation, the edited timeline can be re-mixed without running
IndexTTS again.

## ASR proofreading and subtitle rewrite

ASR runs locally with bundled OpenAI Whisper; an optional faster-whisper backend can be installed separately. It can
proofread a single generated file or every line in a dialogue task. Chinese/Japanese use CER, while English/Spanish/
Arabic use WER. Reports include simplified/traditional Chinese and number normalization, exact differences, word-level
timestamps, similarity, threshold, and pass/fail status. The decoded waveform is passed directly to ASR, so this feature
does not require a system FFmpeg.

The first use downloads the selected `tiny / base / small / medium / turbo` ASR model into Electron's per-user
`data/asr_models` directory. `tiny` is fastest for a quick check; larger models are more accurate but use more disk,
memory, and GPU resources. ASR can be disabled without affecting normal IndexTTS generation.

Dialogue generation also exports `rewritten.srt`. Its timing can preserve the source SRT or use the actual mixed audio
timeline. Text can keep the original, replace every successfully recognized line, or replace only lines whose
similarity passes the selected threshold. Role prefixes are optional.

CFM defaults remain the official `25 steps / 0.7 CFG / 1.0 temperature`. Advanced controls are experimental; fixed seed
plus one-variable-at-a-time comparisons are recommended.

## Optional acceleration

The launcher defaults to `off`. `auto_safe`, BigVGAN CUDA, Torch Compile, GPT acceleration, and DeepSpeed
are opt-in. Missing dependencies, initialization failures, or first-inference failures release the accelerated model,
reload normal mode once, and retry automatically instead of blocking startup or generation.
The Windows desktop package bundles ABI-matched wheels for Python 3.10 and torch 2.8.0+cu128. Their exact URLs and
SHA-256 values are recorded in `desktop_acceleration_manifest.json`; they are optional at runtime and never enabled
without a user selection. GPT acceleration is temporarily bypassed
when the selected sampling parameters cannot be represented without changing generation semantics. It is also bypassed
for long-text or multi-pause-block requests that could hit the upstream synthetic-prompt KV-cache edge case.

## Chinese polyphone quick start

The pronunciation section is expanded by default and includes an in-app Chinese guide plus a
one-click example. The shortest workflow is to write an inline annotation directly in the target text:

```text
小明<要求|YAO4 QIU2>这个题的答案是多少。今天的<行程|XING2 CHENG2>顺利。
```

Use the form `<original text|reading>`. Chinese Pinyin uses one tone-numbered syllable per Han character,
separated by spaces. Annotate the complete word when a polyphonic character is inside a word: use
`<要求|YAO4 QIU2>` rather than `<要|YAO4>求` (the latter is the unreliable form reported in upstream issue #792).
Repeated names and polyphones can instead be stored in the persistent pronunciation table.

## Development

```powershell
cd desktop
npm install
npm start
```

In development mode, the existing project `checkpoints` directory may be preselected. The launcher still waits for the
user to choose an acceleration profile and press the start button before loading the model.

The model directory can also be supplied for managed deployments:

```powershell
T8star-Aix-IndexTTS-2.5.exe --model-dir "D:\Models\IndexTTS-2.5"
```

or through the `T8STAR_INDEXTTS_MODEL_DIR` environment variable.

## Package

```powershell
cd desktop
npm run package
npm run verify:runtime
```

Electron Forge copies the managed CPython runtime and `.venv/Lib/site-packages` into the packaged application's `resources` directory. It does not copy `checkpoints`.
The post-package hook removes legacy 2.0 entrypoints and an explicit list of unused PyTorch static development archives.
It retains CUDA DLLs, PyTorch headers, and import libraries needed by optional BigVGAN/C++ extension compilation, writes
`runtime-prune-report.json`, and is checked by `npm run verify:runtime`.

## Distribution

Use the verified portable ZIP as the default distribution:

```powershell
npm run make
```

This builds only `@electron-forge/maker-zip`. The unpacked application is still
available under `desktop/out/T8star-Aix-IndexTTS-2.5-v0.16.0-win32-x64` for local testing.
The bundled runtime contains tens of thousands of small files, so Squirrel/NuGet
can spend a long time repeatedly rewriting a multi-gigabyte package. It is not the
recommended user distribution. If an installer is specifically required, build it
explicitly after the portable package has passed `npm run verify:runtime`:

```powershell
npm run make:installer
```

Neither distribution includes `checkpoints`; users select or download the complete
IndexTTS 2.5 model directory from the launcher.
