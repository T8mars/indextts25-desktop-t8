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
- generation preflight with inspectable normalized text, estimated duration, token pressure, risk flags, and explicit apply-to-source
- long English/Spanish output guards that retry a smaller segment when mel-token exhaustion or an implausible duration is detected
- punctuation presets and explicit `<pause=0.5>` / `<pause=500ms>` silence
- one-pass native target duration plus legacy natural, pad-only, and sample-exact modes
- streaming playback with cancellation while speech blocks are generated
- optional clarity, narration, deharsh, warm, and normalize post-processing
- persistent character voice library, directly selectable from the single-generation page without re-uploading audio
- searchable Voice Library 2.0 metadata, favorites, quality reports, and portable `.t8voice.zip` import/export shared with ComfyUI
- multi-role batch/JSON dialogue and SRT dubbing with timeline reports
- per-line emotion overrides for the same role (text description, eight-dimensional vector, speaker-following, or role-default inheritance)
- context-aware per-line emotion suggestions that fill the editable timeline without starting synthesis
- crash-safe dialogue task manifests, restart/resume, and selected-line regeneration
- one persistent FIFO queue shared by single-voice, multi-role, and SRT work, with restart recovery, cancellation, and retry
- complete `.indextts-project.zip` export/import containing task state, per-line audio, combined audio, subtitles/reports, and referenced voice bundles
- local Whisper ASR proofreading with OpenAI/faster-whisper backends, CER/WER, normalized diffs, and word timestamps
- rewritten SRT export using original timing or the generated audio's actual timeline
- editable millisecond timeline table plus draggable/resizable tracks, ASR word-boundary snapping, and no-inference re-mixing
- selected-line up/down reordering that moves role/language/text/emotion together while keeping authored SRT time slots chronological
- reference-condition cache statistics and safe clearing for this application's own `safetensors` entries
- automatic reference-audio quality summary plus opt-in detailed analysis/cropping
- persisted model-memory policy with manual release, idle release, generation-count recycling, and lazy reload
- collapsed A/B candidate auditioning with 1–5 star reviews, notes, and durable favorites
- conservative cross-segment speech-rate anomaly detection with selective retry of only a collapsed segment
- visual internal-segment rate audit with original/retry/current audio previews and one-click segment-only regeneration
- deterministic five-language real-model quality regression with optional CER/WER and baseline comparison
- a complete IndexTTS 2.5 CLI for five languages, emotion, duration, sampling, precision, and optional acceleration
- bundled FlashAttention 2.8.3, Triton Windows 3.4.0.post21, and DeepSpeed 0.17.5 Windows acceleration wheels with automatic fallback
- explicit auto/BF16/FP16/FP32 selection before model loading, plus native-BF16 fallback detection
- optional CPU placement for Wav2Vec/CAMPPlus reference encoders and fast default-emotion condition reuse
- no-model acceleration preflight with per-mode availability/reason, exact bundled dependency versions, refresh, and JSON diagnostic export
- signed layered GitHub Release updates for the app and split runtime, with resumable download, per-part/archive/file verification, explicit install confirmation, and automatic rollback
- live model scan/download/verification progress with current file, speed, ETA, conservative disk-space preflight, resumable repair, and precise failure details
- local-only experimental audio.cpp node that accepts user-supplied CLI/GGUF absolute paths and never installs components
- independently configurable output and user-data directories, with saved paths and direct open-folder actions
- contextual fixed-bottom generate/stop controls that remain available while scrolling in single-voice and multi-role workflows
- collapsed-by-default guidance, quality tools, advanced controls, dialogue timing, ASR, timeline, report, and task-recovery workspaces
- direct WebUI actions for returning to setup, opening outputs, opening user data, and opening the exact log directory

The large model files are intentionally external. On first launch, select a complete IndexTTS 2.5 model directory.
Version 0.24.0 is paired with ComfyUI Node 0.23.0 and model bundle `1.0.0` at revision `14166a74`. It adds standalone JSON/CSV editable-timeline transfer, an in-place eight-vector emotion guide, and a compact one-line-per-sentence emotion syntax shared with the node while preserving legacy JSON. Fixed-bottom controls expose live progress, and repeat dialogue generation preserves untimed rows instead of turning them into authored `0/0` slots. Desktop updates remain split into signed app/runtime layers with resume, full verification, health checks, and rollback; models remain independent on Hugging Face. Advanced and engineering workspaces stay collapsed by default. Optional acceleration failures still reload the normal model and complete the task. No benchmark, model load, model download, update download, install, or acceleration mode starts automatically.
The launcher validates official model file sizes, while the downloader performs full SHA-256 verification.
The output directory and user-data directory can be moved independently from the launcher. Voice-library entries,
presets, dialogue tasks, ASR caches, benchmarks, and logs follow the configured user-data directory; generated WAVs
follow the configured output directory. Both choices persist across launches. The active WebUI exposes the resolved
paths and direct folder buttons. Its `返回启动配置（停止模型）` action, or closing the WebUI window, stops the model,
releases its process, and returns to setup instead of forcing the whole application to exit. Closing the setup window
still exits normally.

Optional acceleration fallback is presented as a normal operating state rather than an error: the WebUI shows the
requested mode, the effective mode, and a Chinese explanation first. The complete capability JSON remains available
inside a collapsed troubleshooting section.
This release also supports running the portable package from Windows paths containing Chinese characters.
AAC/M4A and other compressed reference audio is decoded by the bundled PyAV runtime, without requiring a system FFmpeg installation. Streaming previews are also encoded by bundled PyAV, so Gradio no longer calls an external `ffmpeg` or `ffprobe` executable and cannot fail with `[WinError 2]` merely because those programs are absent from `PATH`.
The setup screen and Gradio workspace now adapt from compact windows to full-screen desktop layouts.
The pronunciation workspace supports inline Chinese Pinyin, English CMU phonemes, Japanese kana,
an editable persistent dictionary, preview/validation, search, and YAML/JSON import/export. The dictionary
is stored under Electron's per-user data directory and is never written into the external model directory.

### Development-branch quality audit and CLI

After a non-streaming multi-segment generation, the `跨段语速审计与内部单段重做` panel displays the real
units-per-second value and preceding stable median for every internal segment. It preserves separate original,
automatic-retry, and currently selected WAV artifacts. Selecting a segment and clicking the regeneration button
runs only that segment with a new reproducible seed, rebuilds all speech blocks and pauses, reapplies the selected
duration/post-processing policy, and writes the merged final WAV. The first full result is backed up before any
manual segment replacement.

Run the deterministic multilingual regression against an existing complete model and reference voice:

```powershell
..\.venv\Scripts\python.exe scripts\smoke-multilingual-quality.py `
  --model-dir ..\checkpoints --voice ..\voice.wav `
  --asr-backend auto --output-dir ..\quality-regression --strict
```

The output contains five WAV files and `quality-report.json`. Add `--baseline previous\quality-report.json` to
detect material RTF, CER/WER, clipping, silence, duration, or internal-rate regressions. The runner never downloads
the main model or reference audio.

The repository CLI now uses `indextts.infer_v2_5.IndexTTS2`:

```powershell
..\.venv\Scripts\python.exe -m indextts.cli "A reproducible IndexTTS 2.5 CLI sample." `
  --voice ..\voice.wav --model-dir ..\checkpoints --language EN `
  --precision auto --output-path ..\cli-sample.wav
```

Use `python -m indextts.cli --help` for emotion references/vectors/text, native target duration, sampling/CFM,
reference-device, and acceleration options.

The visible `角色音色库` tab copies named voice and emotion-reference audio into the same user-data directory. The
`语音生成` tab exposes these saved roles in a refreshable dropdown; selecting one reuses its copied timbre audio immediately,
so repeated single-voice generation does not require another upload. Each role
can independently use speaker-following, emotion-reference audio, an eight-dimensional emotion vector, or Qwen emotion
text. Existing roles can be loaded back for auditioning, editing, overwriting, or renaming without changing their library ID. The
`多角色 / 批量台词 / SRT` tab accepts `角色|台词|语言|时长系数|逐句情感`, JSON arrays, or SRT with `[角色] 台词`
and `角色：台词` markers. The optional fifth column supports `text:生气、激动` or
`vector:喜,怒,哀,惧,厌恶,低落,惊喜,平静`; append `;strength=0.75` and, for vector sampling,
`;random=true` when needed. An empty value inherits the saved role emotion. SRT can use
`[角色|emotion=text:平静、从容] 台词`. This lets consecutive lines keep one role and one cloned voice while changing
emotion independently. It exports a combined WAV plus a ZIP containing per-line WAV files and `report.json`.
The UI lists all five language codes and makes clear that the supported 0.5–2.0 duration factor is a unitless
duration multiplier, not a maximum number of seconds. Copyable batch and SRT examples can be loaded with one click.

Voice Library 2.0 adds tag/search/favorite/notes and saved quality metadata. Its `.t8voice.zip` export is portable and
is read directly by the ComfyUI Saved Voice node from `ComfyUI/models/TTS/IndexTTS-2.5/voices/`. Project export uses
`.indextts-project.zip` and includes task state, line WAV files, combined output, subtitle/report files, and the referenced
voice bundles. Imported projects receive a new local task ID and do not overwrite an existing task.
SRT slot finishing now defaults to safe padding: short audio is padded, while overlong speech is preserved so dialogue
is not silently cut off. One-pass native length regulation, natural overrun, legacy retry, and hard-trim modes remain
available as explicit choices with truncation warnings.

The generation page defaults to language-aware segmentation: EN/ES 60, AR 80, JA 100, and ZH 120 tokens.
The preflight panel shows the actual normalized/pronunciation-resolved text, speech-block IDs, planned token counts,
estimated duration, risk flags, and external silence before synthesis. Its text remains editable and only replaces the
main input after an explicit apply action. Punctuation pause
presets perform real chunked inference and silence insertion; explicit pause tags work even when the preset is off.
Target duration can use the native length regulator in one pass. The generation page streams completed model chunks to
an autoplay preview while separately saving the final WAV; legacy two-pass duration modes wait for the final result to
avoid playing an obsolete first pass. Audio post-processing is optional and `off` preserves the unprocessed waveform.

The dialogue workspace writes `task.json` after every completed line. Interrupted tasks can be resumed after restarting
the application, and any selected line can be regenerated without repeating the completed lines before recombining the
timeline and ZIP archive. The preview table is editable and synchronized with the visual tracks: drag a block to move it,
drag either handle to resize it, and release near another line boundary or an ASR word timestamp to snap. Hold Alt while
dragging to bypass snapping. A drag immediately updates the table and selects that line, so text, role, language, timing,
or emotion can be changed and only that line regenerated and merged. The edited timeline can also be re-mixed without
running IndexTTS again. Selecting a table row also enables one-click up/down ordering; content and emotion move as a
unit while source SRT time slots stay in chronological positions.
The compact help beside the table defines all eight vector positions. A collapsed transfer panel exports the current
editable table as lossless JSON or spreadsheet-friendly UTF-8 CSV and imports either format later; this lightweight
timeline file includes role, language, timing, duration factor, text, and per-line emotion, but no model or audio.

The `任务队列` tab persists single-voice, dialogue, and SRT parameter snapshots in `task_queue.json`. A process restart
recovers interrupted work to `pending` without auto-starting inference. Failed/cancelled work can be retried. Optional
generation candidates are kept in a collapsed A/B workspace for audition, star rating, notes, and copying favorites to
`candidate_favorites/`.

### Context-aware per-line emotion suggestions

Desktop 0.17.0 adds an expanded `上下文情感自动标注` section below the editable dialogue table. Choose how
many previous and following lines to include (default: two per side), then click `分析上下文并填入建议`. Local
QwenEmotion evaluates the target line while the prompt keeps previous/target/following roles separate. The result is
written to the table's last column as an editable eight-dimensional vector plus strength. Existing manual `text:` or
`vector:` values are preserved unless `覆盖已有逐句情感` is enabled.

This action never starts TTS. Review, change, or clear every suggestion first; synthesis begins only when the user later
clicks `生成全部台词`. Temporarily loaded QwenEmotion is released again after analysis when it was not already active.

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

The pronunciation section is collapsed by default and includes an in-app Chinese guide plus a
one-click example when opened. The shortest workflow is to write an inline annotation directly in the target text:

```text
小明<要求|YAO4 QIU2>这个题的答案是多少。今天的<行程|XING2 CHENG2>顺利。
```

Use the form `<original text|reading>`. Chinese Pinyin uses one tone-numbered syllable per Han character,
separated by spaces. Annotate the complete word when a polyphonic character is inside a word: use
`<要求|YAO4 QIU2>` rather than `<要|YAO4>求` (the latter is the unreliable form reported in upstream issue #792).
Repeated names and polyphones can instead be stored in the persistent pronunciation table.

## Chinese numbers, dates, and years

Arabic-digit mispronunciation is generally caused by the text-normalization front end, not by voice cloning or the
reference recording. The full Windows portable package installs `wetext>=0.1.7,<0.2` directly; Linux development uses
`WeTextProcessing>=1.2.0,<2`. At application startup, the generation page runs a real smoke test and shows the active
package/version plus the result of `1939年` → `一九三九年` beside **Text normalization (numbers/dates)**.

- A year such as `1939年` is normally read digit by digit: `一九三九年`.
- A quantity such as `1939个人` is normally read as a cardinal number: `一千九百三十九个人`.
- If normalization is disabled or its visible smoke test fails, write the intended spoken form explicitly.

The packaging verification fails unless this conversion works from the bundled runtime, including an installation path
containing Chinese characters. GitHub app-layer incremental updates intentionally do not replace Python/CUDA files, so an
older portable runtime missing this dependency must be replaced by a new full portable build rather than only applying an
app-layer update.

## Desktop updates and external models

The launcher checks the stable GitHub Release channel once per day by default; beta is opt-in. A check never downloads
or installs anything. An automatic portable update is offered only when the Release contains all three assets:

- `desktop-app-update-v<version>-win32-x64.zip`
- `desktop-update-manifest.json`
- `desktop-update-manifest.sig`

The embedded Ed25519 public key verifies the canonical manifest before any package URL or replacement list is trusted.
The updater then verifies the GitHub asset size, ZIP SHA-256, exact extracted file set, and every file hash. The user must
click both download and install. Only `resources/app.asar`, IndexTTS/application source, manifests, launcher assets, and
the update helper are replaceable. Python/CUDA dependencies, external checkpoints, user data, voices, presets, and output
files are not part of the incremental ZIP. Read-only or Squirrel installs are sent to the Release page for manual update.

Model weights are distributed independently at
[`t8star/IndexTTS-2.5-Comfy`](https://huggingface.co/t8star/IndexTTS-2.5-Comfy), pinned to revision
`14166a7401f9f87f53770a1784390e8c0e9da15a`. The repository publishes signed `model-bundle.json` and
`model-bundle.sig` metadata independently from desktop releases. Click `Hugging Face 自动下载／修复完整模型（推荐）`
in the launcher.
When no model directory is selected, choosing a parent folder creates `<parent>\IndexTTS-2.5`; an already selected
incomplete folder is synchronized in place. The Hugging Face cache is kept below that model directory so interrupted
downloads can resume. All 26 required main and auxiliary files are size- and SHA-256-verified before the Start button
is enabled. Updating only the Hugging Face README does not trigger a model update; the signed bundle version must change.
ModelScope remains an explicit fallback and no model is silently overwritten by the desktop update checker.

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
npm run build:runtime
npm run build:update
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
available under `desktop/out/T8star-Aix-IndexTTS-2.5-v0.24.0-win32-x64` for local testing.
The bundled runtime contains tens of thousands of small files, so Squirrel/NuGet
can spend a long time repeatedly rewriting a multi-gigabyte package. It is not the
recommended user distribution. If an installer is specifically required, build it
explicitly after the portable package has passed `npm run verify:runtime`:

```powershell
npm run make:installer
```

Neither distribution includes `checkpoints`; users select or download the complete
IndexTTS 2.5 model directory from the launcher.

`npm run build:runtime` creates a separately versioned Python/Torch/CUDA layer and splits it into GitHub-safe assets
below 2 GiB. Use `npm run release:runtime` only after the packaged runtime passes real startup verification; it is the
explicit, GitHub-writing form. `npm run build:update` reads the verified packaged resources, builds an app-layer ZIP,
writes its exact manifest, optionally pins the published runtime-layer descriptor, and
signs it with `T8_UPDATE_PRIVATE_KEY_BASE64`, `T8_UPDATE_PRIVATE_KEY_FILE`, or the private key passed with
`--private-key`. The private key must never be committed. `.github/workflows/desktop-release.yml` can alternatively
build the same source-only patch on `windows-latest`; configure the repository secret
`T8_UPDATE_PRIVATE_KEY_BASE64` before creating a `v<package version>` tag. The full multi-gigabyte portable archive is
kept on the external distribution mirror because it exceeds GitHub Releases' per-file limit; its URL and SHA-256 can be
placed in `desktop_release_config.json` when available. Models remain separate on Hugging Face. See
[`docs/DESKTOP_LAYERED_UPDATE.md`](../docs/DESKTOP_LAYERED_UPDATE.md) for the release, verification, resume, and rollback flow.
