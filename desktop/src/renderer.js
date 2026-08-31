const elements = {
  statusDot: document.querySelector("#statusDot"),
  statusText: document.querySelector("#statusText"),
  modelPath: document.querySelector("#modelPath"),
  outputPath: document.querySelector("#outputPath"),
  dataPath: document.querySelector("#dataPath"),
  logPath: document.querySelector("#logPath"),
  chooseOutputButton: document.querySelector("#chooseOutputButton"),
  chooseDataButton: document.querySelector("#chooseDataButton"),
  openOutputButton: document.querySelector("#openOutputButton"),
  openDataButton: document.querySelector("#openDataButton"),
  runtimeProfile: document.querySelector("#runtimeProfile"),
  applyRuntimeProfileButton: document.querySelector("#applyRuntimeProfileButton"),
  hardwareSummary: document.querySelector("#hardwareSummary"),
  runtimeProfileHint: document.querySelector("#runtimeProfileHint"),
  accelerationMode: document.querySelector("#accelerationMode"),
  precisionMode: document.querySelector("#precisionMode"),
  referenceDevice: document.querySelector("#referenceDevice"),
  reuseDefaultEmotion: document.querySelector("#reuseDefaultEmotion"),
  refreshDiagnosticsButton: document.querySelector("#refreshDiagnosticsButton"),
  exportDiagnosticsButton: document.querySelector("#exportDiagnosticsButton"),
  diagnosticsVersions: document.querySelector("#diagnosticsVersions"),
  diagnosticsGrid: document.querySelector("#diagnosticsGrid"),
  runBenchmarkButton: document.querySelector("#runBenchmarkButton"),
  cancelBenchmarkButton: document.querySelector("#cancelBenchmarkButton"),
  applyBenchmarkButton: document.querySelector("#applyBenchmarkButton"),
  benchmarkSummary: document.querySelector("#benchmarkSummary"),
  benchmarkResults: document.querySelector("#benchmarkResults"),
  checkUpdatesButton: document.querySelector("#checkUpdatesButton"),
  openDesktopReleaseButton: document.querySelector("#openDesktopReleaseButton"),
  openOfficialButton: document.querySelector("#openOfficialButton"),
  openNodeRepositoryButton: document.querySelector("#openNodeRepositoryButton"),
  autoCheckUpdates: document.querySelector("#autoCheckUpdates"),
  updateChannel: document.querySelector("#updateChannel"),
  updateSummary: document.querySelector("#updateSummary"),
  updateResults: document.querySelector("#updateResults"),
  updateProgressPanel: document.querySelector("#updateProgressPanel"),
  updateProgress: document.querySelector("#updateProgress"),
  updateProgressText: document.querySelector("#updateProgressText"),
  downloadUpdateButton: document.querySelector("#downloadUpdateButton"),
  updateModelButton: document.querySelector("#updateModelButton"),
  cancelUpdateButton: document.querySelector("#cancelUpdateButton"),
  installUpdateButton: document.querySelector("#installUpdateButton"),
  missingFiles: document.querySelector("#missingFiles"),
  chooseModelButton: document.querySelector("#chooseModelButton"),
  startButton: document.querySelector("#startButton"),
  downloadModelscopeButton: document.querySelector("#downloadModelscopeButton"),
  downloadHuggingfaceButton: document.querySelector("#downloadHuggingfaceButton"),
  cancelDownloadButton: document.querySelector("#cancelDownloadButton"),
  modelDownloadPanel: document.querySelector("#modelDownloadPanel"),
  modelDownloadProgress: document.querySelector("#modelDownloadProgress"),
  modelDownloadTitle: document.querySelector("#modelDownloadTitle"),
  modelDownloadDetail: document.querySelector("#modelDownloadDetail"),
  modelDownloadDisk: document.querySelector("#modelDownloadDisk"),
  huggingfaceButton: document.querySelector("#huggingfaceButton"),
  modelscopeButton: document.querySelector("#modelscopeButton"),
  openLogsButton: document.querySelector("#openLogsButton"),
  logOutput: document.querySelector("#logOutput")
};

const profileLabels = {
  low_vram: "6–8GB 省显存",
  balanced: "10–16GB 均衡",
  max_speed: "16GB+ 速度优先",
  compatibility: "稳定兼容 / 排错",
  custom: "手动自定义"
};

const profileDescriptions = {
  low_vram: "FP16、参考编码器放 CPU、关闭可选加速。",
  balanced: "自动精度与参考设备，只启用已具备的安全融合核。",
  max_speed: "参考编码器同显卡并启用 GPT 加速，不兼容采样自动回退。",
  compatibility: "关闭可选加速并保留独立情感编码。",
  custom: "以下精度、参考设备和加速模式由你分别设置。"
};

const accelerationLabels = {
  auto_safe: "自动安全 / BigVGAN",
  bigvgan_cuda: "BigVGAN CUDA",
  torch_compile: "Torch Compile",
  gpt_accel: "GPT 加速",
  deepspeed: "DeepSpeed FP16"
};

let currentState = {};

function formatBytes(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes < 0) return "–";
  if (bytes < 1024) return `${Math.round(bytes)} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let amount = bytes;
  let index = -1;
  do {
    amount /= 1024;
    index += 1;
  } while (amount >= 1024 && index < units.length - 1);
  return `${amount >= 100 ? amount.toFixed(0) : amount.toFixed(1)} ${units[index]}`;
}

function formatEta(value) {
  const seconds = Number(value);
  if (!Number.isFinite(seconds) || seconds < 0) return "计算中";
  if (seconds < 60) return `${Math.ceil(seconds)} 秒`;
  if (seconds < 3600) return `${Math.ceil(seconds / 60)} 分钟`;
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.ceil((seconds % 3600) / 60);
  return `${hours} 小时 ${minutes} 分钟`;
}

function renderModelDownload(download) {
  elements.modelDownloadPanel.hidden = !download;
  if (!download) return;
  elements.modelDownloadProgress.value = Number(download.overallPercent || 0);
  elements.modelDownloadTitle.textContent = download.message || "正在准备模型下载…";

  const details = [];
  if (download.file) {
    const position = download.fileCount
      ? `文件 ${download.fileIndex || 0}/${download.fileCount}`
      : "当前文件";
    details.push(`${position}：${download.file}`);
  }
  if (download.phase === "downloading" && Number(download.total) > 0) {
    details.push(`${formatBytes(download.received)} / ${formatBytes(download.total)}`);
    if (Number(download.bytesPerSecond) > 0) {
      details.push(`${formatBytes(download.bytesPerSecond)}/s`);
      details.push(`预计剩余 ${formatEta(download.etaSeconds)}`);
    }
  } else if (["scanning", "verifying"].includes(download.phase) && Number(download.total) > 0) {
    details.push(`已处理 ${formatBytes(download.received)} / ${formatBytes(download.total)}`);
  }
  elements.modelDownloadDetail.textContent = details.join(" · ") || "进度信息准备中…";

  if (Number.isFinite(Number(download.availableBytes))) {
    const required = formatBytes(download.requiredBytes);
    const available = formatBytes(download.availableBytes);
    elements.modelDownloadDisk.textContent = `磁盘预检：最多还需 ${required}，当前可用 ${available}${download.warning ? `。${download.warning}` : "。"}`;
    elements.modelDownloadDisk.className = `model-download-disk ${download.diskSufficient === false ? "warning" : "ready"}`;
  } else {
    elements.modelDownloadDisk.textContent = "";
    elements.modelDownloadDisk.className = "model-download-disk";
  }
}

function renderProfileHint(selectedProfile) {
  const recommendation = currentState.recommendedProfile || "compatibility";
  if (selectedProfile === "recommended") {
    elements.runtimeProfileHint.textContent = `将应用检测建议：${profileLabels[recommendation]}。${profileDescriptions[recommendation]}`;
    return;
  }
  const selected = selectedProfile || "custom";
  elements.runtimeProfileHint.textContent = `当前：${profileLabels[selected]}。${profileDescriptions[selected]} 检测建议：${profileLabels[recommendation]}。`;
}

function renderAccelerationDiagnostics(report, busy) {
  elements.refreshDiagnosticsButton.disabled = Boolean(busy);
  elements.exportDiagnosticsButton.disabled = Boolean(busy) || !report;
  elements.diagnosticsGrid.replaceChildren();

  if (!report) {
    const pending = document.createElement("div");
    pending.className = "diagnostic-item pending";
    pending.textContent = busy
      ? "正在检测；此过程不会加载 IndexTTS 模型。"
      : "暂无诊断结果，请点击“重新检测”。";
    elements.diagnosticsGrid.appendChild(pending);
    elements.diagnosticsVersions.textContent = "尚未读取运行库版本。";
    return;
  }

  const versions = report.capabilities?.versions || {};
  const versionParts = [
    ["torch", versions.torch],
    ["Torchaudio", versions.torchaudio],
    ["TorchCodec", versions.torchcodec],
    ["CUDA", versions.cuda_runtime],
    ["FlashAttention", versions.flash_attn],
    ["Triton", versions.triton],
    ["DeepSpeed", versions.deepspeed],
    ["Ninja", versions.ninja]
  ].map(([name, value]) => `${name} ${value || "未安装"}`);
  elements.diagnosticsVersions.textContent = versionParts.join(" · ");

  const torchcodec = report.capabilities?.runtime_checks?.torchcodec;
  if (torchcodec) {
    const ready = Boolean(torchcodec.ready);
    const item = document.createElement("div");
    item.className = `diagnostic-item ${ready ? "ready" : "fallback"}`;
    const title = document.createElement("strong");
    title.textContent = `${ready ? "✓" : "!"} TorchCodec / FFmpeg DLL`;
    const description = document.createElement("span");
    description.textContent = torchcodec.reason || "没有返回音频运行时检测结果";
    item.append(title, description);
    elements.diagnosticsGrid.appendChild(item);
  }

  for (const mode of Object.keys(accelerationLabels)) {
    const result = report.modes?.[mode];
    const ready = Boolean(result?.available && result?.effective !== "off");
    const item = document.createElement("div");
    item.className = `diagnostic-item ${ready ? "ready" : "fallback"}`;

    const title = document.createElement("strong");
    title.textContent = `${ready ? "✓" : "–"} ${accelerationLabels[mode]}`;
    const description = document.createElement("span");
    description.textContent = result?.reason || "没有返回检测结果";
    item.append(title, description);
    elements.diagnosticsGrid.appendChild(item);
  }
}

function renderBenchmark(report, busy, modelValid) {
  elements.runBenchmarkButton.disabled = Boolean(busy) || !modelValid;
  elements.cancelBenchmarkButton.hidden = !currentState.benchmarkBusy;
  const recommendation = report?.recommendation;
  elements.applyBenchmarkButton.disabled = Boolean(busy) || !recommendation?.mode;
  elements.benchmarkResults.replaceChildren();

  if (!report) {
    elements.benchmarkSummary.textContent = currentState.benchmarkBusy
      ? "正在逐个加载并测试加速模式；不同显卡可能需要几分钟。"
      : "尚未运行真实基准。请选择完整模型并准备一段参考音频。";
    return;
  }

  elements.benchmarkSummary.textContent = report.summary
    || (recommendation?.mode
      ? `推荐：${accelerationLabels[recommendation.mode] || recommendation.mode}。${recommendation.reason || ""}`
      : "未得到可用推荐，请查看失败原因。");

  const results = Array.isArray(report.results) ? report.results : [];
  if (!results.length) return;

  const table = document.createElement("table");
  table.className = "benchmark-table";
  const header = document.createElement("tr");
  for (const label of ["请求模式", "实际模式", "状态", "初始化", "推理", "音频", "RTF", "峰值显存"]) {
    const cell = document.createElement("th");
    cell.textContent = label;
    header.appendChild(cell);
  }
  const head = document.createElement("thead");
  head.appendChild(header);
  table.appendChild(head);

  const body = document.createElement("tbody");
  for (const result of results) {
    const row = document.createElement("tr");
    if (recommendation?.mode === result.requested_mode) row.className = "recommended";
    const values = [
      accelerationLabels[result.requested_mode] || result.requested_mode || "–",
      accelerationLabels[result.effective_mode] || result.effective_mode || "–",
      result.status === "ok" ? "成功" : (result.reason || "失败"),
      Number.isFinite(result.init_seconds) ? `${result.init_seconds.toFixed(2)}s` : "–",
      Number.isFinite(result.inference_seconds) ? `${result.inference_seconds.toFixed(2)}s` : "–",
      Number.isFinite(result.audio_seconds) ? `${result.audio_seconds.toFixed(2)}s` : "–",
      Number.isFinite(result.rtf) ? result.rtf.toFixed(3) : "–",
      Number.isFinite(result.peak_vram_gb) ? `${result.peak_vram_gb.toFixed(2)}GB` : "–"
    ];
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.appendChild(cell);
    }
    body.appendChild(row);
  }
  table.appendChild(body);
  elements.benchmarkResults.appendChild(table);
}

function renderUpdateReport(report, busy) {
  elements.checkUpdatesButton.disabled = Boolean(busy);
  elements.updateResults.replaceChildren();
  elements.autoCheckUpdates.checked = currentState.autoCheckUpdates !== false;
  elements.updateChannel.value = currentState.updateChannel || "stable";
  const download = currentState.updateDownload;
  const downloading = ["downloading", "verifying", "cancelling"].includes(download?.status);
  elements.updateProgressPanel.hidden = !download;
  elements.updateProgress.value = Number(download?.percent || 0);
  elements.updateProgressText.textContent = download?.message || "尚未下载。";
  elements.cancelUpdateButton.hidden = !downloading;
  elements.installUpdateButton.hidden = !currentState.updateReady;
  elements.downloadUpdateButton.hidden = !report?.desktop?.updateAvailable || currentState.updateReady;
  elements.downloadUpdateButton.disabled = Boolean(busy) || downloading;
  elements.downloadUpdateButton.textContent = report?.desktop?.manualOnly
    ? "打开 Release 手动更新"
    : "下载并校验更新";
  elements.updateModelButton.hidden = !report?.officialModel?.updateAvailable;
  elements.updateModelButton.disabled = Boolean(busy) || currentState.phase === "downloading";
  elements.updateModelButton.textContent = report?.officialModel?.latest
    ? `下载模型包 ${report.officialModel.latest}`
    : "下载并更新模型";
  if (!report) {
    elements.updateSummary.textContent = busy ? "正在联网检查…" : "尚未检查。";
    return;
  }
  elements.updateSummary.textContent = report.summary || "检查完成。";
  const rows = [
    ["桌面程序", report.desktop?.current, report.desktop?.latest, report.desktop?.updateAvailable],
    ["ComfyUI 节点", report.node?.bundled, report.node?.latest, report.node?.updateAvailable],
    ["官方代码", report.officialCode?.pinned, report.officialCode?.latest, report.officialCode?.updateAvailable],
    ["T8star 模型包", report.officialModel?.pinned, report.officialModel?.latest, report.officialModel?.updateAvailable]
  ];
  for (const [label, current, latest, available] of rows) {
    const item = document.createElement("div");
    item.className = `update-item ${available ? "available" : "current"}`;
    const title = document.createElement("strong");
    title.textContent = `${available ? "↑" : "✓"} ${label}`;
    const detail = document.createElement("span");
    const shorten = (value) => String(value || "未知").length > 16 ? String(value).slice(0, 12) : String(value || "未知");
    const signatureNote = label === "桌面程序" && report.desktop?.updateAvailable
      ? report.desktop?.signatureVerified ? " · 签名有效" : " · 仅手动更新"
      : "";
    detail.textContent = `当前 ${shorten(current)} · 最新 ${shorten(latest)}${signatureNote}`;
    item.append(title, detail);
    elements.updateResults.appendChild(item);
  }
  for (const error of report.errors || []) {
    const item = document.createElement("div");
    item.className = "update-item warning";
    item.textContent = error;
    elements.updateResults.appendChild(item);
  }
}

function renderState(state) {
  currentState = state;
  const busy = ["starting", "ready", "stopping", "downloading", "benchmarking"].includes(state.phase);
  elements.statusText.textContent = state.message || "等待操作";
  elements.modelPath.value = state.modelDir || "";
  elements.outputPath.value = state.outputDir || "";
  elements.dataPath.value = state.dataDir || "";
  elements.logPath.textContent = state.logDir || "";
  elements.runtimeProfile.value = state.runtimeProfile || "custom";
  elements.hardwareSummary.textContent = state.hardwareSummary || "显卡信息尚不可用。";
  renderProfileHint(elements.runtimeProfile.value);
  elements.accelerationMode.value = state.accelerationMode || "off";
  elements.precisionMode.value = state.precisionMode || "auto";
  elements.referenceDevice.value = state.referenceDevice || "auto";
  elements.reuseDefaultEmotion.checked = Boolean(state.reuseDefaultEmotion);
  elements.accelerationMode.disabled = busy;
  elements.runtimeProfile.disabled = busy;
  elements.applyRuntimeProfileButton.disabled = busy || elements.runtimeProfile.value === "custom";
  elements.precisionMode.disabled = busy;
  elements.referenceDevice.disabled = busy;
  elements.reuseDefaultEmotion.disabled = busy;
  renderAccelerationDiagnostics(state.accelerationDiagnostics, state.diagnosticsBusy);
  renderBenchmark(state.benchmarkReport, busy, Boolean(state.modelValid));
  renderUpdateReport(state.updateReport, state.updateBusy);
  renderModelDownload(state.modelDownload);
  elements.startButton.disabled = !state.modelValid || busy;
  elements.chooseModelButton.disabled = busy;
  elements.chooseOutputButton.disabled = busy;
  elements.chooseDataButton.disabled = busy;
  elements.downloadModelscopeButton.disabled = busy;
  elements.downloadHuggingfaceButton.disabled = busy;
  elements.cancelDownloadButton.hidden = state.phase !== "downloading";
  elements.statusDot.className = "status-dot";
  if (state.phase === "ready") elements.statusDot.classList.add("ready");
  if (state.phase === "error" || state.phase === "model-required") elements.statusDot.classList.add("error");

  elements.missingFiles.replaceChildren();
  for (const missingFile of state.missingFiles || []) {
    const item = document.createElement("li");
    item.textContent = `缺少：${missingFile}`;
    elements.missingFiles.appendChild(item);
  }
}

function appendLog(line) {
  if (elements.logOutput.textContent === "等待启动…") elements.logOutput.textContent = "";
  elements.logOutput.textContent += `${line}\n`;
  elements.logOutput.scrollTop = elements.logOutput.scrollHeight;
}

elements.chooseModelButton.addEventListener("click", async () => {
  renderState(await window.desktopApi.chooseModelDirectory());
});

elements.chooseOutputButton.addEventListener("click", async () => {
  renderState(await window.desktopApi.chooseOutputDirectory());
});

elements.chooseDataButton.addEventListener("click", async () => {
  renderState(await window.desktopApi.chooseDataDirectory());
});

elements.openOutputButton.addEventListener("click", () => window.desktopApi.openOutputDirectory());
elements.openDataButton.addEventListener("click", () => window.desktopApi.openDataDirectory());

elements.startButton.addEventListener("click", async () => {
  appendLog("正在启动内置 Python 与 IndexTTS 2.5…");
  renderState(await window.desktopApi.startService());
});

elements.runtimeProfile.addEventListener("change", () => {
  renderProfileHint(elements.runtimeProfile.value);
  elements.applyRuntimeProfileButton.disabled = elements.runtimeProfile.value === "custom";
});

elements.applyRuntimeProfileButton.addEventListener("click", async () => {
  appendLog("正在应用一键运行方案；不会自动加载模型…");
  renderState(await window.desktopApi.applyRuntimeProfile(elements.runtimeProfile.value));
});

elements.accelerationMode.addEventListener("change", async () => {
  renderState(await window.desktopApi.setAcceleration(elements.accelerationMode.value));
});

async function saveRuntimeOptions() {
  renderState(await window.desktopApi.setRuntimeOptions({
    precisionMode: elements.precisionMode.value,
    referenceDevice: elements.referenceDevice.value,
    reuseDefaultEmotion: elements.reuseDefaultEmotion.checked
  }));
}

elements.precisionMode.addEventListener("change", saveRuntimeOptions);
elements.referenceDevice.addEventListener("change", saveRuntimeOptions);
elements.reuseDefaultEmotion.addEventListener("change", saveRuntimeOptions);

elements.refreshDiagnosticsButton.addEventListener("click", async () => {
  appendLog("正在重新检测加速环境；不会加载模型…");
  renderState(await window.desktopApi.refreshDiagnostics());
});

elements.exportDiagnosticsButton.addEventListener("click", async () => {
  try {
    const result = await window.desktopApi.exportDiagnostics();
    if (!result.canceled) appendLog(`诊断报告已导出：${result.filePath}`);
  } catch (error) {
    appendLog(`诊断报告导出失败：${error.message}`);
  }
});

elements.runBenchmarkButton.addEventListener("click", async () => {
  appendLog("准备运行真实加速基准；请选择一段参考音频…");
  renderState(await window.desktopApi.runRuntimeBenchmark());
});

elements.cancelBenchmarkButton.addEventListener("click", async () => {
  renderState(await window.desktopApi.cancelRuntimeBenchmark());
});

elements.applyBenchmarkButton.addEventListener("click", async () => {
  appendLog("正在应用真实基准给出的推荐模式；不会自动加载模型…");
  renderState(await window.desktopApi.applyBenchmarkRecommendation());
});

elements.checkUpdatesButton.addEventListener("click", async () => {
  appendLog("正在检查桌面程序、官方代码、模型和节点版本；不会自动下载…");
  renderState(await window.desktopApi.checkUpdates());
});

elements.openDesktopReleaseButton.addEventListener("click", () => window.desktopApi.openUpdatePage("desktop"));
elements.openOfficialButton.addEventListener("click", () => window.desktopApi.openUpdatePage("official"));
elements.openNodeRepositoryButton.addEventListener("click", () => window.desktopApi.openUpdatePage("node"));

async function saveUpdatePreferences() {
  renderState(await window.desktopApi.setUpdatePreferences({
    autoCheckUpdates: elements.autoCheckUpdates.checked,
    updateChannel: elements.updateChannel.value
  }));
}

elements.autoCheckUpdates.addEventListener("change", saveUpdatePreferences);
elements.updateChannel.addEventListener("change", saveUpdatePreferences);

elements.downloadUpdateButton.addEventListener("click", async () => {
  appendLog("准备下载桌面更新；下载后会先校验，不会立即退出或安装…");
  renderState(await window.desktopApi.downloadUpdate());
});

elements.updateModelButton.addEventListener("click", async () => {
  appendLog("准备从 Hugging Face 自动下载并校验最新模型包…");
  renderState(await window.desktopApi.downloadModel("huggingface"));
});

elements.cancelUpdateButton.addEventListener("click", async () => {
  renderState(await window.desktopApi.cancelUpdate());
});

elements.installUpdateButton.addEventListener("click", async () => {
  appendLog("准备退出并安装已校验的桌面更新…");
  renderState(await window.desktopApi.installUpdate());
});

elements.downloadModelscopeButton.addEventListener("click", async () => {
  appendLog("准备从 ModelScope 下载外置模型…");
  renderState(await window.desktopApi.downloadModel("modelscope"));
});

elements.downloadHuggingfaceButton.addEventListener("click", async () => {
  appendLog("准备从 Hugging Face 下载外置模型…");
  renderState(await window.desktopApi.downloadModel("huggingface"));
});

elements.cancelDownloadButton.addEventListener("click", async () => {
  renderState(await window.desktopApi.cancelModelDownload());
});

elements.huggingfaceButton.addEventListener("click", () => window.desktopApi.openModelPage("huggingface"));
elements.modelscopeButton.addEventListener("click", () => window.desktopApi.openModelPage("modelscope"));
elements.openLogsButton.addEventListener("click", () => window.desktopApi.openLogs());

window.desktopApi.onState(renderState);
window.desktopApi.onLog(appendLog);
window.desktopApi.getState().then(renderState);
