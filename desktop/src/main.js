const { app, BrowserWindow, dialog, ipcMain, shell } = require("electron");
const fs = require("node:fs");
const https = require("node:https");
const net = require("node:net");
const os = require("node:os");
const path = require("node:path");
const { spawn } = require("node:child_process");
const { createDiagnosticReport } = require("./diagnostic_report");
const {
  hardwareSummary,
  recommendRuntimeProfile,
  resolveRuntimeProfile
} = require("./runtime_profiles");

const APP_TITLE = "T8star-Aix · IndexTTS 2.5";
const COMFY_NODE_VERSION = "0.15.0";
const MODEL_URLS = {
  huggingface: "https://huggingface.co/IndexTeam/IndexTTS-2.5",
  modelscope: "https://modelscope.cn/models/IndexTeam/IndexTTS-2.5"
};
let modelManifestCache = null;

let mainWindow = null;
let pythonProcess = null;
let downloadProcess = null;
let cancelledDownloadProcess = null;
let benchmarkProcess = null;
let benchmarkCancelled = false;
let activePort = null;
let state = {
  phase: "idle",
  message: "请选择 IndexTTS 2.5 模型目录",
  modelDir: "",
  modelValid: false,
  missingFiles: [],
  accelerationMode: "off",
  precisionMode: "auto",
  referenceDevice: "auto",
  reuseDefaultEmotion: false,
  runtimeProfile: "custom",
  recommendedProfile: "compatibility",
  hardwareSummary: "正在检测显卡（只检测环境，不加载模型）…",
  accelerationDiagnostics: null,
  diagnosticsBusy: true,
  benchmarkBusy: false,
  benchmarkReport: null,
  updateBusy: false,
  updateReport: null,
  serviceUrl: ""
};

function fetchText(url, timeoutMs = 15000) {
  return new Promise((resolve, reject) => {
    const request = https.get(url, {
      headers: {
        "User-Agent": "T8star-Aix-IndexTTS25-Desktop",
        Accept: "application/json, text/plain;q=0.9, */*;q=0.8"
      }
    }, (response) => {
      if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
        response.resume();
        fetchText(new URL(response.headers.location, url).toString(), timeoutMs).then(resolve, reject);
        return;
      }
      let body = "";
      response.setEncoding("utf8");
      response.on("data", (chunk) => { body += chunk; });
      response.on("end", () => {
        if (response.statusCode < 200 || response.statusCode >= 300) {
          reject(new Error(`HTTP ${response.statusCode}: ${url}`));
          return;
        }
        resolve(body);
      });
    });
    request.setTimeout(timeoutMs, () => request.destroy(new Error("检查更新超时")));
    request.on("error", reject);
  });
}

function compareVersions(left, right) {
  const a = String(left || "0").split(".").map((part) => Number.parseInt(part, 10) || 0);
  const b = String(right || "0").split(".").map((part) => Number.parseInt(part, 10) || 0);
  for (let index = 0; index < Math.max(a.length, b.length); index += 1) {
    if ((a[index] || 0) !== (b[index] || 0)) return (a[index] || 0) > (b[index] || 0) ? 1 : -1;
  }
  return 0;
}

async function checkForUpdates() {
  if (state.updateBusy) return state;
  updateState({ updateBusy: true, message: "正在检查官方代码、模型与节点版本…" });
  const manifest = modelManifest();
  const sources = await Promise.allSettled([
    fetchText("https://api.github.com/repos/index-tts/index-tts/commits/main"),
    fetchText("https://huggingface.co/api/models/IndexTeam/IndexTTS-2.5"),
    fetchText("https://raw.githubusercontent.com/T8mars/comfyui-indextts25-t8/main/pyproject.toml")
  ]);
  const errors = [];
  const readJson = (result, label) => {
    if (result.status === "rejected") {
      errors.push(`${label}：${result.reason.message || result.reason}`);
      return {};
    }
    try { return JSON.parse(result.value); } catch (error) {
      errors.push(`${label}：${error.message}`);
      return {};
    }
  };
  const upstream = readJson(sources[0], "官方代码");
  const model = readJson(sources[1], "官方模型");
  let remoteNodeVersion = "";
  if (sources[2].status === "fulfilled") {
    remoteNodeVersion = sources[2].value.match(/^version\s*=\s*["']([^"']+)["']/m)?.[1] || "";
  } else {
    errors.push(`节点仓库：${sources[2].reason.message || sources[2].reason}`);
  }
  const codeRevision = String(upstream.sha || "");
  const modelRevision = String(model.sha || "");
  const report = {
    checkedAt: new Date().toISOString(),
    desktop: { current: app.getVersion() },
    node: {
      bundled: COMFY_NODE_VERSION,
      latest: remoteNodeVersion,
      updateAvailable: Boolean(remoteNodeVersion && compareVersions(remoteNodeVersion, COMFY_NODE_VERSION) > 0)
    },
    officialCode: {
      pinned: String(manifest.codeRevision || ""),
      latest: codeRevision,
      updateAvailable: Boolean(codeRevision && !codeRevision.startsWith(String(manifest.codeRevision || "")) && !String(manifest.codeRevision || "").startsWith(codeRevision))
    },
    officialModel: {
      pinned: String(manifest.modelRevision || ""),
      latest: modelRevision,
      updateAvailable: Boolean(modelRevision && modelRevision !== String(manifest.modelRevision || ""))
    },
    errors
  };
  const updates = [report.node, report.officialCode, report.officialModel].filter((item) => item.updateAvailable).length;
  report.summary = errors.length === 3
    ? "检查失败，请确认网络后重试。"
    : updates
      ? `发现 ${updates} 项新版本；这里只提示，不会自动下载或覆盖。`
      : "当前未发现新版本。";
  updateState({ updateBusy: false, updateReport: report, message: report.summary });
  appendLog(`Update check: ${JSON.stringify(report)}`);
  return state;
}

function projectRoot() {
  return path.resolve(__dirname, "..", "..");
}

function modelManifest() {
  if (modelManifestCache) return modelManifestCache;
  const manifestPath = path.join(
    app.isPackaged ? process.resourcesPath : projectRoot(),
    "desktop_model_manifest.json"
  );
  modelManifestCache = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  return modelManifestCache;
}

function settingsPath() {
  return path.join(app.getPath("userData"), "settings.json");
}

function logsDirectory() {
  return path.join(app.getPath("userData"), "logs");
}

function outputDirectory() {
  return path.join(app.getPath("documents"), "T8star-Aix IndexTTS 2.5", "outputs");
}

function benchmarkDirectory() {
  return path.join(app.getPath("userData"), "benchmarks");
}

function userDataDirectory() {
  return app.getPath("userData");
}

function readSettings() {
  try {
    return JSON.parse(fs.readFileSync(settingsPath(), "utf8"));
  } catch {
    return {};
  }
}

function commandLineModelDirectory() {
  const argumentIndex = process.argv.indexOf("--model-dir");
  if (argumentIndex >= 0 && process.argv[argumentIndex + 1]) {
    return path.resolve(process.argv[argumentIndex + 1]);
  }
  if (process.env.T8STAR_INDEXTTS_MODEL_DIR) {
    return path.resolve(process.env.T8STAR_INDEXTTS_MODEL_DIR);
  }
  return "";
}

function writeSettings(nextSettings) {
  fs.mkdirSync(path.dirname(settingsPath()), { recursive: true });
  fs.writeFileSync(settingsPath(), JSON.stringify(nextSettings, null, 2), "utf8");
}

function validateModelDirectory(modelDir) {
  if (!modelDir || !fs.existsSync(modelDir)) {
    return { valid: false, missingFiles: ["模型目录不存在"] };
  }

  const missingFiles = [];
  for (const [relativePath, metadata] of Object.entries(modelManifest().files)) {
    const localPath = path.join(modelDir, ...relativePath.split("/"));
    if (!fs.existsSync(localPath)) {
      missingFiles.push(relativePath);
      continue;
    }
    if (fs.statSync(localPath).size !== metadata.size) {
      missingFiles.push(`${relativePath}（版本不匹配）`);
    }
  }

  return { valid: missingFiles.length === 0, missingFiles };
}

function updateState(patch) {
  state = { ...state, ...patch };
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("desktop:state", state);
  }
}

function appendLog(rawLine) {
  const line = String(rawLine).trimEnd();
  if (!line) return;
  fs.mkdirSync(logsDirectory(), { recursive: true });
  const stamp = new Date().toISOString();
  fs.appendFileSync(path.join(logsDirectory(), "desktop.log"), `[${stamp}] ${line}\n`, "utf8");
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("desktop:log", line);
  }
}

async function probeRuntimeHardware() {
  const runtime = runtimePaths();
  const pythonPathParts = [runtime.backendRoot, runtime.sitePackages];
  if (process.env.PYTHONPATH) pythonPathParts.push(process.env.PYTHONPATH);
  const probeSource = [
    "import json, torch",
    "from dataclasses import asdict",
    "from runtime_acceleration import MODES, probe_acceleration, recommend_runtime_config, resolve_acceleration",
    "device = 'cuda:0' if torch.cuda.is_available() else 'cpu'",
    "caps = probe_acceleration(device)",
    "gpu = caps.get('gpu', {})",
    "hardware = {'cudaAvailable': bool(caps.get('cuda')), 'deviceName': str(gpu.get('name', '')), 'vramGb': float(gpu.get('total_vram_gb', 0) or 0), 'nativeBf16': bool(caps.get('bf16'))}",
    "modes = {mode: asdict(resolve_acceleration(mode, device, caps)) for mode in MODES}",
    "print(json.dumps({'hardware': hardware, 'device': device, 'capabilities': caps, 'recommended': recommend_runtime_config(caps), 'modes': modes}, ensure_ascii=False))"
  ].join("; ");

  const diagnostics = await new Promise((resolve, reject) => {
    const child = spawn(runtime.pythonExe, ["-c", probeSource], {
      cwd: runtime.backendRoot,
      windowsHide: true,
      env: {
        ...process.env,
        PYTHONUTF8: "1",
        PYTHONPATH: pythonPathParts.join(path.delimiter)
      }
    });
    let stdout = "";
    let stderr = "";
    let settled = false;
    const finish = (callback) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      callback();
    };
    const timeout = setTimeout(() => {
      child.kill();
      finish(() => reject(new Error("显卡检测超时")));
    }, 30000);
    child.stdout.on("data", (chunk) => { stdout += chunk.toString("utf8"); });
    child.stderr.on("data", (chunk) => { stderr += chunk.toString("utf8"); });
    child.on("error", (error) => finish(() => reject(error)));
    child.on("close", (code) => finish(() => {
      if (code !== 0) {
        reject(new Error(stderr.trim() || `显卡检测进程退出码 ${code}`));
        return;
      }
      try {
        const lines = stdout.trim().split(/\r?\n/).filter(Boolean);
        resolve(JSON.parse(lines.at(-1) || "{}"));
      } catch (error) {
        reject(new Error(`无法解析显卡检测结果：${error.message}`));
      }
    }));
  });

  const hardware = diagnostics.hardware || {};
  const recommendedProfile = recommendRuntimeProfile(hardware);
  const summary = hardwareSummary(hardware);
  updateState({
    recommendedProfile,
    hardwareSummary: summary,
    accelerationDiagnostics: diagnostics,
    diagnosticsBusy: false
  });
  appendLog(`Hardware probe only (model not loaded): ${summary}`);
  appendLog(`Acceleration preflight only (model not loaded): ${JSON.stringify(diagnostics.modes || {})}`);
  return hardware;
}

function buildDiagnosticReport() {
  const manifest = modelManifest();
  return createDiagnosticReport({
    appName: APP_TITLE,
    appVersion: app.getVersion(),
    electronVersion: process.versions.electron,
    nodeVersion: process.versions.node,
    platform: process.platform,
    architecture: process.arch,
    osRelease: os.release(),
    osVersion: typeof os.version === "function" ? os.version() : "",
    state,
    manifest
  });
}

async function exportDiagnosticReport() {
  if (!state.accelerationDiagnostics) await probeRuntimeHardware();
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const result = await dialog.showSaveDialog(mainWindow, {
    title: "导出 IndexTTS 2.5 诊断报告",
    defaultPath: path.join(
      app.getPath("documents"),
      "T8star-Aix IndexTTS 2.5",
      `IndexTTS25-diagnostic-${stamp}.json`
    ),
    filters: [{ name: "JSON 诊断报告", extensions: ["json"] }]
  });
  if (result.canceled || !result.filePath) return { canceled: true, filePath: "" };
  fs.mkdirSync(path.dirname(result.filePath), { recursive: true });
  fs.writeFileSync(result.filePath, `${JSON.stringify(buildDiagnosticReport(), null, 2)}\n`, "utf8");
  appendLog(`Diagnostic report exported: ${result.filePath}`);
  return { canceled: false, filePath: result.filePath };
}

async function runRuntimeBenchmark() {
  if (pythonProcess && pythonProcess.exitCode === null) {
    throw new Error("真实基准需要依次重载不同运行模式，请先关闭当前推理服务。");
  }
  if (benchmarkProcess && benchmarkProcess.exitCode === null) return state;
  const validation = validateModelDirectory(state.modelDir);
  if (!validation.valid) throw new Error("请先选择完整的 IndexTTS 2.5 模型目录。");

  const settings = readSettings();
  const selected = await dialog.showOpenDialog(mainWindow, {
    title: "选择一段 3–10 秒的清晰单人参考音频",
    defaultPath: settings.benchmarkReferenceAudio || app.getPath("documents"),
    properties: ["openFile"],
    filters: [
      { name: "音频", extensions: ["wav", "flac", "mp3", "m4a", "ogg"] }
    ]
  });
  if (selected.canceled || selected.filePaths.length === 0) return state;
  const referenceAudio = selected.filePaths[0];
  writeSettings({ ...settings, benchmarkReferenceAudio: referenceAudio });

  const runtime = runtimePaths();
  const reportRoot = benchmarkDirectory();
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const caseOutput = path.join(reportRoot, stamp);
  const reportPath = path.join(reportRoot, `benchmark-${stamp}.json`);
  fs.mkdirSync(caseOutput, { recursive: true });
  const pythonPathParts = [runtime.backendRoot, runtime.sitePackages];
  if (process.env.PYTHONPATH) pythonPathParts.push(process.env.PYTHONPATH);
  const args = [
    "-u",
    path.join(runtime.backendRoot, "desktop_runtime_benchmark.py"),
    "--model-dir", state.modelDir,
    "--reference-audio", referenceAudio,
    "--output-dir", caseOutput,
    "--report-path", reportPath,
    "--precision", state.precisionMode || "auto",
    "--reference-device", state.referenceDevice || "auto"
  ];
  if (state.reuseDefaultEmotion) args.push("--reuse-spk-cond-for-emo");

  updateState({
    phase: "benchmarking",
    message: "正在运行真实加速基准；将依次加载可用模式…",
    benchmarkBusy: true,
    benchmarkReport: null
  });
  appendLog(`Runtime benchmark reference: ${referenceAudio}`);
  appendLog(`Runtime benchmark report: ${reportPath}`);
  benchmarkProcess = spawn(runtime.pythonExe, args, {
    cwd: runtime.backendRoot,
    windowsHide: true,
    env: {
      ...process.env,
      PYTHONUTF8: "1",
      PYTHONUNBUFFERED: "1",
      PYTHONPATH: pythonPathParts.join(path.delimiter),
      HF_HOME: path.join(state.modelDir, "hf_cache"),
      HF_HUB_CACHE: path.join(state.modelDir, "hf_cache"),
      MODELSCOPE_CACHE: path.join(state.modelDir, "modelscope_cache")
    }
  });
  benchmarkCancelled = false;
  const processRef = benchmarkProcess;
  let stdoutBuffer = "";
  let finalReport = null;
  processRef.stdout.on("data", (chunk) => {
    stdoutBuffer += chunk.toString("utf8");
    const lines = stdoutBuffer.split(/\r?\n/);
    stdoutBuffer = lines.pop() || "";
    for (const line of lines) {
      appendLog(line);
      if (!line.startsWith("T8BENCH:")) continue;
      try {
        const event = JSON.parse(line.slice("T8BENCH:".length));
        if (event.event === "case_start") {
          updateState({ message: `正在基准测试 ${event.mode}（实际 ${event.effective}）…` });
        } else if (event.event === "case_complete") {
          const result = event.result || {};
          const detail = result.status === "ok"
            ? `RTF ${Number(result.rtf).toFixed(3)}`
            : result.status === "skipped" ? "依赖不可用，已跳过" : "运行失败，已记录";
          updateState({ message: `${result.requested_mode || "当前模式"}：${detail}` });
        } else if (event.event === "complete") {
          finalReport = event.report || null;
        }
      } catch (error) {
        appendLog(`Benchmark event parse warning: ${error.message}`);
      }
    }
  });
  processRef.stderr.on("data", (chunk) => appendLog(chunk.toString("utf8")));
  const exitCode = await new Promise((resolve) => {
    processRef.on("error", (error) => {
      appendLog(`Benchmark process error: ${error.stack || error.message}`);
      resolve(-1);
    });
    processRef.on("exit", (code, signal) => {
      appendLog(`Benchmark process exited: code=${code}, signal=${signal}`);
      resolve(code ?? -1);
    });
  });
  if (benchmarkProcess === processRef) benchmarkProcess = null;
  const wasCancelled = benchmarkCancelled;
  benchmarkCancelled = false;
  if (!finalReport && fs.existsSync(reportPath)) {
    try { finalReport = JSON.parse(fs.readFileSync(reportPath, "utf8")); } catch { /* reported below */ }
  }
  if (!wasCancelled && exitCode === 0 && finalReport) {
    updateState({
      phase: "idle",
      benchmarkBusy: false,
      benchmarkReport: { ...finalReport, reportPath },
      message: finalReport.summary || "真实加速基准完成"
    });
  } else {
    updateState({
      phase: "error",
      benchmarkBusy: false,
      message: wasCancelled ? "真实基准已取消" : "真实基准未完成，请查看日志"
    });
  }
  return state;
}

function cancelRuntimeBenchmark() {
  if (!benchmarkProcess || benchmarkProcess.exitCode !== null) return;
  const processRef = benchmarkProcess;
  benchmarkProcess = null;
  benchmarkCancelled = true;
  processRef.kill();
  updateState({ phase: "error", benchmarkBusy: false, message: "正在取消真实加速基准…" });
}

function resolvePackagedRuntime() {
  const entries = fs.readdirSync(process.resourcesPath, { withFileTypes: true });
  const pythonDir = entries.find((entry) => {
    return entry.isDirectory() && entry.name.startsWith("cpython-") &&
      fs.existsSync(path.join(process.resourcesPath, entry.name, "python.exe"));
  });
  if (!pythonDir) {
    throw new Error("整合包中的 Python 运行时缺失。");
  }
  return {
    pythonExe: path.join(process.resourcesPath, pythonDir.name, "python.exe"),
    backendRoot: process.resourcesPath,
    sitePackages: path.join(process.resourcesPath, "site-packages")
  };
}

function runtimePaths() {
  if (app.isPackaged) return resolvePackagedRuntime();
  const root = projectRoot();
  return {
    pythonExe: path.join(root, ".venv", "Scripts", "python.exe"),
    backendRoot: root,
    sitePackages: path.join(root, ".venv", "Lib", "site-packages")
  };
}

function findAvailablePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      server.close(() => resolve(address.port));
    });
  });
}

async function waitForService(url, processRef) {
  const deadline = Date.now() + 10 * 60 * 1000;
  while (Date.now() < deadline) {
    if (!pythonProcess || pythonProcess !== processRef || processRef.exitCode !== null) {
      throw new Error("Python 推理服务已退出，请查看日志。");
    }
    try {
      const response = await fetch(`${url}/gradio_api/info`, { signal: AbortSignal.timeout(2000) });
      if (response.ok) return;
    } catch {
      // Model loading can take tens of seconds. Keep polling while showing logs.
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error("模型加载超时，请查看日志或检查显存。");
}

async function startPythonService() {
  if (downloadProcess && downloadProcess.exitCode === null) {
    updateState({ phase: "downloading", message: "模型仍在下载，请等待下载完成" });
    return state;
  }
  if (pythonProcess && pythonProcess.exitCode === null) {
    return state;
  }

  const validation = validateModelDirectory(state.modelDir);
  if (!validation.valid) {
    updateState({
      phase: "model-required",
      modelValid: false,
      missingFiles: validation.missingFiles,
      message: "模型目录不完整"
    });
    return state;
  }

  const runtime = runtimePaths();
  for (const requiredPath of [runtime.pythonExe, runtime.backendRoot, runtime.sitePackages]) {
    if (!fs.existsSync(requiredPath)) {
      throw new Error(`运行时文件缺失：${requiredPath}`);
    }
  }

  fs.mkdirSync(outputDirectory(), { recursive: true });
  fs.mkdirSync(userDataDirectory(), { recursive: true });
  activePort = await findAvailablePort();
  const scriptPath = path.join(runtime.backendRoot, "desktop_webui.py");
  const serviceUrl = `http://127.0.0.1:${activePort}`;
  const pythonPathParts = [runtime.backendRoot, runtime.sitePackages];
  if (process.env.PYTHONPATH) pythonPathParts.push(process.env.PYTHONPATH);

  updateState({
    phase: "starting",
    message: "正在加载 IndexTTS 2.5 模型…",
    modelValid: true,
    missingFiles: [],
    serviceUrl
  });
  appendLog(`Starting bundled Python: ${runtime.pythonExe}`);
  appendLog(`Model directory: ${state.modelDir}`);
  appendLog(`Code revision: ${modelManifest().codeRevision}`);
  appendLog(`Model revision: ${modelManifest().modelRevision}`);
  appendLog(`Optional acceleration mode: ${state.accelerationMode || "off"}`);
  appendLog(`Precision mode: ${state.precisionMode || "auto"}`);
  appendLog(`Reference encoders: ${state.referenceDevice || "auto"}`);
  appendLog(`Fast default emotion: ${Boolean(state.reuseDefaultEmotion)}`);

  const pythonArguments = [
    "-u",
    scriptPath,
    "--model_dir", state.modelDir,
    "--output_dir", outputDirectory(),
    "--data_dir", userDataDirectory(),
    "--host", "127.0.0.1",
    "--port", String(activePort),
    "--acceleration", state.accelerationMode || "off",
    "--precision", state.precisionMode || "auto",
    "--reference-device", state.referenceDevice || "auto"
  ];
  if (state.reuseDefaultEmotion) pythonArguments.push("--reuse-spk-cond-for-emo");

  pythonProcess = spawn(runtime.pythonExe, pythonArguments, {
    cwd: runtime.backendRoot,
    windowsHide: true,
    env: {
      ...process.env,
      PYTHONUTF8: "1",
      PYTHONUNBUFFERED: "1",
      PYTHONPATH: pythonPathParts.join(path.delimiter),
      HF_HOME: path.join(state.modelDir, "hf_cache"),
      HF_HUB_CACHE: path.join(state.modelDir, "hf_cache"),
      MODELSCOPE_CACHE: path.join(state.modelDir, "modelscope_cache")
    }
  });

  const processRef = pythonProcess;
  processRef.stdout.on("data", (chunk) => appendLog(chunk.toString("utf8")));
  processRef.stderr.on("data", (chunk) => appendLog(chunk.toString("utf8")));
  processRef.on("error", (error) => {
    appendLog(`Python process error: ${error.stack || error.message}`);
    updateState({ phase: "error", message: error.message });
  });
  processRef.on("exit", (code, signal) => {
    appendLog(`Python process exited: code=${code}, signal=${signal}`);
    if (pythonProcess === processRef) pythonProcess = null;
    if (!app.quitting && state.phase !== "stopping") {
      updateState({ phase: "error", message: `推理服务已退出（代码 ${code ?? "unknown"}）` });
    }
  });

  try {
    await waitForService(serviceUrl, processRef);
    updateState({ phase: "ready", message: "IndexTTS 2.5 已就绪" });
    await mainWindow.loadURL(serviceUrl);
  } catch (error) {
    appendLog(error.stack || error.message);
    updateState({ phase: "error", message: error.message });
    throw error;
  }
  return state;
}

async function chooseDownloadTarget() {
  if (state.modelDir) return state.modelDir;
  const result = await dialog.showOpenDialog(mainWindow, {
    title: "选择模型保存位置（将在其中创建 IndexTTS-2.5 文件夹）",
    buttonLabel: "选择保存位置",
    properties: ["openDirectory", "createDirectory"]
  });
  if (result.canceled || result.filePaths.length === 0) return "";
  const target = path.join(result.filePaths[0], "IndexTTS-2.5");
  fs.mkdirSync(target, { recursive: true });
  updateState({ modelDir: target, modelValid: false, missingFiles: [] });
  return target;
}

async function downloadExternalModel(source) {
  if (!["huggingface", "modelscope"].includes(source)) {
    throw new Error("Unknown model download source.");
  }
  if (downloadProcess && downloadProcess.exitCode === null) return state;
  if (pythonProcess && pythonProcess.exitCode === null) {
    updateState({ phase: "ready", message: "推理服务运行中，无法同时下载模型" });
    return state;
  }

  const target = await chooseDownloadTarget();
  if (!target) return state;
  const runtime = runtimePaths();
  const scriptPath = path.join(runtime.backendRoot, "desktop_model_download.py");
  const pythonPathParts = [runtime.backendRoot, runtime.sitePackages];
  if (process.env.PYTHONPATH) pythonPathParts.push(process.env.PYTHONPATH);

  updateState({
    phase: "downloading",
    message: `正在从 ${source === "modelscope" ? "ModelScope" : "Hugging Face"} 下载模型…`,
    modelDir: target,
    modelValid: false,
    missingFiles: []
  });
  appendLog(`Downloading external model from ${source} to ${target}`);
  downloadProcess = spawn(runtime.pythonExe, [
    "-u", scriptPath, "--target", target, "--source", source
  ], {
    cwd: runtime.backendRoot,
    windowsHide: true,
    env: {
      ...process.env,
      PYTHONUTF8: "1",
      PYTHONUNBUFFERED: "1",
      PYTHONPATH: pythonPathParts.join(path.delimiter),
      HF_HOME: path.join(target, "hf_cache"),
      HF_HUB_CACHE: path.join(target, "hf_cache"),
      MODELSCOPE_CACHE: path.join(target, "modelscope_cache")
    }
  });

  const processRef = downloadProcess;
  cancelledDownloadProcess = null;
  processRef.stdout.on("data", (chunk) => appendLog(chunk.toString("utf8")));
  processRef.stderr.on("data", (chunk) => appendLog(chunk.toString("utf8")));

  const exitCode = await new Promise((resolve) => {
    processRef.on("error", (error) => {
      appendLog(`Model download process error: ${error.stack || error.message}`);
      resolve(-1);
    });
    processRef.on("exit", (code, signal) => {
      appendLog(`Model download exited: code=${code}, signal=${signal}`);
      resolve(code ?? -1);
    });
  });
  if (downloadProcess === processRef) downloadProcess = null;
  const wasCancelled = cancelledDownloadProcess === processRef;
  if (wasCancelled) cancelledDownloadProcess = null;

  const validation = validateModelDirectory(target);
  if (exitCode === 0 && validation.valid) {
    writeSettings({ ...readSettings(), modelDir: target });
    updateState({
      phase: "idle",
      message: "IndexTTS 2.5 模型下载完成，可以启动",
      modelDir: target,
      modelValid: true,
      missingFiles: []
    });
  } else {
    updateState({
      phase: "error",
      message: wasCancelled ? "模型下载已取消，可稍后继续" : "模型下载未完成，请查看日志后重试",
      modelDir: target,
      modelValid: validation.valid,
      missingFiles: validation.missingFiles
    });
  }
  return state;
}

function cancelModelDownload() {
  if (!downloadProcess || downloadProcess.exitCode !== null) return;
  const processRef = downloadProcess;
  downloadProcess = null;
  cancelledDownloadProcess = processRef;
  processRef.kill();
  updateState({ phase: "error", message: "正在取消模型下载，可稍后继续断点下载" });
}

function stopPythonService() {
  if (!pythonProcess || pythonProcess.exitCode !== null) {
    pythonProcess = null;
    return;
  }
  updateState({ phase: "stopping", message: "正在关闭推理服务…" });
  pythonProcess.kill();
  pythonProcess = null;
}

function isTrustedSetupFrame(frame) {
  if (!frame || !frame.url) return false;
  try {
    const url = new URL(frame.url);
    return url.protocol === "file:" && path.basename(url.pathname) === "index.html";
  } catch {
    return false;
  }
}

function assertTrustedSender(event) {
  if (!isTrustedSetupFrame(event.senderFrame)) {
    throw new Error("Rejected IPC request from an untrusted renderer.");
  }
}

function registerIpcHandlers() {
  ipcMain.handle("desktop:get-state", (event) => {
    assertTrustedSender(event);
    return state;
  });

  ipcMain.handle("desktop:choose-model-directory", async (event) => {
    assertTrustedSender(event);
    const result = await dialog.showOpenDialog(mainWindow, {
      title: "选择 IndexTTS 2.5 模型目录",
      properties: ["openDirectory"]
    });
    if (result.canceled || result.filePaths.length === 0) return state;
    const modelDir = result.filePaths[0];
    const validation = validateModelDirectory(modelDir);
    writeSettings({ ...readSettings(), modelDir });
    updateState({
      modelDir,
      modelValid: validation.valid,
      missingFiles: validation.missingFiles,
      phase: validation.valid ? "idle" : "model-required",
      message: validation.valid ? "模型校验通过，可以启动" : "该目录不是完整的 IndexTTS 2.5 模型"
    });
    return state;
  });

  ipcMain.handle("desktop:start-service", async (event) => {
    assertTrustedSender(event);
    try {
      return await startPythonService();
    } catch {
      return state;
    }
  });

  ipcMain.handle("desktop:apply-runtime-profile", (event, profileName) => {
    assertTrustedSender(event);
    if (pythonProcess && pythonProcess.exitCode === null) {
      throw new Error("推理服务运行中；应用运行方案需要先重启应用。");
    }
    const profile = resolveRuntimeProfile(profileName, state.recommendedProfile);
    const nextSettings = {
      ...readSettings(),
      runtimeProfile: profile.name,
      accelerationMode: profile.accelerationMode,
      precisionMode: profile.precisionMode,
      referenceDevice: profile.referenceDevice,
      reuseDefaultEmotion: profile.reuseDefaultEmotion
    };
    writeSettings(nextSettings);
    updateState({
      runtimeProfile: profile.name,
      accelerationMode: profile.accelerationMode,
      precisionMode: profile.precisionMode,
      referenceDevice: profile.referenceDevice,
      reuseDefaultEmotion: profile.reuseDefaultEmotion,
      message: `已应用“${profile.label}”方案，点击启动后生效`
    });
    appendLog(`Runtime profile applied: ${profile.name} | ${profile.description}`);
    return state;
  });

  ipcMain.handle("desktop:set-acceleration", (event, mode) => {
    assertTrustedSender(event);
    const allowed = ["off", "auto_safe", "bigvgan_cuda", "torch_compile", "gpt_accel", "deepspeed"];
    if (!allowed.includes(mode)) throw new Error("Unknown acceleration mode.");
    if (pythonProcess && pythonProcess.exitCode === null) {
      throw new Error("推理服务运行中；更改加速模式需要先重启应用。");
    }
    writeSettings({ ...readSettings(), accelerationMode: mode, runtimeProfile: "custom" });
    updateState({ accelerationMode: mode, runtimeProfile: "custom", message: "加速模式已保存，启动时生效" });
    return state;
  });

  ipcMain.handle("desktop:set-runtime-options", (event, options) => {
    assertTrustedSender(event);
    if (pythonProcess && pythonProcess.exitCode === null) {
      throw new Error("推理服务运行中；更改运行模式需要先重启应用。");
    }
    const precisionModes = ["auto", "bfloat16", "float16", "float32"];
    const referenceDevices = ["auto", "same", "cpu"];
    const precisionMode = options && options.precisionMode;
    const referenceDevice = options && options.referenceDevice;
    if (!precisionModes.includes(precisionMode)) throw new Error("Unknown precision mode.");
    if (!referenceDevices.includes(referenceDevice)) throw new Error("Unknown reference device.");
    const reuseDefaultEmotion = Boolean(options.reuseDefaultEmotion);
    writeSettings({
      ...readSettings(),
      precisionMode,
      referenceDevice,
      reuseDefaultEmotion,
      runtimeProfile: "custom"
    });
    updateState({
      precisionMode,
      referenceDevice,
      reuseDefaultEmotion,
      runtimeProfile: "custom",
      message: "低显存/精度设置已保存，启动时生效"
    });
    return state;
  });

  ipcMain.handle("desktop:refresh-diagnostics", async (event) => {
    assertTrustedSender(event);
    updateState({ diagnosticsBusy: true, message: "正在重新检测加速环境（不会加载模型）…" });
    try {
      await probeRuntimeHardware();
      updateState({ message: "加速环境检测完成；请选择方案后手动启动" });
    } catch (error) {
      appendLog(`Acceleration preflight failed (model was not loaded): ${error.message}`);
      updateState({
        diagnosticsBusy: false,
        message: `加速环境检测失败：${error.message}`
      });
    }
    return state;
  });

  ipcMain.handle("desktop:export-diagnostics", async (event) => {
    assertTrustedSender(event);
    return exportDiagnosticReport();
  });

  ipcMain.handle("desktop:run-runtime-benchmark", async (event) => {
    assertTrustedSender(event);
    try {
      return await runRuntimeBenchmark();
    } catch (error) {
      appendLog(error.stack || error.message);
      updateState({ phase: "error", benchmarkBusy: false, message: error.message });
      return state;
    }
  });

  ipcMain.handle("desktop:cancel-runtime-benchmark", (event) => {
    assertTrustedSender(event);
    cancelRuntimeBenchmark();
    return state;
  });

  ipcMain.handle("desktop:apply-benchmark-recommendation", (event) => {
    assertTrustedSender(event);
    const mode = state.benchmarkReport?.recommendation?.mode;
    const allowed = ["off", "auto_safe", "bigvgan_cuda", "torch_compile", "gpt_accel", "deepspeed"];
    if (!allowed.includes(mode)) throw new Error("当前没有可应用的真实基准推荐。");
    writeSettings({ ...readSettings(), accelerationMode: mode, runtimeProfile: "custom" });
    updateState({
      accelerationMode: mode,
      runtimeProfile: "custom",
      message: `已应用真实基准推荐“${mode}”，点击启动后生效`
    });
    return state;
  });

  ipcMain.handle("desktop:check-updates", async () => {
    try {
      return await checkForUpdates();
    } catch (error) {
      updateState({ updateBusy: false, message: `检查更新失败：${error.message}` });
      return state;
    }
  });

  ipcMain.handle("desktop:open-update-page", async (event, target) => {
    const urls = {
      official: "https://github.com/index-tts/index-tts",
      node: "https://github.com/T8mars/comfyui-indextts25-t8"
    };
    if (urls[target]) await shell.openExternal(urls[target]);
  });

  ipcMain.handle("desktop:download-model", async (event, source) => {
    assertTrustedSender(event);
    try {
      return await downloadExternalModel(source);
    } catch (error) {
      appendLog(error.stack || error.message);
      updateState({ phase: "error", message: error.message });
      return state;
    }
  });

  ipcMain.handle("desktop:cancel-model-download", (event) => {
    assertTrustedSender(event);
    cancelModelDownload();
    return state;
  });

  ipcMain.handle("desktop:stop-service", (event) => {
    assertTrustedSender(event);
    stopPythonService();
    return state;
  });

  ipcMain.handle("desktop:open-model-page", async (event, source) => {
    assertTrustedSender(event);
    const url = MODEL_URLS[source];
    if (!url) throw new Error("Unknown model source.");
    await shell.openExternal(url);
  });

  ipcMain.handle("desktop:open-logs", async (event) => {
    assertTrustedSender(event);
    fs.mkdirSync(logsDirectory(), { recursive: true });
    await shell.openPath(logsDirectory());
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1240,
    height: 820,
    minWidth: 980,
    minHeight: 680,
    title: APP_TITLE,
    show: false,
    backgroundColor: "#0f1117",
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true
    }
  });

  mainWindow.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  mainWindow.webContents.on("will-navigate", (event, targetUrl) => {
    const allowedLocal = targetUrl.startsWith("file:") ||
      (activePort && targetUrl.startsWith(`http://127.0.0.1:${activePort}`));
    if (!allowedLocal) event.preventDefault();
  });
  mainWindow.once("ready-to-show", () => mainWindow.show());
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
  mainWindow.loadFile(path.join(__dirname, "index.html"));
}

const singleInstance = app.requestSingleInstanceLock();
if (!singleInstance) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });

  app.whenReady().then(() => {
    const settings = readSettings();
    let modelDir = commandLineModelDirectory() || settings.modelDir || "";
    if (!app.isPackaged && !modelDir) modelDir = path.join(projectRoot(), "checkpoints");
    const validation = validateModelDirectory(modelDir);
    state = {
      ...state,
      modelDir,
      modelValid: validation.valid,
      missingFiles: validation.missingFiles,
      accelerationMode: settings.accelerationMode || "off",
      precisionMode: settings.precisionMode || "auto",
      referenceDevice: settings.referenceDevice || "auto",
      reuseDefaultEmotion: Boolean(settings.reuseDefaultEmotion),
      runtimeProfile: settings.runtimeProfile || "custom",
      recommendedProfile: "compatibility",
      hardwareSummary: "正在检测显卡（只检测环境，不加载模型）…",
      accelerationDiagnostics: null,
      diagnosticsBusy: true,
      benchmarkBusy: false,
      benchmarkReport: null,
      updateBusy: false,
      updateReport: null,
      phase: validation.valid ? "idle" : "model-required",
      message: validation.valid ? "模型校验通过，可以启动" : "请选择完整的 IndexTTS 2.5 模型目录"
    };
    registerIpcHandlers();
    createWindow();
    probeRuntimeHardware().catch((error) => {
      appendLog(`Hardware probe failed (model was not loaded): ${error.message}`);
      updateState({
        recommendedProfile: "compatibility",
        hardwareSummary: "显卡检测失败；可手动选择运行方案，模型尚未加载。",
        diagnosticsBusy: false
      });
    });
  });
}

app.on("before-quit", () => {
  app.quitting = true;
  cancelModelDownload();
  cancelRuntimeBenchmark();
  stopPythonService();
});

app.on("window-all-closed", () => app.quit());
