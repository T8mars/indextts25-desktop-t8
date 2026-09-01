const { app, BrowserWindow, dialog, ipcMain, shell } = require("electron");
const crypto = require("node:crypto");
const fs = require("node:fs");
const https = require("node:https");
const net = require("node:net");
const os = require("node:os");
const path = require("node:path");
const { spawn } = require("node:child_process");
const { createDiagnosticReport } = require("./diagnostic_report");
const {
  compareVersions,
  createDownloadTask,
  assembleFileParts,
  extractAndVerifyUpdate,
  extractAndVerifyRuntimeUpdate,
  normalizeChannel,
  resolveDesktopUpdate,
  resolveModelBundleUpdate,
  validateModelBundleManifest,
  verifyModelBundleSignature,
  verifyPayloadFiles
} = require("./update_manager");
const {
  hardwareSummary,
  recommendRuntimeProfile,
  resolveRuntimeProfile
} = require("./runtime_profiles");

const APP_TITLE = "T8star-Aix · IndexTTS 2.5";
const COMFY_NODE_VERSION = "0.23.0";
const MODEL_DOWNLOAD_PROGRESS_PREFIX = "@@T8_MODEL_PROGRESS@@";
const MODEL_URLS = {
  huggingface: "https://huggingface.co/t8star/IndexTTS-2.5-Comfy",
  modelscope: "https://modelscope.cn/models/IndexTeam/IndexTTS-2.5"
};
let modelManifestCache = null;
let runtimeManifestCache = null;

let mainWindow = null;
let pythonProcess = null;
let stoppingPythonProcess = null;
let downloadProcess = null;
let cancelledDownloadProcess = null;
let benchmarkProcess = null;
let benchmarkCancelled = false;
let updateDownloadTask = null;
let preparedDesktopUpdate = null;
let activeModelDownloadBundle = null;
let activePort = null;
let returningToLauncher = false;
let state = {
  phase: "idle",
  message: "请选择 IndexTTS 2.5 模型目录",
  modelDir: "",
  outputDir: "",
  dataDir: "",
  logDir: "",
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
  updateDownload: null,
  updateReady: false,
  modelDownload: null,
  modelBundleVersion: "",
  autoCheckUpdates: true,
  updateChannel: "stable",
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

async function checkForUpdates() {
  if (state.updateBusy) return state;
  updateState({ updateBusy: true, message: "正在检查桌面程序、官方代码、模型与节点版本…" });
  const manifest = modelManifest();
  const installedManifest = modelManifestForDirectory(state.modelDir);
  const sources = await Promise.allSettled([
    resolveDesktopUpdate({
      currentVersion: app.getVersion(),
      currentRuntimeVersion: runtimeManifest().runtimeVersion,
      channel: state.updateChannel || "stable"
    }),
    fetchText("https://api.github.com/repos/index-tts/index-tts/commits/main"),
    resolveModelBundleUpdate({
      currentVersion: installedManifest.bundleVersion,
      desktopVersion: app.getVersion()
    }),
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
  let desktop = {
    current: app.getVersion(),
    latest: "",
    channel: state.updateChannel || "stable",
    updateAvailable: false,
    manualOnly: true,
    releaseUrl: "https://github.com/T8mars/indextts25-desktop-t8/releases/latest",
    manifest: null
  };
  if (sources[0].status === "fulfilled") desktop = sources[0].value;
  else errors.push(`桌面程序：${sources[0].reason.message || sources[0].reason}`);
  desktop.portableInstallSupported = portableUpdateSupported();
  if (!desktop.portableInstallSupported) desktop.manualOnly = true;
  if (desktop.updateAvailable && desktop.manifestError) {
    errors.push(`桌面更新安全校验：${desktop.manifestError}已禁用程序内自动安装。`);
  }
  const upstream = readJson(sources[1], "官方代码");
  let modelBundle = null;
  if (sources[2].status === "fulfilled") {
    modelBundle = sources[2].value;
    if (!modelBundle.compatible) {
      errors.push(
        `模型包 ${modelBundle.latest} 需要 Desktop ${modelBundle.minimumDesktopVersion} 或更高版本。`
      );
    }
  } else {
    errors.push(`T8star-Aix 模型仓库：${sources[2].reason.message || sources[2].reason}`);
  }
  let remoteNodeVersion = "";
  if (sources[3].status === "fulfilled") {
    remoteNodeVersion = sources[3].value.match(/^version\s*=\s*["']([^"']+)["']/m)?.[1] || "";
  } else {
    errors.push(`节点仓库：${sources[3].reason.message || sources[3].reason}`);
  }
  const codeRevision = String(upstream.sha || "");
  const report = {
    checkedAt: new Date().toISOString(),
    desktop,
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
      pinned: String(installedManifest.bundleVersion || "0.0.0"),
      revision: String(installedManifest.modelRevision || manifest.modelRevision || ""),
      latest: String(modelBundle?.latest || ""),
      latestRevision: String(modelBundle?.revision || ""),
      updateAvailable: Boolean(modelBundle?.updateAvailable && modelBundle?.compatible),
      compatible: modelBundle?.compatible !== false,
      signatureVerified: Boolean(modelBundle?.signatureVerified),
      repositoryUrl: modelBundle?.repositoryUrl || MODEL_URLS.huggingface
    },
    errors
  };
  const updates = [report.desktop, report.node, report.officialCode, report.officialModel]
    .filter((item) => item.updateAvailable).length;
  report.summary = errors.length === 4
    ? "检查失败，请确认网络后重试。"
    : report.desktop.updateAvailable && report.desktop.manualOnly
      ? "发现桌面程序或运行库更新；当前版本需要打开 Release 页面手动更新。"
      : report.desktop.desktopUpdateAvailable && report.desktop.runtimeUpdateAvailable
        ? `发现 Desktop ${report.desktop.latest} 与运行库 ${report.desktop.latestRuntime}，可分层下载并安全安装。`
      : report.desktop.desktopUpdateAvailable
        ? `发现 Desktop ${report.desktop.latest}，可在启动器内下载并安全安装。`
      : report.desktop.runtimeUpdateAvailable
        ? `发现运行库 ${report.desktop.latestRuntime}，可分卷续传并安全安装。`
    : updates
      ? `发现 ${updates} 项上游或节点更新；不会自动覆盖模型和运行库。`
      : "当前未发现新版本。";
  updateState({ updateBusy: false, updateReport: report, message: report.summary });
  writeSettings({ ...readSettings(), lastUpdateCheck: report.checkedAt });
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

function runtimeManifest() {
  if (runtimeManifestCache) return runtimeManifestCache;
  const manifestPath = path.join(
    app.isPackaged ? process.resourcesPath : projectRoot(),
    "desktop_runtime_manifest.json"
  );
  try {
    const parsed = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
    const version = String(parsed.runtimeVersion || "").replace(/^v/i, "");
    if (parsed.schemaVersion !== 1 || !/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(version)) {
      throw new Error("invalid schema or runtimeVersion");
    }
    runtimeManifestCache = { ...parsed, runtimeVersion: version };
  } catch (error) {
    appendLog(`Runtime manifest unavailable: ${error.message}`);
    runtimeManifestCache = { schemaVersion: 1, runtimeVersion: "0.0.0", roots: [] };
  }
  return runtimeManifestCache;
}

const INSTALLED_MODEL_MANIFEST = ".t8star-model-bundle.json";
const INSTALLED_MODEL_SIGNATURE = ".t8star-model-bundle.sig";

function modelManifestForDirectory(modelDir) {
  const bundled = modelManifest();
  if (!modelDir || !fs.existsSync(modelDir)) return bundled;
  const manifestPath = path.join(modelDir, INSTALLED_MODEL_MANIFEST);
  const signaturePath = path.join(modelDir, INSTALLED_MODEL_SIGNATURE);
  if (!fs.existsSync(manifestPath) || !fs.existsSync(signaturePath)) return bundled;
  try {
    const rawManifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
    const signature = fs.readFileSync(signaturePath, "ascii");
    verifyModelBundleSignature(rawManifest, signature);
    const installed = validateModelBundleManifest(rawManifest);
    if (compareVersions(app.getVersion(), installed.minimumDesktopVersion) < 0) return bundled;
    return compareVersions(installed.bundleVersion, bundled.bundleVersion) >= 0 ? installed : bundled;
  } catch (error) {
    appendLog(`Ignoring invalid installed model bundle metadata: ${error.message}`);
    return bundled;
  }
}

function modelBundleCacheDirectory(bundleVersion) {
  return path.join(updatesDirectory(), "model-bundles", String(bundleVersion));
}

async function resolveLatestModelBundle() {
  const installed = modelManifestForDirectory(state.modelDir);
  const resolved = await resolveModelBundleUpdate({
    currentVersion: installed.bundleVersion,
    desktopVersion: app.getVersion()
  });
  if (!resolved.signatureVerified) throw new Error("远程模型清单没有通过签名校验。");
  if (!resolved.compatible) {
    throw new Error(
      `模型包 ${resolved.latest} 需要 Desktop ${resolved.minimumDesktopVersion} 或更高版本。`
    );
  }
  const directory = modelBundleCacheDirectory(resolved.latest);
  fs.mkdirSync(directory, { recursive: true });
  const manifestPath = path.join(directory, "model-bundle.json");
  const signaturePath = path.join(directory, "model-bundle.sig");
  fs.writeFileSync(manifestPath, `${JSON.stringify(resolved.signedManifest, null, 2)}\n`, "utf8");
  fs.writeFileSync(signaturePath, `${resolved.signature}\n`, "ascii");
  verifyModelBundleSignature(
    JSON.parse(fs.readFileSync(manifestPath, "utf8")),
    fs.readFileSync(signaturePath, "ascii")
  );
  return { ...resolved, manifestPath, signaturePath };
}

function installModelBundleMetadata(modelDir, bundle) {
  if (!bundle?.signedManifest || !bundle?.signature) return;
  verifyModelBundleSignature(bundle.signedManifest, bundle.signature);
  validateModelBundleManifest(bundle.signedManifest);
  const manifestPath = path.join(modelDir, INSTALLED_MODEL_MANIFEST);
  const signaturePath = path.join(modelDir, INSTALLED_MODEL_SIGNATURE);
  const temporaryManifest = `${manifestPath}.tmp`;
  const temporarySignature = `${signaturePath}.tmp`;
  fs.writeFileSync(temporaryManifest, `${JSON.stringify(bundle.signedManifest, null, 2)}\n`, "utf8");
  fs.writeFileSync(temporarySignature, `${bundle.signature}\n`, "ascii");
  fs.renameSync(temporaryManifest, manifestPath);
  fs.renameSync(temporarySignature, signaturePath);
}

function settingsPath() {
  return path.join(app.getPath("userData"), "settings.json");
}

function defaultOutputDirectory() {
  return path.join(app.getPath("documents"), "T8star-Aix IndexTTS 2.5", "outputs");
}

function outputDirectory() {
  return path.resolve(state.outputDir || defaultOutputDirectory());
}

function defaultDataDirectory() {
  return app.getPath("userData");
}

function dataDirectory() {
  return path.resolve(state.dataDir || defaultDataDirectory());
}

function logsDirectory() {
  return path.join(dataDirectory(), "logs");
}

function benchmarkDirectory() {
  return path.join(dataDirectory(), "benchmarks");
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

function updatesDirectory() {
  return path.join(userDataDirectory(), "updates");
}

function updateResultPath() {
  return path.join(updatesDirectory(), "last-result.json");
}

function updateHelperPath() {
  return app.isPackaged
    ? path.join(process.resourcesPath, "portable-update-helper.ps1")
    : path.join(projectRoot(), "desktop", "scripts", "portable-update-helper.ps1");
}

function safeUpdateVersion(value) {
  const version = String(value || "").trim();
  if (!/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(version)) {
    throw new Error("桌面更新版本号无效。");
  }
  return version;
}

function readLastUpdateResult() {
  try {
    const result = JSON.parse(fs.readFileSync(updateResultPath(), "utf8"));
    if (result.status === "installed") return `Desktop ${result.version} 已更新并通过启动检查。`;
    if (result.status === "rolled-back") return `Desktop ${result.version} 更新失败，已自动回滚。`;
    if (result.status === "failed") return `上次桌面更新失败：${result.message || "未知原因"}`;
  } catch {
    // No previous update result is normal.
  }
  return "";
}

function commandLineValue(name) {
  const index = process.argv.indexOf(name);
  return index >= 0 && process.argv[index + 1] ? process.argv[index + 1] : "";
}

function markUpdateHealthyIfRequested() {
  const token = String(commandLineValue("--update-token") || "");
  const markerArgument = String(commandLineValue("--update-health-marker") || "");
  if (!token || !markerArgument) return;
  if (!/^[a-f0-9-]{16,64}$/.test(token)) {
    appendLog("Rejected invalid update health token.");
    return;
  }
  const root = path.resolve(updatesDirectory());
  const marker = path.resolve(markerArgument);
  if (!marker.startsWith(`${root}${path.sep}`)) {
    appendLog("Rejected update health marker outside the updater data directory.");
    return;
  }
  fs.mkdirSync(path.dirname(marker), { recursive: true });
  fs.writeFileSync(marker, `${token}\n`, "utf8");
  appendLog("Portable update startup health marker written.");
}

function updateInstallRoot() {
  return path.dirname(process.execPath);
}

function portableUpdateSupported() {
  if (!app.isPackaged || process.platform !== "win32" || process.windowsStore) return false;
  const normalized = updateInstallRoot().toLowerCase();
  if (/(^|[\\/])app-\d+\.\d+\.\d+/.test(normalized)) return false;
  try {
    fs.accessSync(updateInstallRoot(), fs.constants.W_OK);
    return true;
  } catch {
    return false;
  }
}

function updateDownloadState(patch) {
  updateState({ updateDownload: { ...(state.updateDownload || {}), ...patch } });
}

function requiredUpdateDiskBytes(appPackage, runtimePackage) {
  const appFiles = (appPackage?.files || []).reduce((sum, entry) => sum + Number(entry.size || 0), 0);
  const downloadBytes = Number(appPackage?.size || 0) + Number(runtimePackage?.archiveSize || 0);
  const runtimeFiles = Number(runtimePackage?.unpackedSize || 0);
  // Keep download parts, the assembled archive, verified payload and rollback
  // backup at the same time. The 1 GiB reserve avoids filling the system disk.
  return downloadBytes + Number(runtimePackage?.archiveSize || 0) +
    (appFiles + runtimeFiles) * 2 + 1024 * 1024 * 1024;
}

function assertUpdateDiskSpace(directory, requiredBytes) {
  if (typeof fs.statfsSync !== "function") return;
  try {
    const stats = fs.statfsSync(directory, { bigint: true });
    const available = Number(stats.bavail * stats.bsize);
    if (Number.isFinite(available) && available < requiredBytes) {
      const needGiB = (requiredBytes / (1024 ** 3)).toFixed(1);
      const freeGiB = (available / (1024 ** 3)).toFixed(1);
      throw new Error(`更新至少需要约 ${needGiB} GiB 可用空间（当前 ${freeGiB} GiB），用于下载、校验与回滚备份。`);
    }
  } catch (error) {
    if (/更新至少需要/.test(error.message)) throw error;
    appendLog(`Update disk preflight unavailable: ${error.message}`);
  }
}

function mergeVerifiedPayloadLayers(layers, destination) {
  fs.rmSync(destination, { recursive: true, force: true });
  fs.mkdirSync(destination, { recursive: true });
  const files = [];
  const seen = new Map();
  for (const layer of layers) {
    if (!layer) continue;
    for (const entry of layer.files) {
      const key = String(entry.path).toLowerCase();
      const previous = seen.get(key);
      if (previous) {
        if (previous.path !== entry.path || previous.size !== entry.size || previous.sha256 !== entry.sha256) {
          throw new Error(`程序层与运行库层包含冲突文件：${entry.path}`);
        }
        continue;
      }
      seen.set(key, entry);
      const source = path.join(layer.payloadRoot, ...entry.path.split("/"));
      const target = path.join(destination, ...entry.path.split("/"));
      fs.mkdirSync(path.dirname(target), { recursive: true });
      fs.copyFileSync(source, target, fs.constants.COPYFILE_EXCL);
      files.push(entry);
    }
  }
  return files.sort((left, right) => left.path.localeCompare(right.path, "en"));
}

async function downloadDesktopUpdate() {
  if (updateDownloadTask) return state;
  if (!state.updateReport?.desktop) await checkForUpdates();
  const desktopUpdate = state.updateReport?.desktop;
  if (!desktopUpdate?.updateAvailable) throw new Error("当前没有可下载的桌面或运行库更新。");
  if (desktopUpdate.manualOnly || !desktopUpdate.manifest) {
    if (desktopUpdate.releaseUrl) await shell.openExternal(desktopUpdate.releaseUrl);
    throw new Error("该版本需要从 Release 页面下载完整包后手动更新。");
  }
  const targetVersion = safeUpdateVersion(desktopUpdate.latest);
  const appPackage = desktopUpdate.desktopUpdateAvailable
    ? desktopUpdate.manifest.portableApp
    : null;
  const runtimePackage = desktopUpdate.runtimeUpdateAvailable
    ? desktopUpdate.manifest.runtimePackage
    : null;
  if (!appPackage && !runtimePackage) throw new Error("更新清单没有当前设备需要的可安装分层包。");
  const targetRuntimeVersion = runtimePackage?.version || runtimeManifest().runtimeVersion;
  const stagingDirectory = path.join(
    updatesDirectory(),
    `desktop-v${targetVersion}-runtime-v${safeUpdateVersion(targetRuntimeVersion)}`
  );
  fs.mkdirSync(stagingDirectory, { recursive: true });
  assertUpdateDiskSpace(stagingDirectory, requiredUpdateDiskBytes(appPackage, runtimePackage));
  const totalDownloadBytes = Number(appPackage?.size || 0) + Number(runtimePackage?.archiveSize || 0);
  let lastProgressAt = 0;
  let completedDownloadBytes = 0;
  updateState({
    updateReady: false,
    updateDownload: {
      status: "downloading",
      version: targetVersion,
      received: 0,
      total: totalDownloadBytes,
      percent: 0,
      message: runtimePackage
        ? `正在下载 Desktop ${targetVersion} / 运行库 ${targetRuntimeVersion}…`
        : `正在下载 Desktop ${targetVersion}…`
    },
    message: `正在下载 Desktop ${targetVersion} 更新包…`
  });
  const downloadAsset = async (asset, destination, label) => {
    updateDownloadTask = createDownloadTask({
      url: asset.url,
      destination,
      expectedSize: asset.size,
      expectedSha256: asset.sha256,
      onProgress: ({ received, total }) => {
      const now = Date.now();
      if (now - lastProgressAt < 200 && received < total) return;
      lastProgressAt = now;
        const aggregateReceived = completedDownloadBytes + received;
        const percent = totalDownloadBytes
          ? Math.min(100, Math.round(aggregateReceived * 1000 / totalDownloadBytes) / 10)
          : 0;
        updateDownloadState({
          received: aggregateReceived,
          total: totalDownloadBytes,
          percent,
          message: `${label}：${percent}%`
        });
      }
    });
    await updateDownloadTask.promise;
    completedDownloadBytes += Number(asset.size);
  };
  try {
    let appLayer = null;
    let runtimeLayer = null;
    const archivePaths = [];
    if (appPackage) {
      const appArchivePath = path.join(stagingDirectory, appPackage.assetName);
      await downloadAsset(appPackage, appArchivePath, `正在下载程序层 Desktop ${targetVersion}`);
      archivePaths.push(appArchivePath);
      updateDownloadState({ status: "verifying", message: "程序层下载完成，正在逐文件校验…" });
      const appPayloadRoot = await extractAndVerifyUpdate(
        appArchivePath,
        path.join(stagingDirectory, "app-layer"),
        appPackage.files
      );
      appLayer = { payloadRoot: appPayloadRoot, files: appPackage.files };
    }
    if (runtimePackage) {
      const partPaths = [];
      for (let index = 0; index < runtimePackage.parts.length; index += 1) {
        const part = runtimePackage.parts[index];
        const partPath = path.join(stagingDirectory, part.assetName);
        await downloadAsset(
          part,
          partPath,
          `正在下载运行库分卷 ${index + 1}/${runtimePackage.parts.length}`
        );
        partPaths.push(partPath);
      }
      updateDownloadState({ status: "verifying", percent: 100, message: "分卷下载完成，正在合并并校验运行库…" });
      const runtimeArchivePath = path.join(stagingDirectory, runtimePackage.archiveName);
      await assembleFileParts({
        partPaths,
        destination: runtimeArchivePath,
        expectedSize: runtimePackage.archiveSize,
        expectedSha256: runtimePackage.archiveSha256,
        onProgress: ({ written, total }) => updateDownloadState({
          message: `正在合并运行库分卷：${Math.round(written * 100 / Math.max(1, total))}%`
        })
      });
      archivePaths.push(runtimeArchivePath);
      runtimeLayer = await extractAndVerifyRuntimeUpdate(
        runtimeArchivePath,
        path.join(stagingDirectory, "runtime-layer"),
        runtimePackage
      );
    }
    updateDownloadState({ status: "verifying", percent: 100, message: "正在合并分层载荷并进行最终校验…" });
    const payloadRoot = path.join(stagingDirectory, "prepared-payload");
    const files = mergeVerifiedPayloadLayers([appLayer, runtimeLayer], payloadRoot);
    await verifyPayloadFiles(payloadRoot, files);
    preparedDesktopUpdate = {
      targetVersion,
      targetRuntimeVersion,
      stagingDirectory,
      archivePaths,
      payloadRoot,
      files,
      releaseUrl: desktopUpdate.releaseUrl
    };
    fs.writeFileSync(
      path.join(stagingDirectory, "prepared-update.json"),
      JSON.stringify(preparedDesktopUpdate, null, 2),
      "utf8"
    );
    updateState({
      updateReady: true,
      updateDownload: {
        status: "ready",
        version: targetVersion,
        received: totalDownloadBytes,
        total: totalDownloadBytes,
        percent: 100,
        message: "更新已校验，可以退出并安装。"
      },
      message: runtimePackage
        ? `Desktop ${targetVersion} / 运行库 ${targetRuntimeVersion} 已下载并通过校验。`
        : `Desktop ${targetVersion} 已下载并通过校验。`
    });
  } catch (error) {
    updateState({
      updateReady: false,
      updateDownload: {
        ...(state.updateDownload || {}),
        status: /取消/.test(error.message) ? "cancelled" : "error",
        message: error.message
      },
      message: `桌面更新未准备完成：${error.message}`
    });
    appendLog(`Desktop update download failed: ${error.stack || error.message}`);
  } finally {
    updateDownloadTask = null;
  }
  return state;
}

function cancelDesktopUpdate() {
  if (updateDownloadTask) updateDownloadTask.cancel();
  updateDownloadState({ status: "cancelling", message: "正在取消更新下载；已下载部分会保留以便续传。" });
  return state;
}

async function installDesktopUpdate() {
  if (!preparedDesktopUpdate || !state.updateReady) throw new Error("没有已校验的桌面更新。");
  if (!portableUpdateSupported()) {
    if (preparedDesktopUpdate.releaseUrl) await shell.openExternal(preparedDesktopUpdate.releaseUrl);
    throw new Error("当前安装目录不可写或不是便携版，请从 Release 页面手动更新。");
  }
  if ((pythonProcess && pythonProcess.exitCode === null) || (downloadProcess && downloadProcess.exitCode === null)) {
    throw new Error("请先停止推理服务并等待模型下载结束，再安装桌面更新。");
  }
  await verifyPayloadFiles(preparedDesktopUpdate.payloadRoot, preparedDesktopUpdate.files);
  const confirmation = await dialog.showMessageBox(mainWindow, {
    type: "question",
    buttons: ["退出并安装", "取消"],
    defaultId: 0,
    cancelId: 1,
    title: `安装 Desktop ${preparedDesktopUpdate.targetVersion}`,
    message: "程序将退出、替换已声明的程序/运行库文件并重新启动。",
    detail: `目标运行库 ${preparedDesktopUpdate.targetRuntimeVersion || runtimeManifest().runtimeVersion}。` +
      "模型、音色库、预设、生成记录和用户设置不会被覆盖；若新版本未通过启动检查会自动回滚。"
  });
  if (confirmation.response !== 0) return state;

  const token = crypto.randomUUID().toLowerCase();
  const backupRoot = path.join(
    updatesDirectory(),
    "backups",
    `${app.getVersion()}-to-${preparedDesktopUpdate.targetVersion}-${token}`
  );
  const healthMarker = path.join(preparedDesktopUpdate.stagingDirectory, `health-${token}.txt`);
  const plan = {
    schemaVersion: 1,
    parentPid: process.pid,
    currentVersion: app.getVersion(),
    targetVersion: preparedDesktopUpdate.targetVersion,
    targetRuntimeVersion: preparedDesktopUpdate.targetRuntimeVersion,
    installRoot: updateInstallRoot(),
    updatesRoot: updatesDirectory(),
    payloadRoot: preparedDesktopUpdate.payloadRoot,
    backupRoot,
    executablePath: process.execPath,
    healthMarker,
    healthToken: token,
    resultPath: updateResultPath(),
    files: preparedDesktopUpdate.files
  };
  const planPath = path.join(preparedDesktopUpdate.stagingDirectory, "apply-plan.json");
  fs.writeFileSync(planPath, JSON.stringify(plan, null, 2), "utf8");
  const helperPath = updateHelperPath();
  if (!fs.existsSync(helperPath)) throw new Error(`便携更新助手缺失：${helperPath}`);
  const detachedHelperPath = path.join(preparedDesktopUpdate.stagingDirectory, "portable-update-helper.ps1");
  fs.copyFileSync(helperPath, detachedHelperPath);
  const powershell = path.join(
    process.env.SystemRoot || "C:\\Windows",
    "System32", "WindowsPowerShell", "v1.0", "powershell.exe"
  );
  const helper = spawn(powershell, [
    "-NoProfile",
    "-NonInteractive",
    "-ExecutionPolicy", "Bypass",
    "-File", detachedHelperPath,
    "-PlanPath", planPath
  ], {
    detached: true,
    windowsHide: true,
    stdio: "ignore"
  });
  helper.unref();
  appendLog(`Portable update helper launched for Desktop ${preparedDesktopUpdate.targetVersion}.`);
  app.quitting = true;
  app.quit();
  return state;
}

function setUpdatePreferences(options) {
  const next = {
    ...readSettings(),
    autoCheckUpdates: options?.autoCheckUpdates !== false,
    updateChannel: normalizeChannel(options?.updateChannel)
  };
  writeSettings(next);
  updateState({
    autoCheckUpdates: next.autoCheckUpdates,
    updateChannel: next.updateChannel,
    message: `更新设置已保存：${next.autoCheckUpdates ? "自动检查" : "仅手动检查"} / ${next.updateChannel}`
  });
  return state;
}

function scheduleAutomaticUpdateCheck() {
  if (!state.autoCheckUpdates) return;
  const settings = readSettings();
  const lastChecked = Date.parse(settings.lastUpdateCheck || "");
  if (Number.isFinite(lastChecked) && Date.now() - lastChecked < 24 * 60 * 60 * 1000) return;
  setTimeout(() => {
    checkForUpdates().catch((error) => {
      appendLog(`Automatic desktop update check failed: ${error.message}`);
      updateState({ updateBusy: false, message: `自动检查更新失败：${error.message}` });
    });
  }, 5000);
}

function validateModelDirectory(modelDir, manifestOverride = null) {
  if (!modelDir || !fs.existsSync(modelDir)) {
    return { valid: false, missingFiles: ["模型目录不存在"], manifest: modelManifest() };
  }

  const manifest = manifestOverride || modelManifestForDirectory(modelDir);
  const missingFiles = [];
  for (const [relativePath, metadata] of Object.entries(manifest.files)) {
    const localPath = path.join(modelDir, ...relativePath.split("/"));
    if (!fs.existsSync(localPath)) {
      missingFiles.push(relativePath);
      continue;
    }
    if (fs.statSync(localPath).size !== metadata.size) {
      missingFiles.push(`${relativePath}（版本不匹配）`);
    }
  }

  return { valid: missingFiles.length === 0, missingFiles, manifest };
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

function handleModelDownloadLine(line) {
  const text = String(line || "").trimEnd();
  if (!text) return;
  if (!text.startsWith(MODEL_DOWNLOAD_PROGRESS_PREFIX)) {
    appendLog(text);
    return;
  }
  try {
    const progress = JSON.parse(text.slice(MODEL_DOWNLOAD_PROGRESS_PREFIX.length));
    updateState({
      modelDownload: {
        ...(state.modelDownload || {}),
        ...progress,
        status: progress.phase === "complete"
          ? "complete"
          : progress.phase === "error"
            ? "error"
            : "active"
      },
      message: progress.message || state.message
    });
  } catch (error) {
    appendLog(`无法解析模型下载进度：${error.message}`);
  }
}

function attachModelDownloadOutput(processRef) {
  let stdoutBuffer = "";
  const flushLines = (final = false) => {
    const parts = stdoutBuffer.split(/\r?\n/);
    const tail = parts.pop() || "";
    stdoutBuffer = final ? "" : tail;
    for (const line of parts) handleModelDownloadLine(line);
    if (final && tail) handleModelDownloadLine(tail);
  };
  processRef.stdout.on("data", (chunk) => {
    stdoutBuffer += chunk.toString("utf8");
    flushLines(false);
  });
  processRef.stderr.on("data", (chunk) => appendLog(chunk.toString("utf8")));
  processRef.on("close", () => flushLines(true));
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
  const manifest = modelManifestForDirectory(state.modelDir);
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
  fs.mkdirSync(dataDirectory(), { recursive: true });
  fs.mkdirSync(logsDirectory(), { recursive: true });
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
  appendLog(`Output directory: ${outputDirectory()}`);
  appendLog(`User data directory: ${dataDirectory()}`);
  appendLog(`Log directory: ${logsDirectory()}`);
  appendLog(`Code revision: ${validation.manifest.codeRevision}`);
  appendLog(`Model bundle: ${validation.manifest.bundleVersion}`);
  appendLog(`Model revision: ${validation.manifest.modelRevision}`);
  appendLog(`Optional acceleration mode: ${state.accelerationMode || "off"}`);
  appendLog(`Precision mode: ${state.precisionMode || "auto"}`);
  appendLog(`Reference encoders: ${state.referenceDevice || "auto"}`);
  appendLog(`Fast default emotion: ${Boolean(state.reuseDefaultEmotion)}`);

  const pythonArguments = [
    "-u",
    scriptPath,
    "--model_dir", state.modelDir,
    "--output_dir", outputDirectory(),
    "--data_dir", dataDirectory(),
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
    const expectedStop = stoppingPythonProcess === processRef;
    if (expectedStop) stoppingPythonProcess = null;
    if (!app.quitting && !expectedStop && state.phase !== "stopping") {
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
  let selectedBundle = null;
  if (source === "huggingface") {
    updateState({ phase: "downloading", message: "正在验证 Hugging Face 模型清单签名…" });
    selectedBundle = await resolveLatestModelBundle();
  }
  activeModelDownloadBundle = selectedBundle;

  updateState({
    phase: "downloading",
    message: `正在从 ${source === "modelscope" ? "ModelScope" : "Hugging Face"} 下载模型…`,
    modelDir: target,
    modelValid: false,
    missingFiles: [],
    modelDownload: {
      status: "active",
      phase: "starting",
      source,
      overallPercent: 0,
      phasePercent: 0,
      message: "正在启动模型检查与下载任务…"
    }
  });
  appendLog(`Downloading external model from ${source} to ${target}`);
  const downloadArguments = ["-u", scriptPath, "--target", target, "--source", source];
  if (selectedBundle?.manifestPath) {
    downloadArguments.push("--manifest", selectedBundle.manifestPath);
  }
  downloadProcess = spawn(runtime.pythonExe, downloadArguments, {
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
  attachModelDownloadOutput(processRef);

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

  const completedBundle = activeModelDownloadBundle;
  activeModelDownloadBundle = null;
  const validation = validateModelDirectory(target, completedBundle?.manifest || null);
  if (exitCode === 0 && validation.valid) {
    installModelBundleMetadata(target, completedBundle);
    writeSettings({ ...readSettings(), modelDir: target });
    updateState({
      phase: "idle",
      message: `IndexTTS 2.5 模型包 ${validation.manifest.bundleVersion} 下载完成，可以启动`,
      modelDir: target,
      modelValid: true,
      missingFiles: [],
      modelBundleVersion: validation.manifest.bundleVersion,
      modelDownload: {
        ...(state.modelDownload || {}),
        status: "complete",
        phase: "complete",
        overallPercent: 100,
        phasePercent: 100,
        message: "完整模型已下载并通过 SHA-256 校验。"
      }
    });
  } else {
    const reportedError = String(state.modelDownload?.error || "").trim();
    updateState({
      phase: "error",
      message: wasCancelled
        ? "模型下载已取消，可稍后继续"
        : reportedError || "模型下载未完成，请查看日志后重试",
      modelDir: target,
      modelValid: false,
      missingFiles: validation.missingFiles,
      modelDownload: {
        ...(state.modelDownload || {}),
        status: wasCancelled ? "cancelled" : "error",
        message: wasCancelled
          ? "下载已取消，保留的断点文件可在下次继续。"
          : reportedError
            ? `失败：${reportedError}`
            : "模型下载或校验失败，请查看错误日志后重试。"
      }
    });
  }
  return state;
}

function cancelModelDownload() {
  if (!downloadProcess || downloadProcess.exitCode !== null) return;
  const processRef = downloadProcess;
  downloadProcess = null;
  cancelledDownloadProcess = processRef;
  activeModelDownloadBundle = null;
  processRef.kill();
  updateState({
    phase: "error",
    message: "正在取消模型下载，可稍后继续断点下载",
    modelDownload: {
      ...(state.modelDownload || {}),
      status: "cancelling",
      message: "正在停止下载；已经写入的断点文件不会删除。"
    }
  });
}

function stopPythonService() {
  if (!pythonProcess || pythonProcess.exitCode !== null) {
    pythonProcess = null;
    return;
  }
  updateState({ phase: "stopping", message: "正在关闭推理服务…" });
  stoppingPythonProcess = pythonProcess;
  pythonProcess.kill();
  pythonProcess = null;
}

function isTrustedRendererFrame(frame) {
  if (!frame || !frame.url) return false;
  try {
    const url = new URL(frame.url);
    if (url.protocol === "file:" && path.basename(url.pathname) === "index.html") return true;
    return Boolean(
      activePort &&
      url.protocol === "http:" &&
      url.hostname === "127.0.0.1" &&
      Number(url.port) === activePort
    );
  } catch {
    return false;
  }
}

function assertTrustedSender(event) {
  if (!isTrustedRendererFrame(event.senderFrame)) {
    throw new Error("Rejected IPC request from an untrusted renderer.");
  }
}

async function chooseWorkingDirectory(kind) {
  if (pythonProcess && pythonProcess.exitCode === null) {
    throw new Error("推理服务运行中；修改存放目录前请先返回启动配置并停止模型。");
  }
  const output = kind === "output";
  const result = await dialog.showOpenDialog(mainWindow, {
    title: output ? "选择生成音频保存目录" : "选择用户数据保存目录（音色库、预设、任务与日志）",
    defaultPath: output ? outputDirectory() : dataDirectory(),
    buttonLabel: "使用此目录",
    properties: ["openDirectory", "createDirectory"]
  });
  if (result.canceled || result.filePaths.length === 0) return state;
  const selected = path.resolve(result.filePaths[0]);
  fs.mkdirSync(selected, { recursive: true });
  fs.accessSync(selected, fs.constants.W_OK);
  const nextSettings = {
    ...readSettings(),
    [output ? "outputDir" : "dataDir"]: selected
  };
  writeSettings(nextSettings);
  const patch = output
    ? { outputDir: selected, message: `生成音频将保存到：${selected}` }
    : {
        dataDir: selected,
        logDir: path.join(selected, "logs"),
        message: `音色库、预设、任务与日志将保存到：${selected}`
      };
  updateState(patch);
  return state;
}

async function showLauncher() {
  if (returningToLauncher) return state;
  returningToLauncher = true;
  const hadRunningService = Boolean(pythonProcess && pythonProcess.exitCode === null);
  try {
    if (hadRunningService) stopPythonService();
    await mainWindow.loadFile(path.join(__dirname, "index.html"));
    activePort = null;
    const validation = validateModelDirectory(state.modelDir);
    updateState({
      phase: validation.valid ? "idle" : "model-required",
      message: hadRunningService
        ? "已返回启动配置并停止模型；可修改设置后重新启动"
        : "已返回启动配置",
      serviceUrl: ""
    });
    return state;
  } finally {
    returningToLauncher = false;
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
      modelBundleVersion: validation.manifest.bundleVersion,
      phase: validation.valid ? "idle" : "model-required",
      message: validation.valid ? "模型校验通过，可以启动" : "该目录不是完整的 IndexTTS 2.5 模型"
    });
    return state;
  });

  ipcMain.handle("desktop:choose-output-directory", async (event) => {
    assertTrustedSender(event);
    return chooseWorkingDirectory("output");
  });

  ipcMain.handle("desktop:choose-data-directory", async (event) => {
    assertTrustedSender(event);
    return chooseWorkingDirectory("data");
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

  ipcMain.handle("desktop:check-updates", async (event) => {
    assertTrustedSender(event);
    try {
      return await checkForUpdates();
    } catch (error) {
      updateState({ updateBusy: false, message: `检查更新失败：${error.message}` });
      return state;
    }
  });

  ipcMain.handle("desktop:open-update-page", async (event, target) => {
    assertTrustedSender(event);
    const urls = {
      desktop: state.updateReport?.desktop?.releaseUrl || "https://github.com/T8mars/indextts25-desktop-t8/releases/latest",
      official: "https://github.com/index-tts/index-tts",
      node: "https://github.com/T8mars/comfyui-indextts25-t8"
    };
    if (urls[target]) await shell.openExternal(urls[target]);
  });

  ipcMain.handle("desktop:download-update", async (event) => {
    assertTrustedSender(event);
    try {
      return await downloadDesktopUpdate();
    } catch (error) {
      appendLog(`Desktop update request failed: ${error.stack || error.message}`);
      updateState({ message: `桌面更新失败：${error.message}` });
      return state;
    }
  });

  ipcMain.handle("desktop:cancel-update", (event) => {
    assertTrustedSender(event);
    return cancelDesktopUpdate();
  });

  ipcMain.handle("desktop:install-update", async (event) => {
    assertTrustedSender(event);
    try {
      return await installDesktopUpdate();
    } catch (error) {
      appendLog(`Desktop update install failed: ${error.stack || error.message}`);
      updateState({ message: `无法安装桌面更新：${error.message}` });
      return state;
    }
  });

  ipcMain.handle("desktop:set-update-preferences", (event, options) => {
    assertTrustedSender(event);
    return setUpdatePreferences(options);
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

  ipcMain.handle("desktop:show-launcher", async (event) => {
    assertTrustedSender(event);
    return showLauncher();
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

  ipcMain.handle("desktop:open-output-directory", async (event) => {
    assertTrustedSender(event);
    fs.mkdirSync(outputDirectory(), { recursive: true });
    await shell.openPath(outputDirectory());
  });

  ipcMain.handle("desktop:reveal-output-item", async (event, requestedPath) => {
    assertTrustedSender(event);
    fs.mkdirSync(outputDirectory(), { recursive: true });
    const outputRoot = fs.realpathSync(outputDirectory());
    const requested = String(requestedPath || "").trim();
    if (!requested) throw new Error("No output item was selected.");
    const candidate = path.resolve(requested);
    if (!fs.existsSync(candidate)) throw new Error("The selected output file no longer exists.");
    const resolved = fs.realpathSync(candidate);
    const relative = path.relative(outputRoot, resolved);
    if (relative.startsWith("..") || path.isAbsolute(relative)) {
      throw new Error("Only files inside the configured output directory can be revealed.");
    }
    const stats = fs.statSync(resolved);
    if (stats.isDirectory()) {
      const failure = await shell.openPath(resolved);
      if (failure) throw new Error(failure);
    } else {
      shell.showItemInFolder(resolved);
    }
    return { ok: true };
  });

  ipcMain.handle("desktop:open-data-directory", async (event) => {
    assertTrustedSender(event);
    fs.mkdirSync(dataDirectory(), { recursive: true });
    await shell.openPath(dataDirectory());
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
  mainWindow.webContents.once("did-finish-load", () => markUpdateHealthyIfRequested());
  mainWindow.once("ready-to-show", () => {
    mainWindow.show();
    scheduleAutomaticUpdateCheck();
  });
  mainWindow.on("close", (event) => {
    const currentUrl = mainWindow?.webContents.getURL() || "";
    const showingWebUi = Boolean(
      activePort && currentUrl.startsWith(`http://127.0.0.1:${activePort}`)
    );
    if (!app.quitting && showingWebUi) {
      event.preventDefault();
      showLauncher().catch((error) => {
        appendLog(`Return to launcher failed: ${error.stack || error.message}`);
        updateState({ phase: "error", message: `返回启动配置失败：${error.message}` });
      });
    }
  });
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
    const previousUpdateMessage = readLastUpdateResult();
    const rollbackMessage = process.argv.includes("--update-rollback")
      ? "新版本未通过启动检查，已恢复到旧版本。"
      : "";
    state = {
      ...state,
      modelDir,
      outputDir: path.resolve(settings.outputDir || defaultOutputDirectory()),
      dataDir: path.resolve(settings.dataDir || defaultDataDirectory()),
      logDir: path.join(path.resolve(settings.dataDir || defaultDataDirectory()), "logs"),
      modelValid: validation.valid,
      missingFiles: validation.missingFiles,
      modelBundleVersion: validation.manifest.bundleVersion,
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
      updateDownload: null,
      updateReady: false,
      autoCheckUpdates: settings.autoCheckUpdates !== false,
      updateChannel: normalizeChannel(settings.updateChannel),
      phase: validation.valid ? "idle" : "model-required",
      message: rollbackMessage || previousUpdateMessage
        || (validation.valid ? "模型校验通过，可以启动" : "请选择完整的 IndexTTS 2.5 模型目录")
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
  if (updateDownloadTask) updateDownloadTask.cancel();
  cancelModelDownload();
  cancelRuntimeBenchmark();
  stopPythonService();
});

app.on("window-all-closed", () => app.quit());
