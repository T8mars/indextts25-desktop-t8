const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {
  validateModelBundleManifest,
  verifyModelBundleSignature
} = require("../src/update_manager");

const desktopRoot = path.resolve(__dirname, "..");
const projectRoot = path.resolve(desktopRoot, "..");
const mainSource = fs.readFileSync(path.join(desktopRoot, "src", "main.js"), "utf8");
const preloadSource = fs.readFileSync(path.join(desktopRoot, "src", "preload.js"), "utf8");
const rendererSource = fs.readFileSync(path.join(desktopRoot, "src", "renderer.js"), "utf8");
const htmlSource = fs.readFileSync(path.join(desktopRoot, "src", "index.html"), "utf8");
const profileSource = fs.readFileSync(path.join(desktopRoot, "src", "runtime_profiles.js"), "utf8");
const diagnosticSource = fs.readFileSync(path.join(desktopRoot, "src", "diagnostic_report.js"), "utf8");
const forgeSource = fs.readFileSync(path.join(desktopRoot, "forge.config.js"), "utf8");
const releaseWorkflowSource = fs.readFileSync(
  path.join(projectRoot, ".github", "workflows", "desktop-release.yml"),
  "utf8"
);
const webuiSource = fs.readFileSync(path.join(projectRoot, "desktop_webui.py"), "utf8");
const desktopVersion = JSON.parse(
  fs.readFileSync(path.join(desktopRoot, "package.json"), "utf8")
).version;
const modelManifest = JSON.parse(
  fs.readFileSync(path.join(projectRoot, "desktop_model_manifest.json"), "utf8")
);
const modelSignature = fs.readFileSync(
  path.join(projectRoot, "model-bundle.sig"),
  "ascii"
);
const nodePyproject = fs.readFileSync(
  path.join(projectRoot, "comfyui-indextts25-T8", "pyproject.toml"),
  "utf8"
);
const nodeVersion = nodePyproject.match(/^version\s*=\s*"([^"]+)"/m)?.[1];

assert.deepEqual(modelManifest.files["bpe.model"], {
  size: 475997,
  sha256: "b2a5ce8090d32da3642cc4f81fdc996376bc6dd3f4cd5e3d165f71120d9f2bc8",
  modelScopeRepository: "IndexTeam/IndexTTS-2",
});
assert.equal(modelManifest.modelRepository, "t8star/IndexTTS-2.5-Comfy");
assert.match(modelManifest.modelRevision, /^[a-f0-9]{40}$/);
assert.equal(validateModelBundleManifest(modelManifest).bundleVersion, "1.0.0");
assert.equal(Object.keys(modelManifest.files).length, 26);
assert.equal(verifyModelBundleSignature(modelManifest, modelSignature), true);

assert.doesNotMatch(
  mainSource,
  /did-finish-load[\s\S]{0,500}startPythonService\s*\(/,
  "The launcher must not start IndexTTS automatically after the setup page loads."
);
assert.match(
  mainSource,
  /ipcMain\.handle\("desktop:start-service"[\s\S]{0,300}startPythonService\s*\(/,
  "Inference must remain available through the explicit start button IPC action."
);
assert.match(
  htmlSource,
  /先选择精度、参考编码器与加速模式，再点击“启动 IndexTTS 2\.5”/,
  "The setup page must explain the manual startup order."
);
for (const option of ["precisionMode", "referenceDevice", "reuseDefaultEmotion"]) {
  assert.ok(htmlSource.includes(`id="${option}"`), `Launcher must expose ${option}.`);
}
const advancedPanels = [...htmlSource.matchAll(/<details\b[^>]*class="[^"]*\badvanced-card\b[^"]*"[^>]*>/g)];
assert.equal(advancedPanels.length, 4, "Launcher must group the four advanced areas into disclosures.");
for (const [openingTag] of advancedPanels) {
  assert.doesNotMatch(openingTag, /\sopen(?:\s|=|>)/, "Advanced areas must be collapsed by default.");
}
for (const title of [
  "手动高级运行设置",
  "加速环境预检",
  "真实加速基准与推荐",
  "桌面自动更新与上游动态",
]) {
  assert.ok(htmlSource.includes(title), `Launcher must expose the collapsed advanced section: ${title}`);
}
for (const storageControl of [
  "outputPath",
  "dataPath",
  "logPath",
  "chooseOutputButton",
  "chooseDataButton",
  "openOutputButton",
  "openDataButton",
]) {
  assert.ok(htmlSource.includes(`id="${storageControl}"`), `Launcher must expose ${storageControl}.`);
}
for (const profileControl of [
  "runtimeProfile",
  "applyRuntimeProfileButton",
  "hardwareSummary",
  "refreshDiagnosticsButton",
  "exportDiagnosticsButton",
  "diagnosticsVersions",
  "diagnosticsGrid",
  "runBenchmarkButton",
  "cancelBenchmarkButton",
  "applyBenchmarkButton",
  "checkUpdatesButton",
  "openDesktopReleaseButton",
  "autoCheckUpdates",
  "updateChannel",
  "updateResults",
  "updateProgress",
  "downloadUpdateButton",
  "updateModelButton",
  "cancelUpdateButton",
  "installUpdateButton",
  "modelDownloadPanel",
  "modelDownloadProgress",
  "modelDownloadTitle",
  "modelDownloadDetail",
  "modelDownloadDisk",
]) {
  assert.ok(htmlSource.includes(`id="${profileControl}"`), `Launcher must expose ${profileControl}.`);
}
for (const profile of ["low_vram", "balanced", "max_speed", "compatibility"]) {
  assert.ok(profileSource.includes(`${profile}:`), `Runtime profile ${profile} must be defined.`);
}
assert.match(mainSource, /ipcMain\.handle\("desktop:apply-runtime-profile"/);
assert.match(mainSource, /ipcMain\.handle\("desktop:refresh-diagnostics"/);
assert.match(mainSource, /ipcMain\.handle\("desktop:export-diagnostics"/);
assert.match(mainSource, /ipcMain\.handle\("desktop:run-runtime-benchmark"/);
assert.match(mainSource, /ipcMain\.handle\("desktop:check-updates"/);
assert.match(mainSource, /ipcMain\.handle\("desktop:download-update"/);
assert.match(mainSource, /ipcMain\.handle\("desktop:install-update"/);
assert.match(mainSource, /ipcMain\.handle\("desktop:choose-output-directory"/);
assert.match(mainSource, /ipcMain\.handle\("desktop:choose-data-directory"/);
assert.match(mainSource, /ipcMain\.handle\("desktop:show-launcher"/);
assert.match(mainSource, /ipcMain\.handle\("desktop:open-output-directory"/);
assert.match(mainSource, /ipcMain\.handle\("desktop:open-data-directory"/);
assert.match(mainSource, /"--output_dir", outputDirectory\(\)/);
assert.match(mainSource, /"--data_dir", dataDirectory\(\)/);
assert.match(mainSource, /stoppingPythonProcess === processRef/);
assert.match(mainSource, /mainWindow\.on\("close"[\s\S]{0,500}showLauncher\(\)/);
assert.match(preloadSource, /showLauncher: \(\) => ipcRenderer\.invoke\("desktop:show-launcher"\)/);
assert.match(preloadSource, /chooseOutputDirectory/);
assert.match(preloadSource, /chooseDataDirectory/);
assert.match(rendererSource, /chooseOutputButton\.addEventListener/);
assert.match(rendererSource, /chooseDataButton\.addEventListener/);
assert.match(mainSource, /resolveDesktopUpdate/);
assert.match(mainSource, /markUpdateHealthyIfRequested/);
assert.match(releaseWorkflowSource, /T8_UPDATE_PRIVATE_KEY_BASE64/);
assert.match(releaseWorkflowSource, /desktop-update-manifest\.sig/);
assert.match(releaseWorkflowSource, /desktop-app-update-/);
assert.match(mainSource, /fetchText\("https:\/\/api\.github\.com\/repos\/index-tts\/index-tts\/commits\/main"/);
assert.match(mainSource, /resolveModelBundleUpdate/);
assert.match(mainSource, /@@T8_MODEL_PROGRESS@@/);
assert.match(mainSource, /attachModelDownloadOutput/);
assert.doesNotMatch(mainSource, /huggingface\.co\/api\/models\/t8star\/IndexTTS-2\.5-Comfy/);
assert.match(mainSource, /Hardware probe only \(model not loaded\)/);
assert.match(mainSource, /probe_acceleration/);
assert.match(diagnosticSource, /预检只检查硬件、依赖与工具链，不加载 IndexTTS 模型/);
assert.match(diagnosticSource, /aio\.lib\/cufile\.lib/);
assert.match(mainSource, /--precision/);
assert.match(mainSource, /--reference-device/);
assert.ok(nodeVersion, "The ComfyUI node version must be readable from pyproject.toml.");
assert.ok(
  mainSource.includes(`const COMFY_NODE_VERSION = "${nodeVersion}";`),
  "The launcher's bundled ComfyUI node version must match the node pyproject version."
);
assert.ok(
  webuiSource.includes(`DESKTOP_VERSION = "${desktopVersion}"`),
  "The Python WebUI version must match the Electron package version."
);
assert.ok(
  forgeSource.includes('path.join(projectRoot, "context_emotion.py")'),
  "The packaged desktop runtime must include the context-emotion helper."
);
assert.ok(
  webuiSource.includes("上下文情感建议 · 默认关闭，确认后才生成"),
  "The desktop WebUI must expose the confirmation-first context emotion flow."
);
assert.ok(webuiSource.includes("t8-action-dock"));
assert.ok(webuiSource.includes("停止语音任务"));
assert.ok(webuiSource.includes("停止多角色任务"));
assert.ok(webuiSource.includes("格式说明与真实示例 · 新手需要时展开"));
assert.ok(webuiSource.includes("任务恢复、单句重试与工程管理 · 按需展开"));
assert.ok(webuiSource.includes("返回启动配置（停止模型）"));
assert.ok(webuiSource.includes("window.desktopApi.showLauncher"));
assert.ok(webuiSource.includes("技术诊断 JSON（排错时展开或复制）"));
for (const expected of [
  `DESKTOP ${desktopVersion}`,
  `Desktop ${desktopVersion}`,
  `ComfyUI Node ${nodeVersion}`,
  `Core ${modelManifest.codeRevision.slice(0, 8)}`,
  `Model ${modelManifest.modelRevision.slice(0, 8)}`,
]) {
  assert.ok(htmlSource.includes(expected), `The launcher must display ${expected}.`);
}

console.log("Launcher manual-start policy OK");
