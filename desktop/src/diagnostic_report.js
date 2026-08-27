"use strict";

function createDiagnosticReport({
  appName,
  appVersion,
  electronVersion,
  nodeVersion,
  platform,
  architecture,
  osRelease,
  osVersion,
  state,
  manifest,
  generatedAt = new Date().toISOString()
}) {
  if (!state || !manifest) throw new Error("Diagnostic report requires state and manifest.");
  return {
    schemaVersion: 1,
    generatedAt,
    application: {
      name: appName,
      desktopVersion: appVersion,
      electronVersion,
      nodeVersion
    },
    system: { platform, architecture, osRelease, osVersion },
    model: {
      directory: state.modelDir || "",
      valid: Boolean(state.modelValid),
      missingFiles: [...(state.missingFiles || [])],
      codeRevision: manifest.codeRevision,
      modelRevision: manifest.modelRevision
    },
    selectedRuntime: {
      profile: state.runtimeProfile,
      accelerationMode: state.accelerationMode,
      precisionMode: state.precisionMode,
      referenceDevice: state.referenceDevice,
      reuseDefaultEmotion: Boolean(state.reuseDefaultEmotion)
    },
    accelerationPreflight: state.accelerationDiagnostics || null,
    runtimeBenchmark: state.benchmarkReport || null,
    updateCheck: state.updateReport || null,
    notes: [
      "预检只检查硬件、依赖与工具链，不加载 IndexTTS 模型。",
      "实际加速初始化失败时会自动回退；真实生效模式以启动日志和 WebUI 环境诊断为准。",
      "DeepSpeed 的 aio.lib/cufile.lib 信息属于可选训练/存储扩展探测，不是语音推理成功的必要条件。"
    ]
  };
}

module.exports = { createDiagnosticReport };
