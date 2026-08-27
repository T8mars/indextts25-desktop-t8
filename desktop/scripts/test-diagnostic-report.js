const assert = require("node:assert/strict");
const { createDiagnosticReport } = require("../src/diagnostic_report");

const diagnostics = {
  capabilities: {
    versions: { torch: "2.8.0+cu128", cuda_runtime: "12.8" }
  },
  modes: {
    gpt_accel: { effective: "gpt_accel", available: true, reason: "ready" }
  }
};
const report = createDiagnosticReport({
  appName: "T8star-Aix · IndexTTS 2.5",
  appVersion: "0.16.0",
  electronVersion: "43.3.0",
  nodeVersion: "24.0.0",
  platform: "win32",
  architecture: "x64",
  osRelease: "test",
  osVersion: "Windows test",
  generatedAt: "2026-08-27T00:00:00.000Z",
  state: {
    modelDir: "D:\\IndexTTS-2.5",
    modelValid: true,
    missingFiles: [],
    runtimeProfile: "max_speed",
    accelerationMode: "gpt_accel",
    precisionMode: "auto",
    referenceDevice: "same",
    reuseDefaultEmotion: true,
    accelerationDiagnostics: diagnostics,
    benchmarkReport: { recommendation: { mode: "gpt_accel" } },
    updateReport: { summary: "当前未发现新版本。" }
  },
  manifest: { codeRevision: "code", modelRevision: "model" }
});

assert.equal(report.schemaVersion, 1);
assert.equal(report.application.desktopVersion, "0.16.0");
assert.equal(report.model.valid, true);
assert.equal(report.selectedRuntime.accelerationMode, "gpt_accel");
assert.equal(report.accelerationPreflight, diagnostics);
assert.equal(report.runtimeBenchmark.recommendation.mode, "gpt_accel");
assert.equal(report.updateCheck.summary, "当前未发现新版本。");
assert.match(report.notes.join("\n"), /不加载 IndexTTS 模型/);
assert.match(report.notes.join("\n"), /aio\.lib\/cufile\.lib/);
assert.throws(() => createDiagnosticReport({}), /requires state and manifest/);

console.log("Diagnostic report schema OK");
