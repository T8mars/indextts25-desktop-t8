const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("desktopApi", {
  getState: () => ipcRenderer.invoke("desktop:get-state"),
  chooseModelDirectory: () => ipcRenderer.invoke("desktop:choose-model-directory"),
  chooseOutputDirectory: () => ipcRenderer.invoke("desktop:choose-output-directory"),
  chooseDataDirectory: () => ipcRenderer.invoke("desktop:choose-data-directory"),
  downloadModel: (source) => ipcRenderer.invoke("desktop:download-model", source),
  cancelModelDownload: () => ipcRenderer.invoke("desktop:cancel-model-download"),
  startService: () => ipcRenderer.invoke("desktop:start-service"),
  applyRuntimeProfile: (profile) => ipcRenderer.invoke("desktop:apply-runtime-profile", profile),
  refreshDiagnostics: () => ipcRenderer.invoke("desktop:refresh-diagnostics"),
  exportDiagnostics: () => ipcRenderer.invoke("desktop:export-diagnostics"),
  runRuntimeBenchmark: () => ipcRenderer.invoke("desktop:run-runtime-benchmark"),
  cancelRuntimeBenchmark: () => ipcRenderer.invoke("desktop:cancel-runtime-benchmark"),
  applyBenchmarkRecommendation: () => ipcRenderer.invoke("desktop:apply-benchmark-recommendation"),
  checkUpdates: () => ipcRenderer.invoke("desktop:check-updates"),
  openUpdatePage: (target) => ipcRenderer.invoke("desktop:open-update-page", target),
  downloadUpdate: () => ipcRenderer.invoke("desktop:download-update"),
  cancelUpdate: () => ipcRenderer.invoke("desktop:cancel-update"),
  installUpdate: () => ipcRenderer.invoke("desktop:install-update"),
  setUpdatePreferences: (options) => ipcRenderer.invoke("desktop:set-update-preferences", options),
  setAcceleration: (mode) => ipcRenderer.invoke("desktop:set-acceleration", mode),
  setRuntimeOptions: (options) => ipcRenderer.invoke("desktop:set-runtime-options", options),
  stopService: () => ipcRenderer.invoke("desktop:stop-service"),
  showLauncher: () => ipcRenderer.invoke("desktop:show-launcher"),
  openModelPage: (source) => ipcRenderer.invoke("desktop:open-model-page", source),
  openLogs: () => ipcRenderer.invoke("desktop:open-logs"),
  openOutputDirectory: () => ipcRenderer.invoke("desktop:open-output-directory"),
  revealOutputItem: (target) => ipcRenderer.invoke("desktop:reveal-output-item", target),
  openDataDirectory: () => ipcRenderer.invoke("desktop:open-data-directory"),
  onState: (callback) => {
    const handler = (_event, state) => callback(state);
    ipcRenderer.on("desktop:state", handler);
    return () => ipcRenderer.removeListener("desktop:state", handler);
  },
  onLog: (callback) => {
    const handler = (_event, line) => callback(line);
    ipcRenderer.on("desktop:log", handler);
    return () => ipcRenderer.removeListener("desktop:log", handler);
  }
});
