const fs = require("node:fs");
const path = require("node:path");
const { prunePackagedRuntime } = require("./scripts/prune-runtime");
const desktopPackage = require("./package.json");
const electronPackage = require("electron/package.json");

const projectRoot = path.resolve(__dirname, "..");

function findCachedElectronZip() {
  const fileName = `electron-v${electronPackage.version}-win32-x64.zip`;
  const roots = [
    process.env.ELECTRON_CACHE,
    process.env.LOCALAPPDATA && path.join(process.env.LOCALAPPDATA, "electron", "Cache")
  ].filter(Boolean);
  for (const root of roots) {
    if (!fs.existsSync(root)) continue;
    const direct = path.join(root, fileName);
    if (fs.existsSync(direct)) return path.dirname(direct);
    for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const candidate = path.join(root, entry.name, fileName);
      if (fs.existsSync(candidate)) return path.dirname(candidate);
    }
  }
  return undefined;
}

function resolvePythonRuntime() {
  const managedPython = path.join(projectRoot, ".python");
  const candidates = fs
    .readdirSync(managedPython, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && entry.name.startsWith("cpython-"))
    .map((entry) => path.join(managedPython, entry.name))
    .filter((candidate) => fs.existsSync(path.join(candidate, "python.exe")));

  if (candidates.length === 0) {
    throw new Error("Bundled CPython runtime was not found under .python/.");
  }

  return fs.realpathSync(candidates[0]);
}

const pythonRuntime = resolvePythonRuntime();
const sitePackages = path.join(projectRoot, ".venv", "Lib", "site-packages");
const electronZipDir = findCachedElectronZip();

if (!fs.existsSync(sitePackages)) {
  throw new Error("Python dependencies were not found under .venv/Lib/site-packages.");
}

module.exports = {
  packagerConfig: {
    name: `T8star-Aix-IndexTTS-2.5-v${desktopPackage.version}`,
    executableName: "T8star-Aix-IndexTTS-2.5",
    electronZipDir,
    asar: true,
    icon: path.join(projectRoot, "assets", "index_icon"),
    extraResource: [
      pythonRuntime,
      sitePackages,
      path.join(projectRoot, "indextts"),
      path.join(projectRoot, "assets"),
      path.join(projectRoot, "desktop_webui.py"),
      path.join(projectRoot, "desktop_runtime_benchmark.py"),
      path.join(projectRoot, "desktop_generation_controls.py"),
      path.join(projectRoot, "desktop_model_lifecycle.py"),
      path.join(projectRoot, "desktop_streaming_audio.py"),
      path.join(projectRoot, "desktop_candidate_workspace.py"),
      path.join(projectRoot, "desktop_job_queue.py"),
      path.join(projectRoot, "desktop_tasks.py"),
      path.join(projectRoot, "desktop_project_bundle.py"),
      path.join(projectRoot, "audio_quality.py"),
      path.join(projectRoot, "audiocpp_backend.py"),
      path.join(projectRoot, "audiocpp_component_manager.py"),
      path.join(projectRoot, "speech_review.py"),
      path.join(projectRoot, "timeline_tools.py"),
      path.join(projectRoot, "context_emotion.py"),
      path.join(projectRoot, "desktop_presets.py"),
      path.join(projectRoot, "desktop_voice_library.py"),
      path.join(projectRoot, "dialogue_runtime.py"),
      path.join(projectRoot, "runtime_acceleration.py"),
      path.join(projectRoot, "runtime_metrics.py"),
      path.join(projectRoot, "runtime_benchmark.py"),
      path.join(projectRoot, "candidate_quality.py"),
      path.join(projectRoot, "segment_rate_workspace.py"),
      path.join(projectRoot, "desktop_model_download.py"),
      path.join(projectRoot, "desktop_model_manifest.json"),
      path.join(projectRoot, "desktop_runtime_manifest.json"),
      path.join(projectRoot, "desktop_acceleration_manifest.json"),
      path.join(__dirname, "scripts", "portable-update-helper.ps1"),
      path.join(projectRoot, "LICENSE"),
      path.join(projectRoot, "LICENSE_ZH.txt"),
      path.join(projectRoot, "DISCLAIMER")
    ],
    ignore: [
      /^\/out($|\/)/,
      /^\/node_modules\/.cache($|\/)/
    ]
  },
  rebuildConfig: {},
  hooks: {
    postPackage: async (_forgeConfig, packageResult) => {
      const legacyPublicEntrypoints = ["infer.py", "infer_v2.py", "cli.py", "cli_v2.py"];
      for (const outputPath of packageResult.outputPaths) {
        const packageModuleRoot = path.join(outputPath, "resources", "indextts");
        for (const filename of legacyPublicEntrypoints) {
          fs.rmSync(path.join(packageModuleRoot, filename), { force: true });
        }
        const report = prunePackagedRuntime(path.join(outputPath, "resources"));
        console.log(`Pruned ${report.savedGiB} GiB of unused PyTorch static development archives.`);
      }
    }
  },
  makers: [
    {
      name: "@electron-forge/maker-zip",
      platforms: ["win32"]
    },
    {
      name: "@electron-forge/maker-squirrel",
      config: {
        name: "T8star_Aix_IndexTTS_2_5",
        authors: "T8star-Aix",
        description: "IndexTTS 2.5 desktop integration by T8star-Aix"
      }
    }
  ]
};
