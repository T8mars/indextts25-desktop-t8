const crypto = require("node:crypto");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { spawnSync } = require("node:child_process");
const asar = require("@electron/asar");
const {
  MAX_GITHUB_ASSET_BYTES,
  canonicalJson,
  sha256File,
  validateUpdateManifest,
  verifyManifestSignature
} = require("../src/update_manager");

const desktopRoot = path.resolve(__dirname, "..");
const projectRoot = path.resolve(desktopRoot, "..");
const packageJson = JSON.parse(fs.readFileSync(path.join(desktopRoot, "package.json"), "utf8"));
const version = packageJson.version;

const APP_RESOURCE_FILES = [
  "desktop_webui.py",
  "desktop_runtime_benchmark.py",
  "desktop_generation_controls.py",
  "desktop_model_lifecycle.py",
  "desktop_streaming_audio.py",
  "desktop_tasks.py",
  "audio_quality.py",
  "audiocpp_backend.py",
  "speech_review.py",
  "timeline_tools.py",
  "context_emotion.py",
  "desktop_presets.py",
  "desktop_voice_library.py",
  "dialogue_runtime.py",
  "runtime_acceleration.py",
  "runtime_metrics.py",
  "runtime_benchmark.py",
  "candidate_quality.py",
  "desktop_model_download.py",
  "desktop_model_manifest.json",
  "desktop_acceleration_manifest.json",
  "portable-update-helper.ps1",
  "LICENSE",
  "LICENSE_ZH.txt",
  "DISCLAIMER"
];

function argumentValue(name, fallback = "") {
  const index = process.argv.indexOf(name);
  return index >= 0 && process.argv[index + 1] ? process.argv[index + 1] : fallback;
}

function copyDirectory(source, destination) {
  fs.cpSync(source, destination, {
    recursive: true,
    filter: (candidate) => {
      const name = path.basename(candidate);
      return name !== "__pycache__" && !name.endsWith(".pyc") && !name.endsWith(".pyo") &&
        !["infer.py", "infer_v2.py", "cli.py", "cli_v2.py"].includes(name);
    }
  });
}

async function buildSourceAppAsar(destination) {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "t8-desktop-app-update-"));
  try {
    const appSource = path.join(temporary, "app");
    fs.mkdirSync(appSource, { recursive: true });
    copyDirectory(path.join(desktopRoot, "src"), path.join(appSource, "src"));
    for (const filename of ["package.json", "package-lock.json"]) {
      fs.copyFileSync(path.join(desktopRoot, filename), path.join(appSource, filename));
    }
    const install = spawnSync(
      "npm",
      ["ci", "--omit=dev", "--ignore-scripts", "--no-audit", "--no-fund"],
      { cwd: appSource, encoding: "utf8", windowsHide: true, shell: true }
    );
    if (install.status !== 0) {
      throw new Error(`Failed to install app update dependencies: ${install.error?.message || install.stderr || install.stdout}`);
    }
    await asar.createPackage(appSource, destination);
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
  }
}

function listFiles(root) {
  const files = [];
  const walk = (directory) => {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const absolute = path.join(directory, entry.name);
      if (entry.isSymbolicLink()) throw new Error(`Update payload cannot contain a symlink: ${absolute}`);
      if (entry.isDirectory()) walk(absolute);
      else if (entry.isFile()) files.push(absolute);
    }
  };
  walk(root);
  return files.sort((left, right) => left.localeCompare(right, "en"));
}

function loadPrivateKey() {
  if (process.env.T8_UPDATE_PRIVATE_KEY_BASE64) {
    return Buffer.from(process.env.T8_UPDATE_PRIVATE_KEY_BASE64, "base64").toString("utf8");
  }
  const configured = argumentValue("--private-key", process.env.T8_UPDATE_PRIVATE_KEY_FILE || "");
  const keyPath = configured || path.join(os.homedir(), ".codex", "secrets", "indextts-desktop-update-private.pem");
  if (!fs.existsSync(keyPath)) {
    throw new Error(
      "Desktop update signing key is missing. Set T8_UPDATE_PRIVATE_KEY_BASE64, " +
      "T8_UPDATE_PRIVATE_KEY_FILE, or pass --private-key."
    );
  }
  return fs.readFileSync(keyPath, "utf8");
}

function readOptionalJson(filePath, fallback) {
  return fs.existsSync(filePath) ? JSON.parse(fs.readFileSync(filePath, "utf8")) : fallback;
}

async function main() {
  if (process.platform !== "win32") throw new Error("The portable desktop update builder currently requires Windows.");
  const packagedRoot = path.resolve(argumentValue(
    "--packaged-root",
    path.join(desktopRoot, "out", `T8star-Aix-IndexTTS-2.5-v${version}-win32-x64`)
  ));
  const packagedResources = path.join(packagedRoot, "resources");
  const appAsar = path.join(packagedResources, "app.asar");
  const sourceMode = process.argv.includes("--source") || !fs.existsSync(appAsar);

  const outputRoot = path.resolve(argumentValue("--output", path.join(desktopRoot, "out", `update-v${version}`)));
  const payloadRoot = path.join(outputRoot, "payload");
  fs.rmSync(outputRoot, { recursive: true, force: true });
  fs.mkdirSync(path.join(payloadRoot, "resources"), { recursive: true });
  const payloadAppAsar = path.join(payloadRoot, "resources", "app.asar");
  if (sourceMode) await buildSourceAppAsar(payloadAppAsar);
  else fs.copyFileSync(appAsar, payloadAppAsar);

  for (const directory of ["indextts", "assets"]) {
    const source = sourceMode ? path.join(projectRoot, directory) : path.join(packagedResources, directory);
    if (!fs.existsSync(source)) throw new Error(`Packaged resource directory is missing: ${directory}`);
    copyDirectory(source, path.join(payloadRoot, "resources", directory));
  }
  for (const filename of APP_RESOURCE_FILES) {
    const source = sourceMode
      ? filename === "portable-update-helper.ps1"
        ? path.join(desktopRoot, "scripts", filename)
        : path.join(projectRoot, filename)
      : path.join(packagedResources, filename);
    if (!fs.existsSync(source)) throw new Error(`Packaged update resource is missing: ${filename}`);
    const destination = path.join(payloadRoot, "resources", filename);
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    fs.copyFileSync(source, destination);
  }

  const files = [];
  for (const absolute of listFiles(payloadRoot)) {
    files.push({
      path: path.relative(payloadRoot, absolute).split(path.sep).join("/"),
      size: fs.statSync(absolute).size,
      sha256: await sha256File(absolute)
    });
  }

  const assetName = `desktop-app-update-v${version}-win32-x64.zip`;
  const archivePath = path.join(outputRoot, assetName);
  const compress = spawnSync(
    "powershell.exe",
    [
      "-NoProfile",
      "-NonInteractive",
      "-ExecutionPolicy", "Bypass",
      "-File", path.join(__dirname, "compress-update.ps1"),
      "-PayloadRoot", payloadRoot,
      "-DestinationPath", archivePath
    ],
    { encoding: "utf8", windowsHide: true }
  );
  if (compress.status !== 0 || !fs.existsSync(archivePath)) {
    throw new Error(`Failed to create update ZIP: ${compress.stderr || compress.stdout}`);
  }
  const archiveSize = fs.statSync(archivePath).size;
  if (archiveSize >= MAX_GITHUB_ASSET_BYTES) {
    throw new Error(`Update ZIP exceeds GitHub's 2 GiB per-file limit: ${archiveSize}`);
  }

  const model = JSON.parse(fs.readFileSync(path.join(projectRoot, "desktop_model_manifest.json"), "utf8"));
  const runtime = readOptionalJson(path.join(projectRoot, "desktop_runtime_manifest.json"), null);
  const releaseConfig = readOptionalJson(path.join(projectRoot, "desktop_release_config.json"), {});
  const channel = version.includes("-") ? "beta" : "stable";
  const manifest = {
    schemaVersion: 1,
    desktopVersion: version,
    channel,
    minimumUpdaterVersion: "0.18.1",
    publishedAt: new Date().toISOString(),
    releaseNotesUrl: `https://github.com/T8mars/indextts25-desktop-t8/releases/tag/v${version}`,
    summary: argumentValue("--summary", `T8star-Aix IndexTTS 2.5 Desktop ${version}`),
    packages: {
      portableApp: {
        assetName,
        size: archiveSize,
        sha256: await sha256File(archivePath),
        restartRequired: true,
        files
      },
      fullPortable: {
        size: Number(releaseConfig.fullPortable?.size || 0),
        sha256: String(releaseConfig.fullPortable?.sha256 || ""),
        urls: Array.isArray(releaseConfig.fullPortable?.urls) ? releaseConfig.fullPortable.urls : []
      }
    },
    model: {
      repository: model.modelRepository,
      revision: model.modelRevision,
      bundleVersion: model.bundleVersion,
      manifestUrl: "https://huggingface.co/t8star/IndexTTS-2.5-Comfy/resolve/main/model-bundle.json",
      signatureUrl: "https://huggingface.co/t8star/IndexTTS-2.5-Comfy/resolve/main/model-bundle.sig"
    },
    runtime: runtime ? {
      version: runtime.runtimeVersion,
      repository: runtime.repository,
      revision: runtime.revision,
      required: Boolean(runtime.required)
    } : null
  };
  if (!manifest.packages.fullPortable.sha256) delete manifest.packages.fullPortable.sha256;
  if (!manifest.packages.fullPortable.size) delete manifest.packages.fullPortable.size;

  const privateKey = loadPrivateKey();
  const signature = crypto.sign(
    null,
    Buffer.from(canonicalJson(manifest), "utf8"),
    privateKey
  ).toString("base64");
  verifyManifestSignature(manifest, signature);

  const manifestPath = path.join(outputRoot, "desktop-update-manifest.json");
  const signaturePath = path.join(outputRoot, "desktop-update-manifest.sig");
  fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  fs.writeFileSync(signaturePath, `${signature}\n`, "ascii");

  const release = {
    tag_name: `v${version}`,
    assets: [
      { name: assetName, size: archiveSize, browser_download_url: `https://example.invalid/${assetName}` }
    ]
  };
  validateUpdateManifest(manifest, release);
  fs.rmSync(payloadRoot, { recursive: true, force: true });
  console.log(JSON.stringify({ version, sourceMode, archivePath, archiveSize, manifestPath, signaturePath, files: files.length }, null, 2));
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});
