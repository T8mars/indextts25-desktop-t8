const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { spawnSync } = require("node:child_process");
const { MAX_GITHUB_ASSET_BYTES, sha256File } = require("../src/update_manager");

const desktopRoot = path.resolve(__dirname, "..");
const projectRoot = path.resolve(desktopRoot, "..");
const packageJson = JSON.parse(fs.readFileSync(path.join(desktopRoot, "package.json"), "utf8"));

function argumentValue(name, fallback = "") {
  const index = process.argv.indexOf(name);
  return index >= 0 && process.argv[index + 1] ? process.argv[index + 1] : fallback;
}

function safeRuntimeRoot(value) {
  const normalized = String(value || "").replace(/\\/g, "/").replace(/^\.\//, "");
  if (!normalized.startsWith("resources/") || normalized.includes("..") || path.isAbsolute(normalized)) {
    throw new Error(`Invalid runtime root: ${value}`);
  }
  return normalized;
}

function listFiles(root, relativeRoot) {
  const absoluteRoot = path.resolve(root, ...relativeRoot.split("/"));
  if (!absoluteRoot.startsWith(`${path.resolve(root)}${path.sep}`) || !fs.existsSync(absoluteRoot)) {
    throw new Error(`Packaged runtime root is missing: ${relativeRoot}`);
  }
  if (fs.statSync(absoluteRoot).isFile()) return [absoluteRoot];
  const files = [];
  const walk = (directory) => {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const absolute = path.join(directory, entry.name);
      if (entry.isSymbolicLink()) throw new Error(`Runtime package cannot contain a symlink: ${absolute}`);
      if (entry.isDirectory()) walk(absolute);
      else if (entry.isFile()) files.push(absolute);
      else throw new Error(`Unsupported runtime file type: ${absolute}`);
    }
  };
  walk(absoluteRoot);
  return files;
}

async function buildFileManifest(packagedRoot, roots, runtimeVersion) {
  const unique = new Map();
  for (const root of roots) {
    for (const absolute of listFiles(packagedRoot, root)) {
      const relative = path.relative(packagedRoot, absolute).split(path.sep).join("/");
      const key = relative.toLowerCase();
      if (unique.has(key)) throw new Error(`Runtime file is included more than once: ${relative}`);
      unique.set(key, absolute);
    }
  }
  const files = [];
  let totalSize = 0;
  let completed = 0;
  for (const [relativeKey, absolute] of [...unique.entries()].sort(([left], [right]) => left.localeCompare(right, "en"))) {
    const relative = path.relative(packagedRoot, absolute).split(path.sep).join("/");
    const size = fs.statSync(absolute).size;
    files.push({ path: relative, size, sha256: await sha256File(absolute) });
    totalSize += size;
    completed += 1;
    if (completed % 1000 === 0 || completed === unique.size) {
      process.stdout.write(`Hashed runtime files: ${completed}/${unique.size}\r`);
    }
  }
  process.stdout.write("\n");
  return { schemaVersion: 1, runtimeVersion, totalSize, files };
}

async function splitArchive(archivePath, outputRoot, archiveName, maximumPartBytes) {
  if (!Number.isSafeInteger(maximumPartBytes) || maximumPartBytes <= 0 || maximumPartBytes >= MAX_GITHUB_ASSET_BYTES) {
    throw new Error("Runtime part size must be positive and remain below GitHub's 2 GiB limit.");
  }
  const parts = [];
  const input = fs.openSync(archivePath, "r");
  const buffer = Buffer.allocUnsafe(8 * 1024 * 1024);
  let archiveOffset = 0;
  try {
    for (let partIndex = 1; archiveOffset < fs.statSync(archivePath).size; partIndex += 1) {
      const assetName = `${archiveName}.part${String(partIndex).padStart(2, "0")}`;
      const partPath = path.join(outputRoot, assetName);
      const output = fs.openSync(partPath, "w");
      let partSize = 0;
      try {
        while (partSize < maximumPartBytes) {
          const wanted = Math.min(buffer.length, maximumPartBytes - partSize);
          const bytesRead = fs.readSync(input, buffer, 0, wanted, archiveOffset);
          if (!bytesRead) break;
          fs.writeSync(output, buffer, 0, bytesRead);
          archiveOffset += bytesRead;
          partSize += bytesRead;
        }
      } finally {
        fs.closeSync(output);
      }
      parts.push({ assetName, size: partSize, sha256: await sha256File(partPath) });
    }
  } finally {
    fs.closeSync(input);
  }
  return parts;
}

async function main() {
  if (process.platform !== "win32") throw new Error("The desktop runtime builder currently requires Windows.");
  const runtimeMetadata = JSON.parse(
    fs.readFileSync(path.join(projectRoot, "desktop_runtime_manifest.json"), "utf8")
  );
  if (runtimeMetadata.schemaVersion !== 1 || !/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(runtimeMetadata.runtimeVersion)) {
    throw new Error("desktop_runtime_manifest.json has an invalid schema or runtimeVersion.");
  }
  const runtimeVersion = runtimeMetadata.runtimeVersion;
  const roots = [...new Set((runtimeMetadata.roots || []).map(safeRuntimeRoot))];
  if (!roots.length) throw new Error("desktop_runtime_manifest.json does not declare runtime roots.");
  const packagedRoot = path.resolve(argumentValue(
    "--packaged-root",
    path.join(desktopRoot, "out", `T8star-Aix-IndexTTS-2.5-v${packageJson.version}-win32-x64`)
  ));
  const outputRoot = path.resolve(argumentValue(
    "--output",
    path.join(desktopRoot, "out", `runtime-v${runtimeVersion}`)
  ));
  fs.rmSync(outputRoot, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
  fs.mkdirSync(outputRoot, { recursive: true });

  const runtimeFileManifest = await buildFileManifest(packagedRoot, roots, runtimeVersion);
  const manifestPath = path.join(outputRoot, "runtime-files.json");
  fs.writeFileSync(manifestPath, `${JSON.stringify(runtimeFileManifest, null, 2)}\n`, "utf8");
  const archiveName = `desktop-runtime-v${runtimeVersion}-win32-x64.zip`;
  const archivePath = path.join(outputRoot, archiveName);
  const tarPath = path.join(process.env.SystemRoot || "C:\\Windows", "System32", "tar.exe");
  const archiveArguments = ["-a", "-cf", archivePath, "-C", outputRoot, "runtime-files.json"];
  for (const root of roots) archiveArguments.push("-C", packagedRoot, root);
  const compressed = spawnSync(tarPath, archiveArguments, {
    encoding: "utf8",
    windowsHide: true,
    maxBuffer: 16 * 1024 * 1024
  });
  if (compressed.status !== 0 || !fs.existsSync(archivePath)) {
    throw new Error(`Failed to build runtime ZIP: ${compressed.stderr || compressed.stdout}`);
  }
  const archiveSize = fs.statSync(archivePath).size;
  const archiveSha256 = await sha256File(archivePath);
  const maximumPartBytes = Number(argumentValue("--part-bytes", String(1792 * 1024 * 1024)));
  const parts = await splitArchive(archivePath, outputRoot, archiveName, maximumPartBytes);
  const releaseTag = String(runtimeMetadata.releaseTag || `runtime-v${runtimeVersion}`);
  if (releaseTag !== `runtime-v${runtimeVersion}`) {
    throw new Error("Runtime releaseTag must match runtime-v<runtimeVersion>.");
  }
  for (const part of parts) {
    part.url = `https://github.com/${runtimeMetadata.repository}/releases/download/${releaseTag}/${encodeURIComponent(part.assetName)}`;
  }
  const descriptor = {
    schemaVersion: 1,
    version: runtimeVersion,
    archiveName,
    archiveSize,
    unpackedSize: runtimeFileManifest.totalSize,
    archiveSha256,
    roots,
    fileManifest: {
      path: "runtime-files.json",
      size: fs.statSync(manifestPath).size,
      sha256: await sha256File(manifestPath)
    },
    parts,
    restartRequired: true
  };
  const descriptorPath = path.join(outputRoot, "desktop-runtime-package.json");
  fs.writeFileSync(descriptorPath, `${JSON.stringify(descriptor, null, 2)}\n`, "utf8");
  if (!process.argv.includes("--keep-archive")) fs.rmSync(archivePath, { force: true });
  fs.rmSync(manifestPath, { force: true });
  if (process.argv.includes("--publish")) {
    const title = `T8star-Aix Desktop Runtime ${runtimeVersion}`;
    const notes = [
      `Python ${runtimeMetadata.pythonVersion} / Torch ${runtimeMetadata.torchVersion} / CUDA ${runtimeMetadata.cudaRuntime}.`,
      "This runtime is installed only through a signed Desktop update manifest.",
      `Every part is below GitHub's 2 GiB asset limit; the launcher verifies part, archive, and per-file SHA-256 values.`
    ].join("\n\n");
    const view = spawnSync("gh", ["release", "view", releaseTag], { encoding: "utf8", windowsHide: true });
    if (view.status !== 0) {
      const create = spawnSync("gh", [
        "release", "create", releaseTag,
        "--title", title,
        "--notes", notes
      ], { encoding: "utf8", windowsHide: true });
      if (create.status !== 0) throw new Error(`Failed to create runtime Release: ${create.stderr || create.stdout}`);
    }
    const upload = spawnSync("gh", [
      "release", "upload", releaseTag,
      descriptorPath,
      ...parts.map((part) => path.join(outputRoot, part.assetName)),
      "--clobber"
    ], { encoding: "utf8", windowsHide: true, maxBuffer: 16 * 1024 * 1024 });
    if (upload.status !== 0) throw new Error(`Failed to upload runtime Release: ${upload.stderr || upload.stdout}`);
  }
  console.log(JSON.stringify({
    runtimeVersion,
    packagedRoot,
    outputRoot,
    descriptorPath,
    archiveSize,
    files: runtimeFileManifest.files.length,
    parts: parts.length
  }, null, 2));
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});
