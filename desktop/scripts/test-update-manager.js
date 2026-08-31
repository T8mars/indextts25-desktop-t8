const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const http = require("node:http");
const os = require("node:os");
const path = require("node:path");
const zlib = require("node:zlib");
const {
  canonicalJson,
  compareVersions,
  createDownloadTask,
  assembleFileParts,
  extractAndVerifyUpdate,
  extractAndVerifyRuntimeUpdate,
  resolveDesktopUpdate,
  resolveModelBundleUpdate,
  safeRelativePath,
  validateModelBundleManifest,
  validateUpdateManifest,
  verifyManifestSignature,
  verifyModelBundleSignature,
  verifyPayloadFiles
} = require("../src/update_manager");

const digest = (value) => crypto.createHash("sha256").update(value).digest("hex");

function storedZip(entries) {
  const localParts = [];
  const centralParts = [];
  let offset = 0;
  for (const entry of entries) {
    const name = Buffer.from(entry.name, "utf8");
    const body = Buffer.from(entry.body || "", "utf8");
    const crc = zlib.crc32(body);
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4);
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(body.length, 18);
    local.writeUInt32LE(body.length, 22);
    local.writeUInt16LE(name.length, 26);
    localParts.push(local, name, body);

    const central = Buffer.alloc(46);
    central.writeUInt32LE(0x02014b50, 0);
    central.writeUInt16LE(0x0314, 4);
    central.writeUInt16LE(20, 6);
    central.writeUInt32LE(crc, 16);
    central.writeUInt32LE(body.length, 20);
    central.writeUInt32LE(body.length, 24);
    central.writeUInt16LE(name.length, 28);
    central.writeUInt32LE(Number(entry.externalAttributes || 0) >>> 0, 38);
    central.writeUInt32LE(offset, 42);
    centralParts.push(central, name);
    offset += local.length + name.length + body.length;
  }
  const centralDirectory = Buffer.concat(centralParts);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(entries.length, 8);
  end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(centralDirectory.length, 12);
  end.writeUInt32LE(offset, 16);
  return Buffer.concat([...localParts, centralDirectory, end]);
}

async function main() {
  assert.equal(compareVersions("0.19.0", "0.18.1"), 1);
  assert.equal(compareVersions("0.19.0-beta.2", "0.19.0-beta.1"), 1);
  assert.equal(compareVersions("0.19.0", "0.19.0-beta.2"), 1);
  assert.equal(compareVersions("v0.19.0", "0.19.0"), 0);
  assert.equal(safeRelativePath("resources/desktop_webui.py"), "resources/desktop_webui.py");
  assert.throws(() => safeRelativePath("../settings.json"), /越界/);
  assert.throws(() => safeRelativePath("C:\\Windows\\system.ini"), /反斜杠/);
  assert.throws(() => safeRelativePath("resources/file.txt:payload"), /Windows/);
  assert.throws(() => safeRelativePath("resources/CON.txt"), /保留名/);

  const fileContent = Buffer.from("verified update payload", "utf8");
  const archiveContent = Buffer.from("fake zip bytes for resumable download", "utf8");
  const runtimeFileContent = Buffer.from("verified runtime payload", "utf8");
  const runtimeFileManifest = Buffer.from(JSON.stringify({
    schemaVersion: 1,
    runtimeVersion: "1.0.0",
    totalSize: runtimeFileContent.length,
    files: [{
      path: "resources/runtime/test.bin",
      size: runtimeFileContent.length,
      sha256: digest(runtimeFileContent)
    }]
  }), "utf8");
  const runtimeArchive = storedZip([
    { name: "runtime-files.json", body: runtimeFileManifest },
    { name: "resources/runtime/test.bin", body: runtimeFileContent }
  ]);
  const runtimeSplit = Math.ceil(runtimeArchive.length / 2);
  const runtimeParts = [
    runtimeArchive.subarray(0, runtimeSplit),
    runtimeArchive.subarray(runtimeSplit)
  ];
  const release = {
    tag_name: "v0.19.0",
    name: "Desktop 0.19.0",
    html_url: "https://github.com/T8mars/indextts25-desktop-t8/releases/tag/v0.19.0",
    published_at: "2026-08-29T00:00:00Z",
    draft: false,
    prerelease: false,
    assets: [
      {
        name: "desktop-update-manifest.json",
        size: 1024,
        browser_download_url: "https://example.test/desktop-update-manifest.json"
      },
      {
        name: "desktop-update-manifest.sig",
        size: 88,
        browser_download_url: "https://example.test/desktop-update-manifest.sig"
      },
      {
        name: "desktop-app-update-v0.19.0-win32-x64.zip",
        size: archiveContent.length,
        browser_download_url: "https://example.test/desktop-app-update-v0.19.0-win32-x64.zip"
      },
      ...runtimeParts.map((body, index) => ({
        name: `desktop-runtime-v1.0.0-win32-x64.zip.part0${index + 1}`,
        size: body.length,
        browser_download_url: `https://github.com/T8mars/indextts25-desktop-t8/releases/download/v0.19.0/desktop-runtime-v1.0.0-win32-x64.zip.part0${index + 1}`
      }))
    ]
  };
  const rawManifest = {
    schemaVersion: 1,
    desktopVersion: "0.19.0",
    channel: "stable",
    minimumUpdaterVersion: "0.18.1",
    summary: "自动更新首版",
    packages: {
      portableApp: {
        assetName: "desktop-app-update-v0.19.0-win32-x64.zip",
        size: archiveContent.length,
        sha256: digest(archiveContent),
        restartRequired: true,
        files: [
          {
            path: "resources/desktop_webui.py",
            size: fileContent.length,
            sha256: digest(fileContent)
          }
        ]
      },
      fullPortable: {
        size: 3845667974,
        sha256: "1".repeat(64),
        urls: ["https://pan.quark.cn/s/example"]
      },
      runtime: {
        version: "1.0.0",
        archiveName: "desktop-runtime-v1.0.0-win32-x64.zip",
        archiveSize: runtimeArchive.length,
        unpackedSize: runtimeFileContent.length,
        archiveSha256: digest(runtimeArchive),
        roots: ["resources/runtime"],
        fileManifest: {
          path: "runtime-files.json",
          size: runtimeFileManifest.length,
          sha256: digest(runtimeFileManifest)
        },
        parts: runtimeParts.map((body, index) => ({
          assetName: `desktop-runtime-v1.0.0-win32-x64.zip.part0${index + 1}`,
          size: body.length,
          sha256: digest(body)
        })),
        restartRequired: true
      }
    },
    model: {
      repository: "t8star/IndexTTS-2.5-Comfy",
      revision: "14166a7401f9f87f53770a1784390e8c0e9da15a",
      bundleVersion: "1.0.0"
    },
    runtime: {
      version: "1.0.0",
      repository: "t8star/IndexTTS-2.5-Desktop-Runtime",
      revision: "main",
      required: false
    }
  };

  const validated = validateUpdateManifest(rawManifest, release);
  assert.equal(validated.desktopVersion, "0.19.0");
  assert.equal(validated.portableApp.url, release.assets[2].browser_download_url);
  assert.equal(validated.portableApp.files[0].path, "resources/desktop_webui.py");
  assert.equal(validated.runtimePackage.parts.length, 2);
  assert.equal(validated.runtimePackage.unpackedSize, runtimeFileContent.length);
  assert.throws(
    () => validateUpdateManifest({ ...rawManifest, desktopVersion: "0.20.0" }, release),
    /不一致/
  );

  const signingKeys = crypto.generateKeyPairSync("ed25519");
  const signature = crypto.sign(
    null,
    Buffer.from(canonicalJson(rawManifest), "utf8"),
    signingKeys.privateKey
  ).toString("base64");
  assert.equal(verifyManifestSignature(rawManifest, signature, signingKeys.publicKey), true);
  assert.throws(
    () => verifyManifestSignature({ ...rawManifest, summary: "tampered" }, signature, signingKeys.publicKey),
    /验证失败/
  );

  const resolved = await resolveDesktopUpdate({
    currentVersion: "0.18.1",
    currentRuntimeVersion: "0.9.0",
    channel: "stable",
    fetchJson: async (url) => url.endsWith("/latest") ? release : rawManifest,
    fetchText: async () => signature,
    publicKey: signingKeys.publicKey
  });
  assert.equal(resolved.updateAvailable, true);
  assert.equal(resolved.desktopUpdateAvailable, true);
  assert.equal(resolved.runtimeUpdateAvailable, true);
  assert.equal(resolved.manualOnly, false);
  assert.equal(resolved.signatureVerified, true);
  assert.equal(resolved.manifest.model.repository, "t8star/IndexTTS-2.5-Comfy");
  assert.equal(resolved.manifest.model.bundleVersion, "1.0.0");

  const runtimeOnly = await resolveDesktopUpdate({
    currentVersion: "0.19.0",
    currentRuntimeVersion: "0.9.0",
    channel: "stable",
    fetchJson: async (url) => url.endsWith("/latest") ? release : rawManifest,
    fetchText: async () => signature,
    publicKey: signingKeys.publicKey
  });
  assert.equal(runtimeOnly.desktopUpdateAvailable, false);
  assert.equal(runtimeOnly.runtimeUpdateAvailable, true);
  assert.equal(runtimeOnly.updateAvailable, true);
  assert.equal(runtimeOnly.manualOnly, false);

  const untrusted = await resolveDesktopUpdate({
    currentVersion: "0.18.1",
    fetchJson: async (url) => url.endsWith("/latest") ? release : rawManifest,
    fetchText: async () => Buffer.alloc(64).toString("base64"),
    publicKey: signingKeys.publicKey
  });
  assert.equal(untrusted.manualOnly, true);
  assert.match(untrusted.manifestError, /验证失败/);

  const remoteModelManifest = {
    schemaVersion: 1,
    bundleVersion: "1.0.0",
    publishedAt: "2026-08-29T00:00:00Z",
    minimumDesktopVersion: "0.19.1",
    minimumNodeVersion: "0.18.0",
    totalSize: 8,
    codeRepository: "index-tts/index-tts",
    codeRevision: "e".repeat(40),
    modelRepository: "t8star/IndexTTS-2.5-Comfy",
    modelRevision: "1".repeat(40),
    files: {
      "config.yaml": { size: 3, sha256: digest("cfg") },
      "hf_cache/test.bin": { size: 5, sha256: digest("model") }
    }
  };
  assert.equal(validateModelBundleManifest(remoteModelManifest).totalSize, 8);
  const modelSignature = crypto.sign(
    null,
    Buffer.from(canonicalJson(remoteModelManifest), "utf8"),
    signingKeys.privateKey
  ).toString("base64");
  assert.equal(
    verifyModelBundleSignature(remoteModelManifest, modelSignature, signingKeys.publicKey),
    true
  );
  const modelUpdate = await resolveModelBundleUpdate({
    currentVersion: "0.9.0",
    desktopVersion: "0.19.1",
    fetchJson: async () => remoteModelManifest,
    fetchText: async () => modelSignature,
    publicKey: signingKeys.publicKey
  });
  assert.equal(modelUpdate.updateAvailable, true);
  assert.equal(modelUpdate.compatible, true);
  assert.equal(modelUpdate.signatureVerified, true);
  assert.equal(modelUpdate.revision, "1".repeat(40));
  await assert.rejects(
    resolveModelBundleUpdate({
      currentVersion: "1.0.0",
      desktopVersion: "0.19.1",
      fetchJson: async () => ({ ...remoteModelManifest, totalSize: 9 }),
      fetchText: async () => modelSignature,
      publicKey: signingKeys.publicKey
    }),
    /签名验证失败/
  );

  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "t8-update-manager-"));
  try {
    const server = http.createServer((request, response) => {
      const range = request.headers.range;
      const start = range ? Number(range.match(/bytes=(\d+)-/)?.[1] || 0) : 0;
      const body = archiveContent.subarray(start);
      response.writeHead(start ? 206 : 200, {
        "Content-Length": body.length,
        ...(start ? { "Content-Range": `bytes ${start}-${archiveContent.length - 1}/${archiveContent.length}` } : {})
      });
      response.end(body);
    });
    await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
    try {
      const destination = path.join(temporary, "update.zip");
      fs.writeFileSync(`${destination}.part`, archiveContent.subarray(0, 7));
      const progress = [];
      const task = createDownloadTask({
        url: `http://127.0.0.1:${server.address().port}/update.zip`,
        destination,
        expectedSize: archiveContent.length,
        expectedSha256: digest(archiveContent),
        allowInsecure: true,
        onProgress: (value) => progress.push(value)
      });
      await task.promise;
      assert.deepEqual(fs.readFileSync(destination), archiveContent);
      assert.equal(progress.at(-1).received, archiveContent.length);
      assert.equal(fs.existsSync(`${destination}.part`), false);
    } finally {
      await new Promise((resolve) => server.close(resolve));
    }

    const payloadRoot = path.join(temporary, "payload");
    const payloadFile = path.join(payloadRoot, "resources", "desktop_webui.py");
    fs.mkdirSync(path.dirname(payloadFile), { recursive: true });
    fs.writeFileSync(payloadFile, fileContent);
    await verifyPayloadFiles(payloadRoot, validated.portableApp.files);
    fs.writeFileSync(path.join(payloadRoot, "unexpected.txt"), "reject me", "utf8");
    await assert.rejects(
      verifyPayloadFiles(payloadRoot, validated.portableApp.files),
      /文件清单不一致/
    );

    const safeBody = Buffer.from("safe", "utf8");
    const safeFiles = [{ path: "resources/safe.txt", size: safeBody.length, sha256: digest(safeBody) }];
    const validZip = path.join(temporary, "valid.zip");
    fs.writeFileSync(validZip, storedZip([{ name: "resources/safe.txt", body: "safe" }]));
    await extractAndVerifyUpdate(validZip, path.join(temporary, "valid-staging"), safeFiles);

    const traversalZip = path.join(temporary, "traversal.zip");
    fs.writeFileSync(traversalZip, storedZip([{ name: "../escaped.txt", body: "unsafe" }]));
    await assert.rejects(
      extractAndVerifyUpdate(traversalZip, path.join(temporary, "traversal-staging"), safeFiles),
      /路径|relative path|invalid/i
    );
    assert.equal(fs.existsSync(path.join(temporary, "escaped.txt")), false);

    const runtimePartPaths = runtimeParts.map((body, index) => {
      const partPath = path.join(temporary, `runtime.part0${index + 1}`);
      fs.writeFileSync(partPath, body);
      return partPath;
    });
    const assembledRuntime = path.join(temporary, "runtime.zip");
    await assembleFileParts({
      partPaths: runtimePartPaths,
      destination: assembledRuntime,
      expectedSize: runtimeArchive.length,
      expectedSha256: digest(runtimeArchive)
    });
    const extractedRuntime = await extractAndVerifyRuntimeUpdate(
      assembledRuntime,
      path.join(temporary, "runtime-staging"),
      validated.runtimePackage
    );
    assert.equal(extractedRuntime.runtimeVersion, "1.0.0");
    assert.deepEqual(
      fs.readFileSync(path.join(extractedRuntime.payloadRoot, "resources", "runtime", "test.bin")),
      runtimeFileContent
    );

    const symlinkZip = path.join(temporary, "symlink.zip");
    fs.writeFileSync(symlinkZip, storedZip([{
      name: "resources/safe.txt",
      body: "safe",
      externalAttributes: (0o120777 << 16) >>> 0
    }]));
    await assert.rejects(
      extractAndVerifyUpdate(symlinkZip, path.join(temporary, "symlink-staging"), safeFiles),
      /符号链接/
    );

    const helperRoot = path.join(temporary, "helper-test");
    const installRoot = path.join(helperRoot, "install");
    const updatesRoot = path.join(helperRoot, "updates");
    const helperPayload = path.join(updatesRoot, "payload");
    const relativeFile = "resources/test.txt";
    fs.mkdirSync(path.join(installRoot, "resources"), { recursive: true });
    fs.mkdirSync(path.join(helperPayload, "resources"), { recursive: true });
    fs.mkdirSync(updatesRoot, { recursive: true });
    fs.writeFileSync(path.join(installRoot, ...relativeFile.split("/")), "old", "utf8");
    fs.writeFileSync(path.join(helperPayload, ...relativeFile.split("/")), "new", "utf8");
    // where.exe exits immediately for our synthetic updater arguments, so the
    // rollback relaunch cannot keep the temporary executable locked on Windows.
    fs.copyFileSync("C:\\Windows\\System32\\where.exe", path.join(installRoot, "app.exe"));
    const plan = {
      targetVersion: "9.9.9",
      parentPid: 2147483647,
      installRoot,
      payloadRoot: helperPayload,
      backupRoot: path.join(updatesRoot, "backup"),
      updatesRoot,
      healthMarker: path.join(updatesRoot, "health.txt"),
      executablePath: path.join(installRoot, "app.exe"),
      healthToken: "0123456789abcdef0123456789abcdef",
      resultPath: path.join(updatesRoot, "result.json"),
      files: [{ path: relativeFile, size: 3, sha256: digest("new") }]
    };
    const planPath = path.join(updatesRoot, "plan.json");
    fs.writeFileSync(planPath, JSON.stringify(plan), "utf8");
    const helperResult = spawnSync(
      "powershell.exe",
      [
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy", "Bypass",
        "-File", path.join(__dirname, "portable-update-helper.ps1"),
        "-PlanPath", planPath
      ],
      { encoding: "utf8", timeout: 15000, windowsHide: true }
    );
    const helperDetails = fs.existsSync(plan.resultPath) ? fs.readFileSync(plan.resultPath, "utf8") : "";
    assert.equal(helperResult.status, 2, helperResult.stderr || helperResult.stdout || helperDetails);
    assert.equal(fs.readFileSync(path.join(installRoot, ...relativeFile.split("/")), "utf8"), "old");
    assert.equal(JSON.parse(fs.readFileSync(plan.resultPath, "utf8")).status, "rolled-back");

    const successRoot = path.join(temporary, "helper-success-test");
    const successInstall = path.join(successRoot, "install");
    const successUpdates = path.join(successRoot, "updates");
    const successPayload = path.join(successUpdates, "payload");
    fs.mkdirSync(path.join(successInstall, "resources"), { recursive: true });
    fs.mkdirSync(path.join(successPayload, "resources"), { recursive: true });
    fs.mkdirSync(successUpdates, { recursive: true });
    fs.writeFileSync(path.join(successInstall, "resources", "test.txt"), "old", "utf8");
    fs.writeFileSync(path.join(successPayload, "resources", "test.txt"), "new", "utf8");
    const healthyApp = path.join(successInstall, "healthy.cmd");
    fs.writeFileSync(healthyApp, [
      "@echo off",
      "set token=",
      "set marker=",
      ":args",
      "if \"%~1\"==\"\" goto done",
      "if \"%~1\"==\"--update-token\" set token=%~2",
      "if \"%~1\"==\"--update-health-marker\" set marker=%~2",
      "shift",
      "goto args",
      ":done",
      "if not \"%marker%\"==\"\" >\"%marker%\" echo %token%",
      "exit /b 0"
    ].join("\r\n"), "ascii");
    const successPlan = {
      ...plan,
      targetVersion: "9.9.10",
      installRoot: successInstall,
      payloadRoot: successPayload,
      backupRoot: path.join(successUpdates, "backup"),
      updatesRoot: successUpdates,
      healthMarker: path.join(successUpdates, "health.txt"),
      executablePath: healthyApp,
      resultPath: path.join(successUpdates, "result.json")
    };
    const successPlanPath = path.join(successUpdates, "plan.json");
    fs.writeFileSync(successPlanPath, JSON.stringify(successPlan), "utf8");
    const successResult = spawnSync(
      "powershell.exe",
      [
        "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
        "-File", path.join(__dirname, "portable-update-helper.ps1"),
        "-PlanPath", successPlanPath
      ],
      { encoding: "utf8", timeout: 15000, windowsHide: true }
    );
    const successDetails = fs.existsSync(successPlan.resultPath)
      ? fs.readFileSync(successPlan.resultPath, "utf8")
      : "";
    assert.equal(successResult.status, 0, successResult.stderr || successResult.stdout || successDetails);
    assert.equal(fs.readFileSync(path.join(successInstall, "resources", "test.txt"), "utf8"), "new");
    assert.equal(JSON.parse(fs.readFileSync(successPlan.resultPath, "utf8")).status, "installed");
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true, maxRetries: 30, retryDelay: 100 });
  }

  console.log("Desktop update manager safety tests OK");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
