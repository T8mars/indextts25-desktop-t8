const crypto = require("node:crypto");
const fs = require("node:fs");
const http = require("node:http");
const https = require("node:https");
const path = require("node:path");
const yauzl = require("yauzl");

const DESKTOP_REPOSITORY = "T8mars/indextts25-desktop-t8";
const RELEASE_API = `https://api.github.com/repos/${DESKTOP_REPOSITORY}/releases`;
const MAX_GITHUB_ASSET_BYTES = 2 * 1024 * 1024 * 1024;
const USER_AGENT = "T8star-Aix-IndexTTS25-Desktop-Updater";
const UPDATE_MANIFEST_ASSET = "desktop-update-manifest.json";
const UPDATE_SIGNATURE_ASSET = "desktop-update-manifest.sig";
const MODEL_BUNDLE_REPOSITORY = "t8star/IndexTTS-2.5-Comfy";
const MODEL_BUNDLE_BASE_URL = `https://huggingface.co/${MODEL_BUNDLE_REPOSITORY}/resolve/main`;
const MODEL_BUNDLE_MANIFEST_URL = `${MODEL_BUNDLE_BASE_URL}/model-bundle.json`;
const MODEL_BUNDLE_SIGNATURE_URL = `${MODEL_BUNDLE_BASE_URL}/model-bundle.sig`;
const UPDATE_PUBLIC_KEY_PEM = `-----BEGIN PUBLIC KEY-----
MCowBQYDK2VwAyEARXizEmHovIcMTdi3Ki/tO9EEMAXh11hLVKepFy66ANI=
-----END PUBLIC KEY-----
`;

function compareVersions(left, right) {
  const parse = (value) => {
    const [core, prerelease = ""] = String(value || "0").replace(/^v/i, "").split("-", 2);
    return {
      numbers: core.split(".").map((part) => Number.parseInt(part, 10) || 0),
      prerelease
    };
  };
  const a = parse(left);
  const b = parse(right);
  for (let index = 0; index < Math.max(a.numbers.length, b.numbers.length); index += 1) {
    if ((a.numbers[index] || 0) !== (b.numbers[index] || 0)) {
      return (a.numbers[index] || 0) > (b.numbers[index] || 0) ? 1 : -1;
    }
  }
  if (a.prerelease === b.prerelease) return 0;
  if (!a.prerelease) return 1;
  if (!b.prerelease) return -1;
  return a.prerelease.localeCompare(b.prerelease, "en", { numeric: true });
}

function normalizeChannel(value) {
  return value === "beta" ? "beta" : "stable";
}

function normalizeVersion(value) {
  return String(value || "").trim().replace(/^v/i, "");
}

function isSemver(value) {
  return /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(normalizeVersion(value));
}

function safeRelativePath(value) {
  const raw = String(value || "").normalize("NFC");
  if (raw.includes("\\")) throw new Error(`更新文件路径不允许反斜杠：${value}`);
  const normalized = raw.replace(/^\.\//, "");
  if (!normalized || normalized.startsWith("/") || /^[A-Za-z]:/.test(normalized)) {
    throw new Error(`更新文件路径无效：${value}`);
  }
  const segments = normalized.split("/");
  if (segments.some((segment) => !segment || segment === "." || segment === "..")) {
    throw new Error(`更新文件路径越界：${value}`);
  }
  for (const segment of segments) {
    if (/[\x00-\x1f<>:"|?*]/.test(segment) || /[. ]$/.test(segment)) {
      throw new Error(`更新文件路径含 Windows 非法字符：${value}`);
    }
    const stem = segment.split(".", 1)[0].toUpperCase();
    if (/^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$/.test(stem)) {
      throw new Error(`更新文件路径使用 Windows 保留名：${value}`);
    }
  }
  return segments.join("/");
}

function assertSha256(value, label) {
  const normalized = String(value || "").toLowerCase();
  if (!/^[a-f0-9]{64}$/.test(normalized)) throw new Error(`${label} 缺少有效 SHA-256。`);
  return normalized;
}

function releaseAssets(release) {
  return new Map(
    (Array.isArray(release?.assets) ? release.assets : []).map((asset) => [asset.name, asset])
  );
}

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map((entry) => canonicalJson(entry)).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) => (
      `${JSON.stringify(key)}:${canonicalJson(value[key])}`
    )).join(",")}}`;
  }
  return JSON.stringify(value);
}

function verifySignedManifest(
  manifest,
  signatureText,
  publicKey = UPDATE_PUBLIC_KEY_PEM,
  label = "清单"
) {
  const encoded = String(signatureText || "").trim();
  if (!/^[A-Za-z0-9+/]+={0,2}$/.test(encoded)) {
    throw new Error(`${label}签名格式无效。`);
  }
  const signature = Buffer.from(encoded, "base64");
  if (signature.length !== 64) throw new Error(`${label}签名长度无效。`);
  const valid = crypto.verify(
    null,
    Buffer.from(canonicalJson(manifest), "utf8"),
    publicKey,
    signature
  );
  if (!valid) throw new Error(`${label}签名验证失败。`);
  return true;
}

function verifyManifestSignature(manifest, signatureText, publicKey = UPDATE_PUBLIC_KEY_PEM) {
  return verifySignedManifest(manifest, signatureText, publicKey, "桌面更新清单");
}

function verifyModelBundleSignature(manifest, signatureText, publicKey = UPDATE_PUBLIC_KEY_PEM) {
  return verifySignedManifest(manifest, signatureText, publicKey, "模型清单");
}

function validateModelBundleManifest(rawManifest) {
  if (!rawManifest || typeof rawManifest !== "object") throw new Error("模型清单不是 JSON 对象。");
  if (rawManifest.schemaVersion !== 1) throw new Error("不支持的模型清单版本。");
  const bundleVersion = normalizeVersion(rawManifest.bundleVersion);
  if (!isSemver(bundleVersion)) throw new Error("模型包版本号无效。");
  const repository = String(rawManifest.modelRepository || "").trim();
  if (repository !== MODEL_BUNDLE_REPOSITORY) {
    throw new Error(`模型清单仓库必须是 ${MODEL_BUNDLE_REPOSITORY}。`);
  }
  const revision = String(rawManifest.modelRevision || "").trim().toLowerCase();
  if (!/^[a-f0-9]{40}$/.test(revision)) throw new Error("模型清单缺少固定的 Git 提交版本。");
  const minimumDesktopVersion = normalizeVersion(rawManifest.minimumDesktopVersion || "0.19.1");
  const minimumNodeVersion = normalizeVersion(rawManifest.minimumNodeVersion || "0.18.0");
  if (!isSemver(minimumDesktopVersion) || !isSemver(minimumNodeVersion)) {
    throw new Error("模型清单最低兼容版本无效。");
  }
  if (!rawManifest.files || typeof rawManifest.files !== "object" || Array.isArray(rawManifest.files)) {
    throw new Error("模型清单没有文件列表。");
  }
  const files = {};
  const seen = new Set();
  let totalSize = 0;
  for (const [rawPath, rawMetadata] of Object.entries(rawManifest.files)) {
    const relativePath = safeRelativePath(rawPath);
    const key = relativePath.toLowerCase();
    if (seen.has(key)) throw new Error(`模型清单存在重复文件路径：${relativePath}`);
    seen.add(key);
    const size = Number(rawMetadata?.size);
    if (!Number.isSafeInteger(size) || size < 0) throw new Error(`模型文件大小无效：${relativePath}`);
    const metadata = {
      size,
      sha256: assertSha256(rawMetadata?.sha256, `模型文件 ${relativePath}`)
    };
    for (const field of ["group", "sourceRepository", "sourceRevision", "modelScopeRepository", "modelScopeRevision"]) {
      if (rawMetadata?.[field]) metadata[field] = String(rawMetadata[field]);
    }
    files[relativePath] = metadata;
    totalSize += size;
  }
  if (!Object.keys(files).length) throw new Error("模型清单文件列表为空。");
  if (Number(rawManifest.totalSize) !== totalSize) throw new Error("模型清单总大小与文件列表不一致。");
  return {
    schemaVersion: 1,
    bundleVersion,
    publishedAt: String(rawManifest.publishedAt || ""),
    minimumDesktopVersion,
    minimumNodeVersion,
    totalSize,
    codeRepository: String(rawManifest.codeRepository || "index-tts/index-tts"),
    codeRevision: String(rawManifest.codeRevision || ""),
    modelRepository: repository,
    modelRevision: revision,
    modelScopeRepository: String(rawManifest.modelScopeRepository || "IndexTeam/IndexTTS-2.5"),
    modelScopeRevision: String(rawManifest.modelScopeRevision || "master"),
    files
  };
}

function validateUpdateManifest(rawManifest, release) {
  if (!rawManifest || typeof rawManifest !== "object") throw new Error("桌面更新清单不是 JSON 对象。");
  if (rawManifest.schemaVersion !== 1) throw new Error("不支持的桌面更新清单版本。");
  const desktopVersion = normalizeVersion(rawManifest.desktopVersion);
  const releaseVersion = normalizeVersion(release?.tag_name);
  if (!isSemver(desktopVersion)) throw new Error("桌面更新清单版本号无效。");
  if (releaseVersion && desktopVersion !== releaseVersion) {
    throw new Error(`Release 标签 ${releaseVersion} 与更新清单 ${desktopVersion} 不一致。`);
  }
  const channel = normalizeChannel(rawManifest.channel);
  if (rawManifest.channel && rawManifest.channel !== channel) throw new Error("桌面更新通道无效。");
  const minimumUpdaterVersion = normalizeVersion(rawManifest.minimumUpdaterVersion || "0.18.1");
  if (!isSemver(minimumUpdaterVersion)) throw new Error("最低更新器版本无效。");

  const assets = releaseAssets(release);
  let portableApp = null;
  if (rawManifest.packages?.portableApp) {
    const source = rawManifest.packages.portableApp;
    const assetName = String(source.assetName || "").trim();
    const releaseAsset = assets.get(assetName);
    if (!assetName || !releaseAsset) throw new Error(`Release 缺少程序更新包：${assetName || "未命名"}`);
    const size = Number(source.size);
    if (!Number.isSafeInteger(size) || size <= 0 || size >= MAX_GITHUB_ASSET_BYTES) {
      throw new Error("程序更新包大小无效或超过 GitHub 2 GiB 单文件限制。");
    }
    if (Number(releaseAsset.size) !== size) throw new Error("程序更新包大小与 GitHub Release 不一致。");
    const files = Array.isArray(source.files) ? source.files.map((entry) => ({
      path: safeRelativePath(entry.path),
      size: Number(entry.size),
      sha256: assertSha256(entry.sha256, `更新文件 ${entry.path}`)
    })) : [];
    if (!files.length) throw new Error("程序更新包没有声明任何可替换文件。");
    const uniquePaths = new Set(files.map((entry) => entry.path.toLowerCase()));
    if (uniquePaths.size !== files.length) throw new Error("程序更新包存在重复文件路径。");
    for (const entry of files) {
      if (!Number.isSafeInteger(entry.size) || entry.size < 0) {
        throw new Error(`更新文件大小无效：${entry.path}`);
      }
    }
    portableApp = {
      assetName,
      url: releaseAsset.browser_download_url,
      size,
      sha256: assertSha256(source.sha256, "程序更新包"),
      files,
      restartRequired: source.restartRequired !== false
    };
  }

  const model = rawManifest.model && typeof rawManifest.model === "object" ? {
    repository: String(rawManifest.model.repository || ""),
    revision: String(rawManifest.model.revision || ""),
    bundleVersion: normalizeVersion(rawManifest.model.bundleVersion || "0.0.0"),
    manifestUrl: String(rawManifest.model.manifestUrl || MODEL_BUNDLE_MANIFEST_URL),
    signatureUrl: String(rawManifest.model.signatureUrl || MODEL_BUNDLE_SIGNATURE_URL)
  } : null;
  if (model) {
    if (model.repository !== MODEL_BUNDLE_REPOSITORY || !/^[a-f0-9]{40}$/.test(model.revision)) {
      throw new Error("桌面更新清单中的模型仓库或固定版本无效。");
    }
    if (!isSemver(model.bundleVersion)) throw new Error("桌面更新清单中的模型包版本无效。");
    if (model.manifestUrl !== MODEL_BUNDLE_MANIFEST_URL || model.signatureUrl !== MODEL_BUNDLE_SIGNATURE_URL) {
      throw new Error("桌面更新清单中的模型清单地址无效。");
    }
  }
  const runtime = rawManifest.runtime && typeof rawManifest.runtime === "object" ? {
    version: String(rawManifest.runtime.version || ""),
    repository: String(rawManifest.runtime.repository || ""),
    revision: String(rawManifest.runtime.revision || ""),
    required: Boolean(rawManifest.runtime.required)
  } : null;
  const fullPortableUrls = Array.isArray(rawManifest.packages?.fullPortable?.urls)
    ? rawManifest.packages.fullPortable.urls.filter((url) => /^https:\/\//i.test(String(url)))
    : [];

  return {
    schemaVersion: 1,
    desktopVersion,
    channel,
    minimumUpdaterVersion,
    publishedAt: String(rawManifest.publishedAt || release?.published_at || ""),
    releaseNotesUrl: String(rawManifest.releaseNotesUrl || release?.html_url || ""),
    summary: String(rawManifest.summary || release?.name || `Desktop ${desktopVersion}`),
    portableApp,
    fullPortable: {
      size: Number(rawManifest.packages?.fullPortable?.size || 0),
      sha256: rawManifest.packages?.fullPortable?.sha256
        ? assertSha256(rawManifest.packages.fullPortable.sha256, "完整便携包")
        : "",
      urls: fullPortableUrls
    },
    model,
    runtime
  };
}

function requestText(url, options = {}) {
  const timeoutMs = Number(options.timeoutMs || 15000);
  const maxBytes = Number(options.maxBytes || 5 * 1024 * 1024);
  const headers = { "User-Agent": USER_AGENT, Accept: "application/json, */*", ...(options.headers || {}) };
  const redirects = Number(options.redirects || 0);
  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const transport = parsed.protocol === "http:" ? http : https;
    if (!options.allowInsecure && parsed.protocol !== "https:") {
      reject(new Error(`拒绝非 HTTPS 更新地址：${url}`));
      return;
    }
    const request = transport.get(parsed, { headers }, (response) => {
      if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
        response.resume();
        if (redirects >= 5) {
          reject(new Error("更新地址重定向次数过多。"));
          return;
        }
        requestText(new URL(response.headers.location, url).toString(), {
          ...options,
          redirects: redirects + 1
        }).then(resolve, reject);
        return;
      }
      if (response.statusCode < 200 || response.statusCode >= 300) {
        response.resume();
        reject(new Error(`HTTP ${response.statusCode}: ${url}`));
        return;
      }
      let body = "";
      response.setEncoding("utf8");
      response.on("data", (chunk) => {
        body += chunk;
        if (Buffer.byteLength(body, "utf8") > maxBytes) request.destroy(new Error("更新响应过大。"));
      });
      response.on("end", () => resolve(body));
    });
    request.setTimeout(timeoutMs, () => request.destroy(new Error("检查更新超时")));
    request.on("error", reject);
  });
}

async function requestJson(url, options = {}) {
  const body = await requestText(url, options);
  try {
    return JSON.parse(body);
  } catch (error) {
    throw new Error(`更新服务返回了无效 JSON：${error.message}`);
  }
}

function chooseRelease(releases, channel) {
  const normalizedChannel = normalizeChannel(channel);
  const candidates = (Array.isArray(releases) ? releases : [releases])
    .filter((release) => release && !release.draft)
    .filter((release) => normalizedChannel === "beta" || !release.prerelease)
    .filter((release) => isSemver(release.tag_name));
  candidates.sort((left, right) => compareVersions(right.tag_name, left.tag_name));
  return candidates[0] || null;
}

async function resolveDesktopUpdate({
  currentVersion,
  channel = "stable",
  fetchJson = requestJson,
  fetchText = requestText,
  publicKey = UPDATE_PUBLIC_KEY_PEM
} = {}) {
  const normalizedChannel = normalizeChannel(channel);
  const releasePayload = normalizedChannel === "stable"
    ? await fetchJson(`${RELEASE_API}/latest`)
    : await fetchJson(`${RELEASE_API}?per_page=20`);
  const release = chooseRelease(releasePayload, normalizedChannel);
  if (!release) throw new Error(`没有找到 ${normalizedChannel} 桌面 Release。`);
  const latestVersion = normalizeVersion(release.tag_name);
  const assets = releaseAssets(release);
  const manifestAsset = assets.get(UPDATE_MANIFEST_ASSET);
  const signatureAsset = assets.get(UPDATE_SIGNATURE_ASSET);
  let manifest = null;
  let manifestError = "";
  let signatureVerified = false;
  if (manifestAsset && signatureAsset) {
    try {
      const rawManifest = await fetchJson(manifestAsset.browser_download_url);
      const signatureText = await fetchText(signatureAsset.browser_download_url);
      signatureVerified = verifyManifestSignature(rawManifest, signatureText, publicKey);
      manifest = validateUpdateManifest(rawManifest, release);
    } catch (error) {
      manifestError = error.message;
    }
  } else if (manifestAsset || signatureAsset) {
    manifestError = "Release 的更新清单或签名文件不完整。";
  } else {
    manifestError = "Release 未提供已签名的桌面更新清单。";
  }
  const updateAvailable = compareVersions(latestVersion, currentVersion) > 0;
  const updaterTooOld = Boolean(
    updateAvailable && manifest && compareVersions(currentVersion, manifest.minimumUpdaterVersion) < 0
  );
  return {
    current: normalizeVersion(currentVersion),
    latest: latestVersion,
    channel: normalizedChannel,
    updateAvailable,
    updaterTooOld,
    manualOnly: !manifest || !manifest.portableApp || updaterTooOld,
    releaseUrl: release.html_url,
    releaseName: release.name || `Desktop ${latestVersion}`,
    publishedAt: release.published_at || "",
    manifest,
    signatureVerified,
    manifestError
  };
}

async function resolveModelBundleUpdate({
  currentVersion,
  desktopVersion,
  fetchJson = requestJson,
  fetchText = requestText,
  publicKey = UPDATE_PUBLIC_KEY_PEM
} = {}) {
  const [rawManifest, signatureText] = await Promise.all([
    fetchJson(MODEL_BUNDLE_MANIFEST_URL),
    fetchText(MODEL_BUNDLE_SIGNATURE_URL)
  ]);
  verifyModelBundleSignature(rawManifest, signatureText, publicKey);
  const manifest = validateModelBundleManifest(rawManifest);
  const installedVersion = normalizeVersion(currentVersion || "0.0.0");
  const currentDesktopVersion = normalizeVersion(desktopVersion || "0.0.0");
  const compatible = compareVersions(currentDesktopVersion, manifest.minimumDesktopVersion) >= 0;
  return {
    current: installedVersion,
    latest: manifest.bundleVersion,
    revision: manifest.modelRevision,
    updateAvailable: compareVersions(manifest.bundleVersion, installedVersion) > 0,
    compatible,
    minimumDesktopVersion: manifest.minimumDesktopVersion,
    manifest,
    signedManifest: rawManifest,
    signature: String(signatureText || "").trim(),
    signatureVerified: true,
    repositoryUrl: `https://huggingface.co/${manifest.modelRepository}`
  };
}

function sha256File(filePath) {
  return new Promise((resolve, reject) => {
    const digest = crypto.createHash("sha256");
    const stream = fs.createReadStream(filePath);
    stream.on("data", (chunk) => digest.update(chunk));
    stream.on("error", reject);
    stream.on("end", () => resolve(digest.digest("hex")));
  });
}

function createDownloadTask({ url, destination, expectedSize, expectedSha256, onProgress, allowInsecure = false }) {
  const partPath = `${destination}.part`;
  let activeRequest = null;
  let cancelled = false;

  const run = async (targetUrl, redirects = 0) => {
    if (cancelled) throw new Error("更新下载已取消。");
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    if (fs.existsSync(partPath) && expectedSize && fs.statSync(partPath).size > Number(expectedSize)) {
      fs.rmSync(partPath, { force: true });
    }
    const existing = fs.existsSync(partPath) ? fs.statSync(partPath).size : 0;
    const headers = { "User-Agent": USER_AGENT, Accept: "application/octet-stream" };
    if (existing > 0) headers.Range = `bytes=${existing}-`;
    const parsed = new URL(targetUrl);
    if (!allowInsecure && parsed.protocol !== "https:") throw new Error(`拒绝非 HTTPS 下载地址：${targetUrl}`);
    const transport = parsed.protocol === "http:" ? http : https;
    await new Promise((resolve, reject) => {
      const request = transport.get(parsed, { headers }, (response) => {
        if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
          response.resume();
          if (redirects >= 5) {
            reject(new Error("更新下载重定向次数过多。"));
            return;
          }
          run(new URL(response.headers.location, targetUrl).toString(), redirects + 1).then(resolve, reject);
          return;
        }
        if (![200, 206].includes(response.statusCode)) {
          response.resume();
          reject(new Error(`更新下载失败：HTTP ${response.statusCode}`));
          return;
        }
        const resumed = response.statusCode === 206 && existing > 0;
        const start = resumed ? existing : 0;
        const output = fs.createWriteStream(partPath, { flags: resumed ? "a" : "w" });
        let received = start;
        response.on("data", (chunk) => {
          received += chunk.length;
          if (typeof onProgress === "function") onProgress({ received, total: Number(expectedSize || 0) });
        });
        response.on("error", reject);
        output.on("error", reject);
        output.on("finish", resolve);
        response.pipe(output);
      });
      activeRequest = request;
      request.setTimeout(30000, () => request.destroy(new Error("更新下载超时。")));
      request.on("error", reject);
    });
  };

  const promise = (async () => {
    if (fs.existsSync(destination)) {
      const sizeMatches = !expectedSize || fs.statSync(destination).size === Number(expectedSize);
      const hashMatches = !expectedSha256 || await sha256File(destination) === String(expectedSha256).toLowerCase();
      if (sizeMatches && hashMatches) return destination;
      fs.rmSync(destination, { force: true });
    }
    await run(url);
    if (cancelled) throw new Error("更新下载已取消。");
    const actualSize = fs.statSync(partPath).size;
    if (expectedSize && actualSize !== Number(expectedSize)) {
      throw new Error(`更新包大小不匹配：${actualSize} != ${expectedSize}`);
    }
    const actualHash = await sha256File(partPath);
    if (expectedSha256 && actualHash !== String(expectedSha256).toLowerCase()) {
      throw new Error("更新包 SHA-256 校验失败，已拒绝安装。");
    }
    fs.rmSync(destination, { force: true });
    fs.renameSync(partPath, destination);
    return destination;
  })();

  return {
    promise,
    cancel() {
      cancelled = true;
      if (activeRequest) activeRequest.destroy(new Error("更新下载已取消。"));
    }
  };
}

function listPayloadFiles(root) {
  const files = [];
  const walk = (directory) => {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const absolute = path.join(directory, entry.name);
      if (entry.isSymbolicLink()) throw new Error(`更新包禁止符号链接：${absolute}`);
      if (entry.isDirectory()) walk(absolute);
      else if (entry.isFile()) files.push(path.relative(root, absolute).split(path.sep).join("/"));
      else throw new Error(`更新包含不支持的文件类型：${absolute}`);
    }
  };
  walk(root);
  return files.sort();
}

async function verifyPayloadFiles(payloadRoot, expectedFiles) {
  const expected = expectedFiles.map((entry) => safeRelativePath(entry.path)).sort();
  const actual = listPayloadFiles(payloadRoot);
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error("更新包文件清单不一致，已拒绝安装。");
  }
  for (const entry of expectedFiles) {
    const relativePath = safeRelativePath(entry.path);
    const absolutePath = path.resolve(payloadRoot, ...relativePath.split("/"));
    if (!absolutePath.startsWith(`${path.resolve(payloadRoot)}${path.sep}`)) {
      throw new Error(`更新文件越界：${relativePath}`);
    }
    if (fs.statSync(absolutePath).size !== Number(entry.size)) {
      throw new Error(`更新文件大小不匹配：${relativePath}`);
    }
    if (await sha256File(absolutePath) !== String(entry.sha256).toLowerCase()) {
      throw new Error(`更新文件 SHA-256 校验失败：${relativePath}`);
    }
  }
  return actual;
}

function extractZipSafely(archivePath, payloadRoot, expectedFiles) {
  const expected = new Map(expectedFiles.map((entry) => [safeRelativePath(entry.path).toLowerCase(), {
    ...entry,
    path: safeRelativePath(entry.path)
  }]));
  const seen = new Set();
  return new Promise((resolve, reject) => {
    yauzl.open(archivePath, {
      lazyEntries: true,
      decodeStrings: true,
      validateEntrySizes: true,
      strictFileNames: true
    }, (openError, zipFile) => {
      if (openError) {
        reject(openError);
        return;
      }
      let settled = false;
      const fail = (error) => {
        if (settled) return;
        settled = true;
        try { zipFile.close(); } catch { /* Already closed. */ }
        reject(error);
      };
      zipFile.on("error", fail);
      zipFile.on("end", () => {
        if (settled) return;
        if (seen.size !== expected.size) {
          fail(new Error("更新 ZIP 缺少清单中的文件。"));
          return;
        }
        settled = true;
        resolve();
      });
      zipFile.on("entry", (entry) => {
        try {
          const directory = entry.fileName.endsWith("/");
          const candidate = safeRelativePath(directory ? entry.fileName.slice(0, -1) : entry.fileName);
          const key = candidate.toLowerCase();
          const unixMode = (entry.externalFileAttributes >>> 16) & 0xffff;
          if ((unixMode & 0o170000) === 0o120000) throw new Error(`更新 ZIP 禁止符号链接：${candidate}`);
          if (directory) {
            if (![...expected.values()].some((item) => item.path.startsWith(`${candidate}/`))) {
              throw new Error(`更新 ZIP 包含未声明的目录：${candidate}`);
            }
            fs.mkdirSync(path.join(payloadRoot, ...candidate.split("/")), { recursive: true });
            zipFile.readEntry();
            return;
          }
          const metadata = expected.get(key);
          if (!metadata) throw new Error(`更新 ZIP 包含未声明文件：${candidate}`);
          if (metadata.path !== candidate || seen.has(key)) throw new Error(`更新 ZIP 文件路径重复或大小写冲突：${candidate}`);
          if (Number(entry.uncompressedSize) !== Number(metadata.size)) {
            throw new Error(`更新 ZIP 文件大小与清单不一致：${candidate}`);
          }
          const destination = path.join(payloadRoot, ...candidate.split("/"));
          fs.mkdirSync(path.dirname(destination), { recursive: true });
          zipFile.openReadStream(entry, (streamError, readStream) => {
            if (streamError) {
              fail(streamError);
              return;
            }
            const output = fs.createWriteStream(destination, { flags: "wx", mode: 0o600 });
            readStream.on("error", fail);
            output.on("error", fail);
            output.on("finish", () => {
              seen.add(key);
              zipFile.readEntry();
            });
            readStream.pipe(output);
          });
        } catch (error) {
          fail(error);
        }
      });
      zipFile.readEntry();
    });
  });
}

async function extractAndVerifyUpdate(archivePath, stagingDirectory, expectedFiles) {
  const payloadRoot = path.join(stagingDirectory, "payload");
  fs.rmSync(payloadRoot, { recursive: true, force: true });
  fs.mkdirSync(payloadRoot, { recursive: true });
  await extractZipSafely(archivePath, payloadRoot, expectedFiles);
  await verifyPayloadFiles(payloadRoot, expectedFiles);
  return payloadRoot;
}

module.exports = {
  DESKTOP_REPOSITORY,
  MAX_GITHUB_ASSET_BYTES,
  MODEL_BUNDLE_MANIFEST_URL,
  MODEL_BUNDLE_REPOSITORY,
  MODEL_BUNDLE_SIGNATURE_URL,
  RELEASE_API,
  UPDATE_MANIFEST_ASSET,
  UPDATE_PUBLIC_KEY_PEM,
  UPDATE_SIGNATURE_ASSET,
  canonicalJson,
  chooseRelease,
  compareVersions,
  createDownloadTask,
  extractAndVerifyUpdate,
  extractZipSafely,
  normalizeChannel,
  requestJson,
  resolveDesktopUpdate,
  resolveModelBundleUpdate,
  safeRelativePath,
  sha256File,
  validateModelBundleManifest,
  validateUpdateManifest,
  verifyManifestSignature,
  verifyModelBundleSignature,
  verifySignedManifest,
  verifyPayloadFiles
};
