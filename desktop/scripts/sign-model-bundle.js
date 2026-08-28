const crypto = require("node:crypto");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {
  canonicalJson,
  validateModelBundleManifest,
  verifyModelBundleSignature
} = require("../src/update_manager");

const desktopRoot = path.resolve(__dirname, "..");
const projectRoot = path.resolve(desktopRoot, "..");

function argumentValue(name, fallback = "") {
  const index = process.argv.indexOf(name);
  return index >= 0 && process.argv[index + 1] ? process.argv[index + 1] : fallback;
}

function loadPrivateKey() {
  if (process.env.T8_UPDATE_PRIVATE_KEY_BASE64) {
    return Buffer.from(process.env.T8_UPDATE_PRIVATE_KEY_BASE64, "base64").toString("utf8");
  }
  const configured = argumentValue("--private-key", process.env.T8_UPDATE_PRIVATE_KEY_FILE || "");
  const keyPath = configured || path.join(
    os.homedir(), ".codex", "secrets", "indextts-desktop-update-private.pem"
  );
  if (!fs.existsSync(keyPath)) {
    throw new Error("Model bundle signing key is missing.");
  }
  return fs.readFileSync(keyPath, "utf8");
}

const manifestPath = path.resolve(argumentValue(
  "--manifest",
  path.join(projectRoot, "desktop_model_manifest.json")
));
const outputPath = path.resolve(argumentValue(
  "--output",
  path.join(projectRoot, "model-bundle.sig")
));
const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
validateModelBundleManifest(manifest);
const signature = crypto.sign(
  null,
  Buffer.from(canonicalJson(manifest), "utf8"),
  loadPrivateKey()
).toString("base64");
verifyModelBundleSignature(manifest, signature);
fs.writeFileSync(outputPath, `${signature}\n`, "ascii");
console.log(JSON.stringify({
  bundleVersion: manifest.bundleVersion,
  modelRevision: manifest.modelRevision,
  files: Object.keys(manifest.files).length,
  totalSize: manifest.totalSize,
  signaturePath: outputPath
}, null, 2));
