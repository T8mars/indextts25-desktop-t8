const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { PRUNABLE_TORCH_LIBS, REQUIRED_EXTENSION_FILES, prunePackagedRuntime } = require("./prune-runtime");

const sandbox = fs.mkdtempSync(path.join(os.tmpdir(), "indextts-runtime-prune-"));
const resources = path.join(sandbox, "resources");
const torchLib = path.join(resources, "site-packages", "torch", "lib");
fs.mkdirSync(path.join(resources, "site-packages", "torch", "include"), { recursive: true });
fs.mkdirSync(torchLib, { recursive: true });
for (const filename of [...PRUNABLE_TORCH_LIBS, ...REQUIRED_EXTENSION_FILES]) {
  fs.writeFileSync(path.join(torchLib, filename), filename);
}

try {
  const report = prunePackagedRuntime(resources);
  assert.equal(report.removed.length, PRUNABLE_TORCH_LIBS.length);
  for (const filename of PRUNABLE_TORCH_LIBS) assert.equal(fs.existsSync(path.join(torchLib, filename)), false);
  for (const filename of REQUIRED_EXTENSION_FILES) assert.equal(fs.existsSync(path.join(torchLib, filename)), true);
  assert.equal(fs.existsSync(path.join(resources, "runtime-prune-report.json")), true);
  assert.throws(() => prunePackagedRuntime(sandbox), /Refusing to prune/);
  console.log("Runtime pruning safety test OK");
} finally {
  fs.rmSync(sandbox, { recursive: true, force: true });
}
