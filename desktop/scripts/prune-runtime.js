const fs = require("node:fs");
const path = require("node:path");

// Static development archives that are not loaded by Python or the packaged
// PyTorch DLLs. Import libraries required by torch.utils.cpp_extension stay.
const PRUNABLE_TORCH_LIBS = Object.freeze([
  "dnnl.lib",
  "libprotoc.lib",
  "libprotobuf.lib",
  "kineto.lib",
  "sleef.lib",
  "microkernels-prod.lib",
  "libprotobuf-lite.lib",
  "XNNPACK.lib",
  "fmt.lib",
  "fbgemm.lib",
  "pthreadpool.lib",
  "cpuinfo.lib",
  "libittnotify.lib",
  "asmjit.lib"
]);

const REQUIRED_EXTENSION_FILES = Object.freeze([
  "c10.lib",
  "c10_cuda.lib",
  "torch.lib",
  "torch_cpu.lib",
  "torch_cuda.lib",
  "torch_python.lib",
  "_C.lib",
  "caffe2_nvrtc.lib"
]);

function assertPackagedResources(resourcesRoot) {
  const resolved = path.resolve(resourcesRoot);
  const torchLib = path.join(resolved, "site-packages", "torch", "lib");
  if (path.basename(resolved).toLowerCase() !== "resources" || !fs.existsSync(torchLib)) {
    throw new Error(`Refusing to prune an unexpected path: ${resolved}`);
  }
  return { resolved, torchLib };
}

function prunePackagedRuntime(resourcesRoot) {
  const { resolved, torchLib } = assertPackagedResources(resourcesRoot);
  const removed = [];
  let savedBytes = 0;
  for (const filename of PRUNABLE_TORCH_LIBS) {
    const target = path.join(torchLib, filename);
    if (!fs.existsSync(target)) continue;
    const size = fs.statSync(target).size;
    fs.rmSync(target, { force: true });
    removed.push({ file: filename, bytes: size });
    savedBytes += size;
  }
  const report = {
    strategy: "explicit-static-development-archives",
    removed,
    savedBytes,
    savedGiB: Number((savedBytes / 1024 ** 3).toFixed(3)),
    preservedForExtensions: REQUIRED_EXTENSION_FILES
  };
  fs.writeFileSync(path.join(resolved, "runtime-prune-report.json"), JSON.stringify(report, null, 2));
  return report;
}

module.exports = { PRUNABLE_TORCH_LIBS, REQUIRED_EXTENSION_FILES, prunePackagedRuntime };
