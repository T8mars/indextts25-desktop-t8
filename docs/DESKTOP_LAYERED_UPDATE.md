# Desktop 分层自动更新

桌面更新分为三层，避免把 3–4 GiB 的完整便携包作为单个 GitHub Asset：

1. `desktop-app-update-vX-win32-x64.zip`：Electron、Python 源码和配置，随普通 `vX` Release 发布。
2. `desktop-runtime-vY-win32-x64.zip.partNN`：CPython、Torch、CUDA 依赖，随独立 `runtime-vY` Release 发布；每卷默认 1792 MiB。
3. IndexTTS 2.5 模型：只放在 `t8star/IndexTTS-2.5-Comfy`，不进入 GitHub 更新包。

启动器验证链是：Ed25519 更新清单 → 每个运行库分卷 SHA-256 → 合并 ZIP 的大小与 SHA-256 → ZIP 内 `runtime-files.json` → 每个解压文件的大小与 SHA-256。程序层与运行库层会合并成一个安装计划；更新助手先备份、替换、重启并等待健康标记，失败时自动回滚。

## 构建运行库层

先完成便携包打包和 `npm run verify:runtime`，确认 `desktop/out/T8star-Aix-IndexTTS-2.5-vX-win32-x64` 可真实启动。随后在 `desktop` 目录运行：

```powershell
npm run build:runtime
```

脚本读取根目录 `desktop_runtime_manifest.json`，只收集声明的 `resources/cpython-*`、`resources/site-packages` 和运行库版本元数据。输出位于 `desktop/out/runtime-vY/`，包含：

- `desktop-runtime-package.json`
- `desktop-runtime-vY-win32-x64.zip.partNN`

发布前必须确保 `runtimeVersion` 与 `releaseTag` 一致。确认 GitHub CLI 已登录后，可显式发布：

```powershell
npm run release:runtime
```

这个命令会创建/更新 `runtime-vY` Release 并上传描述文件和全部分卷；不带 `--publish` 的构建命令不会写入 GitHub。

## 构建签名程序层

`.github/workflows/desktop-release.yml` 会尝试从 `runtime-vY` Release 下载 `desktop-runtime-package.json`，把其固定 URL、大小和哈希写入并签名到桌面更新清单。运行库 Release 尚未发布时，工作流只生成程序层并给出警告。

本地构建可显式指定描述文件：

```powershell
node .\desktop\scripts\build-update-package.js --source `
  --runtime-package .\desktop\out\runtime-v1.0.0\desktop-runtime-package.json
```

模型更新不受此流程影响；启动器继续验证 Hugging Face 上的 `model-bundle.json` 和 `model-bundle.sig`。
