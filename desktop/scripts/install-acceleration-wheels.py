"""Install the exact optional Windows acceleration wheels used by the desktop build."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "desktop_acceleration_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_target(manifest: dict) -> None:
    target = manifest["target"]
    if sys.platform != "win32" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise SystemExit("这些轮子只适用于 64 位 Windows。")
    if sys.version_info[:2] != (3, 10):
        raise SystemExit(f"这些轮子要求 Python 3.10，当前是 {platform.python_version()}。")
    import torch

    if torch.__version__ != target["torch"]:
        raise SystemExit(
            f"这些轮子要求 torch {target['torch']}，当前是 {torch.__version__}；"
            "不要覆盖其他 ComfyUI 环境的 torch。"
        )


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _report(manifest: dict) -> bool:
    expected = {item["distribution"]: item["version"] for item in manifest["packages"]}
    rows = []
    all_ready = True
    for item in manifest["packages"]:
        installed = _installed_version(item["distribution"])
        importable = importlib.util.find_spec(item["importName"]) is not None
        ready = installed == expected[item["distribution"]] and importable
        all_ready &= ready
        rows.append(
            {
                "package": item["distribution"],
                "expected": item["version"],
                "installed": installed,
                "importable": importable,
                "ready": ready,
            }
        )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return all_ready


def _install(manifest: dict) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", *manifest["runtimeDependencies"]]
    )
    with tempfile.TemporaryDirectory(prefix="t8star-accel-") as temporary:
        wheel_dir = Path(temporary)
        wheels: list[str] = []
        for item in manifest["packages"]:
            target = wheel_dir / item["filename"]
            print(f"下载 {item['distribution']} {item['version']} ...")
            urllib.request.urlretrieve(item["url"], target)
            actual = _sha256(target)
            if actual.lower() != item["sha256"].lower():
                raise SystemExit(
                    f"{item['filename']} SHA-256 不匹配：{actual}，已停止安装。"
                )
            wheels.append(str(target))
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", *wheels]
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="只检查，不下载或安装")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    _require_target(manifest)
    if args.check:
        return 0 if _report(manifest) else 1
    _install(manifest)
    return 0 if _report(manifest) else 1


if __name__ == "__main__":
    raise SystemExit(main())
