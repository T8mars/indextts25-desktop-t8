"""Standalone foreground launcher for the bundled IndexTTS 2.5 API service."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


APP_DATA_NAME = "T8star-Aix · IndexTTS 2.5"
DEFAULT_PORT = 7861
DEFAULT_HOST = "127.0.0.1"
DESKTOP_VERSION = "0.26.5"


def resource_root() -> Path:
    return Path(__file__).resolve().parent


def prepare_import_path() -> None:
    root = resource_root()
    candidates = (root, root / "site-packages")
    for candidate in reversed(candidates):
        value = str(candidate)
        if candidate.exists() and value not in sys.path:
            sys.path.insert(0, value)
    inherited = [item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item]
    os.environ["PYTHONPATH"] = os.pathsep.join([*(str(item) for item in candidates), *inherited])


def settings_path() -> Path:
    override = str(os.environ.get("T8STAR_INDEXTTS_SETTINGS") or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    appdata = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    return appdata / APP_DATA_NAME / "settings.json"


def read_settings() -> dict[str, Any]:
    path = settings_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取桌面设置：{path}（{exc}）") from exc
    return payload if isinstance(payload, dict) else {}


def write_settings(settings: dict[str, Any]) -> None:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def ensure_api_settings(settings: dict[str, Any]) -> dict[str, Any]:
    result = dict(settings)
    key = str(result.get("apiKey") or "").strip()
    if len(key) < 16:
        result["apiKey"] = secrets.token_urlsafe(32)
    try:
        port = int(result.get("apiPort") or DEFAULT_PORT)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("API 端口必须是 1–65535 的整数。") from exc
    if not 1 <= port <= 65535:
        raise RuntimeError("API 端口必须在 1–65535 之间。")
    host = str(result.get("apiHost") or DEFAULT_HOST).strip()
    if host not in {"127.0.0.1", "0.0.0.0"}:
        raise RuntimeError("API 监听地址只支持 127.0.0.1 或 0.0.0.0。")
    result.update(
        apiPort=port,
        apiHost=host,
        apiSaveHistory=result.get("apiSaveHistory") is not False,
        apiDefaultVoice=str(result.get("apiDefaultVoice") or "").strip(),
    )
    if not str(result.get("dataDir") or "").strip():
        result["dataDir"] = str(settings_path().parent)
    if not str(result.get("outputDir") or "").strip():
        result["outputDir"] = str(Path.home() / "Documents" / "T8star-Aix IndexTTS 2.5" / "outputs")
    if result != settings:
        write_settings(result)
    return result


def endpoint(settings: dict[str, Any], path: str = "") -> str:
    connect_host = "127.0.0.1" if settings["apiHost"] == "0.0.0.0" else settings["apiHost"]
    return f"http://{connect_host}:{settings['apiPort']}{path}"


def request_json(
    url: str,
    *,
    api_key: str = "",
    method: str = "GET",
    timeout: float = 3.0,
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def running_health(settings: dict[str, Any]) -> dict[str, Any] | None:
    try:
        payload = request_json(endpoint(settings, "/health"), timeout=1.5)
    except (OSError, ValueError, urllib.error.URLError):
        return None
    fingerprint = hashlib.sha256(str(settings["apiKey"]).encode("utf-8")).hexdigest()[:16]
    return payload if (
        payload.get("service") == "t8star-indextts-2.5"
        and payload.get("api_key_fingerprint") == fingerprint
    ) else None


def require_directory(settings: dict[str, Any], key: str, label: str) -> Path:
    raw = str(settings.get(key) or "").strip()
    if not raw:
        raise RuntimeError(f"未配置{label}。请先打开桌面程序选择目录并保存。")
    path = Path(raw).expanduser().resolve()
    if key == "modelDir" and not path.is_dir():
        raise RuntimeError(f"{label}不存在：{path}")
    if key != "modelDir":
        path.mkdir(parents=True, exist_ok=True)
    return path


def serve(settings: dict[str, Any]) -> int:
    current = running_health(settings)
    if current:
        print(f"API 服务已经在运行：{endpoint(settings)}")
        print(f"接口文档：{endpoint(settings, '/docs')}")
        return 0
    model_dir = require_directory(settings, "modelDir", "模型目录")
    output_dir = require_directory(settings, "outputDir", "输出目录")
    data_dir = require_directory(settings, "dataDir", "用户数据目录")
    prepare_import_path()
    os.environ.update(
        T8STAR_INDEXTTS_API_KEY=str(settings["apiKey"]),
        T8STAR_INDEXTTS_API_DEFAULT_VOICE=str(settings["apiDefaultVoice"]),
        T8STAR_INDEXTTS_DESKTOP_VERSION=DESKTOP_VERSION,
        PYTHONUTF8="1",
        PYTHONUNBUFFERED="1",
        HF_HOME=str(model_dir / "hf_cache"),
        HF_HUB_CACHE=str(model_dir / "hf_cache"),
        MODELSCOPE_CACHE=str(model_dir / "modelscope_cache"),
    )
    arguments = [
        "desktop_webui.py",
        "--model_dir", str(model_dir),
        "--output_dir", str(output_dir),
        "--data_dir", str(data_dir),
        "--host", str(settings["apiHost"]),
        "--port", str(settings["apiPort"]),
        "--acceleration", str(settings.get("accelerationMode") or "off"),
        "--precision", str(settings.get("precisionMode") or "auto"),
        "--reference-device", str(settings.get("referenceDevice") or "auto"),
    ]
    if settings.get("reuseDefaultEmotion"):
        arguments.append("--reuse-spk-cond-for-emo")
    if not settings["apiSaveHistory"]:
        arguments.append("--api-no-history")
    for origin in settings.get("apiCorsOrigins") or []:
        origin = str(origin).strip()
        if origin:
            arguments.extend(("--api-cors-origin", origin))
    print("=" * 72)
    print("T8star-Aix · IndexTTS 2.5 常驻 API 服务")
    print(f"服务地址：{endpoint(settings)}")
    print(f"OpenAI 兼容接口：{endpoint(settings, '/v1/audio/speech')}")
    print(f"接口文档：{endpoint(settings, '/docs')}")
    print(f"默认音色：{settings['apiDefaultVoice'] or '未设置（请求必须填写 voice）'}")
    print("按 Ctrl+C 可停止服务；本窗口保持打开即表示服务持续监听。")
    print("=" * 72, flush=True)
    from desktop_webui import main as webui_main

    sys.argv = arguments
    webui_main()
    return 0


def status(settings: dict[str, Any]) -> int:
    health = running_health(settings)
    if not health:
        print(f"API 服务未运行：{endpoint(settings)}")
        return 1
    print(f"API 服务运行正常：{endpoint(settings)}")
    print(json.dumps(health, ensure_ascii=False, indent=2))
    try:
        voices = request_json(
            endpoint(settings, "/v1/audio/voices"),
            api_key=str(settings["apiKey"]),
        )
        print(f"可用音色：{len(voices.get('data') or [])} 个")
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(f"读取音色列表失败：{exc}")
    return 0


def stop(settings: dict[str, Any]) -> int:
    if not running_health(settings):
        print("API 服务当前未运行。")
        return 0
    try:
        result = request_json(
            endpoint(settings, "/api/v1/admin/shutdown"),
            api_key=str(settings["apiKey"]),
            method="POST",
            timeout=5.0,
        )
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise RuntimeError(f"停止 API 服务失败：{exc}") from exc
    print(f"已发送停止请求：{result.get('status', 'stopping')}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T8star-Aix IndexTTS 2.5 local API launcher")
    parser.add_argument("command", nargs="?", choices=["serve", "status", "stop"], default="serve")
    return parser.parse_args()


def main() -> int:
    try:
        settings = ensure_api_settings(read_settings())
        command = parse_args().command
        return {"serve": serve, "status": status, "stop": stop}[command](settings)
    except KeyboardInterrupt:
        print("\nAPI 服务已由用户停止。")
        return 0
    except Exception as exc:
        print(f"错误：{str(exc).strip() or type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
