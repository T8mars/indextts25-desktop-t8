"""Desktop-only Confucius4-R2T2 provider and isolated worker lifecycle.

The ComfyUI node package remains independent.  The desktop application talks
to the same worker protocol through this adapter so its Python 3.12/native
dependencies never leak into the bundled IndexTTS Python 3.10 environment.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import secrets
import socket
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = 1
PROVIDER_NAME = "confucius_r2t2"
MODEL_NAME = "Confucius4-R2T2-Q8_0.gguf"
PROJECTOR_NAME = "mmproj-Confucius4-R2T2-Q8_0.gguf"
MODEL_LICENSE_SHA256 = "064483c5ba1dc20907038108da4f45d5b37cd0a41a351c8a8bb98ba1af48c505"
MODEL_FILES = {
    MODEL_NAME: (1_834_422_208, "151097e43957a19984ea7de66e8144ce69b95039eb31c93da4f58db367e455c3"),
    PROJECTOR_NAME: (348_336_544, "8dc2c67e6a0484114928142d098db7ad94ae9f34c78948ef9d37a9678418cb65"),
}
SUPPORTED_LANGUAGES = (
    "Chinese",
    "English",
    "Cantonese",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
)
LANGUAGE_MAP = {
    "AUTO": "Auto",
    "ZH": "Chinese",
    "YUE": "Cantonese",
    "EN": "English",
    "JA": "Japanese",
    "KO": "Korean",
    "DE": "German",
    "FR": "French",
    "RU": "Russian",
    "PT": "Portuguese",
    "ES": "Spanish",
    "IT": "Italian",
}
CAPABILITIES = {
    "live_stream": True,
    "file_transcription": True,
    "word_timestamps": False,
    "translation": False,
}
_MANAGERS: dict[tuple[str, str], "ConfuciusWorkerManager"] = {}
_MANAGERS_LOCK = threading.RLock()
_GPU_COORDINATOR: Any | None = None
_INTEGRITY_CACHE: dict[str, tuple[int, int, str]] = {}
_INTEGRITY_LOCK = threading.RLock()


class ConfuciusError(RuntimeError):
    """Structured provider failure suitable for UI/API mapping."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        http_status: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.http_status = http_status
        self.retryable = bool(retryable)


@dataclass(frozen=True)
class ConfuciusComponent:
    root: Path
    python: Path
    model_dir: Path
    build_dir: Path
    profile: str
    source: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _inside(root: Path, candidate: Path) -> Path:
    resolved_root = root.expanduser().resolve()
    resolved = candidate.expanduser().resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ConfuciusError("ASR_COMPONENT_PATH", "Confucius 组件路径越出了受控目录。")
    return resolved


def _component_roots(data_dir: str | Path) -> list[tuple[Path, str]]:
    roots: list[tuple[Path, str]] = []
    override = str(os.environ.get("T8STAR_CONFUCIUS_ROOT") or "").strip()
    if override:
        roots.append((Path(override).expanduser(), "environment"))
    resources = Path(__file__).resolve().parent / "confucius_component"
    roots.append((resources, "bundled"))
    component_home = Path(data_dir).expanduser().resolve() / "optional_components" / "confucius"
    active = component_home / "active.json"
    if active.is_file():
        try:
            payload = json.loads(active.read_text(encoding="utf-8"))
            relative = str(payload.get("root") or "").strip() if isinstance(payload, dict) else ""
            if relative:
                roots.append((_inside(component_home, component_home / relative), "installed"))
        except (OSError, UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError):
            pass
    roots.append((component_home / "current", "installed-legacy"))
    unique: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for root, source in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        key = str(resolved).casefold()
        if key not in seen:
            seen.add(key)
            unique.append((resolved, source))
    return unique


def _read_component_manifest(root: Path) -> dict[str, Any]:
    path = root / "confucius-component.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfuciusError("ASR_COMPONENT_MANIFEST", "Confucius 组件清单损坏。") from exc
    try:
        supported = isinstance(value, dict) and int(value.get("schemaVersion", 0)) == 1
    except (TypeError, ValueError):
        supported = False
    if not supported:
        raise ConfuciusError("ASR_COMPONENT_MANIFEST", "Confucius 组件清单版本不受支持。")
    return value


def discover_component(data_dir: str | Path) -> ConfuciusComponent:
    failures: list[str] = []
    for root, source in _component_roots(data_dir):
        if not root.is_dir():
            continue
        manifest = _read_component_manifest(root)
        python_rel = str(manifest.get("workerPython") or ".runtime/worker/Scripts/python.exe")
        model_rel = str(manifest.get("modelDir") or "models/Confucius4-R2T2-GGUF")
        build_rel = str(manifest.get("buildDir") or ".runtime/build-native-cu128")
        profile = str(manifest.get("profile") or "sm120")
        try:
            python = _inside(root, root / Path(python_rel))
            model_dir = _inside(root, root / Path(model_rel))
            build_dir = _inside(root, root / Path(build_rel))
        except ConfuciusError as exc:
            failures.append(f"{source}: {exc}")
            continue
        required = [python, root / "r2t2_core" / "worker.py", root / "r2t2_core" / "native.py"]
        missing = [item.relative_to(root).as_posix() for item in required if not item.is_file()]
        if missing:
            failures.append(f"{source}: missing {', '.join(missing)}")
            continue
        if not build_dir.is_dir():
            failures.append(f"{source}: native build missing")
            continue
        if any(
            not (model_dir / name).is_file()
            or (model_dir / name).stat().st_size != expected[0]
            for name, expected in MODEL_FILES.items()
        ):
            failures.append(f"{source}: Q8 model pair missing")
            continue
        return ConfuciusComponent(root, python, model_dir, build_dir, profile, source)
    detail = "; ".join(failures[-3:])
    raise ConfuciusError(
        "ASR_COMPONENT_MISSING",
        "尚未安装 Confucius4-R2T2 实时识别组件。" + (f"（{detail}）" if detail else ""),
    )


def _profile_compatibility(profile: str) -> tuple[bool, str, str]:
    match = re.fullmatch(r"sm(\d{2,3})", str(profile or "").strip().lower())
    if not match:
        return False, "ASR_GPU_PROFILE_UNKNOWN", "Confucius 组件的 GPU profile 无法识别。"
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "ASR_GPU_UNAVAILABLE", "Confucius 实时识别需要可用的 NVIDIA CUDA 显卡。"
        major, minor = torch.cuda.get_device_capability()
    except (ImportError, RuntimeError, OSError):
        return False, "ASR_GPU_UNAVAILABLE", "无法读取 CUDA 显卡能力，Confucius 组件暂不可用。"
    actual = int(major) * 10 + int(minor)
    required = int(match.group(1))
    if actual != required:
        return (
            False,
            "ASR_GPU_PROFILE_UNSUPPORTED",
            f"当前 Confucius 组件为 sm{required} Preview，检测到的显卡能力为 sm{actual}。",
        )
    return True, "READY", f"GPU profile sm{actual} 已匹配。"


def _verified_file_hash(path: Path) -> str:
    stat = path.stat()
    key = str(path.resolve()).casefold()
    with _INTEGRITY_LOCK:
        cached = _INTEGRITY_CACHE.get(key)
        if cached and cached[:2] == (int(stat.st_size), int(stat.st_mtime_ns)):
            return cached[2]
    digest = _sha256(path)
    with _INTEGRITY_LOCK:
        _INTEGRITY_CACHE[key] = (int(stat.st_size), int(stat.st_mtime_ns), digest)
    return digest


def verify_component_integrity(component: ConfuciusComponent) -> None:
    for name, (expected_size, expected_hash) in MODEL_FILES.items():
        path = component.model_dir / name
        if not path.is_file() or path.stat().st_size != expected_size:
            raise ConfuciusError("ASR_INTEGRITY", f"Confucius 模型文件大小校验失败：{name}")
        if _verified_file_hash(path) != expected_hash:
            raise ConfuciusError("ASR_INTEGRITY", f"Confucius 模型 SHA-256 校验失败：{name}")


def license_acceptance_path(data_dir: str | Path) -> Path:
    return Path(data_dir).expanduser().resolve() / "licenses" / "confucius4-r2t2.json"


def license_accepted(data_dir: str | Path) -> bool:
    path = license_acceptance_path(data_dir)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(value, dict)
        and value.get("accepted") is True
        and str(value.get("licenseSha256") or "").lower() == MODEL_LICENSE_SHA256
    )


def accept_license(data_dir: str | Path, accepted: bool) -> dict[str, Any]:
    path = license_acceptance_path(data_dir)
    if not accepted:
        path.unlink(missing_ok=True)
        return {"accepted": False, "licenseSha256": MODEL_LICENSE_SHA256}
    value = {
        "accepted": True,
        "licenseSha256": MODEL_LICENSE_SHA256,
        "acceptedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": "NetEase Youdao Confucius4-R2T2 Q8",
    }
    _write_json_atomic(path, value)
    return value


def component_status(data_dir: str | Path, *, verify_hash: bool = False) -> dict[str, Any]:
    try:
        component = discover_component(data_dir)
    except ConfuciusError as exc:
        return {
            "available": False,
            "ready": False,
            "code": exc.code,
            "message": str(exc),
            "licenseAccepted": license_accepted(data_dir),
            "capabilities": CAPABILITIES,
        }
    files: dict[str, Any] = {}
    valid = True
    for name, (size, expected_hash) in MODEL_FILES.items():
        path = component.model_dir / name
        actual_size = path.stat().st_size if path.is_file() else 0
        state = actual_size == size
        actual_hash = ""
        if state and verify_hash:
            actual_hash = _verified_file_hash(path)
            state = actual_hash == expected_hash
        valid = valid and state
        files[name] = {
            "size": actual_size,
            "expectedSize": size,
            "sha256": actual_hash,
            "expectedSha256": expected_hash,
            "valid": state,
        }
    accepted = license_accepted(data_dir)
    compatible, compatibility_code, compatibility_message = _profile_compatibility(component.profile)
    return {
        "available": valid,
        "compatible": compatible,
        "ready": valid and compatible and accepted,
        "code": (
            "READY" if valid and compatible and accepted else
            "ASR_INTEGRITY" if not valid else
            compatibility_code if not compatible else
            "ASR_LICENSE_REQUIRED"
        ),
        "message": (
            "Confucius4-R2T2 可以使用。" if valid and compatible and accepted else
            "Confucius4-R2T2 组件文件不完整或大小不匹配。" if not valid else
            compatibility_message if not compatible else
            "请先阅读并接受 Confucius4-R2T2 模型许可。"
        ),
        "source": component.source,
        "profile": component.profile,
        "licenseAccepted": accepted,
        "licenseSha256": MODEL_LICENSE_SHA256,
        "files": files,
        "capabilities": CAPABILITIES,
        "languages": list(SUPPORTED_LANGUAGES),
    }


class ConfuciusWorkerManager:
    """Own one authenticated loopback worker for a resolved component."""

    def __init__(self, component: ConfuciusComponent, data_dir: str | Path) -> None:
        self.component = component
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.process: subprocess.Popen | None = None
        self.port = 0
        self.token = ""
        self.generation = ""
        self._lock = threading.RLock()
        self._model_lock = threading.RLock()
        self._owners: dict[str, str] = {}
        self._browser_tokens: dict[str, str] = {}
        self._session_activity: dict[str, float] = {}
        self._final_results: dict[str, dict[str, Any]] = {}
        self._gpu_leases: dict[str, str] = {}
        self._log_file = None
        self._local_http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @staticmethod
    def _unused_port() -> int:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _log_path(self) -> Path:
        return self.data_dir / "logs" / "confucius_asr" / "worker.log"

    def _start(self) -> None:
        if not license_accepted(self.data_dir):
            raise ConfuciusError("ASR_LICENSE_REQUIRED", "请先阅读并接受 Confucius4-R2T2 模型许可。")
        verify_component_integrity(self.component)
        compatible, code, message = _profile_compatibility(self.component.profile)
        if not compatible:
            raise ConfuciusError(code, message)
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        log_path = self._log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.is_file() and log_path.stat().st_size > 8 * 1024 * 1024:
            previous = log_path.with_suffix(".previous.log")
            previous.unlink(missing_ok=True)
            log_path.replace(previous)
        temp_dir = self.data_dir / "temp" / "confucius_asr"
        temp_dir.mkdir(parents=True, exist_ok=True)
        self.port = self._unused_port()
        self.token = secrets.token_urlsafe(48)
        env_keys = (
            "SYSTEMROOT", "WINDIR", "PATH", "USERPROFILE", "CUDA_VISIBLE_DEVICES",
        )
        env = {key: os.environ[key] for key in env_keys if key in os.environ}
        env.update({
            "R2T2_WORKER_TOKEN": self.token,
            "R2T2_MODEL_DIR": str(self.component.model_dir),
            "R2T2_BUILD_DIR": str(self.component.build_dir),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONFAULTHANDLER": "1",
            "PYTHONPATH": str(self.component.root),
            "TEMP": str(temp_dir),
            "TMP": str(temp_dir),
        })
        self._log_file = log_path.open("ab", buffering=0)
        flags = (
            subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            if os.name == "nt" else 0
        )
        try:
            self.process = subprocess.Popen(
                [str(self.component.python), "-m", "r2t2_core.worker", "--port", str(self.port)],
                cwd=str(self.component.root),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
        except OSError as exc:
            self.process = None
            self._log_file.close()
            self._log_file = None
            raise ConfuciusError(
                "ASR_WORKER_START",
                f"Confucius 识别服务无法启动，请查看脱敏日志：{log_path}",
                retryable=True,
            ) from exc
        self._owners.clear()
        self._browser_tokens.clear()
        self._session_activity.clear()
        self._final_results.clear()
        for _ in range(120):
            if self.process.poll() is not None:
                self.close()
                raise ConfuciusError(
                    "ASR_WORKER_CRASH",
                    f"Confucius 识别服务启动失败，请查看脱敏日志：{log_path}",
                    retryable=True,
                )
            try:
                info = self._request("GET", "/health", timeout=1, invalidate=False)
                if int(info.get("protocol_version", 0)) != PROTOCOL_VERSION:
                    raise ConfuciusError("ASR_WORKER_PROTOCOL", "Confucius 组件协议版本不匹配。")
                self.generation = str(info.get("generation") or "")
                return
            except ConfuciusError as exc:
                if exc.code == "ASR_WORKER_PROTOCOL":
                    self.close()
                    raise
                time.sleep(0.1)
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.1)
        self.close()
        raise ConfuciusError("ASR_WORKER_START_TIMEOUT", "Confucius 识别服务启动超时。", retryable=True)

    def ensure(self) -> None:
        with self._lock:
            if self.process is None or self.process.poll() is not None:
                self._start()

    def _invalidate(self) -> None:
        with self._lock:
            process = self.process
            self.process = None
            self._owners.clear()
            self._browser_tokens.clear()
            self._session_activity.clear()
            self._final_results.clear()
            leases = list(self._gpu_leases.values())
            self._gpu_leases.clear()
            if process is not None and process.poll() is None:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=10,
                        check=False,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                else:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for token in leases:
                self._release_gpu(token)

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 120,
        invalidate: bool = True,
    ) -> dict[str, Any]:
        request_headers = {"Authorization": "Bearer " + self.token, **(headers or {})}
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=body, method=method, headers=request_headers
        )
        try:
            with self._local_http.open(request, timeout=timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("worker response is not an object")
                return value
        except urllib.error.HTTPError as exc:
            try:
                payload = json.loads(exc.read().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = {}
            code = str(payload.get("code") or "ASR_WORKER_ERROR")
            message = str(payload.get("message") or code)
            raise ConfuciusError(code, message, http_status=exc.code) from exc
        except (OSError, TimeoutError, urllib.error.URLError, ValueError) as exc:
            if invalidate:
                self._invalidate()
            raise ConfuciusError(
                "ASR_WORKER_UNAVAILABLE",
                "Confucius 识别服务连接中断，请重试。",
                retryable=True,
            ) from exc

    def request(
        self,
        method: str,
        path: str,
        *,
        value: dict[str, Any] | None = None,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 120,
    ) -> dict[str, Any]:
        self.ensure()
        if value is not None:
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            headers = {"Content-Type": "application/json", **(headers or {})}
        return self._request(method, path, body=body, headers=headers, timeout=timeout)

    @staticmethod
    def default_model_config() -> dict[str, int]:
        return {"n_ctx": 8192, "n_batch": 1024, "n_threads": 8, "gpu_layers": -1}

    def load(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._model_lock:
            return self.request("POST", "/models/load", value=config or self.default_model_config(), timeout=240)

    def unload(self) -> dict[str, Any]:
        with self._model_lock:
            return self.request("POST", "/models/unload", value={}, timeout=120)

    def transcribe(self, pcm_bytes: bytes, options: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        metadata = json.dumps(options, ensure_ascii=False).encode("utf-8")
        if len(metadata) > 16_384:
            raise ConfuciusError("ASR_CONTEXT_LIMIT", "识别参数超过 16 KiB。")
        payload = struct.pack("<I", len(metadata)) + metadata + pcm_bytes
        lease = self._acquire_gpu("confucius_file_asr")
        try:
            with self._model_lock:
                self.load(config)
                return self.request(
                    "POST", "/transcribe", body=payload,
                    headers={"Content-Type": "application/octet-stream"}, timeout=1800,
                )
        finally:
            try:
                if self.process is not None and self.process.poll() is None:
                    self.unload()
            except ConfuciusError:
                pass
            self._release_gpu(lease)

    def start_live(self, options: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        self._prune_stale_sessions()
        lease = self._acquire_gpu("confucius_live_asr")
        try:
            with self._model_lock:
                self.load(config)
                owner = secrets.token_urlsafe(48)
                browser = secrets.token_urlsafe(32)
                result = self.request("POST", "/sessions", value=options, headers={"X-R2T2-Owner": owner}, timeout=300)
                sid = str(result["session_id"])
                self._owners[sid] = owner
                self._browser_tokens[sid] = browser
                self._session_activity[sid] = time.monotonic()
                self._gpu_leases[sid] = lease
                return {**result, "browser_token": browser}
        except Exception:
            try:
                if self.process is not None and self.process.poll() is None:
                    self.unload()
            except ConfuciusError:
                pass
            self._release_gpu(lease)
            raise

    def check_browser(self, sid: str, token: str) -> None:
        self._prune_stale_sessions()
        expected = self._browser_tokens.get(str(sid))
        if not expected or not secrets.compare_digest(expected, str(token or "")):
            raise ConfuciusError("FORBIDDEN", "实时识别会话凭据无效。", http_status=403)

    def session_request(
        self,
        method: str,
        sid: str,
        action: str,
        *,
        value: dict[str, Any] | None = None,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        sid = str(sid)
        self._prune_stale_sessions(exclude=sid)
        if sid in self._final_results and action in {"result", "status"}:
            self._session_activity[sid] = time.monotonic()
            cached = dict(self._final_results[sid])
            if action == "status":
                return {
                    "session_id": sid,
                    "status": str(cached.get("status") or "finalized"),
                    "revision": int(cached.get("revision") or 0),
                    "cached": True,
                }
            return cached
        owner = self._owners.get(sid)
        if not owner:
            raise ConfuciusError("SESSION_NOT_FOUND", "实时识别会话不存在或 worker 已重启。", http_status=404)
        try:
            answer = self.request(
                method, f"/sessions/{sid}/{action}", value=value, body=body,
                headers={"X-R2T2-Owner": owner, **(headers or {})},
                timeout=240 if action in {"feed", "finish"} else 30,
            )
        except ConfuciusError as exc:
            if exc.code == "SESSION_NOT_FOUND":
                self.forget_session(sid)
            raise
        if action == "finish":
            self._final_results[sid] = dict(answer)
            self._release_session_gpu(sid)
        elif action == "cancel":
            self._final_results.pop(sid, None)
            self._release_session_gpu(sid)
        self._session_activity[sid] = time.monotonic()
        return answer

    def forget_session(self, sid: str) -> None:
        self._owners.pop(str(sid), None)
        self._browser_tokens.pop(str(sid), None)
        self._session_activity.pop(str(sid), None)
        self._final_results.pop(str(sid), None)
        self._release_session_gpu(sid)

    def _prune_stale_sessions(self, *, exclude: str = "", max_idle_seconds: float = 4500.0) -> None:
        """Cancel abandoned live sessions and expire cached terminal results."""
        now = time.monotonic()
        stale = [
            sid for sid, touched in list(self._session_activity.items())
            if sid != str(exclude) and now - touched > max_idle_seconds
        ]
        for sid in stale:
            owner = self._owners.get(sid)
            if owner and sid not in self._final_results:
                try:
                    self._request(
                        "POST", f"/sessions/{sid}/cancel",
                        body=b"{}",
                        headers={"X-R2T2-Owner": owner, "Content-Type": "application/json"},
                        timeout=10,
                        invalidate=False,
                    )
                except ConfuciusError:
                    pass
            self.forget_session(sid)

    @staticmethod
    def _acquire_gpu(source: str) -> str:
        coordinator = _GPU_COORDINATOR
        return str(coordinator.acquire_lease(source)) if coordinator is not None else ""

    @staticmethod
    def _release_gpu(token: str) -> None:
        coordinator = _GPU_COORDINATOR
        if coordinator is not None and token:
            coordinator.release_lease(token)

    def _release_session_gpu(self, sid: str) -> None:
        token = self._gpu_leases.pop(str(sid), "")
        try:
            if token and self.process is not None and self.process.poll() is None:
                self.unload()
        except ConfuciusError:
            pass
        self._release_gpu(token)

    def close(self) -> None:
        self._invalidate()
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None


def get_manager(data_dir: str | Path) -> ConfuciusWorkerManager:
    component = discover_component(data_dir)
    key = (str(component.root).casefold(), str(Path(data_dir).expanduser().resolve()).casefold())
    with _MANAGERS_LOCK:
        manager = _MANAGERS.get(key)
        if manager is None:
            manager = ConfuciusWorkerManager(component, data_dir)
            _MANAGERS[key] = manager
        return manager


def close_all_managers() -> None:
    with _MANAGERS_LOCK:
        managers = list(_MANAGERS.values())
        _MANAGERS.clear()
    for manager in managers:
        manager.close()


def set_gpu_coordinator(coordinator: Any | None) -> None:
    global _GPU_COORDINATOR
    _GPU_COORDINATOR = coordinator


def map_language(language: str) -> str:
    value = str(language or "AUTO").upper()
    if value not in LANGUAGE_MAP:
        raise ConfuciusError("ASR_LANGUAGE_UNSUPPORTED", f"Confucius 暂不支持识别语言：{value}")
    return LANGUAGE_MAP[value]


def normalize_result(result: dict[str, Any], *, requested_language: str) -> dict[str, Any]:
    segments = result.get("segments")
    acoustic_segments = []
    if isinstance(segments, list):
        for item in segments:
            if not isinstance(item, dict):
                continue
            try:
                start_sample = max(0, int(item.get("start_sample", 0)))
                end_sample = max(start_sample, int(item.get("end_sample", start_sample)))
            except (TypeError, ValueError):
                continue
            try:
                segment_id = int(item.get("segment_id", len(acoustic_segments)))
            except (TypeError, ValueError):
                segment_id = len(acoustic_segments)
            acoustic_segments.append({
                "segment_id": segment_id,
                "start_ms": round(start_sample / 16, 3),
                "end_ms": round(end_sample / 16, 3),
                "end_reason": str(item.get("end_reason") or ""),
                "truncated": bool(item.get("truncated", False)),
            })
    status = str(result.get("status") or ("truncated" if result.get("truncated") else "complete"))
    try:
        forced_boundaries = max(0, int(result.get("forced_boundaries") or 0))
    except (TypeError, ValueError):
        forced_boundaries = 0
    return {
        "text": str(result.get("text") or "").strip(),
        "detected_language": str(result.get("language") or ""),
        "requested_language": str(requested_language or "AUTO").upper(),
        "backend": PROVIDER_NAME,
        "model": str(result.get("model") or MODEL_NAME),
        "device": "cuda",
        "status": status,
        "quality_status": str(result.get("quality_status") or "standard"),
        "segments": len(acoustic_segments),
        "segment_count": len(acoustic_segments),
        "acoustic_segments": acoustic_segments,
        "word_timestamps": [],
        "timestamps_available": False,
        "capabilities": dict(CAPABILITIES),
        "truncated": bool(result.get("truncated", False)),
        "forced_boundaries": forced_boundaries,
        "provider_result": result,
    }


atexit.register(close_all_managers)


__all__ = [
    "CAPABILITIES", "ConfuciusComponent", "ConfuciusError", "ConfuciusWorkerManager",
    "LANGUAGE_MAP", "MODEL_FILES", "PROVIDER_NAME", "SUPPORTED_LANGUAGES",
    "accept_license", "close_all_managers", "component_status", "discover_component",
    "get_manager", "license_accepted", "map_language", "normalize_result", "set_gpu_coordinator",
    "verify_component_integrity",
]
