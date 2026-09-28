"""Same-origin FastAPI/WebSocket gateway for Desktop Confucius live ASR."""

from __future__ import annotations

import asyncio
import json
import struct
import threading
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from fastapi import APIRouter, HTTPException, Request, Response, WebSocket, WebSocketDisconnect

from confucius_asr import (
    ConfuciusError,
    SUPPORTED_LANGUAGES,
    accept_license,
    component_status,
    discover_component,
    get_manager,
    normalize_result,
)


MAX_START_BODY = 64 * 1024
MAX_WS_MESSAGE = 16000 * 4 + 8
MAX_FILE_METADATA = 16 * 1024
MAX_FILE_PCM_BYTES = 128 * 1024 * 1024


def _loopback(value: str | None) -> bool:
    host = str(value or "").strip().lower().strip("[]")
    return host in {"127.0.0.1", "localhost", "::1"}


def _request_origin(request: Request, *, allow_referer: bool = False) -> None:
    origin = str(request.headers.get("origin") or "")
    if not origin and allow_referer:
        origin = str(request.headers.get("referer") or "")
    host_header = str(request.headers.get("host") or "")
    try:
        hostname = urlsplit("//" + host_header).hostname
        parsed = urlsplit(origin)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail={"code": "BAD_ORIGIN"}) from exc
    if (
        not origin
        or not _loopback(hostname)
        or parsed.scheme not in {"http", "https"}
        or parsed.netloc.lower() != host_header.lower()
    ):
        raise HTTPException(status_code=403, detail={"code": "BAD_ORIGIN"})


def _websocket_origin(websocket: WebSocket) -> bool:
    origin = str(websocket.headers.get("origin") or "")
    host_header = str(websocket.headers.get("host") or "")
    try:
        hostname = urlsplit("//" + host_header).hostname
        parsed = urlsplit(origin)
    except ValueError:
        return False
    return bool(
        origin
        and _loopback(hostname)
        and parsed.scheme in {"http", "https"}
        and parsed.netloc.lower() == host_header.lower()
    )


def _error_status(exc: ConfuciusError) -> int:
    if exc.http_status in {400, 403, 404, 409, 413, 415}:
        return int(exc.http_status)
    if exc.code in {"ASR_COMPONENT_MISSING", "ASR_LICENSE_REQUIRED"}:
        return 409
    return 503 if exc.retryable else 502


def _raise_http(exc: ConfuciusError) -> None:
    raise HTTPException(
        status_code=_error_status(exc),
        detail={"code": exc.code, "message": str(exc), "retryable": exc.retryable},
    ) from exc


class ConfuciusDesktopGateway:
    """Own browser-facing leases while the provider owns worker credentials."""

    def __init__(self, data_dir: str | Path, coordinator=None) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.coordinator = coordinator
        self._pending: set[str] = set()
        self._leases: set[str] = set()
        self._lock = threading.RLock()

    def reserve(self, sid: str) -> None:
        with self._lock:
            self._pending.add(str(sid))

    def claim(self, sid: str) -> bool:
        with self._lock:
            sid = str(sid)
            if sid in self._leases or sid not in self._pending:
                return False
            self._pending.discard(sid)
            self._leases.add(sid)
            return True

    def abandon(self, sid: str) -> bool:
        """Atomically expire a start response that never opened its socket."""

        with self._lock:
            sid = str(sid)
            if sid not in self._pending:
                return False
            self._pending.discard(sid)
            return True

    def release(self, sid: str) -> None:
        with self._lock:
            self._leases.discard(sid)

    def live_busy(self) -> bool:
        with self._lock:
            return bool(self._pending or self._leases)

    def acquire_gpu(self, source: str) -> str:
        if self.coordinator is None:
            return ""
        return str(self.coordinator.acquire_lease(source))

    def release_gpu(self, token: str) -> None:
        if self.coordinator is not None and token:
            self.coordinator.release_lease(token)

    def status(self, *, verify_hash: bool = False) -> dict:
        return component_status(self.data_dir, verify_hash=verify_hash)

    def manager(self):
        return get_manager(self.data_dir)


def build_confucius_router(gateway: ConfuciusDesktopGateway) -> APIRouter:
    router = APIRouter(prefix="/api/confucius", tags=["Confucius ASR"])

    async def authenticated_manager(sid: str, token: str):
        """Resolve and validate a browser session without blocking the ASGI loop.

        ``check_browser`` also prunes stale worker sessions.  That cleanup may
        make authenticated localhost HTTP calls, so it must not run directly
        on the event-loop thread used by the UI and WebSocket heartbeat.
        """

        manager = await asyncio.to_thread(gateway.manager)
        await asyncio.to_thread(manager.check_browser, sid, token)
        return manager

    @router.get("/status")
    async def status(request: Request, verify_hash: bool = False):
        _request_origin(request, allow_referer=True)
        return await asyncio.to_thread(gateway.status, verify_hash=bool(verify_hash))

    @router.post("/license")
    async def set_license(request: Request):
        _request_origin(request)
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
            raise HTTPException(status_code=415, detail={"code": "UNSUPPORTED_MEDIA_TYPE"})
        body = await request.body()
        if len(body) > 4096:
            raise HTTPException(status_code=413, detail={"code": "PAYLOAD_TOO_LARGE"})
        try:
            value = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail={"code": "INVALID_INPUT"}) from exc
        if not isinstance(value, dict) or type(value.get("accepted")) is not bool:
            raise HTTPException(status_code=400, detail={"code": "INVALID_INPUT"})
        accepted = value["accepted"]
        status_value = await asyncio.to_thread(gateway.status)
        if accepted and not status_value.get("available"):
            raise HTTPException(status_code=409, detail={"code": "ASR_COMPONENT_MISSING"})
        return await asyncio.to_thread(accept_license, gateway.data_dir, accepted)

    @router.get("/license-text")
    async def license_text(request: Request):
        _request_origin(request, allow_referer=True)
        try:
            component = await asyncio.to_thread(discover_component, gateway.data_dir)
        except ConfuciusError as exc:
            _raise_http(exc)
        candidates = (
            component.root / "licenses" / "MODEL_LICENSE",
            component.root / "vendor" / "r2t2_native" / "MODEL_LICENSE",
            component.model_dir / "MODEL_LICENSE",
        )
        path = next((item for item in candidates if item.is_file()), None)
        if path is None:
            raise HTTPException(status_code=404, detail={"code": "LICENSE_MISSING"})
        return Response(
            path.read_text(encoding="utf-8"),
            media_type="text/plain; charset=utf-8",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @router.get("/assets/r2t2-worklet.js")
    async def worklet(request: Request):
        _request_origin(request, allow_referer=True)
        path = Path(__file__).resolve().parent / "confucius_live_worklet.js"
        if not path.is_file():
            raise HTTPException(status_code=404, detail={"code": "ASSET_MISSING"})
        return Response(
            path.read_text(encoding="utf-8"),
            media_type="application/javascript",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @router.get("/assets/client.js")
    async def client(request: Request):
        _request_origin(request, allow_referer=True)
        path = Path(__file__).resolve().parent / "confucius_live_client.js"
        if not path.is_file():
            raise HTTPException(status_code=404, detail={"code": "ASSET_MISSING"})
        return Response(
            path.read_text(encoding="utf-8"),
            media_type="application/javascript",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @router.post("/live/start")
    async def live_start(request: Request):
        _request_origin(request)
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
            raise HTTPException(status_code=415, detail={"code": "UNSUPPORTED_MEDIA_TYPE"})
        body = await request.body()
        if len(body) > MAX_START_BODY:
            raise HTTPException(status_code=413, detail={"code": "PAYLOAD_TOO_LARGE"})
        try:
            value = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail={"code": "INVALID_INPUT"}) from exc
        if not isinstance(value, dict):
            raise HTTPException(status_code=400, detail={"code": "INVALID_INPUT"})
        context = str(value.get("context") or "")
        hotwords = str(value.get("hotwords") or "")
        if hotwords:
            context = (context + "\n" if context else "") + "Hotwords: " + hotwords
        if len(context) > 8192:
            raise HTTPException(status_code=400, detail={"code": "CONTEXT_LIMIT"})
        try:
            stream_chunk_ms = int(value.get("stream_chunk_ms", 320))
            min_segment_seconds = int(value.get("min_segment_seconds", 8))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail={"code": "INVALID_INPUT"}) from exc
        if stream_chunk_ms not in {160, 320, 480, 640} or min_segment_seconds not in {0, 4, 8}:
            raise HTTPException(status_code=400, detail={"code": "INVALID_INPUT"})
        language = str(value.get("language") or "Auto")
        if language not in {"Auto", *SUPPORTED_LANGUAGES}:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "ASR_LANGUAGE_UNSUPPORTED",
                    "message": "该语言不支持实时字幕；阿拉伯语请使用上传音频转写。",
                },
            )
        options = {
            "language": language,
            "context": context,
            "stream_chunk_ms": stream_chunk_ms,
            "min_segment_seconds": min_segment_seconds,
        }
        config = value.get("model_config")
        if config is not None:
            raise HTTPException(status_code=400, detail={"code": "MODEL_CONFIG_FORBIDDEN"})

        def start_worker():
            manager = gateway.manager()
            return manager, manager.start_live(options, None)

        try:
            manager, result = await asyncio.to_thread(start_worker)
        except ConfuciusError as exc:
            _raise_http(exc)
        sid = str(result.get("session_id") or "")
        if not sid:
            raise HTTPException(status_code=502, detail={"code": "ASR_WORKER_PROTOCOL"})
        gateway.reserve(sid)

        async def cancel_unclaimed() -> None:
            await asyncio.sleep(30)
            if not gateway.abandon(sid):
                return
            try:
                await asyncio.to_thread(manager.session_request, "POST", sid, "cancel", value={})
            except ConfuciusError:
                pass

        asyncio.create_task(cancel_unclaimed())
        return result

    @router.post("/file/transcribe")
    async def file_transcribe(request: Request):
        """Transcribe browser-decoded mono 16 kHz float32 PCM in memory."""

        _request_origin(request)
        if gateway.live_busy():
            raise HTTPException(
                status_code=409,
                detail={"code": "ASR_LIVE_BUSY", "message": "请先停止实时字幕，再转写音频文件。"},
            )
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/octet-stream":
            raise HTTPException(
                status_code=415,
                detail={"code": "UNSUPPORTED_MEDIA_TYPE", "message": "音频转写请求格式无效。"},
            )
        try:
            declared_size = int(request.headers.get("content-length") or 0)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"code": "INVALID_INPUT"}) from exc
        maximum_body = 4 + MAX_FILE_METADATA + MAX_FILE_PCM_BYTES
        if declared_size > maximum_body:
            raise HTTPException(
                status_code=413,
                detail={"code": "AUDIO_TOO_LARGE", "message": "解码后的音频超过 30 分钟或本地转写上限。"},
            )
        body = await request.body()
        if len(body) > maximum_body:
            raise HTTPException(
                status_code=413,
                detail={"code": "AUDIO_TOO_LARGE", "message": "解码后的音频超过 30 分钟或本地转写上限。"},
            )
        if len(body) < 8:
            raise HTTPException(
                status_code=400,
                detail={"code": "INVALID_AUDIO", "message": "音频数据为空或不完整。"},
            )
        metadata_size = struct.unpack_from("<I", body)[0]
        if metadata_size > MAX_FILE_METADATA or len(body) <= 4 + metadata_size:
            raise HTTPException(
                status_code=400,
                detail={"code": "INVALID_INPUT", "message": "音频转写参数过长或不完整。"},
            )
        pcm_bytes = body[4 + metadata_size:]
        if not pcm_bytes or len(pcm_bytes) > MAX_FILE_PCM_BYTES or len(pcm_bytes) % 4:
            raise HTTPException(
                status_code=400,
                detail={"code": "INVALID_AUDIO", "message": "音频 PCM 数据无效。"},
            )
        try:
            metadata = json.loads(body[4:4 + metadata_size].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail={"code": "INVALID_INPUT"}) from exc
        if not isinstance(metadata, dict):
            raise HTTPException(status_code=400, detail={"code": "INVALID_INPUT"})
        language = str(metadata.get("language") or "Auto")
        if language not in {"Auto", "Arabic", *SUPPORTED_LANGUAGES}:
            raise HTTPException(status_code=400, detail={"code": "ASR_LANGUAGE_UNSUPPORTED"})
        context = str(metadata.get("context") or "")
        hotwords = str(metadata.get("hotwords") or "")
        if len(context) > 4096 or len(hotwords) > 2048:
            raise HTTPException(status_code=400, detail={"code": "CONTEXT_LIMIT"})
        samples = np.frombuffer(pcm_bytes, dtype="<f4")
        if not np.isfinite(samples).all():
            raise HTTPException(
                status_code=400,
                detail={"code": "INVALID_AUDIO", "message": "音频包含无效采样。"},
            )

        if language == "Arabic":
            from speech_review import transcribe_waveform

            lease = await asyncio.to_thread(gateway.acquire_gpu, "whisper_file_asr")
            try:
                try:
                    return await asyncio.to_thread(
                        transcribe_waveform,
                        samples.copy(),
                        16000,
                        language="AR",
                        backend="auto",
                        download_root=gateway.data_dir / "asr_models",
                    )
                except Exception as exc:
                    raise HTTPException(
                        status_code=503,
                        detail={
                            "code": "ASR_FALLBACK_UNAVAILABLE",
                            "message": str(exc).strip() or "Whisper 回退不可用。",
                        },
                    ) from exc
            finally:
                await asyncio.to_thread(gateway.release_gpu, lease)

        options = {
            "sample_rate": 16000,
            "channels": 1,
            "channel": "mean",
            "mode": "offline",
            "stream_chunk_ms": 320,
            "language": language,
            "context": context,
            "hotwords": hotwords,
            "auto_gain": True,
        }
        try:
            manager = await asyncio.to_thread(gateway.manager)
            result = await asyncio.to_thread(manager.transcribe, pcm_bytes, options)
            language_codes = {
                "Chinese": "ZH", "English": "EN", "Japanese": "JA",
                "Spanish": "ES", "Cantonese": "YUE", "Korean": "KO",
                "German": "DE", "French": "FR", "Russian": "RU",
                "Portuguese": "PT", "Italian": "IT", "Auto": "AUTO",
            }
            return normalize_result(result, requested_language=language_codes[language])
        except ConfuciusError as exc:
            _raise_http(exc)

    @router.get("/live/{sid}/result")
    async def live_result(request: Request, sid: str):
        _request_origin(request, allow_referer=True)
        token = request.headers.get("X-R2T2-Session-Token", "")
        try:
            manager = await authenticated_manager(sid, token)
            return await asyncio.to_thread(manager.session_request, "GET", sid, "result")
        except ConfuciusError as exc:
            _raise_http(exc)

    @router.post("/live/{sid}/cancel")
    async def live_cancel(request: Request, sid: str):
        _request_origin(request)
        token = request.headers.get("X-R2T2-Session-Token", "")
        try:
            manager = await authenticated_manager(sid, token)
            gateway.abandon(sid)
            answer = await asyncio.to_thread(manager.session_request, "POST", sid, "cancel", value={})
            return answer
        except ConfuciusError as exc:
            _raise_http(exc)

    @router.websocket("/live/{sid}/stream")
    async def live_stream(websocket: WebSocket, sid: str):
        if not _websocket_origin(websocket):
            await websocket.close(code=1008, reason="bad origin")
            return
        await websocket.accept()
        manager = None
        authenticated = False
        terminal = False
        monitor_task: asyncio.Task | None = None

        async def send_error(code: str, message: str) -> None:
            try:
                await websocket.send_json({"type": "error", "code": code, "message": message})
            except (RuntimeError, WebSocketDisconnect):
                pass

        async def monitor() -> None:
            nonlocal terminal
            while not terminal:
                await asyncio.sleep(5)
                if terminal:
                    return
                try:
                    current = await asyncio.to_thread(manager.session_request, "GET", sid, "status")
                except ConfuciusError:
                    current = {"status": "worker_restarted"}
                if current.get("status") in {"interrupted", "failed", "worker_restarted"}:
                    await send_error(str(current.get("status") or "INTERRUPTED").upper(), "实时识别会话已停止，请重新开始。")
                    try:
                        await asyncio.to_thread(
                            manager.session_request, "POST", sid, "cancel", value={}
                        )
                    except ConfuciusError:
                        # A restarted worker or an already-reaped session can no
                        # longer acknowledge cancellation.  Forgetting the
                        # browser session still releases its long-lived GPU
                        # lease so TTS cannot remain blocked indefinitely.
                        await asyncio.to_thread(manager.forget_session, sid)
                    terminal = True
                    await websocket.close(code=1011)
                    return

        try:
            while True:
                message = (
                    await websocket.receive()
                    if authenticated
                    else await asyncio.wait_for(websocket.receive(), timeout=10.0)
                )
                kind = message.get("type")
                if kind == "websocket.disconnect":
                    break
                if not authenticated:
                    text = message.get("text")
                    if text is None:
                        await send_error("AUTH_REQUIRED", "需要会话凭据。")
                        await websocket.close(code=1008)
                        break
                    if len(text) > 4096:
                        await send_error("PAYLOAD_TOO_LARGE", "会话凭据帧过大。")
                        await websocket.close(code=1009)
                        break
                    try:
                        hello = json.loads(text)
                        token = str(hello.get("browser_token") or "") if isinstance(hello, dict) else ""
                        manager = await authenticated_manager(sid, token)
                    except (json.JSONDecodeError, ConfuciusError):
                        await send_error("FORBIDDEN", "实时识别会话凭据无效。")
                        await websocket.close(code=1008)
                        break
                    if not gateway.claim(sid):
                        await send_error("ALREADY_CONNECTED", "该会话已有音频连接。")
                        await websocket.close(code=1008)
                        break
                    authenticated = True
                    monitor_task = asyncio.create_task(monitor())
                    await websocket.send_json({"type": "ready", "protocol_version": 1})
                    continue
                binary = message.get("bytes")
                text = message.get("text")
                if binary is not None:
                    if len(binary) < 12 or len(binary) > MAX_WS_MESSAGE or (len(binary) - 8) % 4:
                        await send_error("INVALID_FRAME", "音频帧格式无效。")
                        break
                    seq, start_sample = struct.unpack_from("<II", binary)
                    answer = await asyncio.to_thread(
                        manager.session_request,
                        "POST", sid, "feed", body=binary[8:],
                        headers={
                            "X-R2T2-Seq": str(seq),
                            "X-R2T2-Start-Sample": str(start_sample),
                            "Content-Type": "application/octet-stream",
                        },
                    )
                    await websocket.send_json({"type": "ack", **answer})
                    continue
                if text is None:
                    continue
                if len(text) > 4096:
                    await send_error("PAYLOAD_TOO_LARGE", "控制帧过大。")
                    break
                value = json.loads(text)
                if not isinstance(value, dict):
                    raise ValueError("stream command must be an object")
                if value.get("type") == "finish":
                    answer = await asyncio.to_thread(
                        manager.session_request, "POST", sid, "finish",
                        value={"last_seq": int(value["last_seq"]), "total_samples": int(value["total_samples"])},
                    )
                    terminal = True
                    await websocket.send_json({"type": "final", **answer})
                    await websocket.close()
                    break
                if value.get("type") == "cancel":
                    answer = await asyncio.to_thread(manager.session_request, "POST", sid, "cancel", value={})
                    terminal = True
                    await websocket.send_json({"type": "cancelled", **answer})
                    await websocket.close()
                    break
                raise ValueError("unsupported stream command")
        except asyncio.TimeoutError:
            await send_error("AUTH_TIMEOUT", "实时识别连接认证超时。")
        except (ConfuciusError, KeyError, ValueError, json.JSONDecodeError) as exc:
            code = exc.code if isinstance(exc, ConfuciusError) else "STREAM_ERROR"
            message = str(exc) if isinstance(exc, ConfuciusError) else "实时识别控制帧无效。"
            await send_error(code, message)
        except WebSocketDisconnect:
            pass
        finally:
            if monitor_task is not None:
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
            if authenticated and manager is not None and not terminal:
                try:
                    await asyncio.to_thread(manager.session_request, "POST", sid, "cancel", value={})
                except ConfuciusError:
                    pass
            if authenticated:
                gateway.release(sid)
            try:
                await websocket.close()
            except RuntimeError:
                pass

    return router


__all__ = ["ConfuciusDesktopGateway", "build_confucius_router"]
