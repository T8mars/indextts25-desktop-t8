from __future__ import annotations

import json
import struct
import threading

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from confucius_asr import ConfuciusError
from confucius_asr_gateway import ConfuciusDesktopGateway, build_confucius_router


class FakeGateway:
    data_dir = "D:/test-data"

    def status(self, *, verify_hash=False):
        return {"ready": True, "verifyHash": verify_hash}


def client():
    app = FastAPI()
    app.include_router(build_confucius_router(FakeGateway()))
    return TestClient(app)


class InterruptedManager:
    def __init__(self):
        self.cancelled = False

    def check_browser(self, sid, token):
        assert sid == "session"
        assert token == "browser-token"

    def session_request(self, method, sid, action, **kwargs):
        if action == "status":
            return {"status": "interrupted"}
        if action == "cancel":
            self.cancelled = True
            return {"status": "interrupted"}
        raise AssertionError((method, sid, action, kwargs))

    def forget_session(self, sid):
        raise AssertionError(f"cancel should acknowledge session {sid}")


class InterruptedGateway(FakeGateway):
    def __init__(self):
        self.worker = InterruptedManager()
        self.claimed = set()

    def manager(self):
        return self.worker

    def claim(self, sid):
        if sid in self.claimed:
            return False
        self.claimed.add(sid)
        return True

    def release(self, sid):
        self.claimed.discard(sid)


def test_status_rejects_remote_or_missing_origin():
    with client() as browser:
        assert browser.get("/api/confucius/status").status_code == 403
        assert browser.get(
            "/api/confucius/status",
            headers={"Host": "127.0.0.1:7860", "Origin": "https://evil.example"},
        ).status_code == 403
        assert browser.get(
            "/api/confucius/status",
            headers={"Host": "[", "Origin": "http://["},
        ).status_code == 403


def test_status_accepts_loopback_same_origin_and_referer():
    with client() as browser:
        headers = {"Host": "127.0.0.1:7860", "Origin": "http://127.0.0.1:7860"}
        response = browser.get("/api/confucius/status?verify_hash=true", headers=headers)
        assert response.status_code == 200
        assert response.json() == {"ready": True, "verifyHash": True}
        referer = {"Host": "localhost:7860", "Referer": "http://localhost:7860/"}
        assert browser.get("/api/confucius/status", headers=referer).status_code == 200


def test_client_assets_are_same_origin_only():
    with client() as browser:
        headers = {"Host": "127.0.0.1:7860", "Referer": "http://127.0.0.1:7860/"}
        response = browser.get("/api/confucius/assets/client.js", headers=headers)
        assert response.status_code == 200
        assert "getUserMedia" in response.text
        assert response.headers["x-content-type-options"] == "nosniff"


def test_live_start_rejects_unsafe_or_invalid_configuration_before_worker_start():
    with client() as browser:
        headers = {"Host": "127.0.0.1:7860", "Origin": "http://127.0.0.1:7860"}
        assert browser.post(
            "/api/confucius/live/start", headers=headers, content=b"{}"
        ).status_code == 415
        assert browser.post(
            "/api/confucius/live/start",
            headers={**headers, "Content-Type": "application/json"},
            json=[],
        ).status_code == 400
        response = browser.post(
            "/api/confucius/live/start",
            headers={**headers, "Content-Type": "application/json"},
            json={"language": "Arabic"},
        )
        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "ASR_LANGUAGE_UNSUPPORTED"
        response = browser.post(
            "/api/confucius/live/start",
            headers={**headers, "Content-Type": "application/json"},
            json={"language": "Chinese", "model_config": {"n_ctx": 999999}},
        )
        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "MODEL_CONFIG_FORBIDDEN"


def test_license_requires_an_explicit_json_boolean():
    with client() as browser:
        headers = {
            "Host": "127.0.0.1:7860",
            "Origin": "http://127.0.0.1:7860",
            "Content-Type": "application/json",
        }
        assert browser.post(
            "/api/confucius/license", headers=headers, json=[]
        ).status_code == 400
        assert browser.post(
            "/api/confucius/license", headers=headers, json={"accepted": "false"}
        ).status_code == 400


def test_pending_live_session_can_only_be_claimed_once(tmp_path):
    gateway = ConfuciusDesktopGateway(tmp_path)
    gateway.reserve("session")
    assert gateway.claim("session") is True
    assert gateway.claim("session") is False
    gateway.release("session")
    assert gateway.claim("session") is False

    gateway.reserve("abandoned")
    assert gateway.abandon("abandoned") is True
    assert gateway.abandon("abandoned") is False
    assert gateway.claim("abandoned") is False


class _LiveManager:
    def __init__(self):
        self.checked_on = None
        self.actions = []
        self.transcriptions = []

    def check_browser(self, sid, token):
        self.checked_on = threading.current_thread().name
        if sid != "session" or token != "browser-token":
            raise ConfuciusError("FORBIDDEN", "bad browser token", http_status=403)

    def session_request(self, method, sid, action, **kwargs):
        self.actions.append((method, sid, action, kwargs))
        if action == "status":
            return {"status": "streaming"}
        if action == "feed":
            return {"ack_sample": 1, "events": []}
        if action == "finish":
            return {"status": "finalized", "text": "最终字幕"}
        if action == "cancel":
            return {"status": "cancelled"}
        if action == "result":
            return {"status": "finalized", "text": "最终字幕"}
        raise AssertionError(action)

    def transcribe(self, pcm_bytes, options):
        self.transcriptions.append((pcm_bytes, options))
        return {
            "status": "complete",
            "text": "文件转写结果",
            "language": options["language"],
            "segments": [],
        }


class _LiveGateway(ConfuciusDesktopGateway):
    def __init__(self, tmp_path):
        super().__init__(tmp_path)
        self.live_manager = _LiveManager()

    def manager(self):
        return self.live_manager


def test_websocket_rejects_unknown_control_and_cleans_up_session(tmp_path):
    gateway = _LiveGateway(tmp_path)
    gateway.reserve("session")
    app = FastAPI()
    app.include_router(build_confucius_router(gateway))
    headers = {"Host": "127.0.0.1:7860", "Origin": "http://127.0.0.1:7860"}

    with TestClient(app) as browser:
        with browser.websocket_connect(
            "/api/confucius/live/session/stream", headers=headers
        ) as socket:
            socket.send_json({"browser_token": "browser-token"})
            assert socket.receive_json()["type"] == "ready"
            socket.send_json({"type": "unsupported"})
            error = socket.receive_json()
            assert error == {
                "type": "error",
                "code": "STREAM_ERROR",
                "message": "实时识别控制帧无效。",
            }

    assert any(action[2] == "cancel" for action in gateway.live_manager.actions)
    assert gateway.claim("session") is False


def test_result_authentication_and_worker_call_run_off_request_thread(tmp_path):
    gateway = _LiveGateway(tmp_path)
    gateway.live_manager.checked_on = ""
    app = FastAPI()
    app.include_router(build_confucius_router(gateway))
    headers = {
        "Host": "127.0.0.1:7860",
        "Origin": "http://127.0.0.1:7860",
        "X-R2T2-Session-Token": "browser-token",
    }

    with TestClient(app) as browser:
        response = browser.get("/api/confucius/live/session/result", headers=headers)

    assert response.status_code == 200
    assert response.json()["text"] == "最终字幕"
    assert gateway.live_manager.checked_on
    assert (
        gateway.live_manager.checked_on.startswith("asyncio")
        or gateway.live_manager.checked_on.endswith("worker thread")
    )


def _file_body(language="Chinese", context="会议", hotwords="专有名词"):
    metadata = json.dumps(
        {"language": language, "context": context, "hotwords": hotwords},
        ensure_ascii=False,
    ).encode("utf-8")
    pcm = np.zeros(1600, dtype="<f4").tobytes()
    return struct.pack("<I", len(metadata)) + metadata + pcm


def test_file_transcription_uses_memory_pcm_and_shared_options(tmp_path):
    gateway = _LiveGateway(tmp_path)
    app = FastAPI()
    app.include_router(build_confucius_router(gateway))
    headers = {
        "Host": "127.0.0.1:7860",
        "Origin": "http://127.0.0.1:7860",
        "Content-Type": "application/octet-stream",
    }

    with TestClient(app) as browser:
        response = browser.post(
            "/api/confucius/file/transcribe",
            headers=headers,
            content=_file_body(),
        )

    assert response.status_code == 200, response.text
    assert response.json()["text"] == "文件转写结果"
    assert response.json()["backend"] == "confucius_r2t2"
    pcm, options = gateway.live_manager.transcriptions[0]
    assert len(pcm) == 1600 * 4
    assert options["language"] == "Chinese"
    assert options["context"] == "会议"
    assert options["hotwords"] == "专有名词"
    assert not list(tmp_path.rglob("*.wav"))


def test_file_transcription_rejects_concurrent_live_session(tmp_path):
    gateway = _LiveGateway(tmp_path)
    gateway.reserve("pending-live-session")
    app = FastAPI()
    app.include_router(build_confucius_router(gateway))
    headers = {
        "Host": "127.0.0.1:7860",
        "Origin": "http://127.0.0.1:7860",
        "Content-Type": "application/octet-stream",
    }

    with TestClient(app) as browser:
        response = browser.post(
            "/api/confucius/file/transcribe",
            headers=headers,
            content=_file_body(),
        )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "ASR_LIVE_BUSY"


def test_arabic_file_transcription_uses_whisper_fallback(monkeypatch, tmp_path):
    import speech_review

    class Coordinator:
        def __init__(self):
            self.released = []

        def acquire_lease(self, source):
            assert source == "whisper_file_asr"
            return "lease"

        def release_lease(self, token):
            self.released.append(token)

    coordinator = Coordinator()
    gateway = _LiveGateway(tmp_path)
    gateway.coordinator = coordinator
    monkeypatch.setattr(
        speech_review,
        "transcribe_waveform",
        lambda waveform, sample_rate, **kwargs: {
            "text": "مرحبا",
            "backend": "openai_whisper",
            "requested_language": kwargs["language"],
        },
    )
    app = FastAPI()
    app.include_router(build_confucius_router(gateway))
    headers = {
        "Host": "127.0.0.1:7860",
        "Origin": "http://127.0.0.1:7860",
        "Content-Type": "application/octet-stream",
    }

    with TestClient(app) as browser:
        response = browser.post(
            "/api/confucius/file/transcribe",
            headers=headers,
            content=_file_body(language="Arabic"),
        )

    assert response.status_code == 200, response.text
    assert response.json()["text"] == "مرحبا"
    assert response.json()["backend"] == "openai_whisper"
    assert coordinator.released == ["lease"]


def test_interrupted_live_session_is_cancelled_to_release_gpu_lease():
    gateway = InterruptedGateway()
    app = FastAPI()
    app.include_router(build_confucius_router(gateway))
    with TestClient(app) as browser:
        with browser.websocket_connect(
            "/api/confucius/live/session/stream",
            headers={"Host": "localhost", "Origin": "http://localhost"},
        ) as websocket:
            websocket.send_json({"browser_token": "browser-token"})
            assert websocket.receive_json()["type"] == "ready"
            assert websocket.receive_json()["type"] == "error"
    assert gateway.worker.cancelled is True
    assert gateway.claimed == set()
