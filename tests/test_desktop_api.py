import time
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pytest
import torch
import gradio as gr
from fastapi.testclient import TestClient

from desktop_api import (
    DesktopTTSService,
    InferenceCoordinator,
    SpeechOptions,
    create_api_app,
    encode_audio_bytes,
)
from desktop_model_lifecycle import DesktopModelLifecycle
from desktop_voice_library import VoiceLibrary


API_KEY = "test-local-api-key-1234567890"


class _Tokenizer:
    @staticmethod
    def encode(text, allowed_special="all"):
        del allowed_special
        return list(str(text))


class _FakeTTS:
    tokenizer = _Tokenizer()
    qwen_emo = object()

    def __init__(self):
        self.calls = []

    @staticmethod
    def split_text_by_tokens(text, max_tokens, prefix):
        del max_tokens, prefix
        return [text]

    @staticmethod
    def normalize_emo_vec(vector, apply_bias=True):
        del apply_bias
        return np.asarray(vector, dtype=np.float32)

    def infer(self, **kwargs):
        self.calls.append(kwargs)
        samples = np.linspace(-0.2, 0.2, 2400, dtype=np.float32)
        return 24000, samples


def _reference_wave(path: Path) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24000)
        output.writeframes(np.zeros(2400, dtype="<i2").tobytes())


@pytest.fixture()
def api_fixture(tmp_path):
    reference = tmp_path / "voice.wav"
    _reference_wave(reference)
    voices = VoiceLibrary(tmp_path / "data")
    profile = voices.save("旁白", reference, language="ZH")
    model = _FakeTTS()
    coordinator = InferenceCoordinator()
    coordinator.guard_model(model)
    history = []
    service = DesktopTTSService(
        DesktopModelLifecycle(model, lambda: coordinator.guard_model(_FakeTTS())),
        voices,
        tmp_path / "outputs",
        coordinator,
        default_voice=profile.profile_id,
        history_writer=lambda item: history.append(item) or "",
    )
    app = create_api_app(
        service,
        api_key=API_KEY,
        desktop_version="9.9.9",
        coordinator=coordinator,
    )
    return TestClient(app), model, history, tmp_path


def _headers():
    return {"Authorization": f"Bearer {API_KEY}"}


def test_audio_encoding_supports_wav_pcm_and_mp3():
    audio = torch.zeros((1, 2400), dtype=torch.float32)
    wav, wav_type = encode_audio_bytes(audio, 24000, "wav")
    pcm, pcm_type = encode_audio_bytes(audio, 24000, "pcm")
    mp3, mp3_type = encode_audio_bytes(audio, 24000, "mp3")
    assert wav.startswith(b"RIFF") and wav_type == "audio/wav"
    assert len(pcm) == 4800 and pcm_type == "audio/pcm"
    assert len(mp3) > 100 and mp3_type == "audio/mpeg"


def test_openai_compatible_speech_is_authenticated_and_persistent(api_fixture):
    client, model, history, tmp_path = api_fixture
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/v1/models").status_code == 401

    voices = client.get("/v1/audio/voices", headers=_headers()).json()["data"]
    assert voices[0]["name"] == "旁白"
    response = client.post(
        "/v1/audio/speech",
        headers=_headers(),
        json={
            "model": "tts-1",
            "input": "1939年，API 测试。",
            "voice": "旁白",
            "response_format": "wav",
            "speed": 1.25,
        },
    )
    assert response.status_code == 200, response.text
    assert response.content.startswith(b"RIFF")
    assert unquote(response.headers["x-t8-voice"]) == "旁白"
    assert model.calls[-1]["duration_factor"] == pytest.approx(0.8)
    assert history[-1]["source"] == "api"
    assert list((tmp_path / "outputs").glob("api_*.wav"))


def test_native_async_job_reports_progress_and_downloads_result(api_fixture):
    client, _model, _history, _tmp_path = api_fixture
    created = client.post(
        "/api/v1/jobs",
        headers={"X-API-Key": API_KEY},
        json={"input": "异步生成测试。", "voice": "旁白", "response_format": "wav"},
    )
    assert created.status_code == 200, created.text
    job_id = created.json()["job_id"]
    deadline = time.time() + 5
    status = created.json()
    while time.time() < deadline and status["status"] not in {"completed", "failed"}:
        time.sleep(0.02)
        status = client.get(f"/api/v1/jobs/{job_id}", headers=_headers()).json()
    assert status["status"] == "completed", status
    assert status["progress"] == 1.0
    assert status["completed_blocks"] == status["total_blocks"] == 1
    download = client.get(f"/api/v1/jobs/{job_id}/audio", headers=_headers())
    assert download.status_code == 200
    assert download.content.startswith(b"RIFF")


def test_validation_errors_use_stable_error_envelope(api_fixture):
    client, *_ = api_fixture
    response = client.post(
        "/v1/audio/speech",
        headers=_headers(),
        json={"input": "", "voice": "旁白"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "validation_error"


def test_openai_fixed_voice_alias_uses_configured_local_default(api_fixture):
    client, _model, _history, _tmp_path = api_fixture
    response = client.post(
        "/v1/audio/speech",
        headers=_headers(),
        json={"input": "默认音色别名。", "voice": "alloy", "response_format": "wav"},
    )
    assert response.status_code == 200, response.text
    assert unquote(response.headers["x-t8-voice"]) == "旁白"


def test_api_routes_and_gradio_routes_can_share_one_fastapi_app(api_fixture):
    client, _model, _history, tmp_path = api_fixture
    app = client.app
    with gr.Blocks() as demo:
        gr.Markdown("shared service")
    mounted = gr.mount_gradio_app(
        app,
        demo,
        path="/",
        allowed_paths=[str(tmp_path)],
    )
    mounted_client = TestClient(mounted)
    health = mounted_client.get("/health")
    assert health.status_code == 200
    assert health.json()["api_key_fingerprint"]
    assert mounted_client.get("/gradio_api/info").status_code == 200


def test_inference_coordinator_serializes_desktop_and_api_calls():
    coordinator = InferenceCoordinator()
    state_lock = threading.Lock()
    release_first = threading.Event()
    active = 0
    peak_active = 0

    def infer(label):
        nonlocal active, peak_active
        with state_lock:
            active += 1
            peak_active = max(peak_active, active)
        if label == "desktop":
            assert release_first.wait(timeout=2)
        with state_lock:
            active -= 1
        return label

    class _ConcurrentModel:
        pass

    model = _ConcurrentModel()
    model.infer = infer
    coordinator.guard_model(model)

    def run(source):
        with coordinator.source(source):
            return model.infer(source)

    with ThreadPoolExecutor(max_workers=2) as pool:
        desktop = pool.submit(run, "desktop")
        deadline = time.time() + 2
        while time.time() < deadline and not coordinator.status()["active"]:
            time.sleep(0.01)
        api = pool.submit(run, "api")
        deadline = time.time() + 2
        while time.time() < deadline and coordinator.status()["waiting"] < 1:
            time.sleep(0.01)
        assert coordinator.status()["active_source"] == "desktop"
        assert coordinator.status()["waiting"] == 1
        release_first.set()
        assert desktop.result(timeout=2) == "desktop"
        assert api.result(timeout=2) == "api"

    assert peak_active == 1
    assert coordinator.status()["completed"] == 2
