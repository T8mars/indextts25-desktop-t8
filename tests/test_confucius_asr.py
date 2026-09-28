from __future__ import annotations

import json
from pathlib import Path

import pytest

import confucius_asr
import speech_review


def test_license_acceptance_is_hash_bound(tmp_path):
    assert confucius_asr.license_accepted(tmp_path) is False


def test_license_acceptance_rejects_valid_non_object_json(tmp_path):
    path = confucius_asr.license_acceptance_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("[]", encoding="utf-8")
    assert confucius_asr.license_accepted(tmp_path) is False


def test_license_acceptance_rejects_non_utf8_json(tmp_path):
    path = confucius_asr.license_acceptance_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xff\xfe")
    assert confucius_asr.license_accepted(tmp_path) is False


def test_component_roots_ignore_non_object_active_manifest(tmp_path):
    active = tmp_path / "optional_components" / "confucius" / "active.json"
    active.parent.mkdir(parents=True)
    active.write_text("[]", encoding="utf-8")
    roots = confucius_asr._component_roots(tmp_path)
    assert any(source == "installed-legacy" for _root, source in roots)


def test_component_manifest_rejects_non_numeric_schema_as_structured_error(tmp_path):
    (tmp_path / "confucius-component.json").write_text(
        json.dumps({"schemaVersion": "invalid"}), encoding="utf-8"
    )
    with pytest.raises(confucius_asr.ConfuciusError) as error:
        confucius_asr._read_component_manifest(tmp_path)
    assert error.value.code == "ASR_COMPONENT_MANIFEST"


def test_component_manifest_rejects_non_utf8_as_structured_error(tmp_path):
    (tmp_path / "confucius-component.json").write_bytes(b"\xff\xfe")
    with pytest.raises(confucius_asr.ConfuciusError) as error:
        confucius_asr._read_component_manifest(tmp_path)
    assert error.value.code == "ASR_COMPONENT_MANIFEST"
    saved = confucius_asr.accept_license(tmp_path, True)
    assert saved["licenseSha256"] == confucius_asr.MODEL_LICENSE_SHA256
    assert confucius_asr.license_accepted(tmp_path) is True
    path = confucius_asr.license_acceptance_path(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["licenseSha256"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")
    assert confucius_asr.license_accepted(tmp_path) is False


def test_map_language_and_result_contract():
    assert confucius_asr.map_language("ZH") == "Chinese"
    assert confucius_asr.map_language("yue") == "Cantonese"
    with pytest.raises(confucius_asr.ConfuciusError, match="暂不支持"):
        confucius_asr.map_language("AR")
    result = confucius_asr.normalize_result(
        {
            "text": " 测试文本 ",
            "language": "Chinese",
            "segments": [{"segment_id": 2, "start_sample": 1600, "end_sample": 3200}],
        },
        requested_language="ZH",
    )
    assert result["backend"] == "confucius_r2t2"
    assert result["text"] == "测试文本"
    assert result["word_timestamps"] == []
    assert result["timestamps_available"] is False
    assert result["acoustic_segments"][0]["start_ms"] == 100.0


def test_result_contract_tolerates_malformed_optional_worker_metadata():
    result = confucius_asr.normalize_result(
        {
            "text": "ok",
            "segments": [
                {"segment_id": "invalid", "start_sample": 0, "end_sample": 160}
            ],
            "forced_boundaries": "invalid",
        },
        requested_language="EN",
    )
    assert result["acoustic_segments"][0]["segment_id"] == 0
    assert result["forced_boundaries"] == 0


def test_auto_transcription_prefers_ready_confucius(monkeypatch, tmp_path):
    class FakeManager:
        def transcribe(self, data, options):
            assert len(data) == 16000 * 4
            assert options["language"] == "English"
            return {"text": "hello", "language": "English", "segments": []}

    monkeypatch.setattr(speech_review, "_confucius_ready", lambda root, language: True)
    monkeypatch.setattr(confucius_asr, "get_manager", lambda data_dir: FakeManager())
    result = speech_review.transcribe_waveform(
        [0.0] * 16000,
        16000,
        language="EN",
        backend="auto",
        download_root=Path(tmp_path) / "asr_models",
    )
    assert result["backend"] == "confucius_r2t2"
    assert result["text"] == "hello"


def test_arabic_auto_never_selects_confucius(monkeypatch):
    monkeypatch.setattr(speech_review, "_confucius_ready", lambda root, language: False)
    monkeypatch.setattr(speech_review, "resolve_asr_backend", lambda backend: "openai_whisper")

    class FakeWhisper:
        def transcribe(self, samples, **kwargs):
            return {"text": "مرحبا", "language": "ar", "segments": []}

    monkeypatch.setattr(speech_review, "load_asr_model", lambda *args: (FakeWhisper(), "cpu"))
    result = speech_review.transcribe_waveform([0.0] * 16000, 16000, language="AR")
    assert result["backend"] == "openai_whisper"


def test_component_status_redacts_local_paths(monkeypatch, tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for name, (size, _digest) in confucius_asr.MODEL_FILES.items():
        path = model_dir / name
        path.touch()
        path.write_bytes(b"x" * min(size, 2))
    component = confucius_asr.ConfuciusComponent(
        root=tmp_path,
        python=tmp_path / "worker-python.exe",
        model_dir=model_dir,
        build_dir=tmp_path / "native",
        profile="sm120",
        source="test",
    )
    monkeypatch.setattr(confucius_asr, "discover_component", lambda _data_dir: component)
    monkeypatch.setattr(confucius_asr, "license_accepted", lambda _data_dir: True)
    monkeypatch.setattr(
        confucius_asr,
        "_profile_compatibility",
        lambda _profile: (True, "READY", "matched"),
    )
    status = confucius_asr.component_status(tmp_path)
    encoded = json.dumps(status)
    assert str(tmp_path) not in encoded
    assert "root" not in status
    assert all("path" not in item for item in status["files"].values())


def test_terminal_result_is_cached_after_model_unload(monkeypatch, tmp_path):
    component = confucius_asr.ConfuciusComponent(
        root=tmp_path,
        python=tmp_path / "python.exe",
        model_dir=tmp_path,
        build_dir=tmp_path,
        profile="sm120",
        source="test",
    )
    manager = confucius_asr.ConfuciusWorkerManager(component, tmp_path)
    manager._owners["session"] = "owner"
    manager._browser_tokens["session"] = "browser"
    manager._session_activity["session"] = 1.0
    manager._gpu_leases["session"] = "lease"
    calls = []

    def fake_request(method, path, **kwargs):
        calls.append((method, path))
        return {"status": "finalized", "text": "最终字幕"}

    monkeypatch.setattr(manager, "request", fake_request)
    monkeypatch.setattr(manager, "unload", lambda: {"status": "unloaded"})
    monkeypatch.setattr(manager, "_release_gpu", lambda token: calls.append(("release", token)))
    answer = manager.session_request(
        "POST", "session", "finish", value={"last_seq": 0, "total_samples": 1}
    )
    assert answer["text"] == "最终字幕"
    cached = manager.session_request("GET", "session", "result")
    assert cached == answer
    assert manager.session_request("GET", "session", "status") == {
        "session_id": "session",
        "status": "finalized",
        "revision": 0,
        "cached": True,
    }
    assert calls.count(("POST", "/sessions/session/finish")) == 1
    assert ("release", "lease") in calls


def test_failed_finish_keeps_live_gpu_lease_for_retry(monkeypatch, tmp_path):
    component = confucius_asr.ConfuciusComponent(
        root=tmp_path,
        python=tmp_path / "python.exe",
        model_dir=tmp_path,
        build_dir=tmp_path,
        profile="sm120",
        source="test",
    )
    manager = confucius_asr.ConfuciusWorkerManager(component, tmp_path)
    manager._owners["session"] = "owner"
    manager._browser_tokens["session"] = "browser"
    manager._session_activity["session"] = 1.0
    manager._gpu_leases["session"] = "lease"
    calls = []

    def fail_finish(*_args, **_kwargs):
        raise confucius_asr.ConfuciusError("SEQUENCE_GAP", "retry", http_status=409)

    monkeypatch.setattr(manager, "request", fail_finish)
    monkeypatch.setattr(manager, "unload", lambda: calls.append("unload"))
    monkeypatch.setattr(manager, "_release_gpu", lambda token: calls.append(("release", token)))
    with pytest.raises(confucius_asr.ConfuciusError) as error:
        manager.session_request(
            "POST", "session", "finish", value={"last_seq": 0, "total_samples": 1}
        )
    assert error.value.code == "SEQUENCE_GAP"
    assert manager._gpu_leases["session"] == "lease"
    assert calls == []


def test_component_integrity_rejects_same_size_tampering(monkeypatch, tmp_path):
    payload = tmp_path / "model.gguf"
    payload.write_bytes(b"known")
    expected = confucius_asr._sha256(payload)
    monkeypatch.setattr(confucius_asr, "MODEL_FILES", {payload.name: (5, expected)})
    confucius_asr._INTEGRITY_CACHE.clear()
    component = confucius_asr.ConfuciusComponent(
        root=tmp_path,
        python=tmp_path / "python.exe",
        model_dir=tmp_path,
        build_dir=tmp_path,
        profile="sm120",
        source="test",
    )
    confucius_asr.verify_component_integrity(component)
    payload.write_bytes(b"wrong")
    with pytest.raises(confucius_asr.ConfuciusError) as error:
        confucius_asr.verify_component_integrity(component)
    assert error.value.code == "ASR_INTEGRITY"


def test_worker_spawn_failure_is_structured_and_closes_log(monkeypatch, tmp_path):
    component = confucius_asr.ConfuciusComponent(
        root=tmp_path,
        python=tmp_path / "python.exe",
        model_dir=tmp_path,
        build_dir=tmp_path,
        profile="sm120",
        source="test",
    )
    manager = confucius_asr.ConfuciusWorkerManager(component, tmp_path)
    monkeypatch.setattr(confucius_asr, "license_accepted", lambda _data_dir: True)
    monkeypatch.setattr(confucius_asr, "verify_component_integrity", lambda _component: None)
    monkeypatch.setattr(
        confucius_asr, "_profile_compatibility", lambda _profile: (True, "READY", "ready")
    )
    monkeypatch.setattr(
        confucius_asr.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cannot spawn")),
    )
    with pytest.raises(confucius_asr.ConfuciusError) as error:
        manager._start()
    assert error.value.code == "ASR_WORKER_START"
    assert error.value.retryable is True
    assert manager.process is None
    assert manager._log_file is None
