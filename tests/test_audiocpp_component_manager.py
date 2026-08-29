import hashlib
import io
import json
import zipfile
from pathlib import Path

import audiocpp_component_manager as manager
import pytest


class _Response(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def getcode(self):
        return self.status


def test_download_verifies_checksum_and_emits_progress(tmp_path: Path, monkeypatch):
    payload = b"verified-payload"
    monkeypatch.setattr(manager.urllib.request, "urlopen", lambda *args, **kwargs: _Response(payload))
    events = []
    target = manager._download(
        "https://example.invalid/file.zip",
        tmp_path / "file.zip",
        expected_size=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        label="test",
        callback=events.append,
    )
    assert target.read_bytes() == payload
    assert {item["phase"] for item in events} >= {"downloading", "verifying"}


def test_download_recovers_completed_part_without_network(tmp_path: Path, monkeypatch):
    payload = b"complete-part"
    target = tmp_path / "file.zip"
    target.with_suffix(".zip.part").write_bytes(payload)
    monkeypatch.setattr(
        manager.urllib.request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network not expected")),
    )
    restored = manager._download(
        "https://example.invalid/file.zip",
        target,
        expected_size=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        label="test",
    )
    assert restored.read_bytes() == payload


def test_install_runtime_selects_assets_extracts_and_records_manifest(tmp_path: Path, monkeypatch):
    binary_zip = io.BytesIO()
    with zipfile.ZipFile(binary_zip, "w") as archive:
        archive.writestr("audio/bin/audiocpp_cli.exe", b"exe")
        archive.writestr("audio/bin/backend.dll", b"dll")
    runtime_zip = io.BytesIO()
    with zipfile.ZipFile(runtime_zip, "w") as archive:
        archive.writestr("audio/bin/cudart64_12.dll", b"cuda")
    payloads = {
        "https://example.invalid/bin.zip": binary_zip.getvalue(),
        "https://example.invalid/cudart.zip": runtime_zip.getvalue(),
    }

    def asset(name, url):
        data = payloads[url]
        return {
            "name": name,
            "size": len(data),
            "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
            "browser_download_url": url,
        }

    release = {
        "tag_name": "v9.9.9",
        "html_url": "https://example.invalid/release",
        "assets": [
            asset(
                "audio-v9.9.9-bin-windows-x64-cuda12.4.zip",
                "https://example.invalid/bin.zip",
            ),
            asset(
                "audio-v9.9.9-cudart-windows-x64-cuda12.4.zip",
                "https://example.invalid/cudart.zip",
            ),
        ],
    }
    monkeypatch.setattr(manager, "_request_json", lambda url: release)
    monkeypatch.setattr(manager.os, "name", "nt")
    monkeypatch.setattr(
        manager.urllib.request,
        "urlopen",
        lambda request, **kwargs: _Response(payloads[request.full_url]),
    )

    result = manager.install_runtime(tmp_path, "cuda")
    assert result["release"] == "v9.9.9"
    assert Path(result["executable"]).read_bytes() == b"exe"
    assert (Path(result["executable"]).parent / "cudart64_12.dll").read_bytes() == b"cuda"
    status = manager.component_status(tmp_path)
    assert status["runtimeReady"] is True


def test_install_model_uses_hf_revision_size_and_hash(tmp_path: Path, monkeypatch):
    payload = b"gguf-model"
    metadata = {
        "filename": "index-tts2_5-q8_0.gguf",
        "repositoryPath": "IndexTTS2.5-GGUF/index-tts2_5-q8_0.gguf",
        "revision": "a" * 40,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "url": "https://example.invalid/model",
    }
    monkeypatch.setattr(manager, "_model_metadata", lambda quantization: metadata)
    monkeypatch.setattr(manager.urllib.request, "urlopen", lambda *args, **kwargs: _Response(payload))
    result = manager.install_model(tmp_path, "q8_0")
    assert Path(result["modelPath"]).read_bytes() == payload
    saved = json.loads((Path(result["modelPath"]).parent / "t8-model.json").read_text(encoding="utf-8"))
    assert saved["revision"] == "a" * 40


def test_component_status_detects_same_size_model_corruption(tmp_path: Path, monkeypatch):
    payload = b"good-model"
    metadata = {
        "filename": "index-tts2_5-q8_0.gguf",
        "repositoryPath": "IndexTTS2.5-GGUF/index-tts2_5-q8_0.gguf",
        "revision": "b" * 40,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "url": "https://example.invalid/model",
    }
    monkeypatch.setattr(manager, "_model_metadata", lambda _quantization: metadata)
    monkeypatch.setattr(
        manager.urllib.request,
        "urlopen",
        lambda *args, **kwargs: _Response(payload),
    )
    result = manager.install_model(tmp_path, "q8_0")
    Path(result["modelPath"]).write_bytes(b"bad!-model")

    status = manager.component_status(tmp_path, verify_hash=True)

    assert status["modelReady"] is False
    assert status["modelIntegrity"] == "sha256-mismatch"


def test_component_status_uses_relative_paths_after_portable_move(tmp_path: Path, monkeypatch):
    payload = b"portable-model"
    metadata = {
        "filename": "index-tts2_5-q8_0.gguf",
        "repositoryPath": "IndexTTS2.5-GGUF/index-tts2_5-q8_0.gguf",
        "revision": "c" * 40,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "url": "https://example.invalid/model",
    }
    monkeypatch.setattr(manager, "_model_metadata", lambda _quantization: metadata)
    monkeypatch.setattr(manager.urllib.request, "urlopen", lambda *args, **kwargs: _Response(payload))
    original = tmp_path / "original"
    moved = tmp_path / "moved"
    manager.install_model(original, "q8_0")
    original.rename(moved)

    status = manager.component_status(moved, verify_hash=True)

    assert status["modelReady"] is True
    assert status["modelIntegrity"] == "verified"
    assert str(moved.resolve()) in status["modelPath"]


def test_runtime_update_keeps_previous_install_when_staging_copy_fails(
    tmp_path: Path,
    monkeypatch,
):
    binary_zip = io.BytesIO()
    with zipfile.ZipFile(binary_zip, "w") as archive:
        archive.writestr("audio/bin/audiocpp_cli.exe", b"old-exe")
    data = binary_zip.getvalue()
    release = {
        "tag_name": "v1.0.0",
        "html_url": "https://example.invalid/release",
        "assets": [
            {
                "name": "audio-v1.0.0-bin-windows-x64-cpu.zip",
                "size": len(data),
                "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
                "browser_download_url": "https://example.invalid/runtime.zip",
            }
        ],
    }
    monkeypatch.setattr(manager, "_request_json", lambda _url: release)
    monkeypatch.setattr(manager.os, "name", "nt")
    monkeypatch.setattr(manager.urllib.request, "urlopen", lambda *args, **kwargs: _Response(data))
    first = manager.install_runtime(tmp_path, "cpu")
    executable = Path(first["executable"])
    original_copytree = manager.shutil.copytree

    def fail_candidate_copy(source, destination, *args, **kwargs):
        if Path(destination).name.startswith(".cpu.new-"):
            raise OSError("simulated disk failure")
        return original_copytree(source, destination, *args, **kwargs)

    monkeypatch.setattr(manager.shutil, "copytree", fail_candidate_copy)
    with pytest.raises(OSError, match="simulated disk failure"):
        manager.install_runtime(tmp_path, "cpu")

    assert executable.read_bytes() == b"old-exe"
    assert not list(executable.parents[3].glob(".cpu.new-*"))
