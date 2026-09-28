"""Repeatable real-worker smoke test for the Desktop Confucius live gateway."""

# ruff: noqa: E402 -- project imports intentionally follow the local sys.path bootstrap.

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import torch
import torchaudio
from fastapi import FastAPI
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from confucius_asr import accept_license, close_all_managers, set_gpu_coordinator
from confucius_asr_gateway import ConfuciusDesktopGateway, build_confucius_router
from desktop_api import InferenceCoordinator
from indextts.utils.audio_io import load_audio_file


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        help="Optional unpacked component root; omit to test the bundled confucius_component directory.",
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--accept-license", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if args.source_root:
        os.environ["T8STAR_CONFUCIUS_ROOT"] = str(Path(args.source_root).resolve())
    else:
        os.environ.pop("T8STAR_CONFUCIUS_ROOT", None)
    data_dir = Path(args.data_dir).resolve()
    if args.accept_license:
        accept_license(data_dir, True)
    coordinator = InferenceCoordinator()
    set_gpu_coordinator(coordinator)
    app = FastAPI()
    app.include_router(build_confucius_router(ConfuciusDesktopGateway(data_dir)))
    headers = {"Host": "127.0.0.1:7860", "Origin": "http://127.0.0.1:7860"}
    waveform, sample_rate = load_audio_file(Path(args.audio).resolve())
    audio = torch.as_tensor(waveform).detach().float().cpu()
    while audio.ndim > 2:
        audio = audio[0]
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if int(sample_rate) != 16000:
        audio = torchaudio.functional.resample(audio, int(sample_rate), 16000)
    samples = audio.squeeze(0).clamp(-1, 1).numpy().astype("<f4", copy=False)
    try:
        with TestClient(app) as browser:
            created_response = browser.post(
                "/api/confucius/live/start",
                headers={**headers, "Content-Type": "application/json"},
                json={"language": "Chinese", "stream_chunk_ms": 320, "min_segment_seconds": 8},
            )
            created_response.raise_for_status()
            created = created_response.json()
            sid = created["session_id"]
            with browser.websocket_connect(
                f"/api/confucius/live/{sid}/stream",
                headers=headers,
            ) as socket:
                socket.send_text(json.dumps({"browser_token": created["browser_token"]}))
                assert socket.receive_json()["type"] == "ready"
                sent = 0
                sequence = 0
                while sent < len(samples):
                    chunk = samples[sent:sent + 640]
                    socket.send_bytes(struct.pack("<II", sequence, sent) + chunk.tobytes())
                    answer = socket.receive_json()
                    assert answer["type"] == "ack"
                    assert int(answer["ack_sample"]) == sent + len(chunk)
                    sent += len(chunk)
                    sequence += 1
                socket.send_text(json.dumps({
                    "type": "finish", "last_seq": sequence - 1, "total_samples": sent,
                }))
                final = socket.receive_json()
                assert final["type"] == "final"
                assert str(final.get("text") or "").strip()
                assert int(final.get("audio_samples_16k") or 0) == len(samples)
            cached_response = browser.get(
                f"/api/confucius/live/{sid}/result",
                headers={**headers, "X-R2T2-Session-Token": created["browser_token"]},
            )
            cached_response.raise_for_status()
            cached = cached_response.json()
            assert cached.get("text") == final.get("text")
            assert coordinator.status().get("lease_active") is False
            public_status = browser.get("/api/confucius/status", headers=headers).json()
            assert not any(key in public_status for key in ("root", "workerPython", "modelDir", "buildDir"))
            assert all("path" not in item for item in public_status.get("files", {}).values())
            print(json.dumps({
                "text": final["text"],
                "status": final.get("status"),
                "quality_status": final.get("quality_status"),
                "samples": len(samples),
                "cached_result": True,
                "coordinator": coordinator.status(),
            }, ensure_ascii=False))
        return 0
    finally:
        close_all_managers()
        set_gpu_coordinator(None)


if __name__ == "__main__":
    raise SystemExit(main())
