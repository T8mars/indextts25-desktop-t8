import json
from pathlib import Path

import torch

from segment_rate_workspace import (
    compose_segment_workspace,
    load_segment_workspace,
    render_segment_rate_html,
    segment_choices,
    segment_rate_rows,
    select_replacement_audio,
    write_segment_workspace,
)


def test_rate_rows_and_html_mark_suspect_and_accepted_retry():
    reports = [
        {
            "index": 3,
            "speech_block": 1,
            "language": "ZH",
            "duration_seconds": 8.0,
            "units_per_second": 1.0,
            "baseline_units_per_second": 3.0,
            "rate_ratio": 0.333,
            "eligible": True,
            "suspect": True,
            "retried": True,
            "retry_units_per_second": 2.8,
            "accepted": True,
            "text": "第三段台词",
        }
    ]
    rows = segment_rate_rows(reports)
    assert rows[0][7] == "异常偏慢"
    assert rows[0][9] == "重试结果"
    rendered = render_segment_rate_html(reports)
    assert "已采用重试" in rendered
    assert "第三段台词" in rendered


def test_workspace_persists_artifacts_and_rebuilds_selected_segments(tmp_path: Path):
    saved: dict[str, torch.Tensor] = {}

    def save_audio(path, waveform, _sample_rate):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"wav")
        saved[path.name] = torch.as_tensor(waveform).clone()

    records = [
        [
            {
                "index": 1,
                "speech_block": 1,
                "language": "ZH",
                "text": "第一段",
                "sample_rate": 1000,
                "waveform": torch.ones(1, 100),
            },
            {
                "index": 2,
                "speech_block": 1,
                "language": "ZH",
                "text": "第二段",
                "sample_rate": 1000,
                "original_waveform": torch.ones(1, 100) * 0.2,
                "retry_waveform": torch.ones(1, 50) * 0.8,
                "waveform": torch.ones(1, 50) * 0.8,
                "selected_source": "auto_retry",
            },
        ]
    ]
    manifest_path = write_segment_workspace(
        tmp_path / "workspace",
        block_records=records,
        reports=[{"position": 1, "index": 2, "suspect": True, "accepted": True}],
        block_pauses=[
            {"index": 1, "segment_indices": [1, 2], "pause_before_ms": 20, "pause_after_ms": 30}
        ],
        settings={"segment_silence_ms": 10},
        output_path=tmp_path / "final.wav",
        save_audio=save_audio,
    )
    manifest = load_segment_workspace(manifest_path)
    assert manifest["segments"][1]["selected_source"] == "auto_retry"
    assert segment_choices(manifest)[1][1] == "2"

    def load_audio(path):
        return saved[path.name], 1000

    waveform, rate = compose_segment_workspace(manifest, load_audio)
    assert rate == 1000
    assert waveform.shape[-1] == 20 + 100 + 10 + 50 + 30

    replacement = manifest_path.parent / "segment_002_manual.wav"
    replacement.write_bytes(b"wav")
    saved[replacement.name] = torch.ones(1, 25)
    updated = select_replacement_audio(
        manifest_path, 2, replacement, source="manual_retry"
    )
    assert updated["segments"][1]["redo_count"] == 1
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["segments"][1][
        "selected_source"
    ] == "manual_retry"


def test_workspace_rejects_artifacts_outside_directory(tmp_path: Path):
    manifest_path = tmp_path / "workspace" / "segment-workspace.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workspace": str(manifest_path.parent),
                "segments": [
                    {
                        "index": 1,
                        "original_audio": "../outside.wav",
                        "retry_audio": "",
                        "selected_audio": "../outside.wav",
                    }
                ],
                "blocks": [],
            }
        ),
        encoding="utf-8",
    )
    try:
        load_segment_workspace(manifest_path)
    except ValueError as exc:
        assert "inside" in str(exc)
    else:
        raise AssertionError("path traversal should be rejected")
