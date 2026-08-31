from __future__ import annotations

from pathlib import Path

from desktop_candidate_workspace import CandidateWorkspace


def test_candidate_workspace_labels_rates_and_favorites(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    data_dir.mkdir()
    output_dir.mkdir()
    first = data_dir / "quality_candidates" / "run" / "candidate_01.wav"
    second = data_dir / "quality_candidates" / "run" / "candidate_02.wav"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"RIFF-one")
    second.write_bytes(b"RIFF-two")
    workspace = CandidateWorkspace(data_dir, output_dir)

    choices = workspace.choices([str(first), {"path": str(second)}])
    assert choices[0][0].startswith("候选 A")
    assert choices[1][0].startswith("候选 B")
    review = workspace.save_review(first, 5, "音色最自然", favorite=True)
    assert review["rating"] == 5
    assert workspace.review_for(first)["note"] == "音色最自然"
    assert workspace.favorite_dir in Path(review["favorite_file"]).parents
