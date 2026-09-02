from __future__ import annotations

import torch

from candidate_quality import combined_candidate_score, select_best_candidate, technical_audio_review


def test_clipped_audio_scores_lower() -> None:
    clean = technical_audio_review(torch.sin(torch.linspace(0, 20, 1000)) * 0.4, 1000)
    clipped = technical_audio_review(torch.ones(1000), 1000)
    assert clean["score"] > clipped["score"]


def test_asr_dominates_combined_score() -> None:
    assert combined_candidate_score(0.4, 0.95) > combined_candidate_score(1.0, 0.7)


def test_first_candidate_wins_exact_tie() -> None:
    assert select_best_candidate([
        {"combined_score": 0.8, "technical": {"score": 0.9}},
        {"combined_score": 0.8, "technical": {"score": 0.9}},
    ]) == 0


def test_complete_tail_beats_higher_whole_sentence_score() -> None:
    assert select_best_candidate([
        {
            "passed": False,
            "tail_passed": False,
            "tail_similarity": 0.875,
            "similarity": 0.96,
            "combined_score": 0.95,
            "technical": {"score": 0.95},
        },
        {
            "passed": True,
            "tail_passed": True,
            "tail_similarity": 1.0,
            "similarity": 0.9,
            "combined_score": 0.89,
            "technical": {"score": 0.9},
        },
    ]) == 1
