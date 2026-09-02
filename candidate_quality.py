from __future__ import annotations

import math

import torch


def technical_audio_review(waveform: torch.Tensor, sample_rate: int) -> dict:
    samples = torch.as_tensor(waveform).detach().float().cpu().reshape(-1)
    sample_rate = max(1, int(sample_rate))
    if samples.numel() == 0:
        return {
            "valid": False,
            "score": 0.0,
            "duration_seconds": 0.0,
            "clipped_ratio": 1.0,
            "silence_ratio": 1.0,
            "dc_offset": 1.0,
            "peak": 0.0,
        }
    finite = torch.isfinite(samples)
    finite_ratio = float(finite.float().mean())
    clean = torch.where(finite, samples, torch.zeros_like(samples)).clamp(-4.0, 4.0)
    absolute = clean.abs()
    clipped_ratio = float((absolute >= 0.995).float().mean())
    silence_ratio = float((absolute <= 0.002).float().mean())
    dc_offset = abs(float(clean.mean()))
    peak = float(absolute.max())
    duration_seconds = float(clean.numel() / sample_rate)
    score = 1.0
    score -= min(0.65, clipped_ratio * 20.0)
    score -= min(0.35, max(0.0, silence_ratio - 0.72) * 1.25)
    score -= min(0.25, dc_offset * 5.0)
    score -= (1.0 - finite_ratio) * 0.8
    if peak < 0.01:
        score -= 0.5
    if duration_seconds < 0.15:
        score -= 0.5
    return {
        "valid": bool(finite_ratio == 1.0 and duration_seconds >= 0.15 and peak >= 0.01),
        "score": round(max(0.0, min(1.0, score)), 6),
        "duration_seconds": round(duration_seconds, 6),
        "clipped_ratio": round(clipped_ratio, 8),
        "silence_ratio": round(silence_ratio, 8),
        "dc_offset": round(dc_offset, 8),
        "peak": round(peak, 6),
    }


def combined_candidate_score(technical_score: float, asr_similarity=None) -> float:
    technical = max(0.0, min(1.0, float(technical_score)))
    if asr_similarity is None or not math.isfinite(float(asr_similarity)):
        return technical
    similarity = max(0.0, min(1.0, float(asr_similarity)))
    return round(similarity * 0.85 + technical * 0.15, 6)


def select_best_candidate(reviews: list[dict]) -> int:
    if not reviews:
        raise ValueError("候选列表不能为空。")
    return max(
        range(len(reviews)),
        key=lambda index: (
            bool(reviews[index].get("passed")),
            bool(
                reviews[index].get(
                    "tail_passed",
                    reviews[index].get("passed", False),
                )
            ),
            float(
                reviews[index].get(
                    "tail_similarity",
                    reviews[index].get("similarity") or 0.0,
                )
            ),
            float(reviews[index].get("combined_score", 0.0)),
            float(reviews[index].get("technical", {}).get("score", 0.0)),
            -index,
        ),
    )
