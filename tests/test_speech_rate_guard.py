from indextts.speech_rate_guard import (
    assess_segment_speech_rates,
    retry_candidate_improves_rate,
    speech_unit_count,
)


def test_counts_language_aware_speech_units_without_tags():
    assert speech_unit_count("你好，世界！<pause=1s>", "ZH") == 4
    assert speech_unit_count("Hello, carefully-written world.", "EN") == 3
    assert speech_unit_count("今日は晴れです。", "JA") == 7


def test_flags_only_strong_cross_segment_slowdown_after_stable_baseline():
    reports = assess_segment_speech_rates(
        [
            {"text": "one two three four five six seven eight", "language": "EN", "duration_seconds": 3.2},
            {"text": "nine ten eleven twelve thirteen fourteen", "language": "EN", "duration_seconds": 2.5},
            {"text": "this tail suddenly becomes extremely slow now", "language": "EN", "duration_seconds": 10.5},
        ]
    )
    assert not reports[0]["suspect"]
    assert not reports[1]["suspect"]
    assert reports[2]["suspect"]
    assert reports[2]["reason"] == "cross_segment_slowdown"


def test_does_not_flatten_moderately_slow_or_too_short_segments():
    reports = assess_segment_speech_rates(
        [
            {"text": "one two three four five six", "language": "EN", "duration_seconds": 2.5},
            {"text": "seven eight nine ten eleven twelve", "language": "EN", "duration_seconds": 2.6},
            {"text": "a short aside", "language": "EN", "duration_seconds": 4.0},
            {"text": "this emotional sentence remains deliberately slower", "language": "EN", "duration_seconds": 5.0},
        ]
    )
    assert not any(item["suspect"] for item in reports)


def test_accepts_only_materially_better_retry_rate():
    assert retry_candidate_improves_rate(0.8, 2.1, 2.4)
    assert not retry_candidate_improves_rate(0.8, 0.9, 2.4)
    assert not retry_candidate_improves_rate(0.8, 5.0, 2.4)
