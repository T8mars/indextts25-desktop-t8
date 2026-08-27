from runtime_benchmark import benchmark_summary, recommend_benchmark_mode


def test_recommendation_prefers_simpler_mode_within_five_percent():
    results = [
        {"status": "ok", "requested_mode": "off", "effective_mode": "off", "rtf": 0.82},
        {"status": "ok", "requested_mode": "gpt_accel", "effective_mode": "gpt_accel", "rtf": 0.79},
        {"status": "ok", "requested_mode": "deepspeed", "effective_mode": "deepspeed", "rtf": 0.78},
    ]
    recommendation = recommend_benchmark_mode(results)
    assert recommendation["effective_mode"] == "gpt_accel"
    assert recommendation["fastest_rtf"] == 0.78


def test_recommendation_ignores_failed_and_skipped_modes():
    results = [
        {"status": "error", "requested_mode": "gpt_accel", "effective_mode": "gpt_accel", "rtf": 0.1},
        {"status": "skipped", "requested_mode": "deepspeed", "effective_mode": "off"},
        {"status": "ok", "requested_mode": "off", "effective_mode": "off", "rtf": 1.2},
    ]
    recommendation = recommend_benchmark_mode(results)
    assert recommendation["effective_mode"] == "off"
    report = {"results": results, "recommendation": recommendation}
    assert "成功 1" in benchmark_summary(report)


def test_recommendation_falls_back_when_nothing_succeeds():
    recommendation = recommend_benchmark_mode([{"status": "error"}])
    assert recommendation["mode"] == "off"
    assert recommendation["rtf"] is None
