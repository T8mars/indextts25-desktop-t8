from __future__ import annotations

import pytest
import torch

from desktop_generation_controls import (
    assess_long_text_result,
    apply_duration_policy,
    allocate_native_chunk_durations,
    build_desktop_plan,
    concatenate_with_pauses,
    normalize_preflight_text,
    postprocess_waveform,
    preflight_plan_rows,
    run_with_long_text_guard,
    split_speech_chunks,
)


class FakeTokenizer:
    def encode(self, text, allowed_special="all"):
        return list(text)


class FakeTTS:
    tokenizer = FakeTokenizer()

    @staticmethod
    def split_text_by_tokens(text, limit, prefix):
        budget = max(1, limit - len(prefix))
        return [text[index : index + budget] for index in range(0, len(text), budget)]


def test_desktop_auto_segmentation_and_pause_preview():
    plan = build_desktop_plan(
        FakeTTS(),
        "Hello there.<pause=500ms>Second line.",
        "EN",
        "auto",
        120,
        "off",
        0,
        0,
        0,
    )
    assert plan.max_tokens == 60
    assert len(plan.chunks) == 2
    assert plan.chunks[0].pause_after_ms == 500
    assert plan.gpt_accel_risk is False
    assert plan.to_dict()["gpt_accel_cache_fix"] is True


def test_long_text_preflight_normalizes_and_marks_risky_segments():
    import re

    class Processor:
        clean_pattern = re.compile(r"[。]")
        char_rep_map = {"。": "."}

        @staticmethod
        def normalize(text):
            return text.replace("1939", "一九三九")

    fake = FakeTTS()
    fake.text_process = Processor()
    normalized = normalize_preflight_text(fake, "1939年。", "ZH", True)
    assert normalized == "一九三九年."

    plan = build_desktop_plan(fake, "甲" * 18, "ZH", "custom", 20, "off", 0, 0, 0)
    rows = preflight_plan_rows(plan)
    assert rows[0][3] > 0
    assert rows[0][4].startswith("高")
    assert rows[0][-1] == "甲" * 13


def test_desktop_pause_presets_keep_annotations_atomic():
    chunks = split_speech_chunks(
        "<行|XING2>程，继续。",
        "custom",
        100,
        300,
        600,
    )
    assert chunks[0].text == "<行|XING2>程，"
    assert chunks[0].pause_after_ms == 100


def test_desktop_duration_and_concatenation_are_sample_exact():
    joined = concatenate_with_pauses(
        [torch.ones(1, 10), torch.ones(1, 10)], 1000, [100, 0]
    )
    assert joined.shape[-1] == 120
    exact, report = apply_duration_policy(joined, 1000, 0.1, "exact")
    assert exact.shape[-1] == 100
    assert report["action"] == "trimmed"


def test_desktop_normalize_postprocess():
    result, report = postprocess_waveform(torch.ones(1, 100) * 0.25, 22050, "normalize", 1.0)
    assert report["preset"] == "normalize"
    assert 0.88 < float(result.abs().max()) < 0.9
    assert result[..., -1].item() == 0.0


def test_desktop_leading_pause_and_decimal_boundaries_are_preserved():
    chunks = split_speech_chunks(
        "<pause=250ms>Version 3.14 costs 1,000.50 dollars.",
        "custom",
        120,
        300,
        600,
    )
    assert len(chunks) == 1
    assert chunks[0].pause_before_ms == 250
    assert chunks[0].pause_after_ms == 300
    assert chunks[0].text == "Version 3.14 costs 1,000.50 dollars."

    joined = concatenate_with_pauses(
        [torch.ones(1, 10)],
        1000,
        [0],
        leading_pause_ms=chunks[0].pause_before_ms,
    )
    assert joined.shape[-1] == 260
    assert torch.count_nonzero(joined[..., :250]) == 0


def test_native_duration_is_distributed_after_external_pauses():
    plan = build_desktop_plan(
        FakeTTS(),
        "one<pause=500ms>three three three",
        "EN",
        "auto",
        120,
        "off",
        0,
        0,
        0,
    )
    durations = allocate_native_chunk_durations(plan, 4.5)
    assert len(durations) == 2
    assert sum(durations) == pytest.approx(4.0)
    assert durations[1] > durations[0]


def test_long_english_guard_retries_with_smaller_segments_after_max_mel_warning():
    calls = []

    def generate(limit):
        import warnings

        calls.append(limit)
        if len(calls) == 1:
            warnings.warn(
                "generation stopped due to exceeding max_mel_tokens",
                RuntimeWarning,
            )
            return {"duration": 1.0}
        return {"duration": 18.0}

    result, report = run_with_long_text_guard(
        generate,
        lambda value: value["duration"],
        text=(
            "This deliberately long English paragraph contains enough words to verify "
            "that a collapsed decode is detected and regenerated with safer segments "
            "instead of silently returning an incomplete final sentence to the user."
        ),
        language="EN",
        token_count=58,
        max_tokens=60,
    )
    assert result["duration"] == 18.0
    assert calls == [60, 40]
    assert report["retried"] is True
    assert report["recovered"] is True
    assert report["first_reasons"] == ["max_mel_tokens_reached", "suspiciously_short_for_latin_text"]


def test_long_spanish_guard_detects_implausibly_short_audio_but_ignores_chinese():
    spanish = " ".join(["Esta frase contiene palabras para comprobar una salida demasiado corta"] * 4)
    reasons = assess_long_text_result(spanish, "ES", 50, 0.8)
    assert "suspiciously_short_for_latin_text" in reasons
    assert assess_long_text_result("这是一个很长的中文测试。" * 20, "ZH", 100, 0.8) == []
