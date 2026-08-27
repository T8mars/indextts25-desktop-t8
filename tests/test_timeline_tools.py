from __future__ import annotations

import pytest

from dialogue_runtime import DialogueLine
from timeline_tools import apply_timeline_edits, render_timeline_html, rewrite_srt, timeline_rows


def _lines():
    return [
        DialogueLine(1, "小明", "第一句", "ZH", 0, 1000, 1.0),
        DialogueLine(2, "旁白", "second line", "EN", 1200, 2200, 1.0),
    ]


def test_editable_timeline_rows_are_validated_and_applied():
    rows = timeline_rows(_lines())
    rows[0][3:5] = [100, 1500]
    rows[0][6] = "修改后的第一句"
    rows[0][7] = "vector:0,0.8,0,0,0,0,0,0"
    edited = apply_timeline_edits(_lines(), rows)
    assert edited[0].start_ms == 100
    assert edited[0].end_ms == 1500
    assert edited[0].text == "修改后的第一句"
    assert edited[0].emotion_mode == "vector"
    assert edited[0].emotion_vector[1] == pytest.approx(0.8)

    rows[1][4] = ""
    with pytest.raises(ValueError, match="同时填写"):
        apply_timeline_edits(_lines(), rows)


def test_srt_rewrite_uses_actual_timing_and_only_passed_asr_text():
    reports = [
        {
            "index": 1,
            "actual_duration_ms": 800,
            "timeline": {"actual_start_ms": 50, "actual_end_ms": 850},
            "asr": {"recognized_text": "识别后的第一句", "passed": True},
        },
        {
            "index": 2,
            "actual_duration_ms": 900,
            "timeline": {"actual_start_ms": 900, "actual_end_ms": 1800},
            "asr": {"recognized_text": "wrong", "passed": False},
        },
    ]
    content, report = rewrite_srt(_lines(), reports, timing_mode="actual", text_mode="asr_passed")
    assert "00:00:00,050 --> 00:00:00,850" in content
    assert "[小明] 识别后的第一句" in content
    assert "[旁白] second line" in content
    assert report["lines"][0]["source"] == "asr"


def test_srt_rewrite_preserves_per_line_emotion_override():
    line = DialogueLine(
        1,
        "旁白",
        "惊讶台词",
        "ZH",
        0,
        1000,
        1.0,
        emotion_mode="text",
        emotion_text="惊讶、激动",
    )
    content, _report = rewrite_srt([line], [], text_mode="original")
    assert "[旁白|emotion=text:惊讶、激动] 惊讶台词" in content


def test_timeline_visual_escapes_user_text_and_includes_asr_score():
    lines = [DialogueLine(1, "<角色>", "<script>alert(1)</script>", "ZH", 0, 1000)]
    rendered = render_timeline_html(
        lines,
        [{"index": 1, "asr": {"similarity": 0.95}, "timeline": {"actual_start_ms": 0, "actual_end_ms": 900}}],
    )
    assert "<script>" not in rendered
    assert "ASR 95%" in rendered
    assert "总时长 0.90s" in rendered


def test_timeline_visual_includes_word_timestamp_markers():
    rendered = render_timeline_html(
        _lines()[:1],
        [
            {
                "index": 1,
                "actual_duration_ms": 1000,
                "asr": {
                    "word_timestamps": [
                        {"word": "第一", "start": 0.1, "end": 0.35},
                        {"word": "句", "start": 0.5, "end": 0.8},
                    ]
                },
            }
        ],
    )
    assert rendered.count("t8-timeline-word") == 2
    assert "第一" in rendered
    assert "left:10.000%" in rendered
