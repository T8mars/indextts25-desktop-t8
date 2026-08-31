from __future__ import annotations

import pytest

from dialogue_runtime import DialogueLine
from timeline_tools import (
    apply_timeline_drag_payload,
    apply_timeline_edits,
    editable_timeline_document,
    editable_timeline_script,
    move_timeline_row,
    parse_editable_timeline_document,
    render_timeline_html,
    rewrite_srt,
    timeline_rows,
)


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


def test_untimed_batch_rows_survive_repeated_gradio_round_trip():
    original = [DialogueLine(1, "旁白", "重复生成", "ZH")]
    rows = timeline_rows(original)
    assert rows[0][3:5] == ["", ""]

    # Older/current Gradio Dataframe versions may turn both empty numeric
    # cells into zero after the first queued generation.
    rows[0][3:5] = [0, 0]
    edited = apply_timeline_edits(original, rows)
    assert edited[0].start_ms is None
    assert edited[0].end_ms is None


def test_authored_zero_length_timeline_is_still_rejected():
    rows = timeline_rows(_lines())
    rows[0][3:5] = [0, 0]
    with pytest.raises(ValueError, match="结束时间"):
        apply_timeline_edits(_lines(), rows)


def test_move_timeline_row_moves_content_but_preserves_srt_slots():
    rows = timeline_rows(_lines())
    moved, selected, changed = move_timeline_row(rows, 2, -1)

    assert changed is True
    assert selected == 1
    assert [row[0] for row in moved] == [1, 2]
    assert [row[6] for row in moved] == ["second line", "第一句"]
    assert [row[1] for row in moved] == ["旁白", "小明"]
    assert [(row[3], row[4]) for row in moved] == [(0, 1000), (1200, 2200)]

    unchanged, selected, changed = move_timeline_row(moved, 1, -1)
    assert changed is False
    assert selected == 1
    assert unchanged == moved


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
    assert "成品总时长 0.90s" in rendered
    assert 'data-index="1"' in rendered
    assert 'data-start-ms="0"' in rendered
    assert "t8-timeline-handle-start" in rendered


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
    assert 'data-snap-ms="100"' in rendered


def test_drag_payload_updates_only_selected_line_and_validates_bounds():
    edited, payload = apply_timeline_drag_payload(
        _lines(),
        {
            "index": 2,
            "start_ms": 1500,
            "end_ms": 2800,
            "mode": "move",
            "snapped_to_ms": 1500,
        },
    )
    assert edited[0] == _lines()[0]
    assert edited[1].start_ms == 1500
    assert edited[1].end_ms == 2800
    assert payload["snapped_to_ms"] == 1500

    with pytest.raises(ValueError, match="结束时间"):
        apply_timeline_drag_payload(
            _lines(), {"index": 1, "start_ms": 1000, "end_ms": 900, "mode": "resize_end"}
        )


@pytest.mark.parametrize("file_format", ["json", "csv"])
def test_editable_timeline_document_round_trip(file_format):
    source = [
        DialogueLine(
            1,
            "旁白",
            "第一句|包含分隔符\n第二行",
            "ZH",
            100,
            900,
            1.1,
            emotion_mode="text",
            emotion_text="平静、坚定",
            emotion_strength=0.75,
        ),
        DialogueLine(
            2,
            "A",
            "surprised",
            "EN",
            1000,
            1800,
            0.9,
            emotion_mode="vector",
            emotion_vector=(0, 0, 0, 0, 0, 0, 0.8, 0),
            emotion_strength=0.85,
            emotion_use_random=True,
        ),
    ]
    content = editable_timeline_document(source, "srt", file_format)
    script_type, restored = parse_editable_timeline_document(
        content, f".{file_format}", "旁白", "ZH"
    )
    assert script_type == "srt"
    assert restored == source
    assert "喜" in content if file_format == "json" else "emotion" in content

    restored_type, script = editable_timeline_script(restored, script_type)
    assert restored_type == "srt"
    assert "emotion=text:平静、坚定;strength=0.75" in script


def test_editable_timeline_import_accepts_legacy_lines_and_rejects_bad_timing():
    legacy = '{"lines":[{"role":"旁白","text":"旧工程","language":"ZH","duration_factor":1.0}]}'
    script_type, lines = parse_editable_timeline_document(legacy, ".json")
    assert script_type == "batch"
    assert lines[0].text == "旧工程"
    assert editable_timeline_script(lines, script_type)[1].startswith("旁白|旧工程|ZH|1|")

    bad = "role,language,start_ms,end_ms,duration_factor,text,emotion\n旁白,ZH,0,0,1,错误,"
    with pytest.raises(ValueError, match="结束时间"):
        parse_editable_timeline_document(bad, ".csv")
