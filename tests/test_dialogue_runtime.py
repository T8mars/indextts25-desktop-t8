import json

import pytest
import torch

from dialogue_runtime import (
    compose_timeline,
    fit_duration_factor,
    format_batch_script,
    missing_roles,
    parse_batch_script,
    parse_srt,
    parse_timestamp,
)


def test_srt_parses_bom_crlf_multiline_and_roles():
    content = (
        "\ufeff1\r\n00:00:01,000 --> 00:00:02,500\r\n[小明] 第一行\r\n第二行\r\n\r\n"
        "2\r\n00:00:03.000 --> 00:00:04.000\r\n小红：你好\r\n"
    )
    lines = parse_srt(content)
    assert [(line.role, line.start_ms, line.end_ms) for line in lines] == [
        ("小明", 1000, 2500),
        ("小红", 3000, 4000),
    ]
    assert lines[0].text == "第一行\n第二行"


def test_srt_rejects_invalid_ranges():
    with pytest.raises(ValueError, match="结束时间"):
        parse_srt("1\n00:00:02,000 --> 00:00:01,000\n错了")
    with pytest.raises(ValueError, match="无效"):
        parse_timestamp("00:72:00,000")


def test_batch_text_and_json_formats():
    lines = parse_batch_script("# comment\n小明|你好|ZH|0.9\n小红：Hello")
    assert [(line.role, line.language, line.duration_factor) for line in lines] == [
        ("小明", "ZH", 0.9),
        ("小红", "ZH", 1.0),
    ]
    payload = json.dumps([{"role": "A", "text": "Hola", "language": "ES"}])
    assert parse_batch_script(payload)[0].language == "ES"
    with pytest.raises(ValueError, match="第 2 条.*对象"):
        parse_batch_script(json.dumps([{"role": "A", "text": "ok"}, "bad"]))
    with pytest.raises(ValueError, match="不能为空"):
        parse_batch_script("[]")


def test_same_role_supports_per_line_text_and_vector_emotions():
    lines = parse_batch_script(
        "旁白|先平静介绍。|ZH|1.0|text:平静、从容\n"
        "旁白|随后突然生气。|ZH|1.0|vector:0,0.8,0,0,0,0,0,0\n"
        "旁白|恢复默认。|ZH|1.0"
    )
    assert [line.role for line in lines] == ["旁白", "旁白", "旁白"]
    assert lines[0].emotion_mode == "text"
    assert lines[0].emotion_text == "平静、从容"
    assert lines[1].emotion_mode == "vector"
    assert lines[1].emotion_vector == (0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert lines[2].emotion_mode == "inherit"

    json_line = parse_batch_script(
        json.dumps(
            [
                {
                    "role": "旁白",
                    "text": "低强度悲伤。",
                    "emotion": {"mode": "text", "text": "悲伤", "strength": 0.65},
                }
            ],
            ensure_ascii=False,
        )
    )[0]
    assert json_line.emotion_text == "悲伤"
    assert json_line.emotion_strength == pytest.approx(0.65)


def test_human_emotion_options_and_batch_escaping_round_trip():
    lines = parse_batch_script(
        "旁白|台词含\\|和换行\\n下一行|ZH|1.0|text:平静；坚定;强度=0.75\n"
        "旁白|惊讶|ZH|1.0|vector:0,0,0,0,0,0,0.8,0;strength=0.85;random=true"
    )
    assert lines[0].text == "台词含|和换行\n下一行"
    assert lines[0].emotion_text == "平静；坚定"
    assert lines[0].emotion_strength == pytest.approx(0.75)
    assert lines[1].emotion_use_random is True
    assert parse_batch_script(format_batch_script(lines)) == lines


def test_srt_supports_per_line_emotion_tag():
    line = parse_srt(
        "1\n00:00:00,000 --> 00:00:02,000\n"
        "[旁白|emotion=text:惊讶、激动] 怎么会这样？"
    )[0]
    assert line.role == "旁白"
    assert line.emotion_mode == "text"
    assert line.emotion_text == "惊讶、激动"


def test_batch_duration_factor_error_explains_that_it_is_not_seconds():
    with pytest.raises(ValueError, match="不是秒数，也不限制台词长度"):
        parse_batch_script("旁白|这是一句可以超过两秒的台词|ZH|3")


def test_missing_roles_and_duration_fit():
    lines = parse_batch_script("A|one|EN\nB|two|EN")
    assert missing_roles(lines, ["A"]) == ["B"]
    assert fit_duration_factor(1.0, 2000, 1000) == 0.5
    assert fit_duration_factor(1.0, 500, 2000) == 2.0


def test_timeline_shift_and_overlay():
    lines = parse_srt(
        "1\n00:00:00,000 --> 00:00:00,500\n[A] one\n\n"
        "2\n00:00:00,250 --> 00:00:00,750\n[B] two"
    )
    clips = [torch.ones(1, 1, 500), torch.full((1, 1, 500), 0.5)]
    shifted, shift_report = compose_timeline(clips, lines, 1000, "shift")
    overlaid, overlay_report = compose_timeline(clips, lines, 1000, "overlay")
    assert shifted.shape[-1] == 1000
    assert shift_report[1].actual_start_ms == 500
    assert overlaid.shape[-1] == 750
    assert overlay_report[1].actual_start_ms == 250
    assert float(overlaid[..., 300].item()) == pytest.approx(0.75)
