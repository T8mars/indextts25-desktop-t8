from __future__ import annotations

from pathlib import Path

import pytest

from indextts.pronunciation import (
    PronunciationEntry,
    PronunciationValidationError,
    entries_from_rows,
    load_dictionary,
    make_annotation,
    process_pronunciation_text,
    save_dictionary,
    validate_reading,
)
from indextts.infer_v2_5 import apply_pronunciation_annotations


def test_manual_annotations_for_all_official_languages_are_preserved():
    cases = (
        ("他在银<行|XING2>里。", "ZH"),
        ("a <minute|M IH1 . N AH0 T> later", "EN"),
        ("彼は<上手|じょうず>だ。", "JA"),
    )
    for text, language in cases:
        result = process_pronunciation_text(text, language, strict=True)
        assert result.text == text
        assert result.errors == ()


def test_dictionary_uses_longest_match_and_never_rewrites_manual_annotation():
    entries = [
        PronunciationEntry("行", "XING2", "ZH"),
        PronunciationEntry("银行", "YIN2 HANG2", "ZH"),
    ]
    result = process_pronunciation_text(
        "银行可以，银<行|HANG2>也可以。",
        "ZH",
        entries,
        strict=True,
    )
    assert result.text == "<银行|YIN2 HANG2>可以，银<行|HANG2>也可以。"
    assert len(result.replacements) == 1


def test_dictionary_output_is_consumed_by_the_pinned_25_inference_core():
    result = process_pronunciation_text(
        "银行和Bilibili",
        "ZH",
        [
            PronunciationEntry("银行", "YIN2 HANG2", "ZH"),
            PronunciationEntry("Bilibili", "B IY1 . L IY1 . B IY1 . L IY1", "EN", True, False),
        ],
        strict=True,
    )
    converted = apply_pronunciation_annotations(result.text)
    assert "<|SPECIAL_TOKEN_2|>YIN2 HANG2<|SPECIAL_TOKEN_2|>" in converted
    assert "<|SPECIAL_TOKEN_1|>B IY1 . L IY1 . B IY1 . L IY1<|SPECIAL_TOKEN_1|>" in converted


def test_dictionary_normalizes_readings_and_reports_invalid_items():
    rows = [["重庆", "ZH", "chong2 qing4", True, True], ["错误", "ZH", "BAD9", True, True]]
    result = process_pronunciation_text("重庆和错误", "ZH", entries_from_rows(rows))
    assert result.text == "<重庆|CHONG2 QING4>和错误"
    assert result.errors and "BAD9" in result.errors[0]
    with pytest.raises(PronunciationValidationError):
        process_pronunciation_text("重庆和错误", "ZH", entries_from_rows(rows), strict=True)


def test_official_documented_pou3_example_is_structurally_valid():
    normalized, warnings, errors = validate_reading("POU3", "ZH")
    assert normalized == "POU3"
    assert warnings == ()
    assert errors == ()


def test_exact_vocab_mismatch_is_a_warning_not_a_false_rejection(tmp_path: Path):
    vocab = tmp_path / "pinyin.vocab"
    vocab.write_text("XING2\n", encoding="utf-8")
    _normalized, warnings, errors = validate_reading("POU3", "ZH", pinyin_vocab_path=vocab)
    assert warnings
    assert errors == ()


def test_dictionary_yaml_round_trip_and_atomic_save(tmp_path: Path):
    path = tmp_path / "pronunciation.yaml"
    entries = [
        PronunciationEntry("银行", "YIN2 HANG2", "ZH"),
        PronunciationEntry("Bilibili", "B IY1 . L IY1 . B IY1 . L IY1", "EN", True, False),
    ]
    assert save_dictionary(path, entries) == path
    assert load_dictionary(path) == entries
    assert not path.with_suffix(".yaml.tmp").exists()


def test_make_annotation_rejects_bad_reading():
    annotation, warnings = make_annotation("行", "xing2", "ZH")
    assert annotation == "<行|XING2>"
    assert warnings == ()
    with pytest.raises(PronunciationValidationError):
        make_annotation("行", "NOT-PINYIN", "ZH")


def test_malformed_annotation_is_reported():
    result = process_pronunciation_text("这是<行|XING2", "ZH")
    assert result.errors


def test_issue_792_guides_contextual_polyphones_to_whole_word_annotation():
    risky = process_pronunciation_text(
        "小明<要|YAO4>求这个题的答案是多少，该做什么呢？", "ZH", strict=True
    )
    assert any("<要求|YAO4 QIU2>" in item for item in risky.warnings)

    robust = process_pronunciation_text(
        "小明<要求|YAO4 QIU2>这个题的答案是多少，该做什么呢？", "ZH", strict=True
    )
    assert robust.warnings == ()
    converted = apply_pronunciation_annotations(robust.text)
    assert "<|SPECIAL_TOKEN_2|>YAO4 QIU2<|SPECIAL_TOKEN_2|>" in converted


def test_chinese_annotation_warns_when_syllable_count_does_not_match_word():
    result = process_pronunciation_text("请到<银行|HANG2>办理。", "ZH", strict=True)
    assert any("2 个汉字" in item and "1 个拼音音节" in item for item in result.warnings)
