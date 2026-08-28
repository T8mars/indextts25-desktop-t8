from __future__ import annotations

from context_emotion import (
    build_context_prompt,
    normalize_emotion_scores,
    suggest_context_emotions,
)
from dialogue_runtime import DialogueLine


def _lines():
    return [
        DialogueLine(1, "甲", "我们终于成功了。"),
        DialogueLine(2, "乙", "等等，结果好像不对。"),
        DialogueLine(3, "甲", "什么？你再说一遍！"),
    ]


def test_context_prompt_marks_target_and_surrounding_roles() -> None:
    prompt, indexes = build_context_prompt(_lines(), 1, 1)
    assert indexes == [1, 2, 3]
    assert prompt.endswith("【只分析这一句】#2 乙：等等，结果好像不对。")
    assert "【此前对话】" in prompt
    assert "【随后对话】" in prompt
    assert "输出 IndexTTS 八维" not in prompt


def test_normalize_scores_supports_chinese_keys_and_caps_sum() -> None:
    vector, raw = normalize_emotion_scores({"愤怒": 1.2, "惊讶": 0.4})
    assert len(vector) == 8
    assert round(sum(vector), 6) == 0.8
    assert raw["angry"] == 1.2
    assert raw["surprised"] == 0.4


def test_suggestions_preserve_manual_overrides_until_explicitly_replaced() -> None:
    source = _lines()
    source[0] = DialogueLine(
        1,
        "甲",
        source[0].text,
        emotion_mode="text",
        emotion_text="克制地高兴",
    )
    calls = []

    def classifier(prompt):
        calls.append(prompt)
        return {"愤怒": 0.9, "惊讶": 0.3}

    updated, report = suggest_context_emotions(source, classifier, context_window=1)
    assert updated[0].emotion_mode == "text"
    assert updated[1].emotion_mode == "vector"
    assert updated[2].emotion_mode == "vector"
    assert report["preserved_count"] == 1
    assert report["classified_count"] == 2
    assert report["requires_user_confirmation"] is True
    assert report["started_synthesis"] is False
    assert len(calls) == 4
    assert calls[0] == source[1].text
    assert "【只分析这一句】" in calls[1]
