from __future__ import annotations

from types import SimpleNamespace

from indextts.infer_v2_5 import IndexTTS2
from indextts.utils.text_segmentation import (
    CJK_SEGMENT_LANGUAGES,
    language_aware_token_budget,
    language_from_prefix,
)


def test_default_non_cjk_budget_is_capped_by_speech_density():
    assert language_aware_token_budget(120, 7, language="EN") == 81
    assert language_aware_token_budget(300, 7, language="es") == 81
    assert language_aware_token_budget(120, 7, language="AR") == 81


def test_existing_conservative_limit_is_not_scaled_twice():
    assert language_aware_token_budget(60, 7, language="EN") == 53
    assert language_aware_token_budget(80, 7, language="ES") == 73


def test_cjk_and_unknown_languages_keep_requested_budget():
    for language in CJK_SEGMENT_LANGUAGES:
        assert language_aware_token_budget(120, 7, language=language) == 113
    assert language_aware_token_budget(120, 7, language="") == 113


def test_capacity_and_language_prefix_are_respected():
    assert language_from_prefix("<|EN|> ") == "en"
    assert language_from_prefix("plain text") == ""
    assert language_aware_token_budget(300, 7, language="EN", capacity=80) == 51


def test_inference_splitter_applies_cap_and_keeps_conservative_limit():
    class CharacterTokenizer:
        @staticmethod
        def encode(text, allowed_special="all"):
            return list(text)

    splitter = IndexTTS2.__new__(IndexTTS2)
    splitter.tokenizer = CharacterTokenizer()
    splitter.gpt = SimpleNamespace(
        text_pos_embedding=SimpleNamespace(
            emb=SimpleNamespace(num_embeddings=1024)
        )
    )

    capped = splitter.split_text_by_tokens("A" * 90, 120, "<|en|> ")
    assert list(map(len, capped)) == [81, 9]
    assert splitter.split_text_by_tokens("B" * 50, 60, "<|en|> ") == ["B" * 50]
    assert splitter.split_text_by_tokens("甲" * 100, 120, "<|zh|> ") == ["甲" * 100]
