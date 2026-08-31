from __future__ import annotations

from indextts.utils.front import (
    TEXT_NORMALIZATION_SMOKE_EXPECTED,
    TEXT_NORMALIZATION_SMOKE_INPUT,
    TextNormalizer,
    probe_text_normalization,
)


def test_bundled_text_normalizer_distinguishes_years_from_quantities():
    normalizer = TextNormalizer()
    normalizer.load()

    assert normalizer.normalize(TEXT_NORMALIZATION_SMOKE_INPUT) == TEXT_NORMALIZATION_SMOKE_EXPECTED
    assert normalizer.normalize("1939个人") == "一千九百三十九个人"

    report = probe_text_normalization()
    assert report["available"] is True
    assert report["verified"] is True
    assert report["example_output"] == TEXT_NORMALIZATION_SMOKE_EXPECTED
    assert report["backend"] in {"wetext", "WeTextProcessing"}
