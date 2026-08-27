from __future__ import annotations

import pytest
import torch

from indextts.utils.duration_control import (
    allocate_target_frames,
    fit_waveform_length,
    normalize_target_duration,
)


def test_target_duration_validation_and_frame_allocation():
    assert normalize_target_duration(None) is None
    assert normalize_target_duration("2.5") == 2.5
    with pytest.raises(ValueError, match="positive number"):
        normalize_target_duration(0)

    frames, samples = allocate_target_frames(7.3, [1, 2, 3], 22050, 256, 200)
    assert samples == round(7.3 * 22050)
    assert frames[0] < frames[1] < frames[2]
    assert sum(frames) == round((samples - int(22050 * 0.2) * 2) / 256)


def test_fit_waveform_length_is_sample_exact():
    waveform = torch.arange(5).reshape(1, 5)
    assert fit_waveform_length(waveform, 3).tolist() == [[0, 1, 2]]
    assert fit_waveform_length(waveform, 7).tolist() == [[0, 1, 2, 3, 4, 0, 0]]
