from types import SimpleNamespace

import pytest

from runtime_metrics import (
    finish_runtime_measurement,
    format_runtime_metrics,
    start_runtime_measurement,
)


class FakeCuda:
    def __init__(self, *, available=True):
        self.available = available
        self.reset_calls = 0
        self.sync_calls = 0

    def is_available(self):
        return self.available

    def current_device(self):
        return 0

    def synchronize(self, _device):
        self.sync_calls += 1

    def reset_peak_memory_stats(self, _device):
        self.reset_calls += 1

    def memory_allocated(self, _device):
        return 4096 * 1024**2

    def memory_reserved(self, _device):
        return 4608 * 1024**2

    def max_memory_allocated(self, _device):
        return 5120 * 1024**2

    def max_memory_reserved(self, _device):
        return 5632 * 1024**2


def clock(*values):
    iterator = iter(values)
    return lambda: next(iterator)


def test_cuda_measurement_reports_peak_and_generation_delta():
    cuda = FakeCuda()
    torch_module = SimpleNamespace(cuda=cuda)
    measurement = start_runtime_measurement(
        torch_module=torch_module,
        clock=clock(10.0),
    )
    report = finish_runtime_measurement(
        measurement,
        8.0,
        torch_module=torch_module,
        clock=clock(14.0),
    )
    assert report["elapsed_seconds"] == 4.0
    assert report["rtf"] == 0.5
    assert report["cuda_peak_allocated_mb"] == 5120.0
    assert report["cuda_generation_delta_mb"] == 1024.0
    assert report["cuda_peak_reserved_mb"] == 5632.0
    assert cuda.reset_calls == 1
    assert cuda.sync_calls == 2
    assert "峰值显存 5120MB" in format_runtime_metrics(report)


def test_cpu_measurement_keeps_timing_without_fake_vram_numbers():
    torch_module = SimpleNamespace(cuda=FakeCuda(available=False))
    measurement = start_runtime_measurement(
        torch_module=torch_module,
        clock=clock(2.0),
    )
    report = finish_runtime_measurement(
        measurement,
        6.0,
        torch_module=torch_module,
        clock=clock(5.0),
    )
    assert report["rtf"] == pytest.approx(0.5)
    assert report["cuda_measured"] is False
    assert report["cuda_peak_allocated_mb"] is None
    assert "RTF 0.500" in format_runtime_metrics(report)
