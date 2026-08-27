"""Per-generation timing and CUDA peak-memory measurements for the desktop UI."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable


MIB = 1024**2


@dataclass(frozen=True)
class RuntimeMeasurement:
    started_at: float
    cuda_enabled: bool = False
    device_index: int | None = None
    start_allocated_mb: float = 0.0
    start_reserved_mb: float = 0.0
    warning: str = ""


def _torch(torch_module: Any | None) -> Any:
    if torch_module is not None:
        return torch_module
    import torch

    return torch


def start_runtime_measurement(
    *,
    torch_module: Any | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> RuntimeMeasurement:
    """Reset CUDA peak counters and begin a synchronized wall-clock measurement."""

    module = _torch(torch_module)
    try:
        if not bool(module.cuda.is_available()):
            return RuntimeMeasurement(started_at=clock())
        device_index = int(module.cuda.current_device())
        module.cuda.synchronize(device_index)
        module.cuda.reset_peak_memory_stats(device_index)
        allocated = float(module.cuda.memory_allocated(device_index)) / MIB
        reserved = float(module.cuda.memory_reserved(device_index)) / MIB
        return RuntimeMeasurement(
            started_at=clock(),
            cuda_enabled=True,
            device_index=device_index,
            start_allocated_mb=round(allocated, 2),
            start_reserved_mb=round(reserved, 2),
        )
    except Exception as exc:
        return RuntimeMeasurement(
            started_at=clock(),
            warning=f"CUDA 峰值计数初始化失败：{type(exc).__name__}: {exc}",
        )


def finish_runtime_measurement(
    measurement: RuntimeMeasurement,
    audio_duration_seconds: float,
    *,
    torch_module: Any | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Return elapsed time, RTF, and synchronized CUDA peak-memory statistics."""

    module = _torch(torch_module)
    warning = measurement.warning
    peak_allocated = 0.0
    peak_reserved = 0.0
    cuda_enabled = measurement.cuda_enabled
    if cuda_enabled:
        try:
            module.cuda.synchronize(measurement.device_index)
            peak_allocated = float(
                module.cuda.max_memory_allocated(measurement.device_index)
            ) / MIB
            peak_reserved = float(
                module.cuda.max_memory_reserved(measurement.device_index)
            ) / MIB
        except Exception as exc:
            cuda_enabled = False
            detail = f"CUDA 峰值读取失败：{type(exc).__name__}: {exc}"
            warning = f"{warning}；{detail}" if warning else detail

    elapsed = max(0.0, float(clock()) - measurement.started_at)
    audio_duration = max(0.0, float(audio_duration_seconds))
    return {
        "elapsed_seconds": round(elapsed, 3),
        "audio_duration_seconds": round(audio_duration, 3),
        "rtf": round(elapsed / max(audio_duration, 1e-6), 4),
        "cuda_measured": cuda_enabled,
        "cuda_device_index": measurement.device_index if cuda_enabled else None,
        "cuda_start_allocated_mb": measurement.start_allocated_mb if cuda_enabled else None,
        "cuda_start_reserved_mb": measurement.start_reserved_mb if cuda_enabled else None,
        "cuda_peak_allocated_mb": round(peak_allocated, 2) if cuda_enabled else None,
        "cuda_peak_reserved_mb": round(peak_reserved, 2) if cuda_enabled else None,
        "cuda_generation_delta_mb": (
            round(max(0.0, peak_allocated - measurement.start_allocated_mb), 2)
            if cuda_enabled
            else None
        ),
        "warning": warning,
    }


def format_runtime_metrics(report: dict[str, Any]) -> str:
    text = (
        f"耗时 {float(report['elapsed_seconds']):.2f}s，"
        f"音频 {float(report['audio_duration_seconds']):.2f}s，"
        f"RTF {float(report['rtf']):.3f}"
    )
    if report.get("cuda_measured"):
        text += (
            f"，峰值显存 {float(report['cuda_peak_allocated_mb']):.0f}MB"
            f"（本次生成增量 {float(report['cuda_generation_delta_mb']):.0f}MB，"
            f"缓存峰值 {float(report['cuda_peak_reserved_mb']):.0f}MB）"
        )
    elif report.get("warning"):
        text += f"，显存统计不可用：{report['warning']}"
    return text


__all__ = [
    "RuntimeMeasurement",
    "finish_runtime_measurement",
    "format_runtime_metrics",
    "start_runtime_measurement",
]
