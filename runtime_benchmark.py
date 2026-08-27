"""Runtime benchmark reporting shared by the desktop launcher and tests."""

from __future__ import annotations

from typing import Any, Iterable


MODE_STABILITY_ORDER = {
    "off": 0,
    "bigvgan_cuda": 1,
    "auto_safe": 1,
    "gpt_accel": 2,
    "torch_compile": 3,
    "deepspeed": 4,
}


def _successful(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for item in results
        if item.get("status") == "ok"
        and float(item.get("rtf") or 0) > 0
        and item.get("effective_mode")
    ]


def recommend_benchmark_mode(
    results: Iterable[dict[str, Any]], *, near_fastest_ratio: float = 1.05
) -> dict[str, Any]:
    """Recommend the simplest mode within five percent of the fastest valid RTF."""

    candidates = _successful(results)
    if not candidates:
        return {
            "mode": "off",
            "effective_mode": "off",
            "reason": "没有成功的真实基准结果，保留稳定普通模式。",
            "rtf": None,
        }
    fastest_rtf = min(float(item["rtf"]) for item in candidates)
    near_fastest = [
        item
        for item in candidates
        if float(item["rtf"]) <= fastest_rtf * max(1.0, float(near_fastest_ratio))
    ]
    selected = min(
        near_fastest,
        key=lambda item: (
            MODE_STABILITY_ORDER.get(str(item.get("effective_mode")), 99),
            float(item["rtf"]),
        ),
    )
    requested = str(selected.get("requested_mode") or selected["effective_mode"])
    effective = str(selected["effective_mode"])
    return {
        "mode": requested,
        "effective_mode": effective,
        "reason": (
            f"真实基准推荐 {effective}：RTF {float(selected['rtf']):.3f}；"
            "在最快结果 5% 范围内优先选择更稳定、依赖更少的模式。"
        ),
        "rtf": round(float(selected["rtf"]), 4),
        "fastest_rtf": round(fastest_rtf, 4),
    }


def benchmark_summary(report: dict[str, Any]) -> str:
    results = list(report.get("results") or [])
    succeeded = sum(item.get("status") == "ok" for item in results)
    failed = sum(item.get("status") == "error" for item in results)
    skipped = sum(item.get("status") == "skipped" for item in results)
    recommendation = report.get("recommendation") or {}
    return (
        f"真实基准完成：成功 {succeeded}，失败 {failed}，跳过 {skipped}；"
        f"推荐 {recommendation.get('effective_mode', 'off')}。"
    )


__all__ = [
    "MODE_STABILITY_ORDER",
    "benchmark_summary",
    "recommend_benchmark_mode",
]
