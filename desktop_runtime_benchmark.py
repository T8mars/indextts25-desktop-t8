"""Run a user-triggered, sequential IndexTTS 2.5 acceleration benchmark."""

from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
import wave
from pathlib import Path

import torch

from indextts.infer_v2_5 import IndexTTS2
from runtime_acceleration import probe_acceleration, resolve_acceleration
from runtime_benchmark import benchmark_summary, recommend_benchmark_mode


DEFAULT_TEXT = "这是 IndexTTS 2.5 真实加速基准测试，用于选择本机更快且稳定的运行模式。"
DEFAULT_MODES = ("off", "auto_safe", "torch_compile", "gpt_accel", "deepspeed")


def emit(event: str, **payload) -> None:
    print(
        "T8BENCH:" + json.dumps({"event": event, **payload}, ensure_ascii=False),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T8star-Aix IndexTTS 2.5 runtime benchmark")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--reference-audio", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--precision", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--reference-device", choices=["auto", "same", "cpu"], default="auto")
    parser.add_argument("--reuse-spk-cond-for-emo", action="store_true")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    return parser.parse_args()


def select_policy(precision: str, reference_device: str) -> dict:
    if not torch.cuda.is_available():
        return {
            "device": "cpu",
            "precision": "float32",
            "use_bf16": False,
            "use_fp16": False,
            "reference_device": "cpu",
        }
    index = int(torch.cuda.current_device())
    device = f"cuda:{index}"
    try:
        with torch.cuda.device(index):
            native_bf16 = bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        native_bf16 = bool(torch.cuda.is_bf16_supported())
    selected_precision = str(precision)
    if selected_precision == "auto":
        selected_precision = "bfloat16" if native_bf16 else "float16"
    elif selected_precision == "bfloat16" and not native_bf16:
        selected_precision = "float16"
    vram_gb = float(torch.cuda.get_device_properties(index).total_memory) / (1024**3)
    selected_reference = (
        "cpu"
        if reference_device == "cpu" or (reference_device == "auto" and vram_gb < 10)
        else device
    )
    return {
        "device": device,
        "precision": selected_precision,
        "use_bf16": selected_precision == "bfloat16",
        "use_fp16": selected_precision == "float16",
        "reference_device": selected_reference,
        "vram_gb": round(vram_gb, 2),
    }


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as stream:
        return float(stream.getnframes()) / max(1, int(stream.getframerate()))


def release_model() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_case(
    *,
    requested_mode: str,
    model_dir: Path,
    reference_audio: Path,
    output_dir: Path,
    text: str,
    policy: dict,
    capabilities: dict,
    reuse_spk_cond_for_emo: bool,
) -> dict:
    selection = resolve_acceleration(requested_mode, policy["device"], capabilities)
    base = {
        "requested_mode": requested_mode,
        "effective_mode": selection.effective,
        "available": bool(selection.available),
        "reason": selection.reason,
    }
    if requested_mode != "off" and selection.effective == "off":
        return {**base, "status": "skipped"}
    target = output_dir / f"benchmark-{requested_mode}.wav"
    model = None
    try:
        emit("case_start", mode=requested_mode, effective=selection.effective)
        before = time.perf_counter()
        model = IndexTTS2(
            cfg_path=str(model_dir / "config.yaml"),
            model_dir=str(model_dir),
            use_bf16=bool(policy["use_bf16"]),
            use_fp16=bool(policy["use_fp16"]),
            device=policy["device"],
            reference_device=policy["reference_device"],
            reuse_spk_cond_for_emo=bool(reuse_spk_cond_for_emo),
            use_qwen_emo=False,
            **selection.constructor_kwargs(),
        )
        init_seconds = time.perf_counter() - before
        sampling = {
            "do_sample": True,
            "top_p": 1.0,
            "top_k": 0,
            "num_beams": 1,
            "repetition_penalty": 1.0,
            "length_penalty": 0.0,
        }
        warmup_started = time.perf_counter()
        model.infer(
            spk_audio_prompt=str(reference_audio),
            text=str(text),
            output_path=None,
            lang="ZH",
            verbose=False,
            duration_factor=1.0,
            seed=20260827,
            **sampling,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        warmup_seconds = time.perf_counter() - warmup_started
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        before = time.perf_counter()
        model.infer(
            spk_audio_prompt=str(reference_audio),
            text=str(text),
            output_path=str(target),
            lang="ZH",
            verbose=False,
            duration_factor=1.0,
            seed=20260827,
            **sampling,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - before
        duration = wav_duration(target)
        peak_mb = (
            round(float(torch.cuda.max_memory_allocated()) / (1024**2), 2)
            if torch.cuda.is_available()
            else None
        )
        result = {
            **base,
            "status": "ok",
            "init_seconds": round(init_seconds, 3),
            "warmup_seconds": round(warmup_seconds, 3),
            "inference_seconds": round(inference_seconds, 3),
            "audio_seconds": round(duration, 3),
            "audio_duration_seconds": round(duration, 3),
            "rtf": round(inference_seconds / max(duration, 1e-6), 4),
            "peak_vram_gb": round(peak_mb / 1024, 4) if peak_mb is not None else None,
            "peak_allocated_mb": peak_mb,
            "output": str(target),
        }
        emit("case_complete", result=result)
        return result
    except Exception as exc:
        traceback.print_exc()
        result = {
            **base,
            "status": "error",
            "error": f"{type(exc).__name__}: {str(exc).strip() or exc!r}",
        }
        emit("case_complete", result=result)
        return result
    finally:
        model = None
        release_model()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    reference_audio = Path(args.reference_audio).resolve()
    output_dir = Path(args.output_dir).resolve()
    report_path = Path(args.report_path).resolve()
    if not reference_audio.is_file():
        raise FileNotFoundError(f"参考音频不存在：{reference_audio}")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    policy = select_policy(args.precision, args.reference_device)
    capabilities = probe_acceleration(policy["device"])
    modes = list(dict.fromkeys(item.strip() for item in args.modes.split(",") if item.strip()))
    emit("start", modes=modes, policy=policy)
    results = [
        run_case(
            requested_mode=mode,
            model_dir=model_dir,
            reference_audio=reference_audio,
            output_dir=output_dir,
            text=args.text,
            policy=policy,
            capabilities=capabilities,
            reuse_spk_cond_for_emo=args.reuse_spk_cond_for_emo,
        )
        for mode in modes
    ]
    report = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_dir": str(model_dir),
        "reference_audio": str(reference_audio),
        "text": args.text,
        "policy": policy,
        "capabilities": capabilities,
        "results": results,
        "recommendation": recommend_benchmark_mode(results),
    }
    report["summary"] = benchmark_summary(report)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    emit("complete", report=report, report_path=str(report_path))


if __name__ == "__main__":
    main()
