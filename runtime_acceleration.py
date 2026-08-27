"""Optional acceleration capability probing with safe, dependency-free fallback."""

from __future__ import annotations

import importlib.util
from importlib import metadata
import json
import shutil
from dataclasses import asdict, dataclass

import torch


MODES = ("off", "auto_safe", "bigvgan_cuda", "torch_compile", "gpt_accel", "deepspeed")


@dataclass(frozen=True, slots=True)
class AccelerationSelection:
    requested: str
    effective: str
    use_cuda_kernel: bool = False
    use_torch_compile: bool = False
    use_accel: bool = False
    use_deepspeed: bool = False
    available: bool = True
    reason: str = ""

    def constructor_kwargs(self) -> dict[str, bool]:
        return {
            "use_cuda_kernel": self.use_cuda_kernel,
            "use_torch_compile": self.use_torch_compile,
            "use_accel": self.use_accel,
            "use_deepspeed": self.use_deepspeed,
        }


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _distribution_version(*names: str) -> str | None:
    """Return a package version without importing optional acceleration modules."""

    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def probe_acceleration(device: str = "cuda:0") -> dict:
    cuda = bool(torch.cuda.is_available() and str(device).startswith("cuda"))
    bf16 = False
    if cuda:
        index = int(str(device).split(":", 1)[1]) if ":" in str(device) else torch.cuda.current_device()
        try:
            with torch.cuda.device(index):
                bf16 = bool(
                    torch.cuda.is_bf16_supported(including_emulation=False)
                )
        except TypeError:
            try:
                bf16 = bool(torch.cuda.is_bf16_supported(index))
            except TypeError:
                try:
                    with torch.cuda.device(index):
                        bf16 = bool(torch.cuda.is_bf16_supported())
                except Exception:
                    bf16 = False
        except Exception:
            bf16 = False
    gpu: dict[str, object] = {}
    if cuda:
        try:
            properties = torch.cuda.get_device_properties(index)
            gpu = {
                "index": index,
                "name": str(properties.name),
                "total_vram_gb": round(float(properties.total_memory) / (1024**3), 2),
            }
        except Exception:
            gpu = {}
    return {
        "cuda": cuda,
        "bf16": bf16,
        "fp16": cuda,
        "gpu": gpu,
        "versions": {
            "torch": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda or ""),
            "deepspeed": _distribution_version("deepspeed"),
            "flash_attn": _distribution_version("flash-attn", "flash_attn"),
            "triton": _distribution_version("triton-windows", "triton"),
            "ninja": _distribution_version("ninja"),
        },
        "modules": {
            "deepspeed": _has_module("deepspeed"),
            "flash_attn": _has_module("flash_attn"),
            "triton": _has_module("triton"),
            "ninja": _has_module("ninja") or shutil.which("ninja") is not None,
        },
        "tools": {
            "nvcc": shutil.which("nvcc") is not None,
            "cl": shutil.which("cl") is not None,
            "cxx": any(shutil.which(name) is not None for name in ("c++", "g++", "clang++")),
        },
    }


def recommend_runtime_config(capabilities: dict) -> dict[str, str]:
    """Return conservative launcher defaults from a dependency-only preflight."""

    if not capabilities.get("cuda", False):
        return {
            "precision": "float32",
            "reference_device": "cpu",
            "acceleration_mode": "off",
            "reason": "未检测到 CUDA；使用 CPU float32 普通模式。",
        }
    total_vram = capabilities.get("gpu", {}).get("total_vram_gb")
    low_vram = isinstance(total_vram, (int, float)) and float(total_vram) < 10.0
    modules = capabilities.get("modules", {})
    tools = capabilities.get("tools", {})
    kernel_ready = bool(
        modules.get("ninja")
        and tools.get("nvcc")
        and (tools.get("cl") or tools.get("cxx"))
    )
    precision = "bfloat16" if capabilities.get("bf16", False) else "float16"
    reference_device = "cpu" if low_vram else "same"
    acceleration = "auto_safe" if kernel_ready else "off"
    return {
        "precision": precision,
        "reference_device": reference_device,
        "acceleration_mode": acceleration,
        "reason": (
            f"推荐 {precision}；"
            + (
                "显存低于 10GB，参考编码器建议放 CPU。"
                if low_vram
                else "参考编码器可与主模型使用同一设备。"
            )
        ),
    }


def resolve_acceleration(mode: str, device: str = "cuda:0", capabilities: dict | None = None) -> AccelerationSelection:
    requested = str(mode or "off").strip().lower()
    if requested not in MODES:
        raise ValueError(f"未知加速模式：{requested}")
    caps = capabilities or probe_acceleration(device)
    modules, tools = caps["modules"], caps["tools"]
    if requested == "off":
        return AccelerationSelection(requested, "off", reason="已关闭可选加速")
    if not caps["cuda"]:
        return AccelerationSelection(requested, "off", available=False, reason="当前设备没有可用 CUDA，已回退普通模式")
    cuda_kernel_ready = bool(
        modules["ninja"] and tools["nvcc"] and (tools.get("cl", False) or tools.get("cxx", False))
    )
    if requested == "auto_safe":
        if cuda_kernel_ready:
            return AccelerationSelection(requested, "bigvgan_cuda", use_cuda_kernel=True, reason="自动安全模式启用 BigVGAN CUDA 融合核")
        return AccelerationSelection(requested, "off", available=False, reason="缺少 CUDA 编译工具链，自动安全模式使用普通路径")
    if requested == "bigvgan_cuda":
        ready = cuda_kernel_ready
        return AccelerationSelection(requested, requested if ready else "off", use_cuda_kernel=ready, available=ready, reason=("依赖可用" if ready else "缺少 ninja 或 CUDA/C++ 编译工具链，已回退"))
    if requested == "torch_compile":
        ready = bool(modules["triton"])
        return AccelerationSelection(requested, requested if ready else "off", use_torch_compile=ready, available=ready, reason=("Triton 可用；首次生成会编译" if ready else "缺少可选 Triton，已回退"))
    if requested == "gpt_accel":
        ready = bool(modules["flash_attn"] and modules["triton"])
        return AccelerationSelection(requested, requested if ready else "off", use_accel=ready, available=ready, reason=("FlashAttention 与 Triton 可用" if ready else "缺少可选 FlashAttention/Triton，已回退"))
    ready = bool(modules["deepspeed"])
    return AccelerationSelection(requested, "deepspeed" if ready else "off", use_deepspeed=ready, available=ready, reason=("DeepSpeed 可用" if ready else "未安装可选 DeepSpeed，已回退；基础安装不要求它"))


def format_acceleration_report(selection: AccelerationSelection, capabilities: dict) -> str:
    return json.dumps(
        {
            "selection": asdict(selection),
            "recommended": recommend_runtime_config(capabilities),
            "capabilities": capabilities,
        },
        ensure_ascii=False,
        indent=2,
    )


__all__ = [
    "AccelerationSelection",
    "MODES",
    "format_acceleration_report",
    "probe_acceleration",
    "recommend_runtime_config",
    "resolve_acceleration",
]
