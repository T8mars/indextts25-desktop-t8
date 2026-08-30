from runtime_acceleration import (
    describe_acceleration_failure,
    recommend_runtime_config,
    resolve_acceleration,
)


def capabilities(*, cuda=True, deepspeed=False, flash=False, triton=False, ninja=False, nvcc=False, cxx=False):
    return {
        "cuda": cuda,
        "bf16": cuda,
        "fp16": cuda,
        "gpu": {"total_vram_gb": 8.0 if cuda else 0.0},
        "modules": {"deepspeed": deepspeed, "flash_attn": flash, "triton": triton, "ninja": ninja},
        "tools": {"nvcc": nvcc, "cl": False, "cxx": cxx},
    }


def test_optional_modes_fall_back_without_dependencies():
    for mode in ("bigvgan_cuda", "torch_compile", "gpt_accel", "deepspeed"):
        selected = resolve_acceleration(mode, capabilities=capabilities())
        assert selected.effective == "off"
        assert not selected.available


def test_deepspeed_is_opt_in_and_never_part_of_auto_safe():
    auto = resolve_acceleration(
        "auto_safe",
        capabilities=capabilities(deepspeed=True, ninja=True, nvcc=True, cxx=True),
    )
    assert auto.effective == "bigvgan_cuda"
    assert not auto.use_deepspeed
    explicit = resolve_acceleration("deepspeed", capabilities=capabilities(deepspeed=True))
    assert explicit.use_deepspeed


def test_cpu_always_falls_back():
    selected = resolve_acceleration("gpt_accel", device="cpu", capabilities=capabilities(cuda=False, flash=True, triton=True))
    assert selected.effective == "off"


def test_waves_per_eu_error_is_reported_as_a_torch_triton_mismatch():
    message = describe_acceleration_failure(
        RuntimeError("Keyword argument waves_per_eu was specified but unrecognised")
    )
    assert "仅适用于 AMD" in message
    assert "GPT/torch.compile 加速与本机环境不兼容" in message


def test_bigvgan_requires_both_cuda_and_cpp_toolchains():
    missing_cpp = resolve_acceleration(
        "bigvgan_cuda", capabilities=capabilities(ninja=True, nvcc=True)
    )
    ready = resolve_acceleration(
        "bigvgan_cuda", capabilities=capabilities(ninja=True, nvcc=True, cxx=True)
    )
    assert missing_cpp.effective == "off"
    assert ready.use_cuda_kernel


def test_preflight_recommends_low_vram_settings_without_loading_model():
    caps = capabilities(cuda=True)
    caps["bf16"] = False
    recommendation = recommend_runtime_config(caps)
    assert recommendation == {
        "precision": "float16",
        "reference_device": "cpu",
        "acceleration_mode": "off",
        "reason": "推荐 float16；显存低于 10GB，参考编码器建议放 CPU。",
    }
