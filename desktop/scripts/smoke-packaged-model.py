"""Run one real IndexTTS 2.5 inference from the packaged Python runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchaudio

from indextts.infer_v2_5 import IndexTTS2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--speaker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda-kernel", action="store_true")
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the packaged real-model smoke test.")
    model = IndexTTS2(
        cfg_path=str(model_dir / "config.yaml"),
        model_dir=str(model_dir),
        device="cuda:0",
        use_bf16=True,
        use_cuda_kernel=args.cuda_kernel,
        use_qwen_emo=False,
    )
    model.infer(
        str(args.speaker.resolve()),
        "这是零点八点一版本的真实模型验证。",
        str(output),
        "ZH",
        seed=20260824,
        diffusion_steps=8,
        do_sample=True,
        top_p=0.8,
        top_k=30,
        num_beams=3,
        repetition_penalty=10.0,
    )
    waveform, sample_rate = torchaudio.load(str(output))
    if waveform.numel() == 0 or sample_rate != 22050:
        raise RuntimeError("Packaged model produced an invalid waveform.")
    print(json.dumps({
        "output": str(output),
        "sample_rate": sample_rate,
        "duration_seconds": round(waveform.shape[-1] / sample_rate, 3),
        "cuda_kernel": bool(args.cuda_kernel),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
