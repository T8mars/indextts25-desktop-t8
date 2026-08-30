"""Run two real Qwen text-emotion inferences from one packaged model instance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torchaudio

from indextts.infer_v2_5 import IndexTTS2
from indextts.utils.audio_io import load_audio_file


CASES = (
    (
        "01_calm_text.wav",
        "我会耐心地向你解释这件事情。",
        "语气非常平静、温柔而且克制，像在耐心安慰对方。",
    ),
    (
        "02_angry_text.wav",
        "你为什么一直在骗我！",
        "明显愤怒、激动并带有强烈质问，情绪很不满。",
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--speaker", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    speaker = args.speaker.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the packaged Qwen emotion smoke test.")

    model = IndexTTS2(
        cfg_path=str(model_dir / "config.yaml"),
        model_dir=str(model_dir),
        device="cuda:0",
        use_bf16=True,
        use_cuda_kernel=False,
        use_qwen_emo=True,
    )
    if model.qwen_emo is None:
        raise RuntimeError("QwenEmotion was requested but is not loaded.")

    results = []
    for index, (filename, text, emotion_text) in enumerate(CASES):
        output = output_dir / filename
        model.infer(
            str(speaker),
            text,
            str(output),
            "ZH",
            use_emo_text=True,
            emo_text=emotion_text,
            seed=20260828 + index,
            diffusion_steps=8,
            do_sample=True,
            top_p=0.8,
            top_k=30,
            num_beams=3,
            repetition_penalty=10.0,
        )
        waveform, sample_rate = load_audio_file(output)
        if waveform.numel() == 0 or sample_rate != 22050:
            raise RuntimeError(f"Qwen emotion smoke produced an invalid waveform: {output}")
        results.append(
            {
                "output": str(output),
                "text": text,
                "emotion_text": emotion_text,
                "sample_rate": sample_rate,
                "duration_seconds": round(waveform.shape[-1] / sample_rate, 3),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
        )

    if results[0]["sha256"] == results[1]["sha256"]:
        raise RuntimeError("The two per-line emotion outputs are unexpectedly identical.")
    print(
        json.dumps(
            {
                "qwen_text_emotion": True,
                "same_speaker": str(speaker),
                "model_instance_count": 1,
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0),
                "results": results,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
