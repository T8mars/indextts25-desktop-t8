"""Generate fixed-seed audio for the upstream IndexTTS 2.5 issue #792 case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchaudio

from indextts.infer_v2_5 import IndexTTS2


CASES = {
    "01_risky_single_character": "小明<要|YAO4>求这个题的答案是多少，该做什么呢？",
    "02_whole_word_tone4": "小明<要求|YAO4 QIU2>这个题的答案是多少，该做什么呢？",
    "03_whole_word_tone1_control": "小明<要求|YAO1 QIU2>这个题的答案是多少，该做什么呢？",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="polyphone-regression-792")
    parser.add_argument(
        "--acceleration", choices=("off", "gpt_accel"), default="gpt_accel"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tts = IndexTTS2(
        cfg_path=str(root / "checkpoints" / "config.yaml"),
        model_dir=str(root / "checkpoints"),
        use_bf16=True,
        device="cuda:0",
        use_accel=args.acceleration == "gpt_accel",
        use_qwen_emo=False,
    )
    report = []
    for name, text in CASES.items():
        path = output_dir / f"{name}.wav"
        result = list(
            tts.infer_generator(
                str(root / "examples" / "voice_01.wav"),
                text,
                str(path),
                "ZH",
                seed=792,
                verbose=False,
            )
        )
        waveform, sample_rate = torchaudio.load(path)
        if not result or not torch.isfinite(waveform).all() or waveform.shape[-1] < sample_rate // 2:
            raise RuntimeError(f"回归音频无效：{path}")
        if not torch.equal(waveform[..., -1], torch.zeros_like(waveform[..., -1])):
            raise RuntimeError(f"回归音频末采样未归零：{path}")
        report.append(
            {
                "case": name,
                "text": text,
                "file": str(path),
                "seconds": round(waveform.shape[-1] / sample_rate, 3),
                "sample_rate": sample_rate,
                "tail_sample": float(waveform[..., -1].abs().max()),
            }
        )
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
