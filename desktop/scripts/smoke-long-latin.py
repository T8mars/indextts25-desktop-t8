"""Real-model regression for long English/Spanish guarded generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from desktop_generation_controls import (  # noqa: E402
    assess_long_text_result,
    run_with_long_text_guard,
)
from indextts.infer_v2_5 import IndexTTS2  # noqa: E402
from indextts.utils.audio_io import save_audio_file  # noqa: E402


CASES = {
    "EN": (
        "When a long paragraph is synthesized, every sentence must remain complete and clearly "
        "aligned with the intended narration. This regression deliberately includes several "
        "clauses, natural punctuation, and a final sentence that must never disappear. If the "
        "model reaches its acoustic token ceiling or returns audio that is implausibly short, "
        "the integration automatically retries with smaller text segments before returning the "
        "result to the user. The closing words confirm that the paragraph finished correctly."
    ),
    "ES": (
        "Cuando se sintetiza un párrafo largo, cada frase debe conservarse completa y mantener "
        "una narración clara. Esta prueba contiene varias cláusulas, puntuación natural y una "
        "última oración que nunca debe desaparecer. Si el modelo alcanza el límite de tokens "
        "acústicos o devuelve un audio demasiado corto, la integración repite automáticamente "
        "la generación con segmentos más pequeños antes de entregar el resultado. Estas palabras "
        "finales confirman que el párrafo terminó correctamente."
    ),
}


def _waveform(result):
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("IndexTTS did not return an in-memory waveform tuple.")
    sample_rate, raw = result
    tensor = torch.as_tensor(raw).detach().cpu()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2 and tensor.shape[-1] == 1:
        tensor = tensor.transpose(0, 1)
    elif tensor.ndim != 2:
        tensor = tensor.reshape(1, -1)
    if not tensor.dtype.is_floating_point:
        tensor = tensor.float() / 32768.0
    return tensor.contiguous(), int(sample_rate)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--speaker", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--language", choices=["EN", "ES", "ALL"], default="ALL")
    parser.add_argument("--max-tokens", type=int, default=60)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real long-text smoke test.")
    model_dir = args.model_dir.resolve()
    speaker = args.speaker.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = IndexTTS2(
        cfg_path=str(model_dir / "config.yaml"),
        model_dir=str(model_dir),
        device="cuda:0",
        use_bf16=True,
        use_qwen_emo=False,
    )
    selected = CASES if args.language == "ALL" else {args.language: CASES[args.language]}
    reports = []
    for position, (language, text) in enumerate(selected.items()):
        prefix = f"<|{language.lower()}|> "
        token_count = len(model.tokenizer.encode(prefix + text, allowed_special="all"))

        def generate(limit: int):
            return model.infer(
                spk_audio_prompt=str(speaker),
                text=text,
                output_path=None,
                lang=language,
                max_text_tokens_per_segment=limit,
                max_mel_tokens=1500,
                seed=20260828 + position,
                diffusion_steps=8,
                do_sample=True,
                top_p=0.8,
                top_k=30,
                num_beams=3,
                repetition_penalty=10.0,
            )

        result, guard = run_with_long_text_guard(
            generate,
            lambda value: _waveform(value)[0].shape[-1] / _waveform(value)[1],
            text=text,
            language=language,
            token_count=token_count,
            max_tokens=args.max_tokens,
        )
        waveform, sample_rate = _waveform(result)
        output = output_dir / f"long_{language.lower()}_guarded.wav"
        save_audio_file(output, waveform, sample_rate)
        duration = waveform.shape[-1] / sample_rate
        final_reasons = assess_long_text_result(
            text,
            language,
            token_count,
            duration,
            warning_messages=guard.get("retry_warnings") or guard.get("first_warnings") or (),
        )
        if final_reasons:
            raise RuntimeError(f"{language} long-text regression still failed: {final_reasons}")
        reports.append(
            {
                "language": language,
                "tokens": token_count,
                "duration_seconds": round(duration, 3),
                "output": str(output),
                "guard": guard,
            }
        )
    print(json.dumps({"gpu": torch.cuda.get_device_name(0), "cases": reports}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
