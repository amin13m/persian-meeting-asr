
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)


MODEL_PATH = Path("models/distil-whisper-small.en")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test local English ASR: Distil-Whisper Small EN"
    )
    parser.add_argument(
        "audio",
        type=Path,
        help="Path to input audio file",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="Local model directory",
    )
    args = parser.parse_args()

    audio_path = args.audio.resolve()
    model_path = args.model.resolve()

    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {model_path}"
        )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )

    print("=" * 70)
    print("ENGLISH ASR TEST - DISTIL WHISPER SMALL EN")
    print("=" * 70)
    print(f"Audio  : {audio_path}")
    print(f"Model  : {model_path}")
    print(f"Device : {device}")
    print()

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
    )

    print("Loading model...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=True,
    )
    model.to(device)
    model.eval()

    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
        max_new_tokens=128,
    )

    print()
    print("Transcribing...")
    start = time.perf_counter()

    with torch.inference_mode():
        result = asr(str(audio_path))

    elapsed = time.perf_counter() - start
    text = result.get("text", "").strip()

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print(text if text else "[No transcription]")
    print()
    print(f"Inference time: {elapsed:.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    main()
