from pathlib import Path
from time import perf_counter

import librosa
import soundfile as sf
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "whisper_persian_v4"
)

AUDIO_PATH = (
    PROJECT_ROOT
    / "tests"
    / "samples"
    / "male1.wav"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "tests"
    / "whisper_persian_v4"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Configuration
# ============================================================

TARGET_SAMPLE_RATE = 16000


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Whisper Persian V4 - Inference Test")
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = torch.device("cuda")
        dtype = torch.float16

    else:

        device = torch.device("cpu")
        dtype = torch.float32

    print(
        f"Device: {device}"
    )

    print(
        f"DType : {dtype}"
    )

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    audio, original_sr = sf.read(
        AUDIO_PATH,
        always_2d=False,
    )

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype("float32")

    if original_sr != TARGET_SAMPLE_RATE:

        audio = librosa.resample(
            audio,
            orig_sr=original_sr,
            target_sr=TARGET_SAMPLE_RATE,
        )

    duration = (
        len(audio)
        / TARGET_SAMPLE_RATE
    )

    print(
        f"Audio duration: "
        f"{duration:.3f} sec"
    )

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    model.to(device)
    model.eval()

    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    inputs = processor(
        audio,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
    )

    input_features = inputs.input_features.to(
        device,
        dtype=dtype,
    )

    attention_mask = inputs.get(
        "attention_mask"
    )

    if attention_mask is not None:

        attention_mask = attention_mask.to(
            device
        )

    # --------------------------------------------------------
    # CUDA statistics
    # --------------------------------------------------------

    if device.type == "cuda":

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(
            device
        )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    start_time = perf_counter()

    with torch.inference_mode():

        generation_kwargs = {
            "language": "fa",
            "task": "transcribe",
        }

        if attention_mask is not None:

            generation_kwargs[
                "attention_mask"
            ] = attention_mask

        generated_ids = model.generate(
            input_features,
            **generation_kwargs,
        )

    processing_time = (
        perf_counter() - start_time
    )

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    rtf = (
        processing_time / duration
        if duration > 0
        else 0
    )

    realtime_factor = (
        duration / processing_time
        if processing_time > 0
        else 0
    )

    peak_vram_gb = None

    if device.type == "cuda":

        peak_vram_gb = (
            torch.cuda.max_memory_allocated(
                device
            )
            / (1024 ** 3)
        )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    print(
        f"Duration       : "
        f"{duration:.3f} sec"
    )

    print(
        f"Inference time : "
        f"{processing_time:.3f} sec"
    )

    print(
        f"RTF            : "
        f"{rtf:.4f}"
    )

    print(
        f"Realtime       : "
        f"{realtime_factor:.2f}x"
    )

    if peak_vram_gb is not None:

        print(
            f"Peak VRAM      : "
            f"{peak_vram_gb:.3f} GB"
        )

    print()
    print("TEXT:")
    print("-" * 60)
    print(text)
    print("=" * 60)

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    result_path = (
        RESULTS_DIR
        / f"{AUDIO_PATH.stem}.txt"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "Whisper Persian V4\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(
            f"Audio: {AUDIO_PATH.name}\n"
        )

        file.write(
            f"Duration: {duration:.3f} sec\n"
        )

        file.write(
            f"Inference time: "
            f"{processing_time:.3f} sec\n"
        )

        file.write(
            f"RTF: {rtf:.4f}\n"
        )

        file.write(
            f"Realtime factor: "
            f"{realtime_factor:.2f}x\n"
        )

        file.write(
            f"Device: {device}\n"
        )

        file.write(
            f"DType: {dtype}\n"
        )

        if peak_vram_gb is not None:

            file.write(
                f"Peak VRAM: "
                f"{peak_vram_gb:.3f} GB\n"
            )

        file.write(
            "\n"
        )

        file.write(
            "TRANSCRIPTION\n"
        )

        file.write(
            "-" * 60
            + "\n"
        )

        file.write(
            text
        )

        file.write(
            "\n"
        )

    print()
    print(
        f"Result saved to:\n"
        f"{result_path}"
    )


if __name__ == "__main__":
    main()