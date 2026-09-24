from pathlib import Path
import time

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "whisper_persian_v4"
SAMPLES_PATH = PROJECT_ROOT / "tests" / "samples"
RESULTS_PATH = PROJECT_ROOT / "tests" / "whisper_persian_v4" / "results"

RESULTS_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# Configuration
# ============================================================

TARGET_SAMPLE_RATE = 16000

# Audio file to test
AUDIO_PATH = SAMPLES_PATH / "male1.wav"


# ============================================================
# Helpers
# ============================================================

def format_seconds(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    milliseconds = int((seconds % 1) * 1000)
    total_seconds = int(seconds)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def save_result(
    output_path: Path,
    audio_path: Path,
    duration: float,
    sample_rate: int,
    device: torch.device,
    dtype: torch.dtype,
    inference_time: float,
    rtf: float,
    peak_vram_gb: float | None,
    transcription: str,
):
    """Save inference result to a TXT file."""

    gpu_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "N/A"
    )

    with output_path.open("w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Whisper Persian V4 - Inference Test Result\n")
        f.write("=" * 70 + "\n\n")

        f.write("AUDIO\n")
        f.write("-" * 70 + "\n")
        f.write(f"File: {audio_path.name}\n")
        f.write(f"Path: {audio_path}\n")
        f.write(f"Duration: {duration:.3f} sec\n")
        f.write(f"Duration formatted: {format_seconds(duration)}\n")
        f.write(f"Sample rate: {sample_rate} Hz\n\n")

        f.write("MODEL\n")
        f.write("-" * 70 + "\n")
        f.write(f"Model path: {MODEL_PATH}\n")
        f.write("Model: nezamisafa/whisper-persian-v4\n")
        f.write("Base model: openai/whisper-large-v3\n\n")

        f.write("HARDWARE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Device: {device}\n")
        f.write(f"GPU: {gpu_name}\n")
        f.write(f"Dtype: {dtype}\n\n")

        f.write("PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Inference time: {inference_time:.3f} sec\n")
        f.write(f"RTF: {rtf:.4f}\n")

        if peak_vram_gb is not None:
            f.write(f"Peak VRAM: {peak_vram_gb:.3f} GB\n")
        else:
            f.write("Peak VRAM: N/A\n")

        if rtf > 0:
            f.write(
                f"Realtime factor: {1 / rtf:.2f}x realtime\n"
            )

        f.write("\n")

        f.write("TRANSCRIPTION\n")
        f.write("-" * 70 + "\n")
        f.write(transcription.strip() + "\n")

        f.write("\n")
        f.write("=" * 70 + "\n")


# ============================================================
# Main Test
# ============================================================

def main():
    print("=" * 70)
    print("Whisper Persian V4 - Inference Test")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Audio file not found:\n{AUDIO_PATH}"
        )

    print(f"\nModel: {MODEL_PATH}")
    print(f"Audio: {AUDIO_PATH}")

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float16

        print(f"\nDevice: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Dtype: {dtype}")

    else:
        device = torch.device("cpu")
        dtype = torch.float32

        print("\nCUDA is not available.")
        print(f"Device: {device}")
        print(f"Dtype: {dtype}")

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    print("\nLoading audio...")

    audio, sample_rate = sf.read(
        AUDIO_PATH,
        dtype="float32",
    )

    # Stereo -> Mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    original_sample_rate = sample_rate

    duration = len(audio) / sample_rate

    print(f"Original sample rate: {sample_rate} Hz")
    print(f"Duration: {duration:.2f} sec")

    # --------------------------------------------------------
    # Resample
    # --------------------------------------------------------

    if sample_rate != TARGET_SAMPLE_RATE:
        print(
            f"Resampling: {sample_rate} Hz -> "
            f"{TARGET_SAMPLE_RATE} Hz"
        )

        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=TARGET_SAMPLE_RATE,
        )

        sample_rate = TARGET_SAMPLE_RATE

    # --------------------------------------------------------
    # Load processor
    # --------------------------------------------------------

    print("\nLoading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    print("Processor: OK")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    model.to(device)
    model.eval()

    print("Model: OK")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    print("\nPreparing input...")

    inputs = processor(
        audio,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
    )

    input_features = inputs.input_features.to(
        device=device,
        dtype=dtype,
    )

    # IMPORTANT:
    # Pass attention_mask when available to avoid the
    # pad_token/eos_token warning during generation.

    attention_mask = None

    if hasattr(inputs, "attention_mask"):
        attention_mask = inputs.attention_mask.to(device)

    # --------------------------------------------------------
    # CUDA memory reset
    # --------------------------------------------------------

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    print("\nRunning inference...")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.inference_mode():

        generate_kwargs = {
            "input_features": input_features,
            "language": "fa",
            "task": "transcribe",
        }

        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask

        generated_ids = model.generate(
            **generate_kwargs
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    inference_time = time.perf_counter() - start_time

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    transcription = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()

    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    rtf = inference_time / duration

    peak_vram_gb = None

    if torch.cuda.is_available():
        peak_vram_bytes = torch.cuda.max_memory_allocated(device)
        peak_vram_gb = peak_vram_bytes / (1024 ** 3)

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    print(f"\nAudio duration : {duration:.2f} sec")
    print(f"Inference time : {inference_time:.2f} sec")
    print(f"RTF            : {rtf:.4f}")
    print(f"Realtime       : {1 / rtf:.2f}x")

    if peak_vram_gb is not None:
        print(f"Peak VRAM      : {peak_vram_gb:.2f} GB")

    print("\nTranscription:")
    print("-" * 70)
    print(transcription)
    print("-" * 70)

    # --------------------------------------------------------
    # Save TXT
    # --------------------------------------------------------

    result_path = RESULTS_PATH / f"{AUDIO_PATH.stem}.txt"

    save_result(
        output_path=result_path,
        audio_path=AUDIO_PATH,
        duration=duration,
        sample_rate=sample_rate,
        device=device,
        dtype=dtype,
        inference_time=inference_time,
        rtf=rtf,
        peak_vram_gb=peak_vram_gb,
        transcription=transcription,
    )

    print(f"\nResult saved to:")
    print(result_path)

    print("\n" + "=" * 70)
    print("INFERENCE TEST: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()