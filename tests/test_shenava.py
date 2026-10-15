from pathlib import Path
import sys
import time

import torch
import nemo.collections.asr as nemo_asr


PROJECT_ROOT = Path(r"D:\Projects\persian_asr")

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "shenava-koochik-v1.0"
    / "shenava-koochik-v1.0.nemo"
)

# یک فایل WAV فارسی برای تست
AUDIO_PATH = (
    PROJECT_ROOT
    /"output"/"debug_audio"/"asr_chunks"/"SPEAKER_00"/"group_000.wav"
)


def main():
    print("=" * 70)
    print("Shenava Koochik v1.0 - Quick Test")
    print("=" * 70)

    print(f"Model : {MODEL_PATH}")
    print(f"Audio : {AUDIO_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Audio not found:\n{AUDIO_PATH}"
        )

    print("\n[1] Device")

    if torch.cuda.is_available():
        device = "cuda"
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("CUDA not available -> CPU")

    print("\n[2] Loading Shenava Koochik...")

    start = time.perf_counter()

    model = nemo_asr.models.ASRModel.restore_from(
        str(MODEL_PATH)
    )

    load_time = time.perf_counter() - start

    model = model.to(device)
    model.eval()

    print(f"Model loaded in {load_time:.2f}s")

    # مدل اصلی Hybrid RNNT/CTC است.
    # برای تست مشابه deployment منتشرشده، CTC را انتخاب می‌کنیم.
    try:
        model.change_decoding_strategy(
            decoder_type="ctc"
        )
        print("Decoder: CTC")
    except Exception as exc:
        print(f"Could not switch to CTC: {exc}")

    # بهترین context گزارش‌شده توسط سازنده
    try:
        model.encoder.set_default_att_context_size([70, 13])
        print("Attention context: [70, 13]")
    except Exception as exc:
        print(f"Could not set attention context: {exc}")

    print("\n[3] Transcribing...")

    start = time.perf_counter()

    result = model.transcribe(
        [str(AUDIO_PATH)]
    )

    inference_time = time.perf_counter() - start

    print(f"Inference time: {inference_time:.2f}s")

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    # بسته به نسخه NeMo خروجی ممکن است Hypothesis یا string باشد
    if isinstance(result, list) and len(result) > 0:
        item = result[0]

        if hasattr(item, "text"):
            text = item.text
        else:
            text = str(item)
    else:
        text = str(result)

    print(text)

    print("=" * 70)


if __name__ == "__main__":
    main()