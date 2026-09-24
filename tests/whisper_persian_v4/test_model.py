from pathlib import Path

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


# ============================================================
# Test
# ============================================================

def main():

    print("=" * 60)
    print("Whisper Persian V4 - Model Load Test")
    print("=" * 60)

    print(
        f"Model path:\n{MODEL_PATH}"
    )

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    print(
        "Processor: OK"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    print(
        "Model: OK"
    )

    print()
    print(
        "Model loaded successfully."
    )


if __name__ == "__main__":
    main()