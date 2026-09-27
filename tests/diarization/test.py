from __future__ import annotations

import json
import math
import time
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
import yaml
from pyannote.audio import Pipeline


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_FILE = PROJECT_ROOT / "config" / "config.yaml"

SAMPLES_DIR = PROJECT_ROOT / "tests" / "diarization" / "samples"

WORKSPACE_DIR = PROJECT_ROOT / "tests" / "diarization" / "workspace"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "diarization" / "output"

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".ogg",
}


# ============================================================
# HELPERS
# ============================================================

def find_first_audio_file(directory: Path) -> Path:
    """Return the first audio file in deterministic filename order."""

    if not directory.exists():
        raise FileNotFoundError(
            f"Samples directory not found:\n{directory}"
        )

    audio_files = sorted(
        [
            p
            for p in directory.iterdir()
            if p.is_file()
            and p.suffix.lower() in AUDIO_EXTENSIONS
        ],
        key=lambda p: p.name.lower(),
    )

    if not audio_files:
        raise FileNotFoundError(
            f"No audio files found in:\n{directory}"
        )

    return audio_files[0]


def load_config() -> dict:
    """Load YAML configuration."""

    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Config file not found:\n{CONFIG_FILE}"
        )

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_audio(
    audio_file: Path,
    workspace_dir: Path,
    target_sample_rate: int = 16000,
) -> tuple[Path, torch.Tensor, int]:
    """
    Convert input audio to:
        - mono
        - target sample rate
        - float32
        - finite samples

    The normalized WAV filename is derived from the original filename.
    """

    workspace_dir.mkdir(parents=True, exist_ok=True)

    normalized_name = (
        f"{audio_file.stem}_{target_sample_rate // 1000}k_mono.wav"
    )

    normalized_file = workspace_dir / normalized_name

    print()
    print("AUDIO PREPARATION")
    print("-" * 70)

    waveform, sample_rate = torchaudio.load(str(audio_file))

    print(f"Original sample rate : {sample_rate}")
    print(f"Original channels    : {waveform.shape[0]}")
    print(f"Original samples     : {waveform.shape[1]}")

    # --------------------------------------------------------
    # Convert to mono
    # --------------------------------------------------------

    if waveform.shape[0] > 1:
        print("Converting audio to mono...")
        waveform = waveform.mean(dim=0, keepdim=True)

    # --------------------------------------------------------
    # Resample
    # --------------------------------------------------------

    if sample_rate != target_sample_rate:
        print(
            f"Resampling {sample_rate} Hz -> "
            f"{target_sample_rate} Hz"
        )

        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            target_sample_rate,
        )

        sample_rate = target_sample_rate

    # --------------------------------------------------------
    # Float32
    # --------------------------------------------------------

    waveform = waveform.float()

    # --------------------------------------------------------
    # Remove NaN / Inf
    # --------------------------------------------------------

    if not torch.isfinite(waveform).all():
        print("WARNING: NaN/Inf detected. Cleaning audio...")
        waveform = torch.nan_to_num(
            waveform,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    # --------------------------------------------------------
    # Ensure exact shape: [1, samples]
    # --------------------------------------------------------

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    if waveform.shape[0] != 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # --------------------------------------------------------
    # Save normalized audio
    # --------------------------------------------------------

    waveform_np = waveform.squeeze(0).cpu().numpy()

    sf.write(
        str(normalized_file),
        waveform_np,
        sample_rate,
        subtype="PCM_16",
    )

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    verify_waveform, verify_sr = torchaudio.load(
        str(normalized_file)
    )

    verify_samples = verify_waveform.shape[1]

    expected_samples = waveform.shape[1]

    if verify_sr != target_sample_rate:
        raise RuntimeError(
            "Normalized audio sample rate verification failed:\n"
            f"Expected: {target_sample_rate}\n"
            f"Actual  : {verify_sr}"
        )

    if verify_waveform.shape[0] != 1:
        raise RuntimeError(
            "Normalized audio channel verification failed:\n"
            f"Expected: 1\n"
            f"Actual  : {verify_waveform.shape[0]}"
        )

    if verify_samples != expected_samples:
        raise RuntimeError(
            "Normalized audio sample count verification failed:\n"
            f"Expected: {expected_samples}\n"
            f"Actual  : {verify_samples}"
        )

    duration = expected_samples / target_sample_rate

    print()
    print(f"Final sample rate : {target_sample_rate}")
    print("Final channels    : 1")
    print(f"Final samples     : {expected_samples}")
    print(f"Final duration    : {duration:.6f} sec")

    print()
    print("Normalized audio saved:")
    print(normalized_file)

    print()
    print("Verification:")
    print(f"  sample rate : {verify_sr}")
    print(f"  samples     : {verify_samples}")
    print(f"  channels    : {verify_waveform.shape[0]}")
    print("Audio verification: PASS")

    return normalized_file, waveform, target_sample_rate


def annotation_to_turns(annotation) -> list[dict]:
    """Convert pyannote annotation to JSON-friendly speaker turns."""

    turns = []

    for segment, _, speaker in annotation.itertracks(
        yield_label=True
    ):
        start = float(segment.start)
        end = float(segment.end)

        turns.append(
            {
                "start": start,
                "end": end,
                "duration": end - start,
                "speaker": str(speaker),
            }
        )

    return turns


def get_speakers(turns: list[dict]) -> list[str]:
    """Return speakers in first-seen order."""

    speakers = []

    for turn in turns:
        speaker = turn["speaker"]

        if speaker not in speakers:
            speakers.append(speaker)

    return speakers


def find_overlaps(turns: list[dict]) -> list[dict]:
    """
    Find overlapping speaker regions.

    Each overlap record contains the two speakers and
    their common active interval.
    """

    overlaps = []

    for i in range(len(turns)):
        a = turns[i]

        for j in range(i + 1, len(turns)):
            b = turns[j]

            if a["speaker"] == b["speaker"]:
                continue

            overlap_start = max(
                a["start"],
                b["start"],
            )

            overlap_end = min(
                a["end"],
                b["end"],
            )

            if overlap_end > overlap_start:
                overlaps.append(
                    {
                        "start": overlap_start,
                        "end": overlap_end,
                        "duration": overlap_end - overlap_start,
                        "speakers": [
                            a["speaker"],
                            b["speaker"],
                        ],
                    }
                )

    return overlaps


def safe_float(value):
    """Convert a value to finite float."""

    value = float(value)

    if not math.isfinite(value):
        return None

    return value


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("PYANNOTE DIARIZATION TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = load_config()

    diarization_config = config.get("diarization", {})
    audio_config = config.get("audio", {})

    target_sample_rate = int(
        diarization_config.get(
            "sample_rate",
            audio_config.get("sample_rate", 16000),
        )
    )

    model_path = diarization_config.get(
        "model_path",
        "models/diarization/speaker-diarization-community-1",
    )

    model_path = Path(model_path)

    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    # --------------------------------------------------------
    # Find first audio file
    # --------------------------------------------------------

    audio_file = find_first_audio_file(SAMPLES_DIR)

    # --------------------------------------------------------
    # Dynamic output names
    # --------------------------------------------------------

    normalized_file, waveform, sample_rate = prepare_audio(
        audio_file,
        WORKSPACE_DIR,
        target_sample_rate,
    )

    output_json = (
        OUTPUT_DIR
        / f"{audio_file.stem}_diarization.json"
    )

    output_rttm = (
        OUTPUT_DIR
        / f"{audio_file.stem}.rttm"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST CONFIGURATION")
    print("=" * 70)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Config       : {CONFIG_FILE}")
    print(f"Samples dir  : {SAMPLES_DIR}")
    print(f"Original     : {audio_file}")
    print(f"Normalized   : {normalized_file}")
    print(f"Model        : {model_path}")
    print(f"Output JSON  : {output_json}")

    # --------------------------------------------------------
    # Torch / CUDA
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TORCH / CUDA")
    print("=" * 70)

    print(f"Torch version : {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version  : {torch.version.cuda}")
        print(
            "GPU           : "
            f"{torch.cuda.get_device_name(0)}"
        )
    else:
        print("GPU           : CPU")

    # --------------------------------------------------------
    # Load pyannote
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING LOCAL PYANNOTE PIPELINE")
    print("=" * 70)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Pyannote model directory not found:\n{model_path}"
        )

    load_start = time.perf_counter()

    pipeline = Pipeline.from_pretrained(
        str(model_path)
    )

    load_time = time.perf_counter() - load_start

    print(
        f"Model loaded in {load_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    requested_device = diarization_config.get(
        "device",
        "cuda",
    )

    if (
        requested_device == "cuda"
        and torch.cuda.is_available()
    ):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    pipeline.to(device)

    print(f"Pipeline device: {device}")

    # --------------------------------------------------------
    # Prepare pyannote input
    # --------------------------------------------------------

    file_input = {
        "waveform": waveform,
        "sample_rate": sample_rate,
    }

    print()
    print("Waveform:")
    print(f"  shape      : {tuple(waveform.shape)}")
    print(f"  sample rate: {sample_rate}")
    print(f"  samples    : {waveform.shape[1]}")

    duration = waveform.shape[1] / sample_rate

    # --------------------------------------------------------
    # GPU cleanup before inference
    # --------------------------------------------------------

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------------
    # Run diarization
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RUNNING DIARIZATION")
    print("=" * 70)

    print("Processing normalized waveform...")

    inference_start = time.perf_counter()

    output = pipeline(file_input)

    processing_seconds = (
        time.perf_counter() - inference_start
    )

    # --------------------------------------------------------
    # Extract regular diarization
    # --------------------------------------------------------

    diarization = output.speaker_diarization

    turns = annotation_to_turns(diarization)

    speakers = get_speakers(turns)

    # --------------------------------------------------------
    # Extract exclusive diarization if available
    # --------------------------------------------------------

    exclusive_annotation = getattr(
        output,
        "exclusive_speaker_diarization",
        None,
    )

    if exclusive_annotation is not None:

        exclusive_turns = annotation_to_turns(
            exclusive_annotation
        )

        exclusive_speakers = get_speakers(
            exclusive_turns
        )

    else:

        exclusive_turns = []
        exclusive_speakers = []

    # --------------------------------------------------------
    # Overlaps
    # --------------------------------------------------------

    overlaps = find_overlaps(turns)

    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    rtf = processing_seconds / duration

    if torch.cuda.is_available():
        peak_vram_bytes = torch.cuda.max_memory_allocated(
            device
        )
        peak_vram_gb = (
            peak_vram_bytes / (1024 ** 3)
        )
    else:
        peak_vram_gb = None

    # --------------------------------------------------------
    # RTTM
    # --------------------------------------------------------

    try:
        with open(
            output_rttm,
            "w",
            encoding="utf-8",
        ) as f:

            for turn in turns:

                start = turn["start"]
                end = turn["end"]
                duration_turn = turn["duration"]
                speaker = turn["speaker"]

                f.write(
                    "SPEAKER "
                    f"{audio_file.stem} "
                    "1 "
                    f"{start:.3f} "
                    f"{duration_turn:.3f} "
                    "<NA> "
                    "<NA> "
                    f"{speaker} "
                    "<NA> "
                    "<NA>\n"
                )

        rttm_saved = True

    except Exception as exc:

        print()
        print(
            f"WARNING: RTTM save failed: {exc}"
        )

        rttm_saved = False

    # --------------------------------------------------------
    # Result JSON
    # --------------------------------------------------------

    result = {
        "project": {
            "name": config.get(
                "project",
                {},
            ).get(
                "name",
                "persian_asr",
            ),
            "mode": config.get(
                "project",
                {},
            ).get(
                "mode",
                "offline",
            ),
        },

        "audio": {
            "original_file": str(
                audio_file
            ),
            "normalized_file": str(
                normalized_file
            ),
            "duration_seconds": safe_float(
                duration
            ),
            "sample_rate": sample_rate,
            "channels": 1,
            "samples": int(
                waveform.shape[1]
            ),
        },

        "model": {
            "engine": "pyannote",
            "model_path": str(
                model_path
            ),
        },

        "performance": {
            "model_load_seconds": safe_float(
                load_time
            ),
            "processing_seconds": safe_float(
                processing_seconds
            ),
            "rtf": safe_float(
                rtf
            ),
            "peak_vram_gb": (
                safe_float(peak_vram_gb)
                if peak_vram_gb is not None
                else None
            ),
        },

        "speakers": speakers,

        "speaker_count": len(speakers),

        "turn_count": len(turns),

        "turns": turns,

        "overlap_count": len(overlaps),

        "overlaps": overlaps,

        "exclusive_speaker_count": len(
            exclusive_speakers
        ),

        "exclusive_speakers": (
            exclusive_speakers
        ),

        "exclusive_turn_count": len(
            exclusive_turns
        ),

        "exclusive_turns": (
            exclusive_turns
        ),

        "output": {
            "json": str(
                output_json
            ),
            "rttm": (
                str(output_rttm)
                if rttm_saved
                else None
            ),
        },
    }

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    with open(
        output_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULT SUMMARY")
    print("=" * 70)

    print(
        f"Audio duration     : {duration:.2f} sec"
    )

    print(
        f"Processing time    : "
        f"{processing_seconds:.2f} sec"
    )

    print(
        f"RTF                : {rtf:.3f}"
    )

    if peak_vram_gb is not None:
        print(
            f"Peak VRAM          : "
            f"{peak_vram_gb:.2f} GB"
        )

    print()
    print(
        f"Speakers detected  : "
        f"{len(speakers)}"
    )

    for speaker in speakers:
        print(
            f"  - {speaker}"
        )

    print()
    print(
        f"Speaker turns      : "
        f"{len(turns)}"
    )

    print(
        f"Overlap regions    : "
        f"{len(overlaps)}"
    )

    print(
        f"Exclusive turns    : "
        f"{len(exclusive_turns)}"
    )

    print()
    print(
        "Output JSON:"
    )

    print(output_json)

    if rttm_saved:
        print()
        print(
            "Output RTTM:"
        )
        print(output_rttm)

    print()
    print("=" * 70)
    print("STATUS: PASS")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()