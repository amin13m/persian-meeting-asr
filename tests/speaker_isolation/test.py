from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAMPLES_DIR = PROJECT_ROOT / "tests" / "diarization" / "samples"
DIARIZATION_OUTPUT_DIR = PROJECT_ROOT / "tests" / "diarization" / "output"

OUTPUT_DIR = PROJECT_ROOT / "tests" / "speaker_isolation" / "output"
WORKSPACE_DIR = PROJECT_ROOT / "tests" / "speaker_isolation" / "workspace"

TARGET_SAMPLE_RATE = 16000

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".ogg",
}

# Silence inserted between separated speech regions.
GAP_SECONDS = 0.08

# Small fade to reduce clicks at segment boundaries.
FADE_SECONDS = 0.015


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("speaker_isolation")


# ============================================================
# Helpers
# ============================================================

def print_header(title: str) -> None:
    logger.info("")
    logger.info("=" * 72)
    logger.info(title)
    logger.info("=" * 72)


def find_first_audio(directory: Path) -> Path:
    if not directory.exists():
        raise FileNotFoundError(
            f"Samples directory does not exist:\n{directory}"
        )

    files = sorted(
        p for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )

    if not files:
        raise FileNotFoundError(
            f"No supported audio files found in:\n{directory}"
        )

    return files[0]


def find_diarization_json(audio_path: Path) -> Path:
    expected = DIARIZATION_OUTPUT_DIR / f"{audio_path.stem}_diarization.json"

    if expected.exists():
        return expected

    candidates = sorted(DIARIZATION_OUTPUT_DIR.glob("*_diarization.json"))

    if not candidates:
        raise FileNotFoundError(
            "No diarization JSON found.\n"
            f"Expected:\n{expected}\n"
            f"Directory:\n{DIARIZATION_OUTPUT_DIR}"
        )

    if len(candidates) > 1:
        logger.warning(
            "Multiple diarization JSON files found. Using first: %s",
            candidates[0],
        )

    return candidates[0]


def load_diarization(path: Path) -> dict[str, Any]:
    logger.info("Loading diarization JSON:")
    logger.info("  %s", path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def load_audio_16k_mono(
    audio_path: Path,
) -> tuple[np.ndarray, int]:

    logger.info("Loading source audio:")
    logger.info("  %s", audio_path)

    audio, sr = librosa.load(
        str(audio_path),
        sr=TARGET_SAMPLE_RATE,
        mono=True,
    )

    audio = audio.astype(np.float32)

    logger.info("Audio:")
    logger.info("  Sample rate : %d Hz", TARGET_SAMPLE_RATE)
    logger.info("  Duration    : %.3f sec", len(audio) / TARGET_SAMPLE_RATE)
    logger.info("  Samples     : %d", len(audio))

    return audio, TARGET_SAMPLE_RATE


def normalize_turn(turn: dict[str, Any]) -> tuple[float, float, str]:
    start = float(turn["start"])
    end = float(turn["end"])
    speaker = str(turn["speaker"])

    return start, end, speaker


def merge_intervals(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:

    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda x: x[0])

    merged: list[list[float]] = []

    for start, end in intervals:
        if end <= start:
            continue

        if not merged:
            merged.append([start, end])
            continue

        previous = merged[-1]

        if start <= previous[1]:
            previous[1] = max(previous[1], end)
        else:
            merged.append([start, end])

    return [(start, end) for start, end in merged]


def subtract_intervals(
    source: list[tuple[float, float]],
    remove: list[tuple[float, float]],
) -> list[tuple[float, float]]:

    if not source:
        return []

    if not remove:
        return source.copy()

    remove = merge_intervals(remove)

    result: list[tuple[float, float]] = []

    for source_start, source_end in source:
        current = source_start

        for remove_start, remove_end in remove:

            if remove_end <= current:
                continue

            if remove_start >= source_end:
                break

            if remove_start > current:
                result.append(
                    (
                        current,
                        min(remove_start, source_end),
                    )
                )

            current = max(current, remove_end)

            if current >= source_end:
                break

        if current < source_end:
            result.append((current, source_end))

    return [
        (start, end)
        for start, end in result
        if end - start > 0.001
    ]


def detect_overlap_regions(
    turns: list[tuple[float, float, str]],
) -> list[tuple[float, float, list[str]]]:

    events: list[tuple[float, int, str]] = []

    for start, end, speaker in turns:
        events.append((start, 1, speaker))
        events.append((end, -1, speaker))

    # End events first when timestamps are identical.
    events.sort(key=lambda x: (x[0], x[1]))

    active: set[str] = set()
    previous_time: float | None = None

    overlaps: list[tuple[float, float, list[str]]] = []

    for time, event_type, speaker in events:

        if previous_time is not None:
            if time > previous_time and len(active) >= 2:
                overlaps.append(
                    (
                        previous_time,
                        time,
                        sorted(active),
                    )
                )

        if event_type == 1:
            active.add(speaker)
        else:
            active.discard(speaker)

        previous_time = time

    # Merge adjacent overlap intervals with same speaker set.
    merged: list[list[Any]] = []

    for start, end, speakers in overlaps:

        if not merged:
            merged.append([start, end, speakers])
            continue

        previous = merged[-1]

        if (
            abs(previous[1] - start) < 1e-6
            and previous[2] == speakers
        ):
            previous[1] = end
        else:
            merged.append([start, end, speakers])

    return [
        (float(start), float(end), list(speakers))
        for start, end, speakers in merged
    ]


def apply_fade(
    audio: np.ndarray,
    sample_rate: int,
    fade_seconds: float,
) -> np.ndarray:

    if len(audio) < 4:
        return audio

    fade_samples = int(sample_rate * fade_seconds)

    fade_samples = min(
        fade_samples,
        len(audio) // 2,
    )

    if fade_samples <= 0:
        return audio

    output = audio.copy()

    fade_in = np.linspace(
        0.0,
        1.0,
        fade_samples,
        dtype=np.float32,
    )

    fade_out = np.linspace(
        1.0,
        0.0,
        fade_samples,
        dtype=np.float32,
    )

    output[:fade_samples] *= fade_in
    output[-fade_samples:] *= fade_out

    return output


def time_to_sample(
    time_seconds: float,
    sample_rate: int,
) -> int:

    return int(round(time_seconds * sample_rate))


def extract_interval(
    audio: np.ndarray,
    start: float,
    end: float,
    sample_rate: int,
) -> np.ndarray:

    start_sample = max(
        0,
        time_to_sample(start, sample_rate),
    )

    end_sample = min(
        len(audio),
        time_to_sample(end, sample_rate),
    )

    if end_sample <= start_sample:
        return np.empty(0, dtype=np.float32)

    segment = audio[start_sample:end_sample]

    return segment.astype(np.float32, copy=True)


def concatenate_segments(
    audio: np.ndarray,
    intervals: list[tuple[float, float]],
    sample_rate: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:

    if not intervals:
        return np.empty(0, dtype=np.float32), []

    output_parts: list[np.ndarray] = []
    mapping: list[dict[str, float]] = []

    gap = np.zeros(
        int(round(GAP_SECONDS * sample_rate)),
        dtype=np.float32,
    )

    output_position = 0.0

    for index, (start, end) in enumerate(intervals):

        segment = extract_interval(
            audio,
            start,
            end,
            sample_rate,
        )

        if len(segment) == 0:
            continue

        segment = apply_fade(
            segment,
            sample_rate,
            FADE_SECONDS,
        )

        duration = len(segment) / sample_rate

        output_start = output_position
        output_end = output_start + duration

        output_parts.append(segment)

        mapping.append(
            {
                "source_start": float(start),
                "source_end": float(end),
                "source_duration": float(end - start),
                "output_start": float(output_start),
                "output_end": float(output_end),
                "output_duration": float(duration),
            }
        )

        output_position = output_end

        if index < len(intervals) - 1:
            output_parts.append(gap)
            output_position += GAP_SECONDS

    if not output_parts:
        return np.empty(0, dtype=np.float32), mapping

    result = np.concatenate(output_parts)

    return result.astype(np.float32), mapping


# ============================================================
# Main
# ============================================================

def main() -> int:

    print_header("SPEAKER ISOLATION TEST")

    logger.info("Project root:")
    logger.info("  %s", PROJECT_ROOT)

    logger.info("Samples directory:")
    logger.info("  %s", SAMPLES_DIR)

    logger.info("Diarization directory:")
    logger.info("  %s", DIARIZATION_OUTPUT_DIR)

    logger.info("Output directory:")
    logger.info("  %s", OUTPUT_DIR)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    WORKSPACE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Find source audio
    # --------------------------------------------------------

    print_header("1. FIND SOURCE AUDIO")

    audio_path = find_first_audio(SAMPLES_DIR)

    logger.info("Selected audio:")
    logger.info("  %s", audio_path)

    # --------------------------------------------------------
    # 2. Find diarization result
    # --------------------------------------------------------

    print_header("2. FIND DIARIZATION RESULT")

    diarization_path = find_diarization_json(audio_path)

    logger.info("Selected diarization:")
    logger.info("  %s", diarization_path)

    # --------------------------------------------------------
    # 3. Load audio
    # --------------------------------------------------------

    print_header("3. LOAD AUDIO")

    audio, sample_rate = load_audio_16k_mono(audio_path)

    # --------------------------------------------------------
    # 4. Load diarization
    # --------------------------------------------------------

    print_header("4. LOAD DIARIZATION")

    diarization = load_diarization(diarization_path)

    raw_turns = diarization.get("turns", [])

    if not raw_turns:
        raise RuntimeError(
            "Diarization JSON does not contain 'turns'."
        )

    turns = [
        normalize_turn(turn)
        for turn in raw_turns
    ]

    logger.info("Total diarization turns: %d", len(turns))

    speakers = sorted(
        {
            speaker
            for _, _, speaker in turns
        }
    )

    logger.info("Detected speakers: %d", len(speakers))

    for speaker in speakers:
        logger.info("  - %s", speaker)

    # --------------------------------------------------------
    # 5. Detect overlap
    # --------------------------------------------------------

    print_header("5. DETECT OVERLAP")

    overlap_regions = detect_overlap_regions(turns)

    total_overlap_duration = sum(
        end - start
        for start, end, _ in overlap_regions
    )

    logger.info(
        "Overlap regions : %d",
        len(overlap_regions),
    )

    logger.info(
        "Overlap duration : %.3f sec",
        total_overlap_duration,
    )

    for index, (start, end, overlap_speakers) in enumerate(
        overlap_regions,
        start=1,
    ):
        logger.info(
            "  #%02d %.3f -> %.3f | %s",
            index,
            start,
            end,
            ", ".join(overlap_speakers),
        )

    # --------------------------------------------------------
    # 6. Build speaker timelines
    # --------------------------------------------------------

    print_header("6. BUILD SPEAKER TIMELINES")

    speaker_results: dict[str, Any] = {}

    for speaker in speakers:

        logger.info("")
        logger.info("Processing %s", speaker)

        speaker_intervals = [
            (start, end)
            for start, end, turn_speaker in turns
            if turn_speaker == speaker
        ]

        speaker_intervals = merge_intervals(
            speaker_intervals
        )

        # Intervals where another speaker is active.
        other_intervals = [
            (start, end)
            for start, end, turn_speaker in turns
            if turn_speaker != speaker
        ]

        other_intervals = merge_intervals(
            other_intervals
        )

        # Remove overlap regions from the clean/non-overlap
        # speech timeline.
        clean_intervals = subtract_intervals(
            speaker_intervals,
            other_intervals,
        )

        speaker_total = sum(
            end - start
            for start, end in speaker_intervals
        )

        clean_total = sum(
            end - start
            for start, end in clean_intervals
        )

        overlap_for_speaker = [
            (start, end)
            for start, end, overlap_speakers in overlap_regions
            if speaker in overlap_speakers
        ]

        overlap_total = sum(
            end - start
            for start, end in overlap_for_speaker
        )

        logger.info(
            "  Original speech : %.3f sec",
            speaker_total,
        )

        logger.info(
            "  Clean speech    : %.3f sec",
            clean_total,
        )

        logger.info(
            "  Overlap speech  : %.3f sec",
            overlap_total,
        )

        logger.info(
            "  Clean intervals : %d",
            len(clean_intervals),
        )

        # ----------------------------------------------------
        # 7. Generate isolated audio
        # ----------------------------------------------------

        logger.info("  Generating isolated audio...")

        isolated_audio, mapping = concatenate_segments(
            audio,
            clean_intervals,
            sample_rate,
        )

        output_wav = OUTPUT_DIR / f"{speaker}.wav"

        sf.write(
            str(output_wav),
            isolated_audio,
            sample_rate,
            subtype="PCM_16",
        )

        isolated_duration = (
            len(isolated_audio) / sample_rate
            if len(isolated_audio)
            else 0.0
        )

        logger.info(
            "  Output WAV      : %s",
            output_wav,
        )

        logger.info(
            "  Output duration : %.3f sec",
            isolated_duration,
        )

        # ----------------------------------------------------
        # 8. Speaker metadata
        # ----------------------------------------------------

        speaker_json = {
            "speaker": speaker,
            "source_audio": str(audio_path),
            "source_sample_rate": sample_rate,
            "output_audio": str(output_wav),
            "output_sample_rate": sample_rate,
            "method": "diarization_based_non_overlap_isolation",
            "source_speech_duration_seconds": speaker_total,
            "clean_speech_duration_seconds": clean_total,
            "overlap_duration_seconds": overlap_total,
            "output_audio_duration_seconds": isolated_duration,
            "clean_intervals": [
                {
                    "start": start,
                    "end": end,
                    "duration": end - start,
                }
                for start, end in clean_intervals
            ],
            "overlap_intervals": [
                {
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "speakers": [
                        s
                        for s in overlap_speakers
                    ],
                }
                for start, end, overlap_speakers in overlap_regions
                if speaker in overlap_speakers
            ],
            "output_mapping": mapping,
        }

        speaker_json_path = OUTPUT_DIR / f"{speaker}.json"

        with speaker_json_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                speaker_json,
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(
            "  Metadata JSON    : %s",
            speaker_json_path,
        )

        speaker_results[speaker] = speaker_json

    # --------------------------------------------------------
    # 9. Summary
    # --------------------------------------------------------

    print_header("9. SAVE SUMMARY")

    summary = {
        "source_audio": str(audio_path),
        "diarization_json": str(diarization_path),
        "sample_rate": sample_rate,
        "source_duration_seconds": len(audio) / sample_rate,
        "speakers": speakers,
        "speaker_count": len(speakers),
        "overlap": {
            "region_count": len(overlap_regions),
            "duration_seconds": total_overlap_duration,
            "regions": [
                {
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "speakers": overlap_speakers,
                }
                for start, end, overlap_speakers in overlap_regions
            ],
        },
        "method": {
            "name": "diarization_based_non_overlap_isolation",
            "description": (
                "Keeps only regions where the target speaker "
                "is active without another diarized speaker."
            ),
            "overlap_separation": False,
        },
        "outputs": {
            speaker: {
                "wav": str(
                    OUTPUT_DIR / f"{speaker}.wav"
                ),
                "json": str(
                    OUTPUT_DIR / f"{speaker}.json"
                ),
            }
            for speaker in speakers
        },
    }

    summary_path = OUTPUT_DIR / "isolation_summary.json"

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(
        "Summary saved:"
    )
    logger.info(
        "  %s",
        summary_path,
    )

    # --------------------------------------------------------
    # 10. Final report
    # --------------------------------------------------------

    print_header("FINAL REPORT")

    logger.info(
        "Source duration : %.3f sec",
        len(audio) / sample_rate,
    )

    logger.info(
        "Speakers         : %d",
        len(speakers),
    )

    logger.info(
        "Overlap regions  : %d",
        len(overlap_regions),
    )

    logger.info(
        "Overlap duration : %.3f sec",
        total_overlap_duration,
    )

    logger.info("")
    logger.info("Generated files:")

    for speaker in speakers:
        logger.info(
            "  %s.wav",
            speaker,
        )
        logger.info(
            "  %s.json",
            speaker,
        )

    logger.info(
        "  isolation_summary.json"
    )

    logger.info("")
    logger.info("=" * 72)
    logger.info("SPEAKER ISOLATION TEST COMPLETED")
    logger.info("=" * 72)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.error("Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logger.exception(
            "Fatal error: %s",
            exc,
        )
        sys.exit(1)