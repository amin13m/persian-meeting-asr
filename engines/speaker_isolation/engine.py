
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf


logger = logging.getLogger(
    "persian_asr.speaker_isolation"
)


class SpeakerIsolationEngine:
    """
    Creates isolated audio timelines for each speaker.

    Input audio may have any sample rate.

    Internally audio is normalized to:
        - mono
        - configured sample rate (normally 16000 Hz)

    Clean speaker regions are preserved.

    Overlap regions are detected and passed to the
    OverlapSeparationEngine.

    Currently the overlap engine is PASS-THROUGH, so
    overlap audio is intentionally excluded from the
    isolated clean-speaker output.
    """

    def __init__(
        self,
        config: dict,
        overlap_engine,
    ):
        self.config = config
        self.overlap_engine = overlap_engine

        self.sample_rate = int(
            config.get("audio", {}).get(
                "sample_rate",
                16000,
            )
        )

        isolation_config = config.get(
            "speaker_isolation",
            {},
        )

        self.preserve_timeline = bool(
            isolation_config.get(
                "preserve_timeline",
                False,
            )
        )

    # ========================================================
    # Audio preparation
    # ========================================================

    def _load_audio(
        self,
        audio_path: Path,
    ) -> tuple[np.ndarray, int]:

        waveform, sample_rate = sf.read(
            str(audio_path),
            dtype="float32",
            always_2d=False,
        )

        waveform = np.asarray(
            waveform,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Convert stereo/multichannel -> mono
        # ----------------------------------------------------

        if waveform.ndim == 2:

            waveform = np.mean(
                waveform,
                axis=1,
                dtype=np.float32,
            )

        if waveform.ndim != 1:
            raise ValueError(
                "Audio waveform must be 1-dimensional "
                "after channel conversion."
            )

        waveform = np.nan_to_num(
            waveform,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        waveform = np.ascontiguousarray(
            waveform,
            dtype=np.float32,
        )

        return waveform, int(sample_rate)

    def _resample(
        self,
        waveform: np.ndarray,
        source_rate: int,
        target_rate: int,
    ) -> np.ndarray:

        if source_rate == target_rate:
            return waveform

        logger.info(
            "Speaker isolation resampling: "
            "%d Hz -> %d Hz",
            source_rate,
            target_rate,
        )

        import librosa

        waveform = librosa.resample(
            waveform,
            orig_sr=source_rate,
            target_sr=target_rate,
        )

        return np.asarray(
            waveform,
            dtype=np.float32,
        )

    def _prepare_audio(
        self,
        audio_path: Path,
    ) -> tuple[np.ndarray, int, int]:

        waveform, source_rate = self._load_audio(
            audio_path
        )

        source_duration = (
            len(waveform) / source_rate
            if source_rate > 0
            else 0.0
        )

        waveform = self._resample(
            waveform,
            source_rate,
            self.sample_rate,
        )

        waveform = np.ascontiguousarray(
            waveform,
            dtype=np.float32,
        )

        logger.info(
            "Isolation audio: "
            "%d Hz -> %d Hz | "
            "duration=%.3fs",
            source_rate,
            self.sample_rate,
            source_duration,
        )

        return (
            waveform,
            source_rate,
            self.sample_rate,
        )

    # ========================================================
    # Interval helpers
    # ========================================================

    @staticmethod
    def _merge_intervals(
        intervals: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:

        if not intervals:
            return []

        intervals = sorted(intervals)

        merged = [intervals[0]]

        for start, end in intervals[1:]:

            previous_start, previous_end = merged[-1]

            if start <= previous_end:
                merged[-1] = (
                    previous_start,
                    max(
                        previous_end,
                        end,
                    ),
                )
            else:
                merged.append(
                    (start, end)
                )

        return merged

    @staticmethod
    def _subtract_intervals(
        base_intervals: list[tuple[float, float]],
        remove_intervals: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:

        if not remove_intervals:
            return base_intervals

        remove_intervals = (
            SpeakerIsolationEngine._merge_intervals(
                remove_intervals
            )
        )

        result = []

        for base_start, base_end in base_intervals:

            cursor = base_start

            for remove_start, remove_end in remove_intervals:

                if remove_end <= cursor:
                    continue

                if remove_start >= base_end:
                    break

                if remove_start > cursor:
                    result.append(
                        (
                            cursor,
                            min(
                                remove_start,
                                base_end,
                            ),
                        )
                    )

                cursor = max(
                    cursor,
                    remove_end,
                )

                if cursor >= base_end:
                    break

            if cursor < base_end:
                result.append(
                    (
                        cursor,
                        base_end,
                    )
                )

        return [
            (start, end)
            for start, end in result
            if end > start
        ]

    # ========================================================
    # Find overlap
    # ========================================================

    def _find_overlap_regions(
        self,
        turns: list[dict],
    ) -> list[dict]:

        overlap_regions = []

        for i in range(len(turns)):

            a = turns[i]

            for j in range(i + 1, len(turns)):

                b = turns[j]

                if a["speaker"] == b["speaker"]:
                    continue

                start = max(
                    float(a["start"]),
                    float(b["start"]),
                )

                end = min(
                    float(a["end"]),
                    float(b["end"]),
                )

                if end > start:

                    overlap_regions.append(
                        {
                            "start": start,
                            "end": end,
                            "duration": end - start,
                            "speakers": sorted(
                                [
                                    a["speaker"],
                                    b["speaker"],
                                ]
                            ),
                        }
                    )

        return self._merge_overlap_regions(
            overlap_regions
        )

    @staticmethod
    def _merge_overlap_regions(
        regions: list[dict],
    ) -> list[dict]:

        if not regions:
            return []

        regions = sorted(
            regions,
            key=lambda x: (
                x["start"],
                x["end"],
            ),
        )

        result = [
            dict(regions[0])
        ]

        for current in regions[1:]:

            previous = result[-1]

            if (
                current["start"]
                <= previous["end"]
                and current["speakers"]
                == previous["speakers"]
            ):

                previous["end"] = max(
                    previous["end"],
                    current["end"],
                )

                previous["duration"] = (
                    previous["end"]
                    - previous["start"]
                )

            else:

                result.append(
                    dict(current)
                )

        return result

    # ========================================================
    # Build speaker intervals
    # ========================================================

    def _build_speaker_intervals(
        self,
        speaker: str,
        turns: list[dict],
        overlap_regions: list[dict],
    ) -> tuple[
        list[tuple[float, float]],
        list[dict],
    ]:

        speaker_intervals = [
            (
                float(turn["start"]),
                float(turn["end"]),
            )
            for turn in turns
            if turn["speaker"] == speaker
        ]

        speaker_intervals = (
            self._merge_intervals(
                speaker_intervals
            )
        )

        speaker_overlap = []

        for region in overlap_regions:

            if speaker in region["speakers"]:

                speaker_overlap.append(
                    (
                        float(region["start"]),
                        float(region["end"]),
                    )
                )

        clean_intervals = (
            self._subtract_intervals(
                speaker_intervals,
                speaker_overlap,
            )
        )

        return (
            clean_intervals,
            speaker_overlap,
        )

    # ========================================================
    # Build audio
    # ========================================================

    def _extract_audio(
        self,
        waveform: np.ndarray,
        intervals: list[tuple[float, float]],
    ) -> np.ndarray:

        pieces = []

        total_samples = len(waveform)

        for start, end in intervals:

            start_sample = max(
                0,
                int(
                    round(
                        start
                        * self.sample_rate
                    )
                ),
            )

            end_sample = min(
                total_samples,
                int(
                    round(
                        end
                        * self.sample_rate
                    )
                ),
            )

            if end_sample > start_sample:

                pieces.append(
                    waveform[
                        start_sample:end_sample
                    ]
                )

        if not pieces:

            return np.zeros(
                0,
                dtype=np.float32,
            )

        return np.concatenate(
            pieces
        ).astype(
            np.float32,
            copy=False,
        )

    # ========================================================
    # Isolate
    # ========================================================

    def isolate(
        self,
        audio_path: Path,
        diarization: dict,
        output_dir: Path | None = None,
    ) -> dict:

        audio_path = Path(audio_path)

        logger.info(
            "Speaker isolation: %s",
            audio_path.name,
        )

        # ----------------------------------------------------
        # Load + normalize audio
        # ----------------------------------------------------

        (
            waveform,
            source_sample_rate,
            normalized_sample_rate,
        ) = self._prepare_audio(
            audio_path
        )

        duration = (
            len(waveform)
            / normalized_sample_rate
        )

        logger.info(
            "Isolation waveform ready: "
            "sample_rate=%d Hz | duration=%.3fs",
            normalized_sample_rate,
            duration,
        )

        # ----------------------------------------------------
        # Diarization data
        #
        # IMPORTANT:
        # Diarization timestamps are in seconds and therefore
        # remain valid after resampling.
        # ----------------------------------------------------

        turns = diarization.get(
            "turns",
            [],
        )

        speakers = diarization.get(
            "speakers",
            [],
        )

        overlap_regions = (
            self._find_overlap_regions(
                turns
            )
        )

        logger.info(
            "Speakers: %d",
            len(speakers),
        )

        logger.info(
            "Overlap regions: %d",
            len(overlap_regions),
        )

        # ----------------------------------------------------
        # Output directory
        # ----------------------------------------------------

        if output_dir is not None:

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        speaker_results = {}

        # ====================================================
        # Process speakers
        # ====================================================

        for speaker in speakers:

            logger.info(
                "Processing speaker: %s",
                speaker,
            )

            (
                clean_intervals,
                speaker_overlap,
            ) = self._build_speaker_intervals(
                speaker,
                turns,
                overlap_regions,
            )

            clean_duration = sum(
                end - start
                for start, end
                in clean_intervals
            )

            overlap_duration = sum(
                region["duration"]
                for region in overlap_regions
                if speaker
                in region["speakers"]
            )

            # ------------------------------------------------
            # Call overlap engine
            #
            # IMPORTANT:
            # waveform is already normalized to 16 kHz.
            # ------------------------------------------------

            overlap_result = (
                self.overlap_engine.separate(
                    audio=waveform,
                    overlap_regions=[
                        region
                        for region in overlap_regions
                        if speaker
                        in region["speakers"]
                    ],
                    target_speaker=speaker,
                )
            )

            # ------------------------------------------------
            # Current prototype behavior:
            #
            # Only clean intervals are extracted.
            #
            # Overlap audio remains excluded until the
            # real overlap separation engine is implemented.
            # ------------------------------------------------

            isolated_audio = self._extract_audio(
                waveform,
                clean_intervals,
            )

            output_file = None

            if output_dir is not None:

                output_file = (
                    output_dir
                    / f"{speaker}.wav"
                )

                sf.write(
                    str(output_file),
                    isolated_audio,
                    self.sample_rate,
                )

            speaker_results[speaker] = {
                "speaker": speaker,

                "original_speech_duration": round(
                    clean_duration
                    + overlap_duration,
                    6,
                ),

                "clean_speech_duration": round(
                    clean_duration,
                    6,
                ),

                "overlap_duration": round(
                    overlap_duration,
                    6,
                ),

                "clean_intervals": [
                    {
                        "start": round(
                            start,
                            6,
                        ),
                        "end": round(
                            end,
                            6,
                        ),
                        "duration": round(
                            end - start,
                            6,
                        ),
                    }
                    for start, end
                    in clean_intervals
                ],

                "overlap_regions": speaker_overlap,

                "overlap_processed": bool(
                    overlap_result.get(
                        "processed",
                        False,
                    )
                ),

                "output_file": (
                    str(output_file)
                    if output_file
                    else None
                ),
            }

            logger.info(
                "%s | clean=%.3fs | overlap=%.3fs",
                speaker,
                clean_duration,
                overlap_duration,
            )

        # ====================================================
        # Final result
        # ====================================================

        return {
            "audio": {
                "source": str(audio_path),

                "source_sample_rate": (
                    source_sample_rate
                ),

                "sample_rate": (
                    self.sample_rate
                ),

                "duration": round(
                    duration,
                    6,
                ),
            },

            "speakers": speaker_results,

            "overlap": {
                "count": len(
                    overlap_regions
                ),

                "duration": round(
                    sum(
                        r["duration"]
                        for r in overlap_regions
                    ),
                    6,
                ),

                "regions": overlap_regions,
            },
        }
