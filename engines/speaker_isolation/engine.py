from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


logger = logging.getLogger(
    "persian_asr.speaker_isolation"
)


class SpeakerIsolationEngine:
    """
    Builds one timeline-aligned isolated audio buffer per speaker.

    Architecture:

        original audio
              |
              +---- clean speaker regions ----------+
              |                                      |
              +---- overlap regions -> REAL-T ------+
                                                     |
                                                     v
                                           speaker timeline

    The generated timeline has exactly the same duration and
    sample alignment as the original normalized audio.

    IMPORTANT:
        TSE is executed ONLY on overlap regions.
        The full meeting audio is NEVER passed to TSE.

    New debug/reference functionality:

        clean speech
             |
             +--> speaker reference WAV
             |
             +--> isolated speaker timeline
    """

    def __init__(
        self,
        config: dict,
        overlap_engine,
    ):
        self.config = config
        self.overlap_engine = overlap_engine

        self.sample_rate = int(
            config.get(
                "audio",
                {},
            ).get(
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
                True,
            )
        )

        references_dir = isolation_config.get(
            "references_dir"
        )

        self.references_dir = (
            Path(references_dir)
            if references_dir
            else None
        )

        self.auto_create_references = bool(
            isolation_config.get(
                "auto_create_references",
                True,
            )
        )

        self.overwrite_references = bool(
            isolation_config.get(
                "overwrite_references",
                False,
            )
        )

        self.reference_max_seconds = float(
            isolation_config.get(
                "reference_max_seconds",
                8.0,
            )
        )

        self.reference_min_seconds = float(
            isolation_config.get(
                "reference_min_seconds",
                1.0,
            )
        )

        # ====================================================
        # Debug audio
        # ====================================================

        debug_config = (
            config
            .get("debug", {})
            .get("audio", {})
        )

        self.debug_audio_enabled = bool(
            debug_config.get(
                "enabled",
                False,
            )
        )

        debug_root = debug_config.get(
            "root_dir",
            "output/debug_audio",
        )

        self.debug_root_dir = Path(
            debug_root
        )

        self.save_speaker_references = bool(
            debug_config.get(
                "save_speaker_references",
                True,
            )
        )

        # Internal in-memory speaker timelines.
        self._speaker_audio: dict[
            str,
            np.ndarray,
        ] = {}

        self._audio_duration: float = 0.0

        logger.info(
            "SpeakerIsolationEngine initialized | "
            "sample_rate=%d | preserve_timeline=%s | "
            "auto_references=%s | debug_audio=%s",
            self.sample_rate,
            self.preserve_timeline,
            self.auto_create_references,
            self.debug_audio_enabled,
        )

    # ========================================================
    # Public access to in-memory timelines
    # ========================================================

    def get_speaker_audio(
        self,
        speaker: str,
    ) -> np.ndarray | None:

        return self._speaker_audio.get(
            str(speaker)
        )

    def get_all_speaker_audio(
        self,
    ) -> dict[str, np.ndarray]:

        return self._speaker_audio

    def clear_audio_buffers(self) -> None:

        self._speaker_audio.clear()

    # ========================================================
    # Utility
    # ========================================================

    @staticmethod
    def _safe_filename(
        value: str,
    ) -> str:

        value = str(value)

        for char in (
            "\\",
            "/",
            ":",
            "*",
            "?",
            '"',
            "<",
            ">",
            "|",
        ):
            value = value.replace(
                char,
                "_",
            )

        return value

    def _ensure_directory(
        self,
        path: Path,
    ) -> None:

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # Audio
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

        if waveform.ndim == 2:
            waveform = np.mean(
                waveform,
                axis=1,
                dtype=np.float32,
            )

        if waveform.ndim != 1:
            raise ValueError(
                "Audio waveform must be 1-dimensional."
            )

        waveform = np.nan_to_num(
            waveform,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return (
            np.ascontiguousarray(
                waveform,
                dtype=np.float32,
            ),
            int(sample_rate),
        )

    def _resample(
        self,
        waveform: np.ndarray,
        source_rate: int,
        target_rate: int,
    ) -> np.ndarray:

        if source_rate == target_rate:
            return waveform

        import librosa

        logger.info(
            "Speaker isolation resampling: "
            "%d Hz -> %d Hz",
            source_rate,
            target_rate,
        )

        waveform = librosa.resample(
            waveform,
            orig_sr=source_rate,
            target_sr=target_rate,
        )

        return np.ascontiguousarray(
            waveform,
            dtype=np.float32,
        )

    def _prepare_audio(
        self,
        audio_path: Path,
    ) -> tuple[np.ndarray, int, int]:

        waveform, source_rate = (
            self._load_audio(
                audio_path
            )
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

        logger.info(
            "Isolation audio: "
            "%d Hz -> %d Hz | duration=%.3fs",
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
    # Diarization normalization
    # ========================================================

    @staticmethod
    def _get_turns(
        diarization: dict,
    ) -> list[dict]:

        raw_turns = []

        if isinstance(
            diarization.get("turns"),
            list,
        ):
            raw_turns = diarization["turns"]

        elif isinstance(
            diarization.get("segments"),
            list,
        ):
            raw_turns = diarization["segments"]

        elif isinstance(
            diarization.get("diarization"),
            list,
        ):
            raw_turns = diarization["diarization"]

        result = []

        for item in raw_turns:

            if not isinstance(
                item,
                dict,
            ):
                continue

            speaker = str(
                item.get(
                    "speaker",
                    "",
                )
            ).strip()

            try:

                start = float(
                    item.get(
                        "start",
                        0.0,
                    )
                )

                end = float(
                    item.get(
                        "end",
                        0.0,
                    )
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

            if (
                not speaker
                or end <= start
            ):
                continue

            result.append(
                {
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "duration": end - start,
                }
            )

        result.sort(
            key=lambda x: (
                x["start"],
                x["end"],
            )
        )

        return result

    # ========================================================
    # Speakers
    # ========================================================

    @staticmethod
    def _get_speakers(
        diarization: dict,
        turns: list[dict],
    ) -> list[str]:

        speakers = diarization.get(
            "speakers",
            [],
        )

        if isinstance(
            speakers,
            dict,
        ):
            speakers = list(
                speakers.keys()
            )

        if not isinstance(
            speakers,
            list,
        ):
            speakers = []

        result = [
            str(x).strip()
            for x in speakers
            if str(x).strip()
        ]

        if not result:

            result = sorted(
                {
                    turn["speaker"]
                    for turn in turns
                }
            )

        return sorted(
            set(result)
        )

    # ========================================================
    # Interval helpers
    # ========================================================

    @staticmethod
    def _merge_intervals(
        intervals: list[
            tuple[float, float]
        ],
    ) -> list[
        tuple[float, float]
    ]:

        if not intervals:
            return []

        intervals = sorted(
            intervals
        )

        merged = [
            intervals[0]
        ]

        for start, end in intervals[1:]:

            prev_start, prev_end = (
                merged[-1]
            )

            if start <= prev_end:

                merged[-1] = (
                    prev_start,
                    max(
                        prev_end,
                        end,
                    ),
                )

            else:

                merged.append(
                    (
                        start,
                        end,
                    )
                )

        return merged

    @staticmethod
    def _subtract_intervals(
        base_intervals: list[
            tuple[float, float]
        ],
        remove_intervals: list[
            tuple[float, float]
        ],
    ) -> list[
        tuple[float, float]
    ]:

        if not remove_intervals:
            return base_intervals

        remove_intervals = (
            SpeakerIsolationEngine
            ._merge_intervals(
                remove_intervals
            )
        )

        result = []

        for base_start, base_end in (
            base_intervals
        ):

            cursor = base_start

            for remove_start, remove_end in (
                remove_intervals
            ):

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
    # Overlap detection
    # ========================================================

    def _find_overlap_regions(
        self,
        turns: list[dict],
    ) -> list[dict]:

        events = []

        for turn in turns:

            speaker = turn["speaker"]

            start = float(
                turn["start"]
            )

            end = float(
                turn["end"]
            )

            events.append(
                (
                    start,
                    1,
                    speaker,
                )
            )

            events.append(
                (
                    end,
                    -1,
                    speaker,
                )
            )

        if not events:
            return []

        events.sort(
            key=lambda x: (
                x[0],
                x[1],
            )
        )

        active = set()

        regions = []

        previous_time = events[0][0]

        index = 0

        while index < len(events):

            current_time = events[index][0]

            if (
                current_time > previous_time
                and len(active) >= 2
            ):

                regions.append(
                    {
                        "start": previous_time,
                        "end": current_time,
                        "duration": (
                            current_time
                            - previous_time
                        ),
                        "speakers": sorted(
                            active
                        ),
                    }
                )

            while (
                index < len(events)
                and events[index][0]
                == current_time
            ):

                _, event_type, speaker = (
                    events[index]
                )

                if event_type == -1:
                    active.discard(
                        speaker
                    )
                else:
                    active.add(
                        speaker
                    )

                index += 1

            previous_time = current_time

        return self._merge_overlap_regions(
            regions
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
    # Speaker intervals
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

        speaker_overlap = [
            region
            for region in overlap_regions
            if speaker in region["speakers"]
        ]

        overlap_intervals = [
            (
                float(region["start"]),
                float(region["end"]),
            )
            for region in speaker_overlap
        ]

        clean_intervals = (
            self._subtract_intervals(
                speaker_intervals,
                overlap_intervals,
            )
        )

        return (
            clean_intervals,
            speaker_overlap,
        )

    # ========================================================
    # Timeline operations
    # ========================================================

    def _write_interval(
        self,
        output: np.ndarray,
        source: np.ndarray,
        start: float,
        end: float,
    ) -> None:

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
            len(source),
            int(
                round(
                    end
                    * self.sample_rate
                )
            ),
        )

        if end_sample <= start_sample:
            return

        output[
            start_sample:end_sample
        ] = source[
            start_sample:end_sample
        ]

    def _insert_tse_region(
        self,
        output: np.ndarray,
        region_start: float,
        region_end: float,
        extracted_audio: np.ndarray,
    ) -> None:

        start_sample = max(
            0,
            int(
                round(
                    region_start
                    * self.sample_rate
                )
            ),
        )

        end_sample = min(
            len(output),
            int(
                round(
                    region_end
                    * self.sample_rate
                )
            ),
        )

        if end_sample <= start_sample:
            return

        expected = (
            end_sample
            - start_sample
        )

        audio = np.asarray(
            extracted_audio,
            dtype=np.float32,
        ).reshape(-1)

        if len(audio) < expected:

            audio = np.pad(
                audio,
                (
                    0,
                    expected - len(audio),
                ),
            )

        elif len(audio) > expected:

            audio = audio[:expected]

        output[
            start_sample:end_sample
        ] = audio

    # ========================================================
    # Speaker reference creation
    # ========================================================

    def _build_reference_audio(
        self,
        waveform: np.ndarray,
        clean_intervals: list[
            tuple[float, float]
        ],
    ) -> np.ndarray:

        if not clean_intervals:
            return np.zeros(
                0,
                dtype=np.float32,
            )

        max_samples = int(
            self.reference_max_seconds
            * self.sample_rate
        )

        pieces = []

        collected = 0

        for start, end in clean_intervals:

            if collected >= max_samples:
                break

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
                len(waveform),
                int(
                    round(
                        end
                        * self.sample_rate
                    )
                ),
            )

            if end_sample <= start_sample:
                continue

            piece = waveform[
                start_sample:end_sample
            ]

            remaining = (
                max_samples
                - collected
            )

            if len(piece) > remaining:
                piece = piece[
                    :remaining
                ]

            if len(piece) == 0:
                continue

            pieces.append(
                np.asarray(
                    piece,
                    dtype=np.float32,
                )
            )

            collected += len(piece)

        if not pieces:
            return np.zeros(
                0,
                dtype=np.float32,
            )

        return np.ascontiguousarray(
            np.concatenate(
                pieces
            ),
            dtype=np.float32,
        )

    def _create_speaker_reference(
        self,
        speaker: str,
        waveform: np.ndarray,
        clean_intervals: list[
            tuple[float, float]
        ],
        references_dir: Path,
    ) -> Path | None:

        reference_audio = (
            self._build_reference_audio(
                waveform,
                clean_intervals,
            )
        )

        duration = (
            len(reference_audio)
            / self.sample_rate
        )

        if duration < self.reference_min_seconds:

            logger.warning(
                "Reference not created for %s: "
                "only %.3fs clean audio available.",
                speaker,
                duration,
            )

            return None

        self._ensure_directory(
            references_dir
        )

        safe_speaker = (
            self._safe_filename(
                speaker
            )
        )

        reference_path = (
            references_dir
            / f"{safe_speaker}.wav"
        )

        if (
            reference_path.exists()
            and not self.overwrite_references
        ):

            logger.info(
                "Using existing speaker reference: %s",
                reference_path,
            )

            return reference_path

        sf.write(
            str(reference_path),
            reference_audio,
            self.sample_rate,
        )

        logger.info(
            "Speaker reference created: "
            "%s | duration=%.3fs",
            reference_path,
            duration,
        )

        # Optional second debug copy.
        if (
            self.debug_audio_enabled
            and self.save_speaker_references
        ):

            debug_dir = (
                self.debug_root_dir
                / "speaker_references"
            )

            self._ensure_directory(
                debug_dir
            )

            debug_path = (
                debug_dir
                / f"{safe_speaker}.wav"
            )

            sf.write(
                str(debug_path),
                reference_audio,
                self.sample_rate,
            )

            logger.debug(
                "Debug speaker reference: %s",
                debug_path,
            )

        return reference_path

    # ========================================================
    # Isolate
    # ========================================================

    def isolate(
        self,
        audio_path: Path,
        diarization: dict,
        output_dir: Path | None = None,
    ) -> dict:

        audio_path = Path(
            audio_path
        )

        logger.info(
            "Speaker isolation: %s",
            audio_path.name,
        )

        self.clear_audio_buffers()

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

        self._audio_duration = duration

        turns = self._get_turns(
            diarization
        )

        speakers = self._get_speakers(
            diarization,
            turns,
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

        if output_dir is not None:

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        # ====================================================
        # Effective reference directory
        # ====================================================

        effective_references_dir = (
            self.references_dir
        )

        if (
            effective_references_dir
            is None
            and self.auto_create_references
        ):

            if output_dir is not None:

                effective_references_dir = (
                    output_dir.parent
                    / "references"
                )

            else:

                effective_references_dir = (
                    self.debug_root_dir
                    / "references"
                )

        if (
            effective_references_dir
            is not None
        ):

            self._ensure_directory(
                effective_references_dir
            )

        speaker_results = {}

        # ====================================================
        # Per speaker
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
                for region in speaker_overlap
            )

            # ----------------------------------------------
            # Speaker reference
            # ----------------------------------------------

            reference_file = None

            if (
                self.auto_create_references
                and effective_references_dir
                is not None
            ):

                reference_file = (
                    self._create_speaker_reference(
                        speaker,
                        waveform,
                        clean_intervals,
                        effective_references_dir,
                    )
                )

            # ----------------------------------------------
            # Create timeline
            # ----------------------------------------------

            isolated_audio = np.zeros(
                len(waveform),
                dtype=np.float32,
            )

            # ----------------------------------------------
            # Copy clean speech
            # ----------------------------------------------

            for start, end in (
                clean_intervals
            ):

                self._write_interval(
                    isolated_audio,
                    waveform,
                    start,
                    end,
                )

            # ----------------------------------------------
            # REAL-T / TSE
            # ----------------------------------------------

            processed_overlap_count = 0

            if speaker_overlap:

                logger.info(
                    "TSE [%s]: processing %d "
                    "overlap regions",
                    speaker,
                    len(speaker_overlap),
                )

                try:

                    overlap_result = (
                        self.overlap_engine.separate(
                            audio=waveform,
                            overlap_regions=speaker_overlap,
                            target_speaker=speaker,
                            enrollment_dir=(
                                effective_references_dir
                            ),
                        )
                    )

                except Exception:

                    logger.exception(
                        "TSE failed for speaker %s",
                        speaker,
                    )

                    overlap_result = {
                        "regions": []
                    }

            else:

                overlap_result = {
                    "regions": []
                }

            # ----------------------------------------------
            # Insert TSE output
            # ----------------------------------------------

            for region_result in (
                overlap_result.get(
                    "regions",
                    [],
                )
            ):

                if not region_result.get(
                    "processed",
                    False,
                ):
                    continue

                audio = region_result.get(
                    "audio"
                )

                if audio is None:
                    continue

                try:

                    self._insert_tse_region(
                        isolated_audio,
                        float(
                            region_result[
                                "start"
                            ]
                        ),
                        float(
                            region_result[
                                "end"
                            ]
                        ),
                        audio,
                    )

                    processed_overlap_count += 1

                except Exception:

                    logger.exception(
                        "Failed to insert TSE "
                        "region for %s",
                        speaker,
                    )

            # ----------------------------------------------
            # Save timeline
            # ----------------------------------------------

            output_file = None

            if output_dir is not None:

                safe_speaker = (
                    self._safe_filename(
                        speaker
                    )
                )

                output_file = (
                    output_dir
                    / f"{safe_speaker}.wav"
                )

                sf.write(
                    str(output_file),
                    isolated_audio,
                    self.sample_rate,
                )

            # ----------------------------------------------
            # Store in memory
            # ----------------------------------------------

            self._speaker_audio[
                speaker
            ] = np.ascontiguousarray(
                isolated_audio,
                dtype=np.float32,
            )

            # ----------------------------------------------
            # Metadata
            # ----------------------------------------------

            speaker_results[speaker] = {

                "speaker": speaker,

                "reference_file": (
                    str(reference_file)
                    if reference_file
                    else None
                ),

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

                "overlap_regions": [
                    {
                        "start": round(
                            float(
                                region["start"]
                            ),
                            6,
                        ),
                        "end": round(
                            float(
                                region["end"]
                            ),
                            6,
                        ),
                        "duration": round(
                            float(
                                region["duration"]
                            ),
                            6,
                        ),
                        "speakers": list(
                            region["speakers"]
                        ),
                    }
                    for region in speaker_overlap
                ],

                "overlap_processed": (
                    processed_overlap_count > 0
                ),

                "overlap_processed_count": (
                    processed_overlap_count
                ),

                "overlap_total_count": (
                    len(
                        speaker_overlap
                    )
                ),

                "timeline_preserved": (
                    True
                ),

                "sample_rate": (
                    self.sample_rate
                ),

                "timeline_samples": (
                    len(isolated_audio)
                ),

                "output_file": (
                    str(output_file)
                    if output_file
                    else None
                ),
            }

            logger.info(
                "%s | clean=%.3fs | "
                "overlap=%.3fs | "
                "TSE=%d/%d | reference=%s",
                speaker,
                clean_duration,
                overlap_duration,
                processed_overlap_count,
                len(speaker_overlap),
                reference_file is not None,
            )

        return {
            "audio": {
                "source": str(
                    audio_path
                ),
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
                "samples": len(
                    waveform
                ),
            },

            "speakers": speaker_results,

            "overlap": {
                "count": len(
                    overlap_regions
                ),
                "duration": round(
                    sum(
                        region["duration"]
                        for region
                        in overlap_regions
                    ),
                    6,
                ),
                "regions": overlap_regions,
            },
        }