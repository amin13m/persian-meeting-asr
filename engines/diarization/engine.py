from pathlib import Path
from typing import Any

import logging
import time

import numpy as np
import soundfile as sf
import torch

from pyannote.audio import Pipeline


logger = logging.getLogger(
    "persian_asr.diarization"
)


class DiarizationEngine:
    """
    Local pyannote speaker diarization engine.

    Compatible with pyannote.audio 4.x.

    Responsibilities:
        - Load local pyannote pipeline
        - Run speaker diarization
        - Normalize DiarizeOutput
        - Extract speaker turns
        - Detect overlap regions
        - Build exclusive speaker timeline
        - Return normalized project-level data
    """

    # ========================================================
    # Initialization
    # ========================================================

    def __init__(
        self,
        config: dict,
    ) -> None:

        self.config = config

        self.pipeline = None
        self.loaded = False

        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        diarization_config = config.get(
            "diarization",
            {},
        )

        self.model_path = Path(
            diarization_config.get(
                "model_path",
                "models/diarization/"
                "speaker-diarization-community-1",
            )
        )

        if not self.model_path.is_absolute():

            project_root = (
                Path(__file__)
                .resolve()
                .parents[2]
            )

            self.model_path = (
                project_root
                / self.model_path
            )

        self.sample_rate = int(
            diarization_config.get(
                "sample_rate",
                16000,
            )
        )

        self.min_speakers = (
            diarization_config.get(
                "min_speakers"
            )
        )

        self.max_speakers = (
            diarization_config.get(
                "max_speakers"
            )
        )

    # ========================================================
    # Load
    # ========================================================

    def load(self) -> None:

        if self.loaded:
            logger.warning(
                "Diarization engine is already loaded."
            )
            return

        if not self.model_path.exists():

            raise FileNotFoundError(
                "Diarization model not found: "
                f"{self.model_path}"
            )

        logger.info(
            "Loading local pyannote pipeline..."
        )

        start = time.perf_counter()

        try:

            self.pipeline = (
                Pipeline.from_pretrained(
                    str(self.model_path)
                )
            )

            if self.pipeline is None:

                raise RuntimeError(
                    "pyannote pipeline returned None."
                )

            self.pipeline.to(
                self.device
            )

            self.loaded = True

            elapsed = (
                time.perf_counter()
                - start
            )

            logger.info(
                "Diarization engine loaded in %.3f sec.",
                elapsed,
            )

            logger.info(
                "Diarization device: %s",
                self.device,
            )

        except Exception:

            self.pipeline = None
            self.loaded = False

            logger.exception(
                "Failed to load diarization engine."
            )

            raise

    # ========================================================
    # Audio loading
    # ========================================================

    def _load_audio(
        self,
        audio: Any,
        sample_rate: int | None = None,
    ) -> tuple[np.ndarray, int]:

        # ----------------------------------------------------
        # File path
        # ----------------------------------------------------

        if isinstance(
            audio,
            (str, Path),
        ):

            waveform, sr = sf.read(
                str(audio),
                dtype="float32",
                always_2d=False,
            )

            waveform = np.asarray(
                waveform,
                dtype=np.float32,
            )

            # Stereo -> mono

            if waveform.ndim == 2:

                waveform = np.mean(
                    waveform,
                    axis=1,
                    dtype=np.float32,
                )

            if waveform.ndim != 1:

                raise ValueError(
                    "Audio must be mono."
                )

            return waveform, int(sr)

        # ----------------------------------------------------
        # Numpy
        # ----------------------------------------------------

        if isinstance(
            audio,
            np.ndarray,
        ):

            waveform = np.asarray(
                audio,
                dtype=np.float32,
            )

            sr = (
                int(sample_rate)
                if sample_rate is not None
                else self.sample_rate
            )

        # ----------------------------------------------------
        # Torch
        # ----------------------------------------------------

        elif torch.is_tensor(audio):

            waveform = (
                audio.detach()
                .cpu()
                .float()
                .numpy()
            )

            sr = (
                int(sample_rate)
                if sample_rate is not None
                else self.sample_rate
            )

        else:

            raise TypeError(
                "Unsupported audio type: "
                f"{type(audio).__name__}"
            )

        # ----------------------------------------------------
        # Normalize dimensions
        # ----------------------------------------------------

        waveform = np.asarray(
            waveform,
            dtype=np.float32,
        )

        if waveform.ndim == 2:

            # [channels, samples]

            if waveform.shape[0] <= 8:

                waveform = np.mean(
                    waveform,
                    axis=0,
                    dtype=np.float32,
                )

            # [samples, channels]

            elif waveform.shape[1] <= 8:

                waveform = np.mean(
                    waveform,
                    axis=1,
                    dtype=np.float32,
                )

            else:

                raise ValueError(
                    "Cannot determine audio channel dimension."
                )

        if waveform.ndim != 1:

            raise ValueError(
                "Audio must be a 1-dimensional waveform."
            )

        return waveform, sr

    # ========================================================
    # Resampling
    # ========================================================

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
            "Resampling audio: %d Hz -> %d Hz",
            source_rate,
            target_rate,
        )

        waveform = librosa.resample(
            waveform,
            orig_sr=source_rate,
            target_sr=target_rate,
        )

        return np.asarray(
            waveform,
            dtype=np.float32,
        )

    # ========================================================
    # Prepare audio
    # ========================================================

    def _prepare_audio(
        self,
        audio: Any,
        sample_rate: int | None,
    ) -> tuple[np.ndarray, int]:

        waveform, sr = self._load_audio(
            audio,
            sample_rate,
        )

        if sr != self.sample_rate:

            waveform = self._resample(
                waveform,
                sr,
                self.sample_rate,
            )

            sr = self.sample_rate

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

        return waveform, sr

    # ========================================================
    # Extract pyannote Annotation
    # ========================================================

    def _extract_annotation(
        self,
        output: Any,
    ) -> Any:

        """
        pyannote.audio 4.x returns:

            DiarizeOutput

        The actual Annotation is:

            output.speaker_diarization
        """

        annotation = getattr(
            output,
            "speaker_diarization",
            None,
        )

        if annotation is not None:

            logger.debug(
                "Using output.speaker_diarization."
            )

            return annotation

        # Compatibility with older API

        if hasattr(
            output,
            "itertracks",
        ):

            logger.debug(
                "Using direct Annotation output."
            )

            return output

        raise TypeError(
            "Unsupported pyannote output type: "
            f"{type(output).__name__}"
        )

    # ========================================================
    # Extract exclusive annotation
    # ========================================================

    def _extract_exclusive_annotation(
        self,
        output: Any,
    ) -> Any | None:

        return getattr(
            output,
            "exclusive_speaker_diarization",
            None,
        )

    # ========================================================
    # Diarization
    # ========================================================

    def diarize(
        self,
        audio: Any,
        sample_rate: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> dict:

        if not self.loaded:

            raise RuntimeError(
                "Diarization engine is not loaded."
            )

        if self.pipeline is None:

            raise RuntimeError(
                "Diarization pipeline is not initialized."
            )

        # ----------------------------------------------------
        # Prepare audio
        # ----------------------------------------------------

        waveform, sr = self._prepare_audio(
            audio,
            sample_rate,
        )

        duration = (
            len(waveform) / sr
        )

        logger.info(
            "Audio duration: %.3f sec",
            duration,
        )

        if duration <= 0:

            raise ValueError(
                "Audio is empty."
            )

        # ----------------------------------------------------
        # Convert to torch
        # ----------------------------------------------------

        waveform_tensor = (
            torch.from_numpy(
                waveform
            )
            .unsqueeze(0)
        )

        pipeline_input = {
            "waveform": waveform_tensor,
            "sample_rate": sr,
        }

        # ----------------------------------------------------
        # Speaker limits
        # ----------------------------------------------------

        effective_min = (
            min_speakers
            if min_speakers is not None
            else self.min_speakers
        )

        effective_max = (
            max_speakers
            if max_speakers is not None
            else self.max_speakers
        )

        # ----------------------------------------------------
        # Run pyannote
        # ----------------------------------------------------

        start = time.perf_counter()

        logger.info(
            "Running pyannote diarization..."
        )

        kwargs = {}

        if effective_min is not None:

            kwargs[
                "min_speakers"
            ] = int(effective_min)

        if effective_max is not None:

            kwargs[
                "max_speakers"
            ] = int(effective_max)

        output = self.pipeline(
            pipeline_input,
            **kwargs,
        )

        inference_time = (
            time.perf_counter()
            - start
        )

        logger.info(
            "Diarization completed in %.3f sec.",
            inference_time,
        )

        # ====================================================
        # IMPORTANT:
        # DiarizeOutput -> Annotation
        # ====================================================

        diarization = (
            self._extract_annotation(
                output
            )
        )

        exclusive_annotation = (
            self._extract_exclusive_annotation(
                output
            )
        )

        # ====================================================
        # Extract turns
        # ====================================================

        turns = []

        speakers = set()

        for (
            turn,
            _,
            speaker,
        ) in diarization.itertracks(
            yield_label=True
        ):

            start_time = float(
                turn.start
            )

            end_time = float(
                turn.end
            )

            if end_time <= start_time:
                continue

            speaker = str(
                speaker
            )

            speakers.add(
                speaker
            )

            turns.append(
                {
                    "start": round(
                        start_time,
                        3,
                    ),
                    "end": round(
                        end_time,
                        3,
                    ),
                    "duration": round(
                        end_time
                        - start_time,
                        3,
                    ),
                    "speaker": speaker,
                }
            )

        turns.sort(
            key=lambda item: (
                item["start"],
                item["end"],
            )
        )

        # ====================================================
        # Detect overlaps
        # ====================================================

        overlaps = []

        for index, first in enumerate(
            turns
        ):

            for second in turns[
                index + 1:
            ]:

                if (
                    second["start"]
                    >= first["end"]
                ):
                    break

                overlap_start = max(
                    first["start"],
                    second["start"],
                )

                overlap_end = min(
                    first["end"],
                    second["end"],
                )

                if (
                    overlap_end
                    <= overlap_start
                ):
                    continue

                overlaps.append(
                    {
                        "start": round(
                            overlap_start,
                            3,
                        ),
                        "end": round(
                            overlap_end,
                            3,
                        ),
                        "duration": round(
                            overlap_end
                            - overlap_start,
                            3,
                        ),
                        "speakers": sorted(
                            {
                                first[
                                    "speaker"
                                ],
                                second[
                                    "speaker"
                                ],
                            }
                        ),
                    }
                )

        overlaps = (
            self._merge_overlaps(
                overlaps
            )
        )

        # ====================================================
        # Exclusive timeline
        # ====================================================

        if (
            exclusive_annotation
            is not None
        ):

            exclusive_turns = []

            for (
                turn,
                _,
                speaker,
            ) in (
                exclusive_annotation.itertracks(
                    yield_label=True
                )
            ):

                start_time = float(
                    turn.start
                )

                end_time = float(
                    turn.end
                )

                if end_time <= start_time:
                    continue

                exclusive_turns.append(
                    {
                        "start": round(
                            start_time,
                            3,
                        ),
                        "end": round(
                            end_time,
                            3,
                        ),
                        "duration": round(
                            end_time
                            - start_time,
                            3,
                        ),
                        "speaker": str(
                            speaker
                        ),
                    }
                )

        else:

            exclusive_turns = (
                self._build_exclusive_timeline(
                    turns
                )
            )

        exclusive_turns.sort(
            key=lambda item: (
                item["start"],
                item["end"],
            )
        )

        # ====================================================
        # Speaker durations
        # ====================================================

        speaker_durations = {
            speaker: 0.0
            for speaker in sorted(
                speakers
            )
        }

        for turn in turns:

            speaker = turn[
                "speaker"
            ]

            speaker_durations[
                speaker
            ] += turn[
                "duration"
            ]

        for speaker in speaker_durations:

            speaker_durations[
                speaker
            ] = round(
                speaker_durations[
                    speaker
                ],
                3,
            )

        # ====================================================
        # Statistics
        # ====================================================

        overlap_duration = sum(
            item["duration"]
            for item in overlaps
        )

        result = {

            "audio": {
                "sample_rate": sr,
                "duration": round(
                    duration,
                    3,
                ),
            },

            "processing": {
                "inference_time": round(
                    inference_time,
                    3,
                ),
                "rtf": round(
                    inference_time
                    / duration,
                    6,
                ),
                "realtime_factor": round(
                    duration
                    / inference_time,
                    3,
                )
                if inference_time > 0
                else None,
            },

            "speakers": sorted(
                speakers
            ),

            "speaker_count": len(
                speakers
            ),

            "speaker_durations": (
                speaker_durations
            ),

            "turns": turns,

            "overlaps": overlaps,

            "exclusive_turns": (
                exclusive_turns
            ),

            "statistics": {

                "turn_count": len(
                    turns
                ),

                "exclusive_turn_count": len(
                    exclusive_turns
                ),

                "overlap_count": len(
                    overlaps
                ),

                "overlap_duration": round(
                    overlap_duration,
                    3,
                ),
            },
        }

        logger.info(
            "Diarization result: "
            "%d speaker(s), "
            "%d turn(s), "
            "%d overlap(s).",
            len(speakers),
            len(turns),
            len(overlaps),
        )

        return result

    # ========================================================
    # Merge overlaps
    # ========================================================

    def _merge_overlaps(
        self,
        overlaps: list[dict],
    ) -> list[dict]:

        if not overlaps:
            return []

        overlaps = sorted(
            overlaps,
            key=lambda item: (
                item["start"],
                item["end"],
            ),
        )

        merged = []

        current = dict(
            overlaps[0]
        )

        for item in overlaps[1:]:

            if (
                item["start"]
                <= current["end"]
            ):

                current["end"] = max(
                    current["end"],
                    item["end"],
                )

                current["speakers"] = sorted(
                    set(
                        current[
                            "speakers"
                        ]
                        + item[
                            "speakers"
                        ]
                    )
                )

                current["duration"] = (
                    current["end"]
                    - current["start"]
                )

            else:

                current["duration"] = (
                    current["end"]
                    - current["start"]
                )

                merged.append(
                    current
                )

                current = dict(
                    item
                )

        current["duration"] = (
            current["end"]
            - current["start"]
        )

        merged.append(
            current
        )

        for item in merged:

            item["start"] = round(
                item["start"],
                3,
            )

            item["end"] = round(
                item["end"],
                3,
            )

            item["duration"] = round(
                item["duration"],
                3,
            )

        return merged

    # ========================================================
    # Build exclusive timeline
    # ========================================================

    def _build_exclusive_timeline(
        self,
        turns: list[dict],
    ) -> list[dict]:

        if not turns:
            return []

        boundaries = set()

        for turn in turns:

            boundaries.add(
                turn["start"]
            )

            boundaries.add(
                turn["end"]
            )

        boundaries = sorted(
            boundaries
        )

        exclusive = []

        for index in range(
            len(boundaries) - 1
        ):

            start = (
                boundaries[index]
            )

            end = (
                boundaries[index + 1]
            )

            if end <= start:
                continue

            active = []

            for turn in turns:

                if (
                    turn["start"]
                    <= start
                    and turn["end"]
                    >= end
                ):

                    active.append(
                        turn["speaker"]
                    )

            # Exactly one speaker
            if len(active) != 1:
                continue

            speaker = active[0]

            segment = {
                "start": round(
                    start,
                    3,
                ),
                "end": round(
                    end,
                    3,
                ),
                "duration": round(
                    end - start,
                    3,
                ),
                "speaker": speaker,
            }

            if (
                exclusive
                and exclusive[-1][
                    "speaker"
                ] == speaker
                and abs(
                    exclusive[-1]["end"]
                    - segment["start"]
                ) < 0.0001
            ):

                exclusive[-1][
                    "end"
                ] = segment["end"]

                exclusive[-1][
                    "duration"
                ] = round(
                    exclusive[-1]["end"]
                    - exclusive[-1]["start"],
                    3,
                )

            else:

                exclusive.append(
                    segment
                )

        return exclusive

    # ========================================================
    # Unload
    # ========================================================

    def unload(self) -> None:

        if self.pipeline is None:

            self.loaded = False
            return

        logger.info(
            "Unloading diarization engine..."
        )

        try:

            del self.pipeline

        except Exception:

            logger.exception(
                "Error unloading pyannote pipeline."
            )

        finally:

            self.pipeline = None
            self.loaded = False

            if torch.cuda.is_available():

                try:

                    torch.cuda.empty_cache()

                except Exception:

                    logger.exception(
                        "Failed to clear CUDA cache."
                    )

        logger.info(
            "Diarization engine unloaded."
        )