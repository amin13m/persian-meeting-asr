from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from engines.asr_grouping import (
    ASRGroupingEngine,
)
from engines.diarization.engine import (
    DiarizationEngine,
)
from engines.overlap_separation.engine import (
    OverlapSeparationEngine,
)
from engines.speaker_isolation.engine import (
    SpeakerIsolationEngine,
)


logger = logging.getLogger(
    "persian_asr.pipeline"
)


class MeetingPipeline:

    def __init__(
        self,
        config: dict,
        asr_engine,
    ):
        self.config = config
        self.asr_engine = asr_engine

        # ====================================================
        # Engines
        # ====================================================

        self.diarization = (
            DiarizationEngine(
                config
            )
        )

        self.overlap = (
            OverlapSeparationEngine(
                config
            )
        )

        self.isolation = (
            SpeakerIsolationEngine(
                config,
                self.overlap,
            )
        )

        # ====================================================
        # Dedicated ASR grouping engine
        # ====================================================

        self.grouping = (
            ASRGroupingEngine(
                config
            )
        )

        # ====================================================
        # Audio
        # ====================================================

        audio_config = config.get(
            "audio",
            {},
        )

        self.target_sample_rate = int(
            audio_config.get(
                "sample_rate",
                16000,
            )
        )

        # ====================================================
        # ASR
        # ====================================================

        asr_config = config.get(
            "asr",
            {},
        )

        self.min_turn_seconds = float(
            asr_config.get(
                "min_turn_seconds",
                0.20,
            )
        )

        self.max_turn_seconds = float(
            asr_config.get(
                "max_turn_seconds",
                30.0,
            )
        )

        self.trim_silence = bool(
            asr_config.get(
                "trim_silence",
                False,
            )
        )

        # ====================================================
        # Debug
        # ====================================================

        debug_cfg = (
            config
            .get(
                "debug",
                {},
            )
            .get(
                "audio",
                {},
            )
        )

        self.debug_audio_enabled = bool(
            debug_cfg.get(
                "enabled",
                False,
            )
        )

        self.save_asr_debug_audio = bool(
            debug_cfg.get(
                "save_asr_chunks",
                False,
            )
        )

        debug_root = debug_cfg.get(
            "root_dir",
            "output/debug_audio",
        )

        self.debug_root = Path(
            debug_root
        )

        # ====================================================
        # Runtime
        # ====================================================

        self._original_waveform = None

        self._original_sample_rate = (
            self.target_sample_rate
        )

    # ========================================================
    # Lifecycle
    # ========================================================

    def load(
        self,
    ) -> None:

        logger.info(
            "Loading meeting pipeline..."
        )

        self.diarization.load()
        self.overlap.load()

        logger.info(
            "Meeting pipeline loaded."
        )

    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:

        try:
            return float(value)
        except (
            TypeError,
            ValueError,
        ):
            return default

    @staticmethod
    def _normalize_turn(
        turn: dict,
    ) -> dict | None:

        speaker = str(
            turn.get(
                "speaker",
                "",
            )
        ).strip()

        start = MeetingPipeline._safe_float(
            turn.get(
                "start",
                0.0,
            )
        )

        end = MeetingPipeline._safe_float(
            turn.get(
                "end",
                0.0,
            )
        )

        if (
            not speaker
            or end <= start
        ):
            return None

        return {
            "speaker": speaker,
            "start": start,
            "end": end,
            "duration": end - start,
        }

    def _get_diarization_turns(
        self,
        diarization: dict,
    ) -> list[dict]:

        raw_turns = []

        if isinstance(
            diarization.get("turns"),
            list,
        ):

            raw_turns = (
                diarization[
                    "turns"
                ]
            )

        elif isinstance(
            diarization.get("segments"),
            list,
        ):

            raw_turns = (
                diarization[
                    "segments"
                ]
            )

        elif isinstance(
            diarization.get(
                "diarization"
            ),
            list,
        ):

            raw_turns = (
                diarization[
                    "diarization"
                ]
            )

        normalized = []

        for item in raw_turns:

            if not isinstance(
                item,
                dict,
            ):
                continue

            turn = (
                self._normalize_turn(
                    item
                )
            )

            if turn is not None:

                normalized.append(
                    turn
                )

        normalized.sort(
            key=lambda x: (
                x["start"],
                x["end"],
            )
        )

        return normalized

    # ========================================================
    # Original audio
    # ========================================================

    @staticmethod
    def _resample_audio(
        waveform: np.ndarray,
        source_rate: int,
        target_rate: int,
    ) -> np.ndarray:

        if waveform.size == 0:

            return np.zeros(
                0,
                dtype=np.float32,
            )

        if source_rate == target_rate:

            return np.ascontiguousarray(
                waveform,
                dtype=np.float32,
            )

        source_length = len(
            waveform
        )

        target_length = int(
            round(
                source_length
                * target_rate
                / source_rate
            )
        )

        if target_length <= 0:

            return np.zeros(
                0,
                dtype=np.float32,
            )

        source_positions = np.linspace(
            0.0,
            source_length - 1,
            num=source_length,
            dtype=np.float64,
        )

        target_positions = np.linspace(
            0.0,
            source_length - 1,
            num=target_length,
            dtype=np.float64,
        )

        resampled = np.interp(
            target_positions,
            source_positions,
            waveform,
        )

        return np.ascontiguousarray(
            resampled,
            dtype=np.float32,
        )

    def _load_original_audio(
        self,
        audio_path: Path,
    ) -> tuple[
        np.ndarray,
        int,
    ]:

        waveform, sample_rate = sf.read(
            str(audio_path),
            always_2d=False,
            dtype="float32",
        )

        waveform = np.asarray(
            waveform,
            dtype=np.float32,
        )

        if waveform.ndim > 1:

            waveform = np.mean(
                waveform,
                axis=1,
                dtype=np.float32,
            )

        waveform = np.ascontiguousarray(
            waveform,
            dtype=np.float32,
        )

        source_rate = int(
            sample_rate
        )

        if (
            source_rate
            != self.target_sample_rate
        ):

            waveform = (
                self._resample_audio(
                    waveform,
                    source_rate,
                    self.target_sample_rate,
                )
            )

            sample_rate = (
                self.target_sample_rate
            )

        return (
            waveform,
            int(sample_rate),
        )

    # ========================================================
    # Audio validation / preparation
    # ========================================================

    def _prepare_group_audio(
        self,
        group: dict,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:

        return self.grouping.build_audio(
            group,
            waveform,
            sample_rate,
        )

    def _validate_groups(
        self,
        groups: list[dict],
        waveform: np.ndarray,
        sample_rate: int,
    ) -> list[dict]:

        return (
            self.grouping
            .validate_groups_against_audio(
                groups,
                waveform,
                sample_rate,
            )
        )

    # ========================================================
    # Debug
    # ========================================================

    @staticmethod
    def _safe_name(
        value: str,
    ) -> str:

        result = str(
            value
        )

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

            result = result.replace(
                char,
                "_",
            )

        return result

    def _save_asr_chunk(
        self,
        group: dict,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Path | None:

        if not (
            self.debug_audio_enabled
            and self.save_asr_debug_audio
        ):

            return None

        if audio.size == 0:
            return None

        speaker = self._safe_name(
            group["speaker"]
        )

        path = (
            self.debug_root
            / "asr_chunks"
            / speaker
            / (
                f"group_"
                f"{group['group_index']:03d}.wav"
            )
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # IMPORTANT:
        # Save only after validation.
        sf.write(
            str(path),
            audio,
            sample_rate,
        )

        return path

    # ========================================================
    # ASR
    # ========================================================

    def _transcribe_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        start: float,
        end: float,
    ) -> str:

        duration = (
            end - start
        )

        if duration <= 0:
            return ""

        if (
            duration
            < self.grouping.min_group_seconds
        ):
            return ""

        if audio.size == 0:
            return ""

        try:

            result = (
                self.asr_engine.transcribe(
                    audio,
                    sample_rate=int(
                        sample_rate
                    ),
                )
            )

            if isinstance(
                result,
                dict,
            ):

                text = result.get(
                    "text",
                    "",
                )

            else:

                text = result

            return str(
                text or ""
            ).strip()

        except Exception:

            logger.exception(
                "ASR failed for "
                "%.3f -> %.3f",
                start,
                end,
            )

            return ""

    def _transcribe_group(
        self,
        group: dict,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> tuple[
        str,
        Path | None,
        np.ndarray,
    ]:

        start = float(
            group["start"]
        )

        end = float(
            group["end"]
        )

        # ----------------------------------------------------
        # Rebuild from source intervals.
        #
        # We intentionally do NOT simply crop
        # waveform[start:end].
        #
        # This prevents ignored short speaker turns from
        # leaking into the ASR group.
        # ----------------------------------------------------

        audio = (
            self._prepare_group_audio(
                group,
                waveform,
                sample_rate,
            )
        )

        if audio.size == 0:

            return (
                "",
                None,
                audio,
            )

        # ----------------------------------------------------
        # Debug WAV is written ONLY here, after grouping
        # validation has already succeeded.
        # ----------------------------------------------------

        debug_path = (
            self._save_asr_chunk(
                group,
                audio,
                sample_rate,
            )
        )

        text = (
            self._transcribe_audio(
                audio,
                sample_rate,
                start,
                end,
            )
        )

        return (
            text,
            debug_path,
            audio,
        )

    # ========================================================
    # Group logging
    # ========================================================

    def _log_grouping_stats(
        self,
        raw_turns: list[dict],
        groups: list[dict],
    ) -> None:

        stats = (
            self.grouping.last_stats
        )

        logger.info(
            "ASR grouping: "
            "raw_turns=%d | "
            "ignored_short=%d | "
            "candidate_groups=%d | "
            "rejected=%d | "
            "valid_groups=%d",
            len(raw_turns),
            int(
                stats.get(
                    "ignored_short_turns",
                    0,
                )
            ),
            int(
                stats.get(
                    "groups_before_audio_validation",
                    0,
                )
            ),
            int(
                stats.get(
                    "groups_rejected",
                    0,
                )
            ),
            len(groups),
        )

    # ========================================================
    # Process
    # ========================================================

    def process(
        self,
        audio_path: Path,
        output_dir: Path,
    ) -> dict:

        audio_path = Path(
            audio_path
        )

        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_start = (
            time.perf_counter()
        )

        logger.info("")
        logger.info(
            "=" * 70
        )
        logger.info(
            "MEETING PIPELINE: %s",
            audio_path.name,
        )
        logger.info(
            "=" * 70
        )

        self._original_waveform = None

        # ====================================================
        # 1. Diarization
        # ====================================================

        logger.info(
            "[1/4] Speaker diarization"
        )

        diarization = (
            self.diarization.diarize(
                audio_path
            )
        )

        turns = (
            self._get_diarization_turns(
                diarization
            )
        )

        logger.info(
            "Diarization turns: %d",
            len(turns),
        )

        # ====================================================
        # 2. Isolation + TSE
        # ====================================================

        logger.info(
            "[2/4] Speaker isolation + "
            "overlap separation"
        )

        isolation_dir = (
            output_dir
            / "speakers"
        )

        isolation = (
            self.isolation.isolate(
                audio_path,
                diarization,
                isolation_dir,
            )
        )

        isolation_enabled = bool(
            getattr(
                self.isolation,
                "enabled",
                True,
            )
        )

        tse_enabled = bool(
            getattr(
                self.overlap,
                "enabled",
                False,
            )
        )

        speaker_audio = {}

        isolation_sample_rate = (
            self.target_sample_rate
        )

        if isolation_enabled:

            speaker_audio = (
                self.isolation
                .get_all_speaker_audio()
            )

            isolation_sample_rate = int(
                isolation.get(
                    "audio",
                    {},
                ).get(
                    "sample_rate",
                    self.target_sample_rate,
                )
            )

        else:

            (
                self._original_waveform,
                self._original_sample_rate,
            ) = (
                self._load_original_audio(
                    audio_path
                )
            )

        # ====================================================
        # 3. Build ASR groups
        # ====================================================

        logger.info(
            "[3/4] Building ASR groups"
        )

        candidate_groups = (
            self.grouping.build_groups(
                turns
            )
        )

        # ----------------------------------------------------
        # Determine audio source BEFORE validation.
        #
        # This is critical:
        # validation must inspect the actual waveform that
        # Whisper will receive.
        # ----------------------------------------------------

        if isolation_enabled:

            validation_waveforms = (
                speaker_audio
            )

            validation_sample_rate = (
                isolation_sample_rate
            )

        else:

            validation_waveforms = {
                "__original__": (
                    self._original_waveform
                )
            }

            validation_sample_rate = (
                self._original_sample_rate
            )

        # ----------------------------------------------------
        # Validate each group against its real audio source.
        # ----------------------------------------------------

        valid_groups = []

        for group in candidate_groups:

            speaker = group[
                "speaker"
            ]

            if isolation_enabled:

                waveform = (
                    validation_waveforms.get(
                        speaker
                    )
                )

            else:

                waveform = (
                    validation_waveforms.get(
                        "__original__"
                    )
                )

            if waveform is None:

                logger.warning(
                    "Rejecting ASR group %03d: "
                    "no waveform for speaker %s",
                    int(
                        group.get(
                            "group_index",
                            -1,
                        )
                    ),
                    speaker,
                )

                self.grouping.last_stats[
                    "groups_rejected"
                ] += 1

                continue

            validated = (
                self.grouping
                .validate_groups_against_audio(
                    [group],
                    waveform,
                    validation_sample_rate,
                )
            )

            if validated:

                valid_groups.append(
                    validated[0]
                )

        # ----------------------------------------------------
        # Re-index ONLY valid groups.
        #
        # This eliminates:
        #
        # group_036.wav
        # group_037.wav   <- rejected
        # group_038.wav
        #
        # and gives:
        #
        # group_036.wav
        # group_037.wav
        # group_038.wav
        # ----------------------------------------------------

        for index, group in enumerate(
            valid_groups
        ):

            group[
                "group_index"
            ] = index

        asr_groups = valid_groups

        self.grouping.last_stats[
            "groups_output"
        ] = len(asr_groups)

        self._log_grouping_stats(
            turns,
            asr_groups,
        )

        logger.info(
            "Diarization turns: %d -> "
            "ASR groups: %d",
            len(turns),
            len(asr_groups),
        )

        conversation_turns = []

        # ====================================================
        # Transcribe groups
        # ====================================================

        for group in asr_groups:

            speaker = group[
                "speaker"
            ]

            start = float(
                group["start"]
            )

            end = float(
                group["end"]
            )

            duration = float(
                group["duration"]
            )

            source_indices = list(
                group[
                    "source_turn_indices"
                ]
            )

            ignored_indices = list(
                group.get(
                    "ignored_turn_indices",
                    [],
                )
            )

            logger.info(
                "ASR group %03d: %s "
                "[%.3f -> %.3f] "
                "(%.3fs) | "
                "source=%s | "
                "ignored=%s",
                group["group_index"],
                speaker,
                start,
                end,
                duration,
                source_indices,
                ignored_indices,
            )

            text = ""
            debug_path = None
            actual_audio_duration = 0.0
            audio_rms = 0.0
            audio_active_ratio = 0.0

            # ------------------------------------------------
            # Speaker isolated mode
            # ------------------------------------------------

            if isolation_enabled:

                speaker_waveform = (
                    speaker_audio.get(
                        speaker
                    )
                )

                if (
                    speaker_waveform
                    is None
                ):

                    logger.warning(
                        "No isolated timeline "
                        "for speaker %s",
                        speaker,
                    )

                else:

                    (
                        text,
                        debug_path,
                        audio,
                    ) = (
                        self._transcribe_group(
                            group,
                            speaker_waveform,
                            isolation_sample_rate,
                        )
                    )

                    if audio.size:

                        actual_audio_duration = (
                            audio.size
                            / isolation_sample_rate
                        )

                        audio_rms = float(
                            np.sqrt(
                                np.mean(
                                    np.square(
                                        audio.astype(
                                            np.float64
                                        )
                                    )
                                )
                            )
                        )

                        audio_active_ratio = float(
                            np.mean(
                                np.abs(audio)
                                > max(
                                    1e-6,
                                    float(
                                        np.max(
                                            np.abs(
                                                audio
                                            )
                                        )
                                    )
                                    * 0.05,
                                )
                            )
                        )

            # ------------------------------------------------
            # Original audio mode
            # ------------------------------------------------

            else:

                if (
                    self._original_waveform
                    is None
                ):

                    logger.error(
                        "Original ASR waveform "
                        "is unavailable."
                    )

                else:

                    (
                        text,
                        debug_path,
                        audio,
                    ) = (
                        self._transcribe_group(
                            group,
                            self._original_waveform,
                            self._original_sample_rate,
                        )
                    )

                    if audio.size:

                        actual_audio_duration = (
                            audio.size
                            / self._original_sample_rate
                        )

                        audio_rms = float(
                            np.sqrt(
                                np.mean(
                                    np.square(
                                        audio.astype(
                                            np.float64
                                        )
                                    )
                                )
                            )
                        )

                        audio_active_ratio = float(
                            np.mean(
                                np.abs(audio)
                                > max(
                                    1e-6,
                                    float(
                                        np.max(
                                            np.abs(
                                                audio
                                            )
                                        )
                                    )
                                    * 0.05,
                                )
                            )
                        )

            conversation_turns.append(
                {
                    "group_index": group[
                        "group_index"
                    ],

                    "speaker": speaker,

                    "start": round(
                        start,
                        3,
                    ),

                    "end": round(
                        end,
                        3,
                    ),

                    "duration": round(
                        duration,
                        3,
                    ),

                    "actual_audio_duration": round(
                        actual_audio_duration,
                        3,
                    ),

                    "audio_rms": round(
                        audio_rms,
                        7,
                    ),

                    "audio_active_ratio": round(
                        audio_active_ratio,
                        5,
                    ),

                    "text": text,

                    "source_turn_indices": (
                        source_indices
                    ),

                    "ignored_turn_indices": (
                        ignored_indices
                    ),

                    "source_turns": [
                        {
                            "speaker": item[
                                "speaker"
                            ],

                            "start": round(
                                float(
                                    item[
                                        "start"
                                    ]
                                ),
                                3,
                            ),

                            "end": round(
                                float(
                                    item[
                                        "end"
                                    ]
                                ),
                                3,
                            ),

                            "duration": round(
                                float(
                                    item[
                                        "duration"
                                    ]
                                ),
                                3,
                            ),
                        }
                        for item
                        in group[
                            "source_turns"
                        ]
                    ],

                    "ignored_turns": [
                        {
                            "speaker": item[
                                "speaker"
                            ],

                            "start": round(
                                float(
                                    item[
                                        "start"
                                    ]
                                ),
                                3,
                            ),

                            "end": round(
                                float(
                                    item[
                                        "end"
                                    ]
                                ),
                                3,
                            ),

                            "duration": round(
                                float(
                                    item[
                                        "duration"
                                    ]
                                ),
                                3,
                            ),
                        }
                        for item
                        in group.get(
                            "ignored_turns",
                            [],
                        )
                    ],

                    "debug_audio": (
                        str(
                            debug_path
                        )
                        if debug_path
                        else None
                    ),
                }
            )

        # ====================================================
        # Speaker transcription
        # ====================================================

        speaker_transcriptions = {}

        for item in conversation_turns:

            speaker = item[
                "speaker"
            ]

            speaker_transcriptions.setdefault(
                speaker,
                [],
            ).append(
                {
                    "group_index": item[
                        "group_index"
                    ],

                    "start": item[
                        "start"
                    ],

                    "end": item[
                        "end"
                    ],

                    "duration": item[
                        "duration"
                    ],

                    "text": item[
                        "text"
                    ],

                    "source_turn_indices": item[
                        "source_turn_indices"
                    ],
                }
            )

        # ====================================================
        # 4. Result
        # ====================================================

        logger.info(
            "[4/4] Building final meeting JSON"
        )

        total_time = (
            time.perf_counter()
            - file_start
        )

        if isolation_enabled:

            transcription_mode = (
                "speaker_isolated_grouped_asr"
            )

            asr_source = (
                "isolated_speaker_timeline"
            )

            isolation_strategy = (
                "speaker_timeline"
            )

        else:

            transcription_mode = (
                "diarized_original_audio_grouped_asr"
            )

            asr_source = (
                "original_normalized_audio"
            )

            isolation_strategy = (
                "disabled_original_audio"
            )

        if tse_enabled:

            overlap_strategy = (
                "REAL-T_target_speaker_extraction"
            )

            tse_source = (
                "overlap_regions_only"
            )

        else:

            overlap_strategy = (
                "disabled"
            )

            tse_source = (
                "none"
            )

        result = {

            "project": {
                "name": self.config.get(
                    "project",
                    {},
                ).get(
                    "name",
                    "persian_asr",
                ),
            },

            "file": {
                "name": audio_path.name,
                "path": str(
                    audio_path
                ),
            },

            "diarization": diarization,

            "speaker_isolation": isolation,

            "transcription": {

                "turns": conversation_turns,

                "raw_turns": [
                    {
                        "turn_index": index,

                        "speaker": turn[
                            "speaker"
                        ],

                        "start": round(
                            float(
                                turn[
                                    "start"
                                ]
                            ),
                            3,
                        ),

                        "end": round(
                            float(
                                turn[
                                    "end"
                                ]
                            ),
                            3,
                        ),

                        "duration": round(
                            float(
                                turn[
                                    "duration"
                                ]
                            ),
                            3,
                        ),
                    }
                    for index, turn
                    in enumerate(
                        turns
                    )
                ],

                "speakers": (
                    speaker_transcriptions
                ),

                "grouping": {

                    "enabled": (
                        self.grouping.enabled
                    ),

                    "max_gap_seconds": (
                        self.grouping
                        .max_gap_seconds
                    ),

                    "max_chunk_seconds": (
                        self.grouping
                        .max_chunk_seconds
                    ),

                    "min_group_seconds": (
                        self.grouping
                        .min_group_seconds
                    ),

                    "preserve_gaps": (
                        self.grouping
                        .preserve_gaps
                    ),

                    "short_turn_filter": {

                        "enabled": (
                            self.grouping
                            .short_turn_filter_enabled
                        ),

                        "ignore_below_seconds": (
                            self.grouping
                            .ignore_below_seconds
                        ),

                        "bridge_same_speaker": (
                            self.grouping
                            .bridge_same_speaker
                        ),

                        "bridge_max_gap_seconds": (
                            self.grouping
                            .bridge_max_gap_seconds
                        ),
                    },

                    "overlap_filter": {

                        "enabled": (
                            self.grouping
                            .overlap_filter_enabled
                        ),

                        "ignore_below_seconds": (
                            self.grouping
                            .ignore_overlap_below_seconds
                        ),

                        "do_not_break_group": (
                            self.grouping
                            .do_not_break_on_short_overlap
                        ),
                    },

                    "noise_filter": {

                        "enabled": (
                            self.grouping
                            .noise_filter_enabled
                        ),

                        "max_noise_turn_seconds": (
                            self.grouping
                            .max_noise_turn_seconds
                        ),

                        "min_rms": (
                            self.grouping
                            .min_rms
                        ),

                        "min_active_ratio": (
                            self.grouping
                            .min_active_ratio
                        ),
                    },

                    "audio_validation": {

                        "enabled": (
                            self.grouping
                            .audio_validation_enabled
                        ),

                        "min_audio_seconds": (
                            self.grouping
                            .min_audio_seconds
                        ),

                        "min_active_ratio": (
                            self.grouping
                            .validation_min_active_ratio
                        ),

                        "reject_silence_only": (
                            self.grouping
                            .reject_silence_only
                        ),

                        "reject_noise_only": (
                            self.grouping
                            .reject_noise_only
                        ),
                    },

                    "group_count": (
                        len(asr_groups)
                    ),

                    "raw_turn_count": (
                        len(turns)
                    ),

                    "ignored_short_turns": int(
                        self.grouping
                        .last_stats
                        .get(
                            "ignored_short_turns",
                            0,
                        )
                    ),

                    "rejected_groups": int(
                        self.grouping
                        .last_stats
                        .get(
                            "groups_rejected",
                            0,
                        )
                    ),
                },
            },

            "processing": {

                "total_seconds": round(
                    total_time,
                    3,
                ),

                "transcription_mode": (
                    transcription_mode
                ),

                "isolation_strategy": (
                    isolation_strategy
                ),

                "overlap_strategy": (
                    overlap_strategy
                ),

                "asr_source": (
                    asr_source
                ),

                "tse_source": (
                    tse_source
                ),

                "speaker_isolation_enabled": (
                    isolation_enabled
                ),

                "overlap_separation_enabled": (
                    tse_enabled
                ),

                "asr_grouping": (
                    self.grouping.enabled
                ),
            },
        }

        # ====================================================
        # JSON
        # ====================================================

        json_path = (
            output_dir
            / f"{audio_path.stem}_meeting.json"
        )

        json_path.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        # ====================================================
        # TXT
        # ====================================================

        txt_lines = []

        for item in conversation_turns:

            text = str(
                item.get(
                    "text",
                    "",
                )
            ).strip()

            if not text:
                continue

            txt_lines.append(
                "[{} -> {}] {}: {}".format(
                    item["start"],
                    item["end"],
                    item["speaker"],
                    text,
                )
            )

        txt_path = (
            output_dir
            / f"{audio_path.stem}_meeting.txt"
        )

        txt_path.write_text(
            "\n".join(
                txt_lines
            )
            + (
                "\n"
                if txt_lines
                else ""
            ),
            encoding="utf-8",
        )

        logger.info(
            "Meeting JSON: %s",
            json_path,
        )

        logger.info(
            "Meeting TXT : %s",
            txt_path,
        )

        logger.info(
            "Raw diarization turns: %d",
            len(turns),
        )

        logger.info(
            "Ignored short turns: %d",
            int(
                self.grouping
                .last_stats
                .get(
                    "ignored_short_turns",
                    0,
                )
            ),
        )

        logger.info(
            "Rejected groups: %d",
            int(
                self.grouping
                .last_stats
                .get(
                    "groups_rejected",
                    0,
                )
            ),
        )

        logger.info(
            "Valid ASR groups: %d",
            len(
                conversation_turns
            ),
        )

        logger.info(
            "Transcribed groups: %d",
            sum(
                bool(
                    item.get(
                        "text",
                        "",
                    ).strip()
                )
                for item
                in conversation_turns
            ),
        )

        logger.info(
            "Total processing time: %.3f sec",
            total_time,
        )

        return result

    # ========================================================
    # Unload
    # ========================================================

    def unload(
        self,
    ) -> None:

        logger.info(
            "Unloading meeting pipeline..."
        )

        self._original_waveform = None

        self.isolation.clear_audio_buffers()

        self.diarization.unload()
        self.overlap.unload()

        logger.info(
            "Meeting pipeline unloaded."
        )