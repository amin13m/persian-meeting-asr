from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from engines.audio_chunker import AudioChunker
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

        self.chunker = AudioChunker(
            config
        )

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

        # ----------------------------------------------------
        # ASR grouping
        # ----------------------------------------------------

        grouping = asr_config.get(
            "grouping",
            {},
        )

        self.grouping_enabled = bool(
            grouping.get(
                "enabled",
                True,
            )
        )

        self.group_max_gap_seconds = float(
            grouping.get(
                "max_gap_seconds",
                1.50,
            )
        )

        self.group_max_chunk_seconds = float(
            grouping.get(
                "max_chunk_seconds",
                self.max_turn_seconds,
            )
        )

        self.group_min_seconds = float(
            grouping.get(
                "min_group_seconds",
                0.20,
            )
        )

        self.group_preserve_gaps = bool(
            grouping.get(
                "preserve_gaps",
                True,
            )
        )

        # ----------------------------------------------------
        # Debug ASR
        # ----------------------------------------------------

        debug_cfg = (
            config.get("debug", {})
            .get("audio", {})
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

        self.debug_root = Path(
            debug_cfg.get(
                "root_dir",
                "output/debug_audio",
            )
        )

    # ========================================================
    # Lifecycle
    # ========================================================

    def load(self) -> None:

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

        start = (
            MeetingPipeline
            ._safe_float(
                turn.get(
                    "start",
                    0.0,
                )
            )
        )

        end = (
            MeetingPipeline
            ._safe_float(
                turn.get(
                    "end",
                    0.0,
                )
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

            raw_turns = diarization[
                "turns"
            ]

        elif isinstance(
            diarization.get("segments"),
            list,
        ):

            raw_turns = diarization[
                "segments"
            ]

        elif isinstance(
            diarization.get("diarization"),
            list,
        ):

            raw_turns = diarization[
                "diarization"
            ]

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
    # ASR grouping
    # ========================================================

    def _build_asr_groups(
        self,
        turns: list[dict],
    ) -> list[dict]:
        """
        Convert many tiny consecutive turns of the same
        speaker into larger ASR chunks.

        Example:

            SPEAKER_00  1.1 -> 19.0
            SPEAKER_00 19.9 -> 20.6
            SPEAKER_00 20.9 -> 21.4

        becomes:

            SPEAKER_00  1.1 -> 21.4

        if the gaps and maximum chunk duration permit it.

        Original turn indices are preserved.
        """

        if not turns:
            return []

        if not self.grouping_enabled:

            return [
                {
                    "group_index": index,
                    "speaker": turn["speaker"],
                    "start": turn["start"],
                    "end": turn["end"],
                    "duration": turn["duration"],
                    "source_turn_indices": [
                        index
                    ],
                    "source_turns": [
                        dict(turn)
                    ],
                }
                for index, turn
                in enumerate(turns)
            ]

        groups = []

        current = None

        for index, turn in enumerate(
            turns
        ):

            if current is None:

                current = {
                    "group_index": len(
                        groups
                    ),
                    "speaker": turn[
                        "speaker"
                    ],
                    "start": turn[
                        "start"
                    ],
                    "end": turn[
                        "end"
                    ],
                    "duration": turn[
                        "duration"
                    ],
                    "source_turn_indices": [
                        index
                    ],
                    "source_turns": [
                        dict(turn)
                    ],
                }

                continue

            same_speaker = (
                turn["speaker"]
                == current["speaker"]
            )

            gap = (
                turn["start"]
                - current["end"]
            )

            candidate_duration = (
                turn["end"]
                - current["start"]
            )

            can_merge = (
                same_speaker
                and gap >= 0.0
                and gap
                <= self.group_max_gap_seconds
                and candidate_duration
                <= self.group_max_chunk_seconds
            )

            if can_merge:

                current["end"] = turn[
                    "end"
                ]

                current["duration"] = (
                    current["end"]
                    - current["start"]
                )

                current[
                    "source_turn_indices"
                ].append(index)

                current[
                    "source_turns"
                ].append(
                    dict(turn)
                )

            else:

                groups.append(
                    current
                )

                current = {
                    "group_index": len(
                        groups
                    ),
                    "speaker": turn[
                        "speaker"
                    ],
                    "start": turn[
                        "start"
                    ],
                    "end": turn[
                        "end"
                    ],
                    "duration": turn[
                        "duration"
                    ],
                    "source_turn_indices": [
                        index
                    ],
                    "source_turns": [
                        dict(turn)
                    ],
                }

        if current is not None:
            groups.append(
                current
            )

        return groups

    # ========================================================
    # Audio
    # ========================================================

    @staticmethod
    def _extract_segment(
        waveform: np.ndarray,
        sample_rate: int,
        start: float,
        end: float,
    ) -> np.ndarray:

        if end <= start:
            return np.zeros(
                0,
                dtype=np.float32,
            )

        start_sample = int(
            round(
                start
                * sample_rate
            )
        )

        end_sample = int(
            round(
                end
                * sample_rate
            )
        )

        start_sample = max(
            0,
            start_sample,
        )

        end_sample = min(
            len(waveform),
            end_sample,
        )

        if end_sample <= start_sample:
            return np.zeros(
                0,
                dtype=np.float32,
            )

        return np.ascontiguousarray(
            waveform[
                start_sample:end_sample
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _is_silent(
        audio: np.ndarray,
    ) -> bool:

        if audio.size == 0:
            return True

        return float(
            np.max(
                np.abs(audio)
            )
        ) < 1e-5

    # ========================================================
    # Debug ASR audio
    # ========================================================

    @staticmethod
    def _safe_name(
        value: str,
    ) -> str:

        result = str(value)

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

        sf.write(
            str(path),
            audio,
            sample_rate,
        )

        logger.debug(
            "Saved ASR debug chunk: %s",
            path,
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

        duration = end - start

        if duration <= 0:
            return ""

        if (
            duration
            < self.group_min_seconds
        ):

            return ""

        if audio.size == 0:
            return ""

        if self._is_silent(
            audio
        ):

            logger.debug(
                "Skipping silent ASR group: "
                "%.3f -> %.3f",
                start,
                end,
            )

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
        speaker_waveform: np.ndarray,
        sample_rate: int,
    ) -> tuple[str, Path | None]:

        start = float(
            group["start"]
        )

        end = float(
            group["end"]
        )

        # ----------------------------------------------------
        # Timeline-preserving extraction.
        #
        # Because speaker_waveform contains only this speaker,
        # any speech belonging to other speakers is zero.
        # ----------------------------------------------------

        audio = self._extract_segment(
            speaker_waveform,
            sample_rate,
            start,
            end,
        )

        if audio.size == 0:
            return "", None

        debug_path = (
            self._save_asr_chunk(
                group,
                audio,
                sample_rate,
            )
        )

        text = self._transcribe_audio(
            audio,
            sample_rate,
            start,
            end,
        )

        return text, debug_path

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
        # 2. Speaker isolation + TSE
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
                16000,
            )
        )

        logger.info(
            "Speaker timelines available: %d",
            len(speaker_audio),
        )

        # ====================================================
        # 3. Build ASR groups
        # ====================================================

        logger.info(
            "[3/4] Building ASR groups"
        )

        asr_groups = (
            self._build_asr_groups(
                turns
            )
        )

        logger.info(
            "Diarization turns: %d -> "
            "ASR groups: %d",
            len(turns),
            len(asr_groups),
        )

        conversation_turns = []

        # ----------------------------------------------------
        # ASR groups
        # ----------------------------------------------------

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

            logger.info(
                "ASR group %d: %s "
                "[%.3f -> %.3f] "
                "(%.3fs) | source turns=%s",
                group["group_index"],
                speaker,
                start,
                end,
                duration,
                source_indices,
            )

            speaker_waveform = (
                speaker_audio.get(
                    speaker
                )
            )

            debug_path = None

            if speaker_waveform is None:

                logger.warning(
                    "No isolated timeline "
                    "for speaker %s",
                    speaker,
                )

                text = ""

            else:

                text, debug_path = (
                    self._transcribe_group(
                        group,
                        speaker_waveform,
                        isolation_sample_rate,
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

                    "text": text,

                    "source_turn_indices": (
                        source_indices
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
                        for item in group[
                            "source_turns"
                        ]
                    ],

                    "debug_audio": (
                        str(debug_path)
                        if debug_path
                        else None
                    ),
                }
            )

        # ====================================================
        # Speaker transcriptions
        # ====================================================

        speaker_transcriptions = {}

        for item in (
            conversation_turns
        ):

            speaker = item[
                "speaker"
            ]

            if speaker not in (
                speaker_transcriptions
            ):

                speaker_transcriptions[
                    speaker
                ] = []

            speaker_transcriptions[
                speaker
            ].append(
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

                    "source_turn_indices": (
                        item[
                            "source_turn_indices"
                        ]
                    ),
                }
            )

        # ====================================================
        # 4. Final JSON
        # ====================================================

        logger.info(
            "[4/4] Building final meeting JSON"
        )

        total_time = (
            time.perf_counter()
            - file_start
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

                # New grouped ASR output.
                "turns": conversation_turns,

                # Original diarization turns.
                "raw_turns": [
                    {
                        "turn_index": index,
                        "speaker": turn[
                            "speaker"
                        ],
                        "start": round(
                            float(
                                turn["start"]
                            ),
                            3,
                        ),
                        "end": round(
                            float(
                                turn["end"]
                            ),
                            3,
                        ),
                        "duration": round(
                            float(
                                turn["duration"]
                            ),
                            3,
                        ),
                    }
                    for index, turn
                    in enumerate(turns)
                ],

                "speakers": (
                    speaker_transcriptions
                ),

                "grouping": {
                    "enabled": (
                        self.grouping_enabled
                    ),
                    "max_gap_seconds": (
                        self.group_max_gap_seconds
                    ),
                    "max_chunk_seconds": (
                        self.group_max_chunk_seconds
                    ),
                    "group_count": len(
                        asr_groups
                    ),
                    "raw_turn_count": len(
                        turns
                    ),
                },
            },

            "processing": {

                "total_seconds": round(
                    total_time,
                    3,
                ),

                "transcription_mode": (
                    "speaker_isolated_grouped_asr"
                ),

                "overlap_strategy": (
                    "REAL-T_target_speaker_extraction"
                ),

                "asr_source": (
                    "isolated_speaker_timeline"
                ),

                "tse_source": (
                    "overlap_regions_only"
                ),

                "asr_grouping": (
                    self.grouping_enabled
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

        for item in (
            conversation_turns
        ):

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
            "ASR groups: %d",
            len(conversation_turns),
        )

        logger.info(
            "Total processing time: %.3f sec",
            total_time,
        )

        return result

    # ========================================================
    # Unload
    # ========================================================

    def unload(self) -> None:

        logger.info(
            "Unloading meeting pipeline..."
        )

        self.isolation.clear_audio_buffers()

        self.diarization.unload()
        self.overlap.unload()

        logger.info(
            "Meeting pipeline unloaded."
        )