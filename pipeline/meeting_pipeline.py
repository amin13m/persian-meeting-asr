from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from engines.audio_chunker import AudioChunker
from engines.diarization.engine import DiarizationEngine
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

    # ========================================================
    # Load
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
        """
        Normalize a diarization turn.

        Expected structure:

        {
            "speaker": "SPEAKER_00",
            "start": 1.234,
            "end": 4.567
        }

        Returns None for invalid/empty turns.
        """

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

        if not speaker:
            return None

        if end <= start:
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
        """
        Extract chronological speaker turns from
        the diarization result.

        The diarization engine may expose turns using
        slightly different container keys. This helper
        keeps the pipeline tolerant of those structures.
        """

        raw_turns = []

        # ----------------------------------------------------
        # Normal expected structure
        # ----------------------------------------------------

        if isinstance(
            diarization.get("turns"),
            list,
        ):
            raw_turns = diarization["turns"]

        # ----------------------------------------------------
        # Fallback structures
        # ----------------------------------------------------

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

        normalized = []

        for item in raw_turns:

            if not isinstance(
                item,
                dict,
            ):
                continue

            turn = self._normalize_turn(
                item
            )

            if turn is not None:
                normalized.append(
                    turn
                )

        # ----------------------------------------------------
        # Always chronological
        # ----------------------------------------------------

        normalized.sort(
            key=lambda x: (
                x["start"],
                x["end"],
            )
        )

        return normalized

    def _transcribe_turn(
        self,
        audio_path: Path,
        start: float,
        end: float,
    ) -> str:
        """
        Extract one original-time audio segment and
        send only that segment to Whisper.

        IMPORTANT:
        The returned text belongs to [start, end]
        in the ORIGINAL meeting timeline.
        """

        duration = end - start

        if duration <= 0:
            return ""

        logger.debug(
            "Extracting ASR segment: %.3f -> %.3f",
            start,
            end,
        )

        # ----------------------------------------------------
        # AudioChunker works on the original file and
        # returns an audio chunk with its own audio data.
        #
        # We use a dedicated chunk window here so the
        # Whisper timestamps do not replace the original
        # meeting timestamps.
        # ----------------------------------------------------

        chunks = self.chunker.chunk_file(
            audio_path
        )

        target_chunk = None

        for chunk in chunks:

            chunk_start = float(
                chunk.start
            )

            chunk_end = float(
                chunk.end
            )

            # Find a chunk containing the requested
            # diarization interval.
            if (
                chunk_start <= start
                and chunk_end >= end
            ):
                target_chunk = chunk
                break

        if target_chunk is None:

            logger.warning(
                "Could not find audio chunk for "
                "%.3f -> %.3f",
                start,
                end,
            )

            return ""

        # ----------------------------------------------------
        # If the chunk is larger than the diarization turn,
        # extract the exact sub-window from the chunk.
        #
        # This keeps Whisper focused on the actual speaker
        # turn instead of transcribing neighbouring speech.
        # ----------------------------------------------------

        relative_start = max(
            0.0,
            start
            - float(
                target_chunk.start
            ),
        )

        relative_end = min(
            float(
                target_chunk.duration
            ),
            end
            - float(
                target_chunk.start
            ),
        )

        if relative_end <= relative_start:
            return ""

        audio = target_chunk.audio

        try:
            import numpy as np

            sample_rate = int(
                self.chunker.sample_rate
            )

            sample_start = int(
                round(
                    relative_start
                    * sample_rate
                )
            )

            sample_end = int(
                round(
                    relative_end
                    * sample_rate
                )
            )

            if sample_start < 0:
                sample_start = 0

            if sample_end > len(audio):
                sample_end = len(audio)

            if sample_end <= sample_start:
                return ""

            segment_audio = audio[
                sample_start:sample_end
            ]

            # Make sure Whisper receives a numpy
            # array in the expected form.
            if not isinstance(
                segment_audio,
                np.ndarray,
            ):
                segment_audio = np.asarray(
                    segment_audio
                )

            result = (
                self.asr_engine.transcribe(
                    segment_audio,
                    sample_rate=sample_rate,
                )
            )

        except Exception:

            logger.exception(
                "ASR failed for segment "
                "%.3f -> %.3f",
                start,
                end,
            )

            return ""

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

        file_start = time.perf_counter()

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
        # 2. Speaker isolation
        # ====================================================

        logger.info(
            "[2/4] Speaker isolation"
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

        # ====================================================
        # 3. ASR
        # ====================================================

        logger.info(
            "[3/4] Turn-level ASR"
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Do NOT transcribe:
        #
        #   SPEAKER_00.wav
        #   SPEAKER_01.wav
        #
        # as one large file.
        #
        # That destroys the original conversation timeline.
        #
        # Instead:
        #
        #   diarization turn
        #          ↓
        #   original audio [start -> end]
        #          ↓
        #   Whisper
        #          ↓
        #   preserve original start/end
        # ----------------------------------------------------

        conversation_turns = []

        for index, turn in enumerate(
            turns
        ):

            speaker = turn[
                "speaker"
            ]

            start = float(
                turn["start"]
            )

            end = float(
                turn["end"]
            )

            duration = float(
                turn["duration"]
            )

            logger.info(
                "Transcribing turn %d: "
                "%s [%.3f -> %.3f]",
                index,
                speaker,
                start,
                end,
            )

            text = (
                self._transcribe_turn(
                    audio_path,
                    start,
                    end,
                )
            )

            item = {
                "turn_index": index,

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
            }

            # ------------------------------------------------
            # Keep the turn even if Whisper returns no text.
            #
            # This preserves the diarization timeline.
            # ------------------------------------------------

            conversation_turns.append(
                item
            )

        # ----------------------------------------------------
        # Final chronological ordering
        # ----------------------------------------------------

        conversation_turns.sort(
            key=lambda x: (
                x["start"],
                x["end"],
                x["turn_index"],
            )
        )

        # ====================================================
        # Build speaker-oriented transcription
        # ====================================================

        speaker_transcriptions = {}

        for item in conversation_turns:

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
                    "turn_index": item[
                        "turn_index"
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
                }
            )

        # ====================================================
        # 4. Build meeting result
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

                # --------------------------------------------
                # Primary representation:
                # actual chronological conversation
                # --------------------------------------------

                "turns": (
                    conversation_turns
                ),

                # --------------------------------------------
                # Secondary representation:
                # grouped by speaker, but with ORIGINAL
                # timestamps preserved.
                # --------------------------------------------

                "speakers": (
                    speaker_transcriptions
                ),

            },

            "processing": {
                "total_seconds": round(
                    total_time,
                    3,
                ),

                "transcription_mode": (
                    "turn_level"
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

        # IMPORTANT:
        #
        # Iterate over conversation_turns,
        # NOT speaker_transcriptions.
        #
        # Therefore the TXT follows the real
        # chronological conversation.
        # ====================================================

        txt_lines = []

        for item in conversation_turns:

            text = str(
                item.get(
                    "text",
                    "",
                )
            ).strip()

            # Do not write empty Whisper results
            # into the human-readable transcript.
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

        # ====================================================
        # Logs
        # ====================================================

        logger.info(
            "Meeting JSON: %s",
            json_path,
        )

        logger.info(
            "Meeting TXT : %s",
            txt_path,
        )

        logger.info(
            "Conversation turns: %d",
            len(
                conversation_turns
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

    def unload(self) -> None:

        logger.info(
            "Unloading meeting pipeline..."
        )

        self.diarization.unload()
        self.overlap.unload()

        logger.info(
            "Meeting pipeline unloaded."
        )