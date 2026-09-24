from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


@dataclass
class AudioChunk:
    """
    One processable audio chunk.
    """

    index: int

    start: float
    end: float

    audio: np.ndarray

    duration: float

    source_file: str


class AudioChunker:
    """
    Silence-aware audio chunker.

    Responsibilities:

        1. Load audio
        2. Convert to mono
        3. Resample
        4. Detect speech regions
        5. Respect silence boundaries
        6. Prevent oversized chunks
        7. Add boundary padding

    It does NOT perform transcription.
    """

    def __init__(
        self,
        config: dict,
    ):

        self.sample_rate = int(
            config["audio"].get(
                "sample_rate",
                16000,
            )
        )

        chunk_config = config.get(
            "chunking",
            {},
        )

        self.enabled = chunk_config.get(
            "enabled",
            True,
        )

        self.max_chunk_seconds = float(
            chunk_config.get(
                "max_chunk_seconds",
                30,
            )
        )

        self.min_chunk_seconds = float(
            chunk_config.get(
                "min_chunk_seconds",
                0.5,
            )
        )

        self.silence_top_db = float(
            chunk_config.get(
                "silence_top_db",
                30,
            )
        )

        self.min_silence_seconds = float(
            chunk_config.get(
                "min_silence_seconds",
                0.35,
            )
        )

        self.padding_before = float(
            chunk_config.get(
                "padding_before_seconds",
                0.20,
            )
        )

        self.padding_after = float(
            chunk_config.get(
                "padding_after_seconds",
                0.20,
            )
        )

    # ========================================================
    # Load
    # ========================================================

    def load_audio(
        self,
        audio_path: Path,
    ):

        audio_path = Path(
            audio_path
        )

        audio, sample_rate = sf.read(
            str(audio_path),
            always_2d=False,
        )

        # ----------------------------------------------------
        # Mono
        # ----------------------------------------------------

        if audio.ndim > 1:

            audio = audio.mean(
                axis=1
            )

        audio = audio.astype(
            "float32"
        )

        # ----------------------------------------------------
        # Resample
        # ----------------------------------------------------

        if sample_rate != self.sample_rate:

            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=self.sample_rate,
            )

        return audio

    # ========================================================
    # Detect Speech
    # ========================================================

    def detect_speech_regions(
        self,
        audio: np.ndarray,
    ):
        """
        Detect non-silent regions.

        Returns:

            [
                (start_sample, end_sample),
                ...
            ]
        """

        if len(audio) == 0:

            return []

        intervals = librosa.effects.split(
            audio,
            top_db=self.silence_top_db,
        )

        if len(intervals) == 0:

            return []

        # ----------------------------------------------------
        # Merge regions separated by very short silence
        # ----------------------------------------------------

        merged = []

        max_gap = int(
            self.min_silence_seconds
            * self.sample_rate
        )

        current_start = int(
            intervals[0][0]
        )

        current_end = int(
            intervals[0][1]
        )

        for start, end in intervals[1:]:

            start = int(start)
            end = int(end)

            gap = start - current_end

            if gap <= max_gap:

                current_end = end

            else:

                merged.append(
                    (
                        current_start,
                        current_end,
                    )
                )

                current_start = start
                current_end = end

        merged.append(
            (
                current_start,
                current_end,
            )
        )

        return merged

    # ========================================================
    # Build Chunks
    # ========================================================

    def build_chunks(
        self,
        audio: np.ndarray,
        speech_regions,
        source_file: str,
    ):

        total_samples = len(audio)

        max_samples = int(
            self.max_chunk_seconds
            * self.sample_rate
        )

        min_samples = int(
            self.min_chunk_seconds
            * self.sample_rate
        )

        padding_before = int(
            self.padding_before
            * self.sample_rate
        )

        padding_after = int(
            self.padding_after
            * self.sample_rate
        )

        chunks = []

        # ----------------------------------------------------
        # No speech detected
        # ----------------------------------------------------

        if not speech_regions:

            return chunks

        # ----------------------------------------------------
        # Add padding
        # ----------------------------------------------------

        padded_regions = []

        for start, end in speech_regions:

            start = max(
                0,
                start - padding_before,
            )

            end = min(
                total_samples,
                end + padding_after,
            )

            padded_regions.append(
                (
                    start,
                    end,
                )
            )

        # ----------------------------------------------------
        # Build chunks
        # ----------------------------------------------------

        chunk_index = 1

        current_start = None
        current_end = None

        for region_start, region_end in padded_regions:

            # ------------------------------------------------
            # Ignore extremely tiny regions
            # ------------------------------------------------

            if (
                region_end - region_start
                < min_samples
            ):
                continue

            # ------------------------------------------------
            # Start first chunk
            # ------------------------------------------------

            if current_start is None:

                current_start = region_start
                current_end = region_end

                continue

            candidate_end = region_end

            candidate_duration = (
                candidate_end
                - current_start
            )

            # ------------------------------------------------
            # Region fits
            # ------------------------------------------------

            if candidate_duration <= max_samples:

                current_end = candidate_end

                continue

            # ------------------------------------------------
            # Current chunk is full
            # ------------------------------------------------

            chunk_audio = audio[
                current_start:current_end
            ]

            chunks.append(
                AudioChunk(
                    index=chunk_index,
                    start=(
                        current_start
                        / self.sample_rate
                    ),
                    end=(
                        current_end
                        / self.sample_rate
                    ),
                    audio=chunk_audio,
                    duration=(
                        len(chunk_audio)
                        / self.sample_rate
                    ),
                    source_file=source_file,
                )
            )

            chunk_index += 1

            # ------------------------------------------------
            # Start new chunk
            # ------------------------------------------------

            current_start = region_start
            current_end = region_end

        # ----------------------------------------------------
        # Final chunk
        # ----------------------------------------------------

        if current_start is not None:

            chunk_audio = audio[
                current_start:current_end
            ]

            if len(chunk_audio) >= min_samples:

                chunks.append(
                    AudioChunk(
                        index=chunk_index,
                        start=(
                            current_start
                            / self.sample_rate
                        ),
                        end=(
                            current_end
                            / self.sample_rate
                        ),
                        audio=chunk_audio,
                        duration=(
                            len(chunk_audio)
                            / self.sample_rate
                        ),
                        source_file=source_file,
                    )
                )

        return chunks

    # ========================================================
    # Process
    # ========================================================

    def chunk_file(
        self,
        audio_path: Path,
    ):

        audio_path = Path(
            audio_path
        )

        audio = self.load_audio(
            audio_path
        )

        total_duration = (
            len(audio)
            / self.sample_rate
        )

        # ----------------------------------------------------
        # Chunking disabled
        # ----------------------------------------------------

        if not self.enabled:

            return [
                AudioChunk(
                    index=1,
                    start=0.0,
                    end=total_duration,
                    audio=audio,
                    duration=total_duration,
                    source_file=audio_path.name,
                )
            ]

        # ----------------------------------------------------
        # Detect speech
        # ----------------------------------------------------

        speech_regions = (
            self.detect_speech_regions(
                audio
            )
        )

        # ----------------------------------------------------
        # Build chunks
        # ----------------------------------------------------

        chunks = self.build_chunks(
            audio,
            speech_regions,
            audio_path.name,
        )

        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        if not chunks:

            chunks = [
                AudioChunk(
                    index=1,
                    start=0.0,
                    end=total_duration,
                    audio=audio,
                    duration=total_duration,
                    source_file=audio_path.name,
                )
            ]

        return chunks