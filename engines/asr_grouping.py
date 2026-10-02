from __future__ import annotations

import logging
from typing import Any

import numpy as np


logger = logging.getLogger(
    "persian_asr.asr_grouping"
)


class ASRGroupingEngine:
    """
    Build robust ASR groups from diarization turns.

    Responsibilities
    ----------------
    1. Ignore configurable ultra-short diarization turns.
    2. Ignore configurable very-short overlap/interruption effects.
    3. Continue a speaker group across ignored short turns.
    4. Merge same-speaker turns when the gap is acceptable.
    5. Never exceed max_chunk_seconds.
    6. Split turns longer than max_chunk_seconds.
    7. Validate the actual extracted audio.
    8. Prevent zero-duration / empty / silence-only groups.
    9. Mask ignored short-speaker turns from original audio.
    10. Work identically whether speaker isolation is enabled
        or disabled.

    IMPORTANT
    ---------
    This class does NOT perform diarization.
    This class does NOT perform TSE.
    This class does NOT perform Whisper.

    It only decides which diarization turns should become
    ASR groups and which audio intervals belong to each group.
    """

    def __init__(
        self,
        config: dict,
    ) -> None:

        self.config = config

        asr_config = config.get(
            "asr",
            {},
        )

        grouping = asr_config.get(
            "grouping",
            {},
        )

        # ====================================================
        # Master
        # ====================================================

        self.enabled = bool(
            grouping.get(
                "enabled",
                True,
            )
        )

        # ====================================================
        # Basic grouping
        # ====================================================

        self.max_gap_seconds = max(
            0.0,
            float(
                grouping.get(
                    "max_gap_seconds",
                    1.50,
                )
            ),
        )

        self.max_chunk_seconds = max(
            0.05,
            float(
                grouping.get(
                    "max_chunk_seconds",
                    asr_config.get(
                        "max_turn_seconds",
                        30.0,
                    ),
                )
            ),
        )

        self.min_group_seconds = max(
            0.0,
            float(
                grouping.get(
                    "min_group_seconds",
                    0.20,
                )
            ),
        )

        self.preserve_gaps = bool(
            grouping.get(
                "preserve_gaps",
                True,
            )
        )

        # ====================================================
        # Short turn filter
        # ====================================================

        short_turn_filter = grouping.get(
            "short_turn_filter",
            {},
        )

        self.short_turn_filter_enabled = bool(
            short_turn_filter.get(
                "enabled",
                True,
            )
        )

        self.ignore_below_seconds = max(
            0.0,
            float(
                short_turn_filter.get(
                    "ignore_below_seconds",
                    0.50,
                )
            ),
        )

        self.bridge_same_speaker = bool(
            short_turn_filter.get(
                "bridge_same_speaker",
                True,
            )
        )

        self.bridge_max_gap_seconds = max(
            0.0,
            float(
                short_turn_filter.get(
                    "bridge_max_gap_seconds",
                    self.max_gap_seconds,
                )
            ),
        )

        # ====================================================
        # Overlap filter
        # ====================================================

        overlap_filter = grouping.get(
            "overlap_filter",
            {},
        )

        self.overlap_filter_enabled = bool(
            overlap_filter.get(
                "enabled",
                True,
            )
        )

        self.ignore_overlap_below_seconds = max(
            0.0,
            float(
                overlap_filter.get(
                    "ignore_below_seconds",
                    0.50,
                )
            ),
        )

        self.do_not_break_on_short_overlap = bool(
            overlap_filter.get(
                "do_not_break_group",
                True,
            )
        )

        # ====================================================
        # Noise filter
        # ====================================================

        noise_filter = grouping.get(
            "noise_filter",
            {},
        )

        self.noise_filter_enabled = bool(
            noise_filter.get(
                "enabled",
                True,
            )
        )

        self.max_noise_turn_seconds = max(
            0.0,
            float(
                noise_filter.get(
                    "max_noise_turn_seconds",
                    self.ignore_below_seconds,
                )
            ),
        )

        self.noise_bridge_same_speaker = bool(
            noise_filter.get(
                "bridge_same_speaker",
                True,
            )
        )

        self.min_rms_ratio = max(
            0.0,
            float(
                noise_filter.get(
                    "min_rms_ratio",
                    0.08,
                )
            ),
        )

        self.min_rms = max(
            0.0,
            float(
                noise_filter.get(
                    "min_rms",
                    0.0005,
                )
            ),
        )

        self.min_active_ratio = max(
            0.0,
            float(
                noise_filter.get(
                    "min_active_ratio",
                    0.02,
                )
            ),
        )

        # ====================================================
        # Audio validation
        # ====================================================

        audio_validation = grouping.get(
            "audio_validation",
            {},
        )

        self.audio_validation_enabled = bool(
            audio_validation.get(
                "enabled",
                True,
            )
        )

        self.validate_actual_duration = bool(
            audio_validation.get(
                "validate_actual_duration",
                True,
            )
        )

        self.min_audio_seconds = max(
            0.0,
            float(
                audio_validation.get(
                    "min_audio_seconds",
                    0.20,
                )
            ),
        )

        self.validation_min_active_ratio = max(
            0.0,
            float(
                audio_validation.get(
                    "min_active_ratio",
                    self.min_active_ratio,
                )
            ),
        )

        self.reject_silence_only = bool(
            audio_validation.get(
                "reject_silence_only",
                True,
            )
        )

        self.reject_noise_only = bool(
            audio_validation.get(
                "reject_noise_only",
                True,
            )
        )

        # ====================================================
        # Debug
        # ====================================================

        debug_cfg = grouping.get(
            "debug",
            {},
        )

        self.save_only_valid_groups = bool(
            debug_cfg.get(
                "save_only_valid_groups",
                True,
            )
        )

        self.contiguous_group_numbers = bool(
            debug_cfg.get(
                "contiguous_group_numbers",
                True,
            )
        )

        # ====================================================
        # Runtime statistics
        # ====================================================

        self.last_stats: dict[str, Any] = {}

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
    def _duration(
        turn: dict,
    ) -> float:

        return max(
            0.0,
            float(turn["end"])
            - float(turn["start"]),
        )

    @staticmethod
    def _copy_turn(
        turn: dict,
    ) -> dict:

        result = dict(turn)

        result["start"] = float(
            turn["start"]
        )

        result["end"] = float(
            turn["end"]
        )

        result["duration"] = (
            result["end"]
            - result["start"]
        )

        return result

    # ========================================================
    # Turn filtering
    # ========================================================

    def _is_short_turn(
        self,
        turn: dict,
    ) -> bool:

        duration = self._duration(
            turn
        )

        if (
            self.short_turn_filter_enabled
            and duration
            < self.ignore_below_seconds
        ):
            return True

        if (
            self.noise_filter_enabled
            and duration
            <= self.max_noise_turn_seconds
        ):
            return True

        return False

    def _filter_short_turns(
        self,
        turns: list[dict],
    ) -> tuple[
        list[dict],
        list[dict],
    ]:

        if not turns:
            return [], []

        kept = []
        ignored = []

        for index, turn in enumerate(
            turns
        ):

            item = self._copy_turn(
                turn
            )

            item["_source_index"] = index

            if self._is_short_turn(
                item
            ):

                item["_ignored_reason"] = (
                    "short_turn"
                )

                ignored.append(
                    item
                )

                continue

            kept.append(
                item
            )

        return kept, ignored

    # ========================================================
    # Overlap
    # ========================================================

    def _overlap_duration(
        self,
        first: dict,
        second: dict,
    ) -> float:

        start = max(
            float(first["start"]),
            float(second["start"]),
        )

        end = min(
            float(first["end"]),
            float(second["end"]),
        )

        return max(
            0.0,
            end - start,
        )

    def _is_ignorable_overlap(
        self,
        first: dict,
        second: dict,
    ) -> bool:

        if not self.overlap_filter_enabled:
            return False

        if (
            first["speaker"]
            == second["speaker"]
        ):
            return False

        overlap = self._overlap_duration(
            first,
            second,
        )

        return (
            overlap > 0.0
            and overlap
            <= self.ignore_overlap_below_seconds
        )

    # ========================================================
    # Group construction
    # ========================================================

    def _new_group(
        self,
        turn: dict,
        source_index: int,
    ) -> dict:

        start = float(
            turn["start"]
        )

        end = float(
            turn["end"]
        )

        return {
            "group_index": -1,
            "speaker": turn["speaker"],
            "start": start,
            "end": end,
            "duration": end - start,
            "source_turn_indices": [
                int(source_index)
            ],
            "source_turns": [
                self._copy_turn(turn)
            ],
            "ignored_turn_indices": [],
            "ignored_turns": [],
        }

    def _append_turn(
        self,
        group: dict,
        turn: dict,
        source_index: int,
    ) -> None:

        group["end"] = max(
            float(group["end"]),
            float(turn["end"]),
        )

        group["duration"] = (
            group["end"]
            - group["start"]
        )

        group[
            "source_turn_indices"
        ].append(
            int(source_index)
        )

        group[
            "source_turns"
        ].append(
            self._copy_turn(turn)
        )

    def _append_ignored_turn(
        self,
        group: dict,
        turn: dict,
    ) -> None:

        source_index = int(
            turn.get(
                "_source_index",
                -1,
            )
        )

        group[
            "ignored_turn_indices"
        ].append(
            source_index
        )

        group[
            "ignored_turns"
        ].append(
            self._copy_turn(turn)
        )

    def _split_long_turn(
        self,
        turn: dict,
    ) -> list[dict]:

        start = float(
            turn["start"]
        )

        end = float(
            turn["end"]
        )

        pieces = []

        cursor = start

        while cursor < end:

            piece_end = min(
                end,
                cursor
                + self.max_chunk_seconds,
            )

            duration = (
                piece_end
                - cursor
            )

            piece = {
                "group_index": -1,
                "speaker": turn["speaker"],
                "start": cursor,
                "end": piece_end,
                "duration": duration,
                "source_turn_indices": [
                    int(
                        turn.get(
                            "_source_index",
                            -1,
                        )
                    )
                ],
                "source_turns": [
                    {
                        **self._copy_turn(
                            turn
                        ),
                        "start": cursor,
                        "end": piece_end,
                        "duration": duration,
                    }
                ],
                "ignored_turn_indices": [],
                "ignored_turns": [],
                "split_from_long_turn": True,
            }

            pieces.append(
                piece
            )

            cursor = piece_end

        return pieces

    def _can_merge(
        self,
        current: dict,
        turn: dict,
        had_ignored_turn: bool,
    ) -> bool:

        if (
            current["speaker"]
            != turn["speaker"]
        ):
            return False

        current_end = float(
            current["end"]
        )

        next_start = float(
            turn["start"]
        )

        gap = (
            next_start
            - current_end
        )

        # ----------------------------------------------------
        # Normal positive gap
        # ----------------------------------------------------

        if gap >= 0.0:

            max_gap = (
                self.max_gap_seconds
            )

            if (
                had_ignored_turn
                and self.bridge_same_speaker
            ):
                max_gap = max(
                    max_gap,
                    self.bridge_max_gap_seconds,
                )

            if gap > max_gap:
                return False

        # ----------------------------------------------------
        # Negative gap = overlap
        # ----------------------------------------------------

        else:

            overlap = -gap

            if not (
                self.do_not_break_on_short_overlap
                and self._is_ignorable_overlap(
                    current[
                        "source_turns"
                    ][-1],
                    turn,
                )
            ):
                return False

            if (
                overlap
                > self.ignore_overlap_below_seconds
            ):
                return False

        candidate_end = max(
            float(current["end"]),
            float(turn["end"]),
        )

        candidate_duration = (
            candidate_end
            - float(current["start"])
        )

        return (
            candidate_duration
            <= self.max_chunk_seconds
        )

    def build_groups(
        self,
        turns: list[dict],
    ) -> list[dict]:

        self.last_stats = {
            "input_turns": len(turns),
            "ignored_short_turns": 0,
            "groups_before_audio_validation": 0,
            "groups_rejected": 0,
            "groups_output": 0,
        }

        if not turns:
            return []

        # ----------------------------------------------------
        # Normalize order
        # ----------------------------------------------------

        normalized = []

        for index, turn in enumerate(
            turns
        ):

            if not isinstance(
                turn,
                dict,
            ):
                continue

            speaker = str(
                turn.get(
                    "speaker",
                    "",
                )
            ).strip()

            start = self._safe_float(
                turn.get(
                    "start",
                    0.0,
                )
            )

            end = self._safe_float(
                turn.get(
                    "end",
                    0.0,
                )
            )

            if (
                not speaker
                or end <= start
            ):
                continue

            item = {
                "speaker": speaker,
                "start": start,
                "end": end,
                "duration": end - start,
                "_source_index": index,
            }

            normalized.append(
                item
            )

        normalized.sort(
            key=lambda x: (
                x["start"],
                x["end"],
                x["speaker"],
            )
        )

        # ----------------------------------------------------
        # Remove short turns
        # ----------------------------------------------------

        filtered, ignored = (
            self._filter_short_turns(
                normalized
            )
        )

        self.last_stats[
            "ignored_short_turns"
        ] = len(ignored)

        if not filtered:
            return []

        # ----------------------------------------------------
        # Grouping disabled
        #
        # We STILL apply short-turn filtering.
        # This is intentional.
        #
        # speaker_isolation.enabled has absolutely no effect
        # on this behavior.
        # ----------------------------------------------------

        if not self.enabled:

            groups = []

            for turn in filtered:

                duration = (
                    self._duration(
                        turn
                    )
                )

                if (
                    duration
                    > self.max_chunk_seconds
                ):

                    groups.extend(
                        self._split_long_turn(
                            turn
                        )
                    )

                else:

                    groups.append(
                        self._new_group(
                            turn,
                            int(
                                turn[
                                    "_source_index"
                                ]
                            ),
                        )
                    )

            self.last_stats[
                "groups_before_audio_validation"
            ] = len(groups)

            return self._finalize_groups(
                groups,
                ignored,
            )

        # ----------------------------------------------------
        # Normal grouping
        # ----------------------------------------------------

        groups = []

        current = None

        ignored_cursor = 0

        for turn in filtered:

            source_index = int(
                turn[
                    "_source_index"
                ]
            )

            # ----------------------------------------------
            # Collect ignored turns that happened before
            # this accepted turn.
            # ----------------------------------------------

            ignored_between = []

            while (
                ignored_cursor
                < len(ignored)
                and int(
                    ignored[
                        ignored_cursor
                    ]["_source_index"]
                )
                < source_index
            ):

                ignored_between.append(
                    ignored[
                        ignored_cursor
                    ]
                )

                ignored_cursor += 1

            # ----------------------------------------------
            # Very long turn
            # ----------------------------------------------

            if (
                self._duration(turn)
                > self.max_chunk_seconds
            ):

                if current is not None:

                    for ignored_turn in (
                        ignored_between
                    ):
                        self._append_ignored_turn(
                            current,
                            ignored_turn,
                        )

                    groups.append(
                        current
                    )

                    current = None

                groups.extend(
                    self._split_long_turn(
                        turn
                    )
                )

                continue

            # ----------------------------------------------
            # First group
            # ----------------------------------------------

            if current is None:

                current = (
                    self._new_group(
                        turn,
                        source_index,
                    )
                )

                for ignored_turn in (
                    ignored_between
                ):
                    self._append_ignored_turn(
                        current,
                        ignored_turn,
                    )

                continue

            # ----------------------------------------------
            # Check if ignored turns are only short
            # interruptions.
            # ----------------------------------------------

            had_ignored_turn = bool(
                ignored_between
            )

            # ----------------------------------------------
            # Same speaker merge
            # ----------------------------------------------

            if self._can_merge(
                current,
                turn,
                had_ignored_turn,
            ):

                for ignored_turn in (
                    ignored_between
                ):
                    self._append_ignored_turn(
                        current,
                        ignored_turn,
                    )

                self._append_turn(
                    current,
                    turn,
                    source_index,
                )

                continue

            # ----------------------------------------------
            # Current group ends
            # ----------------------------------------------

            groups.append(
                current
            )

            current = (
                self._new_group(
                    turn,
                    source_index,
                )
            )

            for ignored_turn in (
                ignored_between
            ):
                self._append_ignored_turn(
                    current,
                    ignored_turn,
                )

        if current is not None:

            # Remaining ignored turns.
            while (
                ignored_cursor
                < len(ignored)
            ):

                self._append_ignored_turn(
                    current,
                    ignored[
                        ignored_cursor
                    ],
                )

                ignored_cursor += 1

            groups.append(
                current
            )

        self.last_stats[
            "groups_before_audio_validation"
        ] = len(groups)

        return self._finalize_groups(
            groups,
            ignored,
        )

    # ========================================================
    # Audio extraction / masking
    # ========================================================

    @staticmethod
    def _extract(
        waveform: np.ndarray,
        sample_rate: int,
        start: float,
        end: float,
    ) -> np.ndarray:

        if (
            waveform.size == 0
            or end <= start
        ):
            return np.zeros(
                0,
                dtype=np.float32,
            )

        start_sample = max(
            0,
            int(
                round(
                    start
                    * sample_rate
                )
            ),
        )

        end_sample = min(
            len(waveform),
            int(
                round(
                    end
                    * sample_rate
                )
            ),
        )

        if (
            end_sample
            <= start_sample
        ):
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
    def _copy_interval(
        source: np.ndarray,
        target: np.ndarray,
        sample_rate: int,
        source_start: float,
        target_start: float,
        interval_start: float,
        interval_end: float,
    ) -> None:

        if interval_end <= interval_start:
            return

        source_start_sample = int(
            round(
                (
                    interval_start
                    - source_start
                )
                * sample_rate
            )
        )

        source_end_sample = int(
            round(
                (
                    interval_end
                    - source_start
                )
                * sample_rate
            )
        )

        target_start_sample = int(
            round(
                (
                    interval_start
                    - target_start
                )
                * sample_rate
            )
        )

        target_end_sample = int(
            round(
                (
                    interval_end
                    - target_start
                )
                * sample_rate
            )
        )

        source_start_sample = max(
            0,
            source_start_sample,
        )

        source_end_sample = min(
            len(source),
            source_end_sample,
        )

        target_start_sample = max(
            0,
            target_start_sample,
        )

        target_end_sample = min(
            len(target),
            target_end_sample,
        )

        if (
            source_end_sample
            <= source_start_sample
        ):
            return

        if (
            target_end_sample
            <= target_start_sample
        ):
            return

        length = min(
            source_end_sample
            - source_start_sample,
            target_end_sample
            - target_start_sample,
        )

        if length <= 0:
            return

        target[
            target_start_sample:
            target_start_sample + length
        ] = source[
            source_start_sample:
            source_start_sample + length
        ]

    def build_audio(
        self,
        group: dict,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:

        start = float(
            group["start"]
        )

        end = float(
            group["end"]
        )

        if end <= start:
            return np.zeros(
                0,
                dtype=np.float32,
            )

        raw = self._extract(
            waveform,
            sample_rate,
            start,
            end,
        )

        if raw.size == 0:
            return raw

        # ----------------------------------------------------
        # If gaps should be preserved, construct a clean
        # speaker-only timeline.
        #
        # This is extremely important when isolation is OFF:
        # a short ignored SPEAKER_02 turn must NOT leak into
        # SPEAKER_01's ASR group.
        # ----------------------------------------------------

        if self.preserve_gaps:

            output = np.zeros_like(
                raw,
                dtype=np.float32,
            )

            for turn in group.get(
                "source_turns",
                [],
            ):

                self._copy_interval(
                    source=raw,
                    target=output,
                    sample_rate=sample_rate,
                    source_start=start,
                    target_start=start,
                    interval_start=max(
                        start,
                        float(
                            turn["start"]
                        ),
                    ),
                    interval_end=min(
                        end,
                        float(
                            turn["end"]
                        ),
                    ),
                )

            return np.ascontiguousarray(
                output,
                dtype=np.float32,
            )

        # ----------------------------------------------------
        # No gaps:
        # concatenate accepted source turns.
        # ----------------------------------------------------

        pieces = []

        for turn in group.get(
            "source_turns",
            [],
        ):

            piece = self._extract(
                waveform,
                sample_rate,
                float(
                    turn["start"]
                ),
                float(
                    turn["end"]
                ),
            )

            if piece.size:
                pieces.append(
                    piece
                )

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

    # ========================================================
    # Audio validation
    # ========================================================

    @staticmethod
    def _rms(
        audio: np.ndarray,
    ) -> float:

        if audio.size == 0:
            return 0.0

        audio64 = np.asarray(
            audio,
            dtype=np.float64,
        )

        return float(
            np.sqrt(
                np.mean(
                    np.square(
                        audio64
                    )
                )
            )
        )

    @staticmethod
    def _peak(
        audio: np.ndarray,
    ) -> float:

        if audio.size == 0:
            return 0.0

        return float(
            np.max(
                np.abs(audio)
            )
        )

    def _active_ratio(
        self,
        audio: np.ndarray,
    ) -> float:

        if audio.size == 0:
            return 0.0

        peak = self._peak(
            audio
        )

        threshold = max(
            self.min_rms,
            peak * 0.05,
        )

        active = np.abs(
            audio
        ) > threshold

        return float(
            np.mean(active)
        )

    def validate_audio(
        self,
        group: dict,
        audio: np.ndarray,
        sample_rate: int,
    ) -> tuple[
        bool,
        dict,
    ]:

        expected_duration = max(
            0.0,
            float(group["end"])
            - float(group["start"]),
        )

        actual_duration = 0.0

        if (
            sample_rate > 0
            and audio.size > 0
        ):
            actual_duration = (
                float(audio.size)
                / float(sample_rate)
            )

        rms = self._rms(
            audio
        )

        peak = self._peak(
            audio
        )

        active_ratio = (
            self._active_ratio(
                audio
            )
        )

        diagnostics = {
            "expected_duration": (
                expected_duration
            ),
            "actual_duration": (
                actual_duration
            ),
            "rms": rms,
            "peak": peak,
            "active_ratio": active_ratio,
            "sample_count": int(
                audio.size
            ),
        }

        # ----------------------------------------------------
        # Basic duration
        # ----------------------------------------------------

        if expected_duration <= 0.0:

            diagnostics[
                "reject_reason"
            ] = "zero_timeline_duration"

            return False, diagnostics

        if audio.size == 0:

            diagnostics[
                "reject_reason"
            ] = "empty_audio"

            return False, diagnostics

        if (
            self.validate_actual_duration
            and actual_duration
            < self.min_audio_seconds
        ):

            diagnostics[
                "reject_reason"
            ] = "audio_too_short"

            return False, diagnostics

        if (
            expected_duration
            < self.min_group_seconds
        ):

            diagnostics[
                "reject_reason"
            ] = "group_too_short"

            return False, diagnostics

        # ----------------------------------------------------
        # Silence
        # ----------------------------------------------------

        if (
            self.audio_validation_enabled
            and self.reject_silence_only
        ):

            if (
                peak <= 1e-6
                or rms <= 1e-7
            ):

                diagnostics[
                    "reject_reason"
                ] = "silence_only"

                return False, diagnostics

        # ----------------------------------------------------
        # Noise / extremely weak audio
        # ----------------------------------------------------

        if (
            self.audio_validation_enabled
            and self.reject_noise_only
        ):

            if (
                rms < self.min_rms
                and peak < self.min_rms
            ):

                diagnostics[
                    "reject_reason"
                ] = "noise_only"

                return False, diagnostics

            if (
                active_ratio
                < self.validation_min_active_ratio
            ):

                diagnostics[
                    "reject_reason"
                ] = "inactive_audio"

                return False, diagnostics

        diagnostics[
            "valid"
        ] = True

        return True, diagnostics

    # ========================================================
    # Finalize
    # ========================================================

    def _finalize_groups(
        self,
        groups: list[dict],
        ignored_turns: list[dict],
    ) -> list[dict]:

        result = []

        for group in groups:

            duration = (
                float(group["end"])
                - float(group["start"])
            )

            if (
                duration
                < self.min_group_seconds
            ):

                self.last_stats[
                    "groups_rejected"
                ] += 1

                continue

            # Remove private fields before returning.
            cleaned = {
                key: value
                for key, value in group.items()
                if not key.startswith("_")
            }

            cleaned[
                "duration"
            ] = duration

            result.append(
                cleaned
            )

        if self.contiguous_group_numbers:

            for index, group in enumerate(
                result
            ):
                group[
                    "group_index"
                ] = index

        self.last_stats[
            "groups_output"
        ] = len(result)

        return result

    # ========================================================
    # Group audio validation pass
    # ========================================================

    def validate_groups_against_audio(
        self,
        groups: list[dict],
        waveform: np.ndarray,
        sample_rate: int,
    ) -> list[dict]:

        if not groups:
            return []

        valid_groups = []

        for group in groups:

            audio = self.build_audio(
                group,
                waveform,
                sample_rate,
            )

            valid, diagnostics = (
                self.validate_audio(
                    group,
                    audio,
                    sample_rate,
                )
            )

            group[
                "audio_validation"
            ] = diagnostics

            if not valid:

                self.last_stats[
                    "groups_rejected"
                ] += 1

                logger.info(
                    "Rejected ASR group %03d: "
                    "%s | %.3fs -> %.3fs | "
                    "duration=%.3fs rms=%.6f "
                    "active=%.4f",
                    int(
                        group.get(
                            "group_index",
                            -1,
                        )
                    ),
                    diagnostics.get(
                        "reject_reason",
                        "unknown",
                    ),
                    float(
                        group["start"]
                    ),
                    float(
                        group["end"]
                    ),
                    float(
                        diagnostics.get(
                            "actual_duration",
                            0.0,
                        )
                    ),
                    float(
                        diagnostics.get(
                            "rms",
                            0.0,
                        )
                    ),
                    float(
                        diagnostics.get(
                            "active_ratio",
                            0.0,
                        )
                    ),
                )

                continue

            group[
                "_validated_audio"
            ] = audio

            valid_groups.append(
                group
            )

        if self.contiguous_group_numbers:

            for index, group in enumerate(
                valid_groups
            ):

                group[
                    "group_index"
                ] = index

        self.last_stats[
            "groups_output"
        ] = len(valid_groups)

        return valid_groups