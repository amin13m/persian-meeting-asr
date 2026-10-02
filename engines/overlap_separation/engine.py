from __future__ import annotations

import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


logger = logging.getLogger(
    "persian_asr.overlap_separation"
)


class OverlapSeparationEngine:
    """
    REAL-T / WeSep based target speaker extraction.

    IMPORTANT:
        TSE is executed ONLY on overlap regions.

    The complete meeting audio is NEVER sent to TSE.

    Debug audio can be enabled from:

        debug:
          audio:
            enabled: true
            root_dir: output/debug_audio
            save_tse_inputs: true
            save_tse_outputs: true
            save_tse_overlap: true
            keep_tse_temp: true
    """

    def __init__(self, config: dict):
        self.config = config

        cfg = config.get(
            "overlap_separation",
            {},
        )

        debug_cfg = (
            config.get("debug", {})
            .get("audio", {})
        )

        self.enabled = bool(
            cfg.get("enabled", True)
        )

        # ----------------------------------------------------
        # Project root
        # ----------------------------------------------------

        self.project_root = Path(
            config.get(
                "_project_root",
                Path(__file__).resolve().parents[2],
            )
        ).resolve()

        # ----------------------------------------------------
        # Debug
        # ----------------------------------------------------

        self.debug_audio_enabled = bool(
            debug_cfg.get(
                "enabled",
                False,
            )
        )

        debug_root = debug_cfg.get(
            "root_dir",
            "output/debug_audio",
        )

        self.debug_root = self._resolve_project_path(
            debug_root
        )

        self.save_tse_inputs = bool(
            debug_cfg.get(
                "save_tse_inputs",
                False,
            )
        )

        self.save_tse_outputs = bool(
            debug_cfg.get(
                "save_tse_outputs",
                False,
            )
        )

        self.save_tse_overlap = bool(
            debug_cfg.get(
                "save_tse_overlap",
                False,
            )
        )

        self.keep_tse_temp = bool(
            debug_cfg.get(
                "keep_tse_temp",
                False,
            )
        )

        # Explicit overlap config has priority.
        if bool(
            cfg.get(
                "keep_temp_files",
                False,
            )
        ):
            self.keep_tse_temp = True

        self.temp_dir = self._resolve_project_path(
            cfg.get(
                "temp_dir",
                "output/.tse_tmp",
            )
        )

        # ----------------------------------------------------
        # REAL-T model
        # ----------------------------------------------------

        model_path_value = cfg.get(
            "model_path",
            cfg.get(
                "model_dir",
                "",
            ),
        )

        self.model_dir = (
            self._resolve_project_path(
                model_path_value
            )
        )

        # ----------------------------------------------------
        # WeSep
        # ----------------------------------------------------

        wesep_path_value = cfg.get(
            "wesep_path",
            cfg.get(
                "wesep_root",
                "",
            ),
        )

        self.wesep_root = (
            self._resolve_project_path(
                wesep_path_value
            )
        )

        # ----------------------------------------------------
        # WeSpeaker
        # ----------------------------------------------------

        wespeaker_path_value = cfg.get(
            "wespeaker_path",
            "",
        )

        if wespeaker_path_value:
            self.wespeaker_root = (
                self._resolve_project_path(
                    wespeaker_path_value
                )
            )
        else:
            self.wespeaker_root = (
                self.wesep_root.parent
                / "WeSpeaker"
            )

        # ----------------------------------------------------
        # Audio
        # ----------------------------------------------------

        self.sample_rate = int(
            cfg.get(
                "sample_rate",
                config.get(
                    "audio",
                    {},
                ).get(
                    "sample_rate",
                    16000,
                ),
            )
        )

        # ----------------------------------------------------
        # Context
        # ----------------------------------------------------

        legacy_context = float(
            cfg.get(
                "context_seconds",
                0.50,
            )
        )

        self.context_before_seconds = float(
            cfg.get(
                "context_before_seconds",
                legacy_context,
            )
        )

        self.context_after_seconds = float(
            cfg.get(
                "context_after_seconds",
                legacy_context,
            )
        )

        # ----------------------------------------------------
        # Limits
        # ----------------------------------------------------

        self.min_overlap_seconds = float(
            cfg.get(
                "min_overlap_seconds",
                0.20,
            )
        )

        self.max_region_seconds = float(
            cfg.get(
                "max_region_seconds",
                8.0,
            )
        )

        self.merge_gap_seconds = float(
            cfg.get(
                "merge_gap_seconds",
                0.10,
            )
        )

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        self.device = str(
            cfg.get(
                "device",
                "cuda",
            )
        ).lower()

        # ----------------------------------------------------
        # Output normalization
        # ----------------------------------------------------

        self.output_norm = bool(
            cfg.get(
                "output_norm",
                True,
            )
        )

        # ----------------------------------------------------
        # Runtime
        # ----------------------------------------------------

        self._model = None
        self._loaded = False

        logger.info(
            "OverlapSeparationEngine initialized | "
            "enabled=%s | sample_rate=%d | device=%s",
            self.enabled,
            self.sample_rate,
            self.device,
        )

    # ========================================================
    # Paths
    # ========================================================

    def _resolve_project_path(
        self,
        value: Any,
    ) -> Path:

        if value is None:
            return Path("")

        value = str(value).strip()

        if not value:
            return Path("")

        path = Path(value)

        if not path.is_absolute():
            path = self.project_root / path

        return path.resolve()

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

    # ========================================================
    # Local imports
    # ========================================================

    def _setup_local_import_paths(self) -> None:
        """
        Handles both:

            models/WeSep/
                wesep/

        and:

            models/WeSep/wesep/
                __init__.py

        The previous implementation only inserted the
        repository path, which could fail when wesep_path
        itself pointed to the package directory.
        """

        if not self.wesep_root:
            raise ValueError(
                "overlap_separation.wesep_path "
                "is not configured."
            )

        if not self.wesep_root.exists():
            raise FileNotFoundError(
                "WeSep directory does not exist: "
                f"{self.wesep_root}"
            )

        if not self.wespeaker_root.exists():
            raise FileNotFoundError(
                "WeSpeaker directory does not exist: "
                f"{self.wespeaker_root}"
            )

        paths: list[Path] = []

        # Repository root.
        paths.append(
            self.wesep_root
        )

        # If wesep_root itself is the package directory,
        # Python needs its parent.
        if (
            self.wesep_root.name.lower()
            == "wesep"
        ):
            paths.append(
                self.wesep_root.parent
            )

        nested_wesep = (
            self.wesep_root / "wesep"
        )

        if nested_wesep.exists():
            paths.append(
                self.wesep_root
            )

        # WeSpeaker repository.
        paths.append(
            self.wespeaker_root
        )

        if (
            self.wespeaker_root.name.lower()
            == "wespeaker"
        ):
            paths.append(
                self.wespeaker_root.parent
            )

        nested_wespeaker = (
            self.wespeaker_root
            / "wespeaker"
        )

        if nested_wespeaker.exists():
            paths.append(
                self.wespeaker_root
            )

        for path in paths:

            path = path.resolve()

            path_string = str(path)

            if path_string not in sys.path:
                sys.path.insert(
                    0,
                    path_string,
                )

        logger.info(
            "WeSep path     : %s",
            self.wesep_root,
        )

        logger.info(
            "WeSpeaker path : %s",
            self.wespeaker_root,
        )

        logger.info(
            "Local WeSep / WeSpeaker import paths configured."
        )

    # ========================================================
    # Lifecycle
    # ========================================================

    def load(self) -> None:

        if not self.enabled:
            logger.info(
                "Overlap separation disabled."
            )
            return

        if self._loaded:
            return

        if not self.model_dir:
            raise ValueError(
                "overlap_separation.model_path "
                "is not configured."
            )

        if not self.model_dir.exists():
            raise FileNotFoundError(
                "REAL-T model directory does not exist: "
                f"{self.model_dir}"
            )

        logger.info(
            "Loading REAL-T / WeSep model..."
        )

        self._setup_local_import_paths()

        try:
            import wesep
        except ModuleNotFoundError as exc:

            logger.exception(
                "Could not import local WeSep."
            )

            raise ModuleNotFoundError(
                "Could not import 'wesep'. "
                "Checked local WeSep repository: "
                f"{self.wesep_root}"
            ) from exc

        logger.info(
            "WeSep imported from: %s",
            getattr(
                wesep,
                "__file__",
                "unknown",
            ),
        )

        self._model = (
            wesep.load_model_local(
                str(self.model_dir)
            )
        )

        self._model.set_resample_rate(
            self.sample_rate
        )

        self._model.set_vad(
            False
        )

        if (
            self.device == "cuda"
            and torch.cuda.is_available()
        ):

            self._model.set_device(
                "cuda"
            )

            logger.info(
                "TSE device: CUDA"
            )

            logger.info(
                "CUDA device: %s",
                torch.cuda.get_device_name(0),
            )

        else:

            self._model.set_device(
                "cpu"
            )

            logger.info(
                "TSE device: CPU"
            )

        self._model.set_output_norm(
            self.output_norm
        )

        self._loaded = True

        logger.info(
            "REAL-T / WeSep model loaded."
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
    def _to_numpy(
        audio: Any,
    ) -> np.ndarray:

        if isinstance(
            audio,
            torch.Tensor,
        ):

            audio = (
                audio.detach()
                .float()
                .cpu()
                .numpy()
            )

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        audio = np.squeeze(
            audio
        )

        if audio.ndim == 2:

            if audio.shape[0] <= 8:

                audio = np.mean(
                    audio,
                    axis=0,
                    dtype=np.float32,
                )

            else:

                audio = np.mean(
                    audio,
                    axis=1,
                    dtype=np.float32,
                )

        if audio.ndim != 1:
            raise ValueError(
                "TSE output could not be converted "
                "to a mono 1-D waveform."
            )

        audio = np.nan_to_num(
            audio,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return np.ascontiguousarray(
            audio,
            dtype=np.float32,
        )

    @staticmethod
    def _normalize_audio(
        audio: np.ndarray,
    ) -> np.ndarray:

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        audio = np.nan_to_num(
            audio,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return np.ascontiguousarray(
            audio,
            dtype=np.float32,
        )

    # ========================================================
    # Debug
    # ========================================================

    def _debug_region_dir(
        self,
        target_speaker: str,
        index: int,
    ) -> Path:

        path = (
            self.debug_root
            / "tse"
            / self._safe_name(
                target_speaker
            )
            / f"overlap_{index:03d}"
        )

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        return path

    def _save_debug_audio(
        self,
        path: Path,
        audio: np.ndarray,
    ) -> None:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        sf.write(
            str(path),
            self._normalize_audio(audio),
            self.sample_rate,
        )

        logger.debug(
            "Saved debug audio: %s",
            path,
        )

    # ========================================================
    # Segment extraction
    # ========================================================

    def _extract_segment(
        self,
        audio: np.ndarray,
        start: float,
        end: float,
    ) -> tuple[
        np.ndarray,
        float,
        float,
    ]:

        total_duration = (
            len(audio)
            / self.sample_rate
        )

        segment_start = max(
            0.0,
            start
            - self.context_before_seconds,
        )

        segment_end = min(
            total_duration,
            end
            + self.context_after_seconds,
        )

        if (
            self.max_region_seconds > 0
            and (
                segment_end
                - segment_start
            )
            > self.max_region_seconds
        ):

            center = (
                start + end
            ) / 2.0

            half = (
                self.max_region_seconds
                / 2.0
            )

            segment_start = max(
                0.0,
                center - half,
            )

            segment_end = min(
                total_duration,
                center + half,
            )

        start_sample = int(
            round(
                segment_start
                * self.sample_rate
            )
        )

        end_sample = int(
            round(
                segment_end
                * self.sample_rate
            )
        )

        start_sample = max(
            0,
            start_sample,
        )

        end_sample = min(
            len(audio),
            end_sample,
        )

        if end_sample <= start_sample:
            return (
                np.zeros(
                    0,
                    dtype=np.float32,
                ),
                segment_start,
                segment_end,
            )

        segment = audio[
            start_sample:end_sample
        ]

        return (
            np.ascontiguousarray(
                segment,
                dtype=np.float32,
            ),
            segment_start,
            segment_end,
        )

    # ========================================================
    # Enrollment
    # ========================================================

    def _resolve_enrollment(
        self,
        target_speaker: str,
        enrollment_dir: Path | None,
    ) -> Path | None:

        if enrollment_dir is None:
            return None

        enrollment_dir = Path(
            enrollment_dir
        )

        if not enrollment_dir.exists():

            logger.warning(
                "Enrollment directory does not exist: %s",
                enrollment_dir,
            )

            return None

        candidates = [
            enrollment_dir
            / f"{target_speaker}.wav",

            enrollment_dir
            / f"{target_speaker}.WAV",
        ]

        for candidate in candidates:

            if candidate.exists():
                return candidate.resolve()

        logger.warning(
            "No enrollment found for %s in %s",
            target_speaker,
            enrollment_dir,
        )

        return None

    # ========================================================
    # TSE
    # ========================================================

    def _run_tse(
        self,
        mixture: np.ndarray,
        enrollment: Path,
        debug_dir: Path | None = None,
    ) -> np.ndarray:

        if not self._loaded:
            self.load()

        if self._model is None:
            raise RuntimeError(
                "TSE model is not loaded."
            )

        use_persistent_temp = (
            self.keep_tse_temp
            or (
                self.debug_audio_enabled
                and self.save_tse_inputs
            )
        )

        if use_persistent_temp:

            self.temp_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            operation_dir = (
                self.temp_dir
                / (
                    debug_dir.name
                    if debug_dir is not None
                    else "tse"
                )
            )

            operation_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            mixture_path = (
                operation_dir
                / "mixture.wav"
            )

            sf.write(
                str(mixture_path),
                mixture,
                self.sample_rate,
            )

            with torch.inference_mode():

                result = (
                    self._model.extract_speech(
                        str(mixture_path),
                        str(enrollment),
                    )
                )

            extracted = self._to_numpy(
                result
            )

            if (
                self.debug_audio_enabled
                and self.save_tse_inputs
                and debug_dir is not None
            ):

                self._save_debug_audio(
                    debug_dir
                    / "mixture.wav",
                    mixture,
                )

            if (
                self.debug_audio_enabled
                and self.save_tse_outputs
                and debug_dir is not None
            ):

                self._save_debug_audio(
                    debug_dir
                    / "extracted.wav",
                    extracted,
                )

            return extracted

        # ----------------------------------------------------
        # Normal temporary mode
        # ----------------------------------------------------

        self.temp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        with tempfile.TemporaryDirectory(
            prefix="persian_tse_",
            dir=str(self.temp_dir),
        ) as temp_dir:

            temp_dir = Path(
                temp_dir
            )

            mixture_path = (
                temp_dir
                / "mixture.wav"
            )

            sf.write(
                str(mixture_path),
                mixture,
                self.sample_rate,
            )

            with torch.inference_mode():

                result = (
                    self._model.extract_speech(
                        str(mixture_path),
                        str(enrollment),
                    )
                )

            extracted = self._to_numpy(
                result
            )

            if (
                self.debug_audio_enabled
                and self.save_tse_inputs
                and debug_dir is not None
            ):

                self._save_debug_audio(
                    debug_dir
                    / "mixture.wav",
                    mixture,
                )

            if (
                self.debug_audio_enabled
                and self.save_tse_outputs
                and debug_dir is not None
            ):

                self._save_debug_audio(
                    debug_dir
                    / "extracted.wav",
                    extracted,
                )

            return extracted

    # ========================================================
    # Crop
    # ========================================================

    def _crop_center(
        self,
        extracted: np.ndarray,
        segment_start: float,
        overlap_start: float,
        overlap_end: float,
    ) -> np.ndarray:

        relative_start = (
            overlap_start
            - segment_start
        )

        relative_end = (
            overlap_end
            - segment_start
        )

        start_sample = int(
            round(
                relative_start
                * self.sample_rate
            )
        )

        end_sample = int(
            round(
                relative_end
                * self.sample_rate
            )
        )

        start_sample = max(
            0,
            start_sample,
        )

        end_sample = min(
            len(extracted),
            end_sample,
        )

        if end_sample <= start_sample:
            return np.zeros(
                0,
                dtype=np.float32,
            )

        return np.ascontiguousarray(
            extracted[
                start_sample:end_sample
            ],
            dtype=np.float32,
        )

    # ========================================================
    # Result
    # ========================================================

    @staticmethod
    def _empty_result(
        start: float,
        end: float,
        duration: float,
        reason: str,
    ) -> dict:

        return {
            "start": start,
            "end": end,
            "duration": duration,
            "processed": False,
            "reason": reason,
            "audio": np.zeros(
                0,
                dtype=np.float32,
            ),
        }

    # ========================================================
    # Public API
    # ========================================================

    def separate(
        self,
        audio: np.ndarray,
        overlap_regions: list[dict],
        target_speaker: str,
        enrollment_dir: Path | None = None,
    ) -> dict:

        audio = self._normalize_audio(
            audio
        )

        if not self.enabled:

            return {
                "processed": False,
                "reason": "disabled",
                "target_speaker": target_speaker,
                "regions": [],
            }

        enrollment = (
            self._resolve_enrollment(
                target_speaker,
                enrollment_dir,
            )
        )

        if enrollment is None:

            return {
                "processed": False,
                "reason": "missing_enrollment",
                "target_speaker": target_speaker,
                "regions": [],
            }

        if not overlap_regions:

            return {
                "processed": False,
                "reason": "no_overlap",
                "target_speaker": target_speaker,
                "regions": [],
            }

        results = []

        for index, region in enumerate(
            overlap_regions
        ):

            start = self._safe_float(
                region.get("start")
            )

            end = self._safe_float(
                region.get("end")
            )

            duration = end - start

            if end <= start:

                results.append(
                    self._empty_result(
                        start,
                        end,
                        duration,
                        "invalid_interval",
                    )
                )

                continue

            if (
                duration
                < self.min_overlap_seconds
            ):

                logger.info(
                    "Skipping tiny overlap: "
                    "%s %.3f -> %.3f (%.3fs)",
                    target_speaker,
                    start,
                    end,
                    duration,
                )

                results.append(
                    self._empty_result(
                        start,
                        end,
                        duration,
                        "too_short",
                    )
                )

                continue

            logger.info(
                "TSE [%s] overlap %d: "
                "%.3f -> %.3f (%.3fs)",
                target_speaker,
                index,
                start,
                end,
                duration,
            )

            debug_dir = None

            if self.debug_audio_enabled:

                debug_dir = (
                    self._debug_region_dir(
                        target_speaker,
                        index,
                    )
                )

                logger.debug(
                    "TSE debug directory: %s",
                    debug_dir,
                )

            (
                mixture_segment,
                segment_start,
                segment_end,
            ) = self._extract_segment(
                audio,
                start,
                end,
            )

            if len(mixture_segment) == 0:

                results.append(
                    self._empty_result(
                        start,
                        end,
                        duration,
                        "empty_segment",
                    )
                )

                continue

            try:

                extracted = self._run_tse(
                    mixture_segment,
                    enrollment,
                    debug_dir,
                )

                overlap_audio = (
                    self._crop_center(
                        extracted,
                        segment_start,
                        start,
                        end,
                    )
                )

                expected_samples = int(
                    round(
                        duration
                        * self.sample_rate
                    )
                )

                if (
                    len(overlap_audio)
                    < expected_samples
                ):

                    overlap_audio = np.pad(
                        overlap_audio,
                        (
                            0,
                            expected_samples
                            - len(overlap_audio),
                        ),
                    )

                elif (
                    len(overlap_audio)
                    > expected_samples
                ):

                    overlap_audio = (
                        overlap_audio[
                            :expected_samples
                        ]
                    )

                overlap_audio = np.ascontiguousarray(
                    overlap_audio,
                    dtype=np.float32,
                )

                if (
                    self.debug_audio_enabled
                    and self.save_tse_overlap
                    and debug_dir is not None
                ):

                    self._save_debug_audio(
                        debug_dir
                        / "overlap.wav",
                        overlap_audio,
                    )

                results.append(
                    {
                        "start": start,
                        "end": end,
                        "duration": duration,
                        "processed": True,
                        "reason": None,
                        "audio": overlap_audio,
                    }
                )

            except torch.cuda.OutOfMemoryError:

                logger.exception(
                    "CUDA OOM during TSE: "
                    "%s %.3f -> %.3f",
                    target_speaker,
                    start,
                    end,
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                results.append(
                    self._empty_result(
                        start,
                        end,
                        duration,
                        "cuda_oom",
                    )
                )

            except Exception:

                logger.exception(
                    "TSE failed: "
                    "%s %.3f -> %.3f",
                    target_speaker,
                    start,
                    end,
                )

                results.append(
                    self._empty_result(
                        start,
                        end,
                        duration,
                        "exception",
                    )
                )

        processed_count = sum(
            1
            for item in results
            if item["processed"]
        )

        return {
            "processed": (
                processed_count > 0
            ),
            "target_speaker": target_speaker,
            "enrollment": str(
                enrollment
            ),
            "processed_count": processed_count,
            "total_count": len(results),
            "regions": results,
        }

    # ========================================================
    # Unload
    # ========================================================

    def unload(self) -> None:

        logger.info(
            "Unloading overlap separation engine..."
        )

        self._model = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(
            "Overlap separation engine unloaded."
        )