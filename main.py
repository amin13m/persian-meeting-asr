from pathlib import Path
import logging
import os
import sys
import time

import yaml

from engines.whisper_persian_v4.engine import (
    WhisperPersianV4Engine,
)

from engines.shenava_koochik.engine import (
    ShenavaKoochikEngine,
)

from pipeline.meeting_pipeline import (
    MeetingPipeline,
)


# ============================================================
# Offline mode
# ============================================================

os.environ.setdefault(
    "HF_HUB_OFFLINE",
    "1",
)

os.environ.setdefault(
    "TRANSFORMERS_OFFLINE",
    "1",
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "config.yaml"
)


# ============================================================
# Logging
# ============================================================

logger = logging.getLogger(
    "persian_asr"
)


def setup_logging(
    config: dict,
) -> None:

    level_name = (
        config
        .get("logging", {})
        .get("level", "INFO")
        .upper()
    )

    level = getattr(
        logging,
        level_name,
        logging.INFO,
    )

    logging.basicConfig(
        level=level,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
        datefmt="%H:%M:%S",
    )


# ============================================================
# Configuration
# ============================================================

def load_config(
    path: Path,
) -> dict:

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(
            "config.yaml is empty or invalid."
        )

    return config


# ============================================================
# Input discovery
# ============================================================

def find_audio_files(
    input_dir: Path,
    extensions: list[str],
) -> list[Path]:

    input_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalized_extensions = {
        (
            ext.lower()
            if ext.startswith(".")
            else f".{ext.lower()}"
        )
        for ext in extensions
    }

    audio_files = [
        path
        for path in input_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in normalized_extensions
        )
    ]

    return sorted(audio_files)


# ============================================================
# Resolve model paths
# ============================================================

def resolve_path(
    project_root: Path,
    configured_path: str,
) -> Path:

    path = Path(
        configured_path
    )

    if path.is_absolute():
        return path

    return (
        project_root
        / path
    )


# ============================================================
# Validate Whisper model
# ============================================================

def validate_whisper_model(
    model_path: Path,
) -> None:

    if not model_path.exists():

        raise FileNotFoundError(
            "Whisper model directory not found: "
            f"{model_path}"
        )

    if not model_path.is_dir():

        raise ValueError(
            "Whisper model path is not a directory: "
            f"{model_path}"
        )


# ============================================================
# Validate Shenava model
# ============================================================

def validate_shenava_model(
    model_path: Path,
) -> None:

    if not model_path.exists():

        raise FileNotFoundError(
            "Shenava model checkpoint not found: "
            f"{model_path}"
        )

    if not model_path.is_file():

        raise ValueError(
            "Shenava model path is not a file: "
            f"{model_path}"
        )

    if model_path.suffix.lower() != ".nemo":

        raise ValueError(
            "Shenava model must be a .nemo checkpoint: "
            f"{model_path}"
        )


# ============================================================
# Main
# ============================================================

def main() -> int:

    process_start = time.perf_counter()

    whisper_engine = None
    shenava_engine = None
    meeting_pipeline = None

    successful = 0
    failed = 0
    audio_files = []

    try:

        print("=" * 70)

        # ====================================================
        # Configuration
        # ====================================================

        config = load_config(
            CONFIG_PATH
        )

        setup_logging(
            config
        )

        logger.info(
            "Persian ASR / Meeting Pipeline starting..."
        )

        logger.info(
            "Project root: %s",
            PROJECT_ROOT,
        )

        logger.info(
            "Configuration: %s",
            CONFIG_PATH,
        )

        # ====================================================
        # Read ASR configuration
        # ====================================================

        asr_config = config.get(
            "asr",
            {},
        )

        if not isinstance(
            asr_config,
            dict,
        ):
            raise ValueError(
                "config.asr must be a dictionary."
            )

        primary_config = asr_config.get(
            "primary",
            {},
        )

        secondary_config = asr_config.get(
            "secondary",
            {},
        )

        if not isinstance(
            primary_config,
            dict,
        ):
            raise ValueError(
                "config.asr.primary must be a dictionary."
            )

        if not isinstance(
            secondary_config,
            dict,
        ):
            raise ValueError(
                "config.asr.secondary must be a dictionary."
            )

        # ====================================================
        # Primary / Secondary enabled
        # ====================================================

        primary_enabled = bool(
            primary_config.get(
                "enabled",
                True,
            )
        )

        secondary_enabled = bool(
            secondary_config.get(
                "enabled",
                False,
            )
        )

        if not primary_enabled:
            raise ValueError(
                "Primary ASR is disabled. "
                "This pipeline currently requires a primary ASR."
            )

        primary_engine_name = (
            primary_config.get(
                "engine",
                "whisper_persian_v4",
            )
        )

        secondary_engine_name = (
            secondary_config.get(
                "engine",
                "shenava_koochik",
            )
        )

        logger.info(
            "Primary ASR engine  : %s",
            primary_engine_name,
        )

        logger.info(
            "Secondary ASR       : %s",
            (
                secondary_engine_name
                if secondary_enabled
                else "disabled"
            ),
        )

        # ====================================================
        # Paths
        # ====================================================

        audio_config = config.get(
            "audio",
            {},
        )

        input_dir = resolve_path(
            PROJECT_ROOT,
            audio_config.get(
                "input_dir",
                "input",
            ),
        )

        output_dir = resolve_path(
            PROJECT_ROOT,
            audio_config.get(
                "output_dir",
                "output",
            ),
        )

        # ====================================================
        # Whisper model path
        #
        # Preferred:
        #   asr.primary.model.path
        #
        # Legacy fallback:
        #   model.path
        # ====================================================

        whisper_model_config = (
            primary_config
            .get(
                "model",
                {},
            )
        )

        if isinstance(
            whisper_model_config,
            dict,
        ):

            whisper_model_path_str = (
                whisper_model_config.get(
                    "path",
                    "",
                )
            )

        else:

            whisper_model_path_str = (
                str(
                    whisper_model_config
                )
            )

        if not whisper_model_path_str:

            whisper_model_path_str = (
                config
                .get(
                    "model",
                    {},
                )
                .get(
                    "path",
                    "",
                )
            )

        whisper_model_path = resolve_path(
            PROJECT_ROOT,
            whisper_model_path_str,
        )

        # ====================================================
        # Shenava model path
        # ====================================================

        shenava_model_config = (
            secondary_config
            .get(
                "model",
                {},
            )
        )

        if isinstance(
            shenava_model_config,
            dict,
        ):

            shenava_model_path_str = (
                shenava_model_config.get(
                    "path",
                    "",
                )
            )

        else:

            shenava_model_path_str = (
                str(
                    shenava_model_config
                )
            )

        shenava_model_path = resolve_path(
            PROJECT_ROOT,
            shenava_model_path_str,
        )

        # ====================================================
        # Create directories
        # ====================================================

        input_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.info(
            "Input directory : %s",
            input_dir,
        )

        logger.info(
            "Output directory: %s",
            output_dir,
        )

        logger.info(
            "Whisper model   : %s",
            whisper_model_path,
        )

        if secondary_enabled:

            logger.info(
                "Shenava model   : %s",
                shenava_model_path,
            )

        # ====================================================
        # Validate models
        # ====================================================

        validate_whisper_model(
            whisper_model_path
        )

        if secondary_enabled:

            validate_shenava_model(
                shenava_model_path
            )

        # ====================================================
        # Find input audio files
        # ====================================================

        extensions = audio_config.get(
            "extensions",
            [
                ".wav",
                ".mp3",
                ".flac",
                ".m4a",
                ".ogg",
            ],
        )

        if not isinstance(
            extensions,
            list,
        ):

            raise ValueError(
                "audio.extensions must be a list."
            )

        audio_files = find_audio_files(
            input_dir,
            extensions,
        )

        if not audio_files:

            logger.warning(
                "No audio files found in: %s",
                input_dir,
            )

            logger.info(
                "Supported extensions: %s",
                ", ".join(
                    str(ext)
                    for ext in extensions
                ),
            )

            return 0

        logger.info(
            "Found %d audio file(s).",
            len(audio_files),
        )

        for index, audio_path in enumerate(
            audio_files,
            start=1,
        ):

            logger.info(
                "[%d/%d] %s",
                index,
                len(audio_files),
                audio_path.name,
            )

        # ====================================================
        # Create Primary ASR
        # ====================================================

        logger.info(
            "Creating primary ASR engine: %s",
            primary_engine_name,
        )

        whisper_engine = (
            WhisperPersianV4Engine(
                config
            )
        )

        # ====================================================
        # Load Primary ASR
        # ====================================================

        logger.info(
            "Loading primary ASR engine..."
        )

        asr_start = time.perf_counter()

        whisper_engine.load()

        asr_load_time = (
            time.perf_counter()
            - asr_start
        )

        logger.info(
            "Primary ASR loaded in %.3f sec.",
            asr_load_time,
        )

        # ====================================================
        # Create Secondary ASR
        # ====================================================

        if secondary_enabled:

            logger.info(
                "Creating secondary ASR engine: %s",
                secondary_engine_name,
            )

            shenava_engine = (
                ShenavaKoochikEngine(
                    config
                )
            )

            # ------------------------------------------------
            # Load Shenava
            # ------------------------------------------------

            logger.info(
                "Loading secondary ASR engine..."
            )

            shenava_start = (
                time.perf_counter()
            )

            shenava_engine.load()

            shenava_load_time = (
                time.perf_counter()
                - shenava_start
            )

            logger.info(
                "Secondary ASR loaded in %.3f sec.",
                shenava_load_time,
            )

        else:

            logger.info(
                "Secondary ASR is disabled."
            )

        # ====================================================
        # Create Meeting Pipeline
        # ====================================================

        logger.info(
            "Creating meeting pipeline..."
        )

        meeting_pipeline = (
            MeetingPipeline(
                config=config,
                asr_engine=whisper_engine,
                secondary_asr_engine=shenava_engine,
            )
        )

        # ====================================================
        # Load Meeting Pipeline
        # ====================================================

        logger.info(
            "Loading meeting pipeline..."
        )

        pipeline_load_start = (
            time.perf_counter()
        )

        meeting_pipeline.load()

        pipeline_load_time = (
            time.perf_counter()
            - pipeline_load_start
        )

        logger.info(
            "Meeting pipeline loaded in %.3f sec.",
            pipeline_load_time,
        )

        # ====================================================
        # ASR architecture summary
        # ====================================================

        logger.info("")
        logger.info(
            "ASR configuration:"
        )

        logger.info(
            "  Primary   : %s",
            primary_engine_name,
        )

        logger.info(
            "  Secondary : %s",
            (
                secondary_engine_name
                if secondary_enabled
                else "disabled"
            ),
        )

        logger.info(
            "  Same group audio -> both engines"
        )

        # ====================================================
        # Process files
        # ====================================================

        for index, audio_path in enumerate(
            audio_files,
            start=1,
        ):

            logger.info("")

            logger.info(
                "=" * 70
            )

            logger.info(
                "FILE [%d/%d] %s",
                index,
                len(audio_files),
                audio_path.name,
            )

            logger.info(
                "=" * 70
            )

            file_start = (
                time.perf_counter()
            )

            # ------------------------------------------------
            # Output directory per input file
            # ------------------------------------------------

            file_output_dir = (
                output_dir
                / audio_path.stem
            )

            file_output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            try:

                # --------------------------------------------
                # Run complete meeting pipeline
                # --------------------------------------------

                result = (
                    meeting_pipeline.process(
                        audio_path,
                        file_output_dir,
                    )
                )

                # --------------------------------------------
                # File completed
                # --------------------------------------------

                file_time = (
                    time.perf_counter()
                    - file_start
                )

                successful += 1

                logger.info("")
                logger.info(
                    "-" * 70
                )

                logger.info(
                    "FILE COMPLETED"
                )

                logger.info(
                    "File       : %s",
                    audio_path.name,
                )

                logger.info(
                    "Output     : %s",
                    file_output_dir,
                )

                logger.info(
                    "Time       : %.3f sec",
                    file_time,
                )

                if result is not None:

                    logger.debug(
                        "Pipeline result type: %s",
                        type(result).__name__,
                    )

                logger.info(
                    "-" * 70
                )

            except Exception:

                failed += 1

                file_time = (
                    time.perf_counter()
                    - file_start
                )

                logger.exception(
                    "Failed processing %s "
                    "after %.3f sec.",
                    audio_path.name,
                    file_time,
                )

                # --------------------------------------------
                # Continue with next input file
                # --------------------------------------------

                continue

        # ====================================================
        # Processing summary
        # ====================================================

        total_time = (
            time.perf_counter()
            - process_start
        )

        logger.info("")
        logger.info(
            "=" * 70
        )

        logger.info(
            "PROCESSING FINISHED"
        )

        logger.info(
            "Total files : %d",
            len(audio_files),
        )

        logger.info(
            "Successful  : %d",
            successful,
        )

        logger.info(
            "Failed      : %d",
            failed,
        )

        logger.info(
            "Total time  : %.3f sec",
            total_time,
        )

        logger.info(
            "Output      : %s",
            output_dir,
        )

        logger.info(
            "=" * 70
        )

        return (
            0
            if failed == 0
            else 1
        )

    # ========================================================
    # Keyboard interrupt
    # ========================================================

    except KeyboardInterrupt:

        logger.warning(
            "Process interrupted by user."
        )

        return 130

    # ========================================================
    # Fatal error
    # ========================================================

    except Exception:

        logger.exception(
            "Fatal error."
        )

        return 1

    # ========================================================
    # Cleanup
    # ========================================================

    finally:

        # ----------------------------------------------------
        # Meeting pipeline
        #
        # Pipeline owns diarization / TSE / grouping.
        # It should NOT unload ASR engines if main owns them.
        # ----------------------------------------------------

        if meeting_pipeline is not None:

            logger.info(
                "Unloading meeting pipeline..."
            )

            try:

                meeting_pipeline.unload()

                logger.info(
                    "Meeting pipeline unloaded."
                )

            except Exception:

                logger.exception(
                    "Meeting pipeline unload failed."
                )

        # ----------------------------------------------------
        # Secondary ASR
        # ----------------------------------------------------

        if shenava_engine is not None:

            logger.info(
                "Unloading Shenava engine..."
            )

            try:

                shenava_engine.unload()

                logger.info(
                    "Shenava engine unloaded."
                )

            except Exception:

                logger.exception(
                    "Shenava unload failed."
                )

        # ----------------------------------------------------
        # Primary ASR
        # ----------------------------------------------------

        if whisper_engine is not None:

            logger.info(
                "Unloading Whisper engine..."
            )

            try:

                whisper_engine.unload()

                logger.info(
                    "Whisper engine unloaded."
                )

            except Exception:

                logger.exception(
                    "Whisper unload failed."
                )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    sys.exit(
        main()
    )