from pathlib import Path
import logging
import os
import sys
import time

import yaml

from engines.whisper_persian_v4.engine import (
    WhisperPersianV4Engine,
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
# Main
# ============================================================

def main() -> int:

    process_start = time.perf_counter()

    asr_engine = None
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
        # Paths
        # ====================================================

        audio_config = config.get(
            "audio",
            {},
        )

        input_dir = (
            PROJECT_ROOT
            / audio_config.get(
                "input_dir",
                "input",
            )
        )

        output_dir = (
            PROJECT_ROOT
            / audio_config.get(
                "output_dir",
                "output",
            )
        )

        whisper_model_path = (
            PROJECT_ROOT
            / config
            .get("model", {})
            .get(
                "path",
                "",
            )
        )

        # ----------------------------------------------------
        # Create directories
        # ----------------------------------------------------

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

        # ====================================================
        # Validate Whisper model
        # ====================================================

        if not whisper_model_path.exists():
            raise FileNotFoundError(
                "Whisper model directory not found: "
                f"{whisper_model_path}"
            )

        if not whisper_model_path.is_dir():
            raise ValueError(
                "Whisper model path is not a directory: "
                f"{whisper_model_path}"
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
        # Create Whisper ASR engine
        # ====================================================

        logger.info(
            "Creating Whisper engine..."
        )

        asr_engine = (
            WhisperPersianV4Engine(
                config
            )
        )

        # ====================================================
        # Load Whisper
        # ====================================================

        logger.info(
            "Loading Whisper engine..."
        )

        asr_start = time.perf_counter()

        asr_engine.load()

        asr_load_time = (
            time.perf_counter()
            - asr_start
        )

        logger.info(
            "Whisper loaded in %.3f sec.",
            asr_load_time,
        )

        # ====================================================
        # Create Meeting Pipeline
        # ====================================================

        logger.info(
            "Creating meeting pipeline..."
        )

        meeting_pipeline = (
            MeetingPipeline(
                config,
                asr_engine,
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

                # ------------------------------------------------
                # Optional result logging
                # ------------------------------------------------

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
        # Whisper
        # ----------------------------------------------------

        if asr_engine is not None:

            logger.info(
                "Unloading Whisper engine..."
            )

            try:

                asr_engine.unload()

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