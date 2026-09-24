from pathlib import Path
import json
import logging
import os
import sys
import time

import yaml

from engines.audio_chunker import AudioChunker
from engines.whisper_persian_v4.engine import WhisperPersianV4Engine


# ============================================================
# Offline mode
# ============================================================

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


# ============================================================
# Engine registry
# ============================================================

ENGINE_REGISTRY = {
    "whisper_persian_v4": WhisperPersianV4Engine,
}


# ============================================================
# Logging
# ============================================================

def setup_logging(config: dict) -> None:
    level_name = config.get("logging", {}).get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


logger = logging.getLogger("persian_asr")


# ============================================================
# Configuration
# ============================================================

def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Configuration file is empty or invalid.")

    return config


# ============================================================
# Engine
# ============================================================

def create_engine(config: dict):
    engine_name = config.get("engine", {}).get("name")

    if not engine_name:
        raise ValueError("engine.name is missing in config.yaml")

    engine_class = ENGINE_REGISTRY.get(engine_name)

    if engine_class is None:
        available = ", ".join(ENGINE_REGISTRY.keys())

        raise ValueError(
            f"Unknown engine: {engine_name}\n"
            f"Available engines: {available}"
        )

    logger.info("Selected engine: %s", engine_name)

    return engine_class(config)


# ============================================================
# Input discovery
# ============================================================

def find_audio_files(
    input_dir: Path,
    extensions: list[str],
) -> list[Path]:

    if not input_dir.exists():
        input_dir.mkdir(parents=True, exist_ok=True)
        return []

    normalized_extensions = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in extensions
    }

    files = [
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in normalized_extensions
    ]

    return sorted(files)


# ============================================================
# Formatting helpers
# ============================================================

def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"

    return f"{minutes:02d}:{secs:05.2f}"


# ============================================================
# Process one audio file
# ============================================================

def process_audio(
    engine,
    chunker,
    audio_path: Path,
    output_dir: Path,
    config: dict,
) -> bool:

    logger.info("")
    logger.info("=" * 60)
    logger.info("Processing: %s", audio_path.name)

    file_start_time = time.perf_counter()

    try:
        # ----------------------------------------------------
        # Chunk audio
        # ----------------------------------------------------

        logger.info("Analyzing audio and creating chunks...")

        chunks = chunker.chunk_file(audio_path)

        if not chunks:
            logger.error(
                "No audio chunks were created for %s",
                audio_path.name,
            )
            return False

        logger.info(
            "Created %d chunk(s) from %s",
            len(chunks),
            audio_path.name,
        )

        # ----------------------------------------------------
        # Chunk information
        # ----------------------------------------------------

        for chunk in chunks:
            logger.info(
                "Chunk %03d | %s -> %s | %.2f sec",
                chunk.index,
                format_seconds(chunk.start),
                format_seconds(chunk.end),
                chunk.duration,
            )

        # ----------------------------------------------------
        # Transcription
        # ----------------------------------------------------

        chunk_results = []
        texts = []

        total_audio_duration = 0.0
        total_inference_time = 0.0

        logger.info("")
        logger.info("Starting transcription...")

        for position, chunk in enumerate(chunks, start=1):

            logger.info(
                "[%d/%d] Transcribing chunk %03d...",
                position,
                len(chunks),
                chunk.index,
            )

            inference_start = time.perf_counter()

            result = engine.transcribe(
                chunk.audio,
                sample_rate=chunker.sample_rate,
            )

            inference_time = time.perf_counter() - inference_start

            text = str(result.get("text", "")).strip()

            chunk_duration = float(chunk.duration)

            total_audio_duration += chunk_duration
            total_inference_time += inference_time

            if text:
                texts.append(text)

            chunk_result = {
                "index": int(chunk.index),
                "start": round(float(chunk.start), 3),
                "end": round(float(chunk.end), 3),
                "duration": round(chunk_duration, 3),
                "inference_time": round(inference_time, 3),
                "rtf": round(
                    inference_time / chunk_duration,
                    6,
                )
                if chunk_duration > 0
                else None,
                "realtime_factor": round(
                    chunk_duration / inference_time,
                    3,
                )
                if inference_time > 0
                else None,
                "text": text,
            }

            chunk_results.append(chunk_result)

            logger.info(
                "Chunk %03d completed | "
                "audio=%.2fs | inference=%.2fs | RTF=%.3f",
                chunk.index,
                chunk_duration,
                inference_time,
                chunk_result["rtf"]
                if chunk_result["rtf"] is not None
                else 0.0,
            )

            if text:
                logger.info(
                    "Text: %s",
                    text,
                )
            else:
                logger.info("Text: [empty]")

        # ----------------------------------------------------
        # Merge final text
        # ----------------------------------------------------

        final_text = " ".join(
            text.strip()
            for text in texts
            if text and text.strip()
        ).strip()

        # ----------------------------------------------------
        # Aggregate metrics
        # ----------------------------------------------------

        file_processing_time = (
            time.perf_counter() - file_start_time
        )

        overall_rtf = (
            total_inference_time / total_audio_duration
            if total_audio_duration > 0
            else None
        )

        overall_realtime_factor = (
            total_audio_duration / total_inference_time
            if total_inference_time > 0
            else None
        )

        # ----------------------------------------------------
        # Output paths
        # ----------------------------------------------------

        txt_path = output_dir / f"{audio_path.stem}.txt"
        json_path = output_dir / f"{audio_path.stem}.json"

        # ----------------------------------------------------
        # Save TXT
        # ----------------------------------------------------

        if config.get("output", {}).get("save_txt", True):

            txt_path.write_text(
                final_text + "\n",
                encoding="utf-8",
            )

            logger.info(
                "TXT saved: %s",
                txt_path,
            )

        # ----------------------------------------------------
        # Save JSON
        # ----------------------------------------------------

        if config.get("output", {}).get("save_json", True):

            json_data = {
                "file": {
                    "name": audio_path.name,
                    "path": str(audio_path),
                    "extension": audio_path.suffix.lower(),
                },

                "engine": {
                    "name": config.get("engine", {}).get("name"),
                    "model_path": str(
                        PROJECT_ROOT
                        / config.get("model", {}).get(
                            "path",
                            "",
                        )
                    ),
                },

                "audio": {
                    "sample_rate": chunker.sample_rate,
                    "language": config.get("audio", {}).get(
                        "language",
                        "fa",
                    ),
                    "task": config.get("audio", {}).get(
                        "task",
                        "transcribe",
                    ),
                    "duration": round(
                        total_audio_duration,
                        3,
                    ),
                },

                "processing": {
                    "chunks": len(chunks),
                    "inference_time": round(
                        total_inference_time,
                        3,
                    ),
                    "total_processing_time": round(
                        file_processing_time,
                        3,
                    ),
                    "rtf": round(
                        overall_rtf,
                        6,
                    )
                    if overall_rtf is not None
                    else None,
                    "realtime_factor": round(
                        overall_realtime_factor,
                        3,
                    )
                    if overall_realtime_factor is not None
                    else None,
                },

                "transcription": {
                    "text": final_text,
                },

                "chunks": chunk_results,
            }

            json_path.write_text(
                json.dumps(
                    json_data,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            logger.info(
                "JSON saved: %s",
                json_path,
            )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        logger.info("")
        logger.info("-" * 60)
        logger.info("FILE COMPLETED")
        logger.info("File       : %s", audio_path.name)
        logger.info(
            "Audio      : %.2f sec",
            total_audio_duration,
        )
        logger.info(
            "Chunks     : %d",
            len(chunks),
        )
        logger.info(
            "Inference  : %.2f sec",
            total_inference_time,
        )
        logger.info(
            "Total time : %.2f sec",
            file_processing_time,
        )

        if overall_rtf is not None:
            logger.info(
                "RTF        : %.3f",
                overall_rtf,
            )

        if overall_realtime_factor is not None:
            logger.info(
                "Realtime   : %.2fx",
                overall_realtime_factor,
            )

        logger.info("-" * 60)

        return True

    except Exception:
        logger.exception(
            "Failed processing file: %s",
            audio_path.name,
        )

        return False


# ============================================================
# Main
# ============================================================

def main() -> int:

    start_time = time.perf_counter()

    try:

        print("=" * 60)

        # ----------------------------------------------------
        # Load configuration
        # ----------------------------------------------------

        config = load_config(CONFIG_PATH)

        setup_logging(config)

        logger.info("Persian ASR starting...")
        logger.info(
            "Project root: %s",
            PROJECT_ROOT,
        )
        logger.info(
            "Loading configuration: %s",
            CONFIG_PATH,
        )

        # ----------------------------------------------------
        # Paths
        # ----------------------------------------------------

        audio_config = config.get("audio", {})

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

        model_path = (
            PROJECT_ROOT
            / config.get("model", {}).get(
                "path",
                "",
            )
        )

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
            "Model directory : %s",
            model_path,
        )

        # ----------------------------------------------------
        # Validate model
        # ----------------------------------------------------

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model directory not found: {model_path}"
            )

        # ----------------------------------------------------
        # Find input files
        # ----------------------------------------------------

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
                ", ".join(extensions),
            )

            return 0

        logger.info(
            "Found %d audio file(s).",
            len(audio_files),
        )

        for index, audio_file in enumerate(
            audio_files,
            start=1,
        ):
            logger.info(
                "[%d/%d] %s",
                index,
                len(audio_files),
                audio_file.name,
            )

        # ----------------------------------------------------
        # Create engine
        # ----------------------------------------------------

        engine = create_engine(config)

        # ----------------------------------------------------
        # Create chunker
        # ----------------------------------------------------

        chunker = AudioChunker(config)

        # ----------------------------------------------------
        # Load engine
        # ----------------------------------------------------

        logger.info("Loading engine...")

        engine_load_start = time.perf_counter()

        engine.load()

        engine_load_time = (
            time.perf_counter()
            - engine_load_start
        )

        logger.info(
            "Engine loaded successfully in %.3f sec.",
            engine_load_time,
        )

        # ----------------------------------------------------
        # Process files
        # ----------------------------------------------------

        successful = 0
        failed = 0

        try:

            for index, audio_path in enumerate(
                audio_files,
                start=1,
            ):

                logger.info("")
                logger.info(
                    "FILE [%d/%d]",
                    index,
                    len(audio_files),
                )

                success = process_audio(
                    engine,
                    chunker,
                    audio_path,
                    output_dir,
                    config,
                )

                if success:
                    successful += 1
                else:
                    failed += 1

        finally:

            # ------------------------------------------------
            # Unload engine
            # ------------------------------------------------

            logger.info("Unloading engine...")

            try:
                engine.unload()
            except Exception:
                logger.exception(
                    "Error while unloading engine."
                )

            logger.info("Engine unloaded.")

        # ----------------------------------------------------
        # Final summary
        # ----------------------------------------------------

        total_time = (
            time.perf_counter()
            - start_time
        )

        logger.info("")
        logger.info("=" * 60)
        logger.info("PROCESSING FINISHED")
        logger.info(
            "Total files : %d",
            len(audio_files),
        )
        logger.info(
            "Successful   : %d",
            successful,
        )
        logger.info(
            "Failed       : %d",
            failed,
        )
        logger.info(
            "Total time   : %.3f sec",
            total_time,
        )
        logger.info(
            "Output       : %s",
            output_dir,
        )
        logger.info("=" * 60)

        return 0 if failed == 0 else 1

    except KeyboardInterrupt:

        logger.warning(
            "Process interrupted by user."
        )

        return 130

    except Exception:

        logger.exception("Fatal error.")

        return 1


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    sys.exit(main())