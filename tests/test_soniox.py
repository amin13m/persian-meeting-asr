
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

# File location:
# D:\Projects\persian_asr\tests\test_soniox.py
#
# Project root:
# D:\Projects\persian_asr

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_FILE, override=False)


# ============================================================
# SONIOX CONFIGURATION
# ============================================================

API_BASE_URL = "https://api.soniox.com/v1"

API_KEY = os.getenv("SONIOX_API_KEY", "").strip()

MODEL = "stt-async-v4"

LANGUAGE = "fa"

ENABLE_SPEAKER_DIARIZATION = True

ENABLE_LANGUAGE_IDENTIFICATION = True

POLL_INTERVAL_SECONDS = 2.0

MAX_WAIT_SECONDS = 600

REQUEST_TIMEOUT_SECONDS = 60

OUTPUT_DIR = PROJECT_ROOT / "output" / "soniox"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("soniox_test")


# ============================================================
# EXCEPTIONS
# ============================================================

class SonioxAPIError(RuntimeError):
    """Raised when Soniox API returns an error."""


# ============================================================
# CLIENT
# ============================================================

class SonioxTestClient:

    def __init__(self, api_key: str) -> None:

        if not api_key:
            raise ValueError(
                "SONIOX_API_KEY is missing. "
                f"Check the environment file: {ENV_FILE}"
            )

        self.session = requests.Session()

        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
            }
        )

    # --------------------------------------------------------
    # HTTP REQUEST
    # --------------------------------------------------------

    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> requests.Response:

        url = f"{API_BASE_URL}{endpoint}"

        kwargs.setdefault(
            "timeout",
            REQUEST_TIMEOUT_SECONDS,
        )

        try:
            response = self.session.request(
                method=method,
                url=url,
                **kwargs,
            )

        except requests.Timeout as exc:
            raise SonioxAPIError(
                f"Request timed out: {method} {endpoint}"
            ) from exc

        except requests.RequestException as exc:
            raise SonioxAPIError(
                f"Network error: {method} {endpoint}: {exc}"
            ) from exc

        if not response.ok:

            try:
                error_data = response.json()

            except ValueError:
                error_data = {
                    "message": response.text[:2000],
                }

            message = (
                f"HTTP {response.status_code}\n"
                f"Endpoint: {endpoint}\n"
                f"Response: "
                f"{json.dumps(error_data, ensure_ascii=False)}"
            )

            if response.status_code == 401:
                message += (
                    "\nCheck whether SONIOX_API_KEY is valid "
                    "and belongs to the intended project."
                )

            elif response.status_code == 402:
                message += (
                    "\nCheck the Soniox organization balance "
                    "and project budget."
                )

            elif response.status_code == 403:
                message += (
                    "\nThe API key may not have permission "
                    "to perform this operation."
                )

            elif response.status_code == 429:
                message += (
                    "\nThe API rate or capacity limit was reached."
                )

            raise SonioxAPIError(message)

        return response

    # --------------------------------------------------------
    # UPLOAD AUDIO
    # --------------------------------------------------------

    def upload_file(
        self,
        audio_path: Path,
    ) -> dict[str, Any]:

        logger.info(
            "Uploading audio: %s",
            audio_path.name,
        )

        with audio_path.open("rb") as audio_file:

            response = self._request(
                "POST",
                "/files",
                files={
                    "file": (
                        audio_path.name,
                        audio_file,
                    )
                },
            )

        result = response.json()

        file_id = result.get("id")

        if not file_id:
            raise SonioxAPIError(
                "Upload response did not contain a file ID."
            )

        logger.info(
            "Upload completed. File ID: %s",
            file_id,
        )

        return result

    # --------------------------------------------------------
    # CREATE TRANSCRIPTION
    # --------------------------------------------------------

    def create_transcription(
        self,
        file_id: str,
    ) -> dict[str, Any]:

        payload = {
            "model": MODEL,
            "file_id": file_id,
            "language_hints": [LANGUAGE],
            "enable_speaker_diarization": (
                ENABLE_SPEAKER_DIARIZATION
            ),
            "enable_language_identification": (
                ENABLE_LANGUAGE_IDENTIFICATION
            ),
        }

        logger.info("Creating transcription job...")

        response = self._request(
            "POST",
            "/transcriptions",
            json=payload,
        )

        result = response.json()

        transcription_id = result.get("id")

        if not transcription_id:
            raise SonioxAPIError(
                "Transcription response did not contain an ID."
            )

        logger.info(
            "Transcription ID: %s",
            transcription_id,
        )

        logger.info(
            "Initial status: %s",
            result.get("status"),
        )

        return result

    # --------------------------------------------------------
    # WAIT FOR COMPLETION
    # --------------------------------------------------------

    def wait_for_transcription(
        self,
        transcription_id: str,
    ) -> dict[str, Any]:

        start_time = time.perf_counter()

        while True:

            elapsed = time.perf_counter() - start_time

            if elapsed >= MAX_WAIT_SECONDS:
                raise TimeoutError(
                    "Soniox transcription timed out after "
                    f"{MAX_WAIT_SECONDS} seconds. "
                    f"Transcription ID: {transcription_id}"
                )

            response = self._request(
                "GET",
                f"/transcriptions/{transcription_id}",
            )

            result = response.json()

            status = str(
                result.get("status", "")
            ).lower()

            logger.info(
                "Status: %s | elapsed: %.1f sec",
                status,
                elapsed,
            )

            if status == "completed":
                return result

            if status == "error":
                raise SonioxAPIError(
                    "Transcription failed.\n"
                    f"Error type: {result.get('error_type')}\n"
                    f"Message: {result.get('error_message')}\n"
                    f"Transcription ID: {transcription_id}"
                )

            if status not in {"queued", "processing"}:
                raise SonioxAPIError(
                    f"Unexpected transcription status: {status}"
                )

            remaining = MAX_WAIT_SECONDS - elapsed

            time.sleep(
                min(POLL_INTERVAL_SECONDS, max(remaining, 0))
            )

    # --------------------------------------------------------
    # GET TRANSCRIPT
    # --------------------------------------------------------

    def get_transcript(
        self,
        transcription_id: str,
    ) -> dict[str, Any]:

        response = self._request(
            "GET",
            f"/transcriptions/{transcription_id}/transcript",
        )

        return response.json()

    # --------------------------------------------------------
    # CLOSE SESSION
    # --------------------------------------------------------

    def close(self) -> None:
        self.session.close()


# ============================================================
# TRANSCRIPT HELPERS
# ============================================================

def extract_text(
    transcript_data: dict[str, Any],
) -> str:

    tokens = transcript_data.get("tokens", [])

    pieces: list[str] = []

    for token in tokens:

        if not isinstance(token, dict):
            continue

        text = token.get("text", "")

        if text is not None:
            pieces.append(str(text))

    return "".join(pieces).strip()


def extract_speakers(
    transcript_data: dict[str, Any],
) -> list[str]:

    speakers: set[str] = set()

    for token in transcript_data.get("tokens", []):

        if not isinstance(token, dict):
            continue

        speaker = token.get("speaker")

        if speaker is not None:
            speakers.add(str(speaker))

    return sorted(speakers)


# ============================================================
# AUDIO DURATION
# ============================================================

def get_audio_duration(
    audio_path: Path,
) -> float | None:

    try:
        import soundfile as sf

        info = sf.info(str(audio_path))

        if info.samplerate <= 0:
            return None

        return info.frames / info.samplerate

    except Exception as exc:
        logger.warning(
            "Could not determine audio duration: %s",
            exc,
        )

        return None


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_results(
    audio_path: Path,
    transcript_data: dict[str, Any],
    transcription_info: dict[str, Any],
    upload_info: dict[str, Any],
    processing_time: float,
    audio_duration: float | None,
) -> tuple[Path, Path]:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_stem = audio_path.stem

    text = extract_text(transcript_data)

    txt_path = OUTPUT_DIR / f"{output_stem}_soniox.txt"

    json_path = OUTPUT_DIR / f"{output_stem}_soniox.json"

    txt_path.write_text(
        text + "\n",
        encoding="utf-8",
    )

    rtf = None
    realtime_factor = None

    if audio_duration is not None and audio_duration > 0:

        rtf = processing_time / audio_duration

        if processing_time > 0:
            realtime_factor = audio_duration / processing_time

    result = {
        "engine": "soniox",
        "model": MODEL,
        "language_hint": LANGUAGE,
        "audio_file": str(audio_path),
        "audio_duration_seconds": audio_duration,
        "processing_time_seconds": round(processing_time, 3),
        "rtf": round(rtf, 4) if rtf is not None else None,
        "realtime_factor": (
            round(realtime_factor, 3)
            if realtime_factor is not None
            else None
        ),
        "transcription_id": transcription_info.get("id"),
        "upload_file_id": upload_info.get("id"),
        "status": transcription_info.get("status"),
        "text": text,
        "speakers": extract_speakers(transcript_data),
        "tokens": transcript_data.get("tokens", []),
        "transcript_response": transcript_data,
        "transcription_response": transcription_info,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    json_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return txt_path, json_path


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Test Soniox asynchronous speech-to-text "
            "using a local audio file."
        )
    )

    parser.add_argument(
        "audio",
        nargs="?",
        help="Path to the input audio file.",
    )

    args = parser.parse_args()

    # Check .env first, so a missing key is reported clearly.
    if not ENV_FILE.is_file():
        logger.error(
            "Environment file not found: %s",
            ENV_FILE,
        )
        return 1

    if not API_KEY:
        logger.error(
            "SONIOX_API_KEY is missing or empty.\n"
            "Expected file: %s",
            ENV_FILE,
        )
        return 1

    audio_input = args.audio

    if not audio_input:
        try:
            audio_input = input(
                "Enter audio file path: "
            ).strip().strip('"')

        except (EOFError, KeyboardInterrupt):
            logger.warning("No audio path provided.")
            return 130

    if not audio_input:
        logger.error("Audio path is empty.")
        return 1

    audio_path = Path(
        audio_input
    ).expanduser().resolve()

    if not audio_path.exists():
        logger.error(
            "Audio file not found: %s",
            audio_path,
        )
        return 1

    if not audio_path.is_file():
        logger.error(
            "Audio path is not a file: %s",
            audio_path,
        )
        return 1

    client: SonioxTestClient | None = None

    try:

        logger.info("=" * 60)
        logger.info("SONIOX ASR TEST")
        logger.info("Project root: %s", PROJECT_ROOT)
        logger.info("Environment file: %s", ENV_FILE)
        logger.info("Audio: %s", audio_path)
        logger.info("Model: %s", MODEL)
        logger.info("Language hint: %s", LANGUAGE)
        logger.info(
            "Speaker diarization: %s",
            ENABLE_SPEAKER_DIARIZATION,
        )
        logger.info("=" * 60)

        audio_duration = get_audio_duration(audio_path)

        if audio_duration is not None:
            logger.info(
                "Audio duration: %.2f sec",
                audio_duration,
            )

        client = SonioxTestClient(API_KEY)

        start_time = time.perf_counter()

        # 1. Upload the local audio file.
        upload_info = client.upload_file(audio_path)

        file_id = upload_info["id"]

        # 2. Submit the transcription job.
        transcription_info = client.create_transcription(file_id)

        transcription_id = transcription_info["id"]

        # 3. Wait until the job completes.
        transcription_info = client.wait_for_transcription(
            transcription_id
        )

        # 4. Fetch the transcript tokens.
        transcript_data = client.get_transcript(transcription_id)

        processing_time = time.perf_counter() - start_time

        # 5. Save TXT and JSON results.
        txt_path, json_path = save_results(
            audio_path=audio_path,
            transcript_data=transcript_data,
            transcription_info=transcription_info,
            upload_info=upload_info,
            processing_time=processing_time,
            audio_duration=audio_duration,
        )

        text = extract_text(transcript_data)

        logger.info("=" * 60)
        logger.info("TRANSCRIPTION COMPLETED")
        logger.info("Processing time: %.2f sec", processing_time)

        if audio_duration is not None and audio_duration > 0:
            logger.info(
                "RTF: %.4f",
                processing_time / audio_duration,
            )

            if processing_time > 0:
                logger.info(
                    "Realtime factor: %.2fx",
                    audio_duration / processing_time,
                )

        logger.info("Text output: %s", txt_path)
        logger.info("JSON output: %s", json_path)
        logger.info("=" * 60)

        print("\n" + "=" * 60)
        print("TRANSCRIPT")
        print("=" * 60)
        print(text)
        print("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130

    except Exception as exc:
        logger.error(
            "Soniox test failed: %s",
            exc,
            exc_info=True,
        )
        return 1

    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    sys.exit(main())

