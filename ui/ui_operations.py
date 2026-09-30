from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QFileDialog, QMessageBox

from pipeline.meeting_pipeline import MeetingPipeline
from ui.recorder import AudioRecorder

from .ui_constants import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    ENGINE_REGISTRY,
    PROJECT_ROOT,
    RECORDINGS_DIR,
    TEXTS,
)


class UIOperationsMixin:

    # ========================================================
    # UI thread bridge
    # ========================================================

    def ui_call(self, fn):
        """
        Schedule a callable on the Qt UI thread.

        Worker threads must never directly modify Qt widgets.
        """

        if not hasattr(self, "ui_bridge"):
            return

        self.ui_bridge.signal.emit(fn)

    # ========================================================
    # Model loading
    # ========================================================

    def load_model(self):
        if self.processing or self.recording:
            return

        model_name = self.model_var.strip()

        if not model_name:
            return QMessageBox.warning(
                self,
                self.TEXT["warning"],
                self.TEXT["select_model"],
            )

        engine_class = ENGINE_REGISTRY.get(model_name)

        if engine_class is None:
            self.logger.error(
                "MODEL | unknown model: %s",
                model_name,
            )

            return QMessageBox.critical(
                self,
                self.TEXT["error"],
                f"Unknown model: {model_name}",
            )

        self.processing = True
        self.operation = "loading"

        self.logger.info(
            "MODEL | loading started | model=%s",
            model_name,
        )

        try:
            self.main_page.set_processing_running(
                "در حال بارگذاری مدل...",
                indeterminate=True,
            )
        except Exception:
            self.set_status("loading")
            self.set_progress(0, "")

        self.refresh_controls()

        threading.Thread(
            target=self._load_model_worker,
            args=(model_name, engine_class),
            daemon=True,
        ).start()

    def _load_model_worker(
        self,
        model_name,
        engine_class,
    ):
        engine = None
        pipeline = None

        try:
            self.logger.info(
                "MODEL | preparing configuration"
            )

            config = self.load_config()

            configured = (
                config
                .get("engine", {})
                .get("name")
            )

            self.logger.info(
                "MODEL | configured engine=%s | selected=%s",
                configured,
                model_name,
            )

            if configured and configured != model_name:
                raise ValueError(
                    "Selected model does not match "
                    "engine.name in config.yaml."
                )

            model_path_value = (
                config
                .get("model", {})
                .get("path", "")
            )

            model_path = Path(model_path_value)

            if not model_path.is_absolute():
                model_path = PROJECT_ROOT / model_path

            self.logger.info(
                "MODEL | model directory: %s",
                model_path,
            )

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model directory not found: {model_path}"
                )

            # ------------------------------------------------
            # Whisper engine
            # ------------------------------------------------

            self.logger.info(
                "MODEL | initializing engine: %s",
                engine_class.__name__,
            )

            started = time.perf_counter()

            engine = engine_class(config)

            self.logger.info(
                "MODEL | calling engine.load()"
            )

            engine.load()

            engine_elapsed = (
                time.perf_counter()
                - started
            )

            self.logger.info(
                "MODEL | engine.load() completed | "
                "elapsed=%.3f sec",
                engine_elapsed,
            )

            # ------------------------------------------------
            # Meeting Pipeline
            # ------------------------------------------------

            self.logger.info(
                "MEETING | initializing MeetingPipeline"
            )

            pipeline = MeetingPipeline(
                config,
                engine,
            )

            self.logger.info(
                "MEETING | calling MeetingPipeline.load()"
            )

            pipeline_started = time.perf_counter()

            pipeline.load()

            pipeline_elapsed = (
                time.perf_counter()
                - pipeline_started
            )

            self.logger.info(
                "MEETING | MeetingPipeline.load() completed | "
                "elapsed=%.3f sec",
                pipeline_elapsed,
            )

            # ------------------------------------------------
            # Store loaded objects
            # ------------------------------------------------

            self.engine = engine
            self.pipeline = pipeline

            self.model_loaded = True

            self.logger.info(
                "MODEL | loading completed successfully | "
                "model=%s | engine=%.3fs | pipeline=%.3fs",
                model_name,
                engine_elapsed,
                pipeline_elapsed,
            )

            self.ui_call(
                self._model_loaded_ui
            )

        except Exception:
            self.logger.exception(
                "MODEL | loading failed | model=%s",
                model_name,
            )

            if pipeline is not None:
                try:
                    pipeline.unload()
                except Exception:
                    self.logger.exception(
                        "MEETING | failed to unload pipeline "
                        "after loading error"
                    )

            if engine is not None:
                try:
                    engine.unload()
                except Exception:
                    self.logger.exception(
                        "MODEL | failed to unload engine "
                        "after loading error"
                    )

            self.pipeline = None
            self.engine = None
            self.model_loaded = False

            self.ui_call(
                self._model_load_failed_ui
            )

    def _model_loaded_ui(self):
        self.model_loaded = True
        self.processing = False
        self.operation = None

        try:
            self.main_page.set_model_loading(False)
        except Exception:
            pass

        try:
            self.main_page.set_processing_success(
                "مدل و موتور پردازش جلسه با موفقیت بارگذاری شدند."
            )
        except Exception:
            self.set_status("loaded")
            self.set_progress(0, "")

        self.refresh_controls()

        self.logger.info(
            "MODEL | UI updated | "
            "model_loaded=%s | current_audio=%s | "
            "engine=%s | pipeline=%s",
            self.model_loaded,
            self.current_audio,
            self.engine is not None,
            self.pipeline is not None,
        )

    def _model_load_failed_ui(self):
        self.processing = False
        self.operation = None
        self.model_loaded = False
        self.engine = None
        self.pipeline = None

        try:
            self.main_page.set_model_loading(False)
        except Exception:
            pass

        try:
            self.main_page.set_processing_error(
                "خطا در بارگذاری مدل و Pipeline."
            )
        except Exception:
            self.set_status("error")

        self.refresh_controls()

        QMessageBox.critical(
            self,
            self.TEXT["error"],
            self.TEXT["model_load_failed"],
        )

    # ========================================================
    # Model unloading
    # ========================================================

    def unload_model(self):
        if (
            self.processing
            or self.recording
            or not self.model_loaded
        ):
            return

        self.processing = True
        self.operation = "unloading"

        self.set_status("unloading")
        self.set_progress(0, "")
        self.refresh_controls()

        self.logger.info(
            "MODEL | unloading started"
        )

        threading.Thread(
            target=self._unload_model_worker,
            daemon=True,
        ).start()

    def _unload_model_worker(self):
        try:
            # Pipeline must be unloaded before ASR engine.
            if self.pipeline is not None:
                self.logger.info(
                    "MEETING | unloading MeetingPipeline"
                )

                try:
                    self.pipeline.unload()
                except Exception:
                    self.logger.exception(
                        "MEETING | MeetingPipeline unload failed"
                    )

                self.pipeline = None

            if self.engine is not None:
                self.logger.info(
                    "MODEL | unloading Whisper engine"
                )

                self.engine.unload()
                self.engine = None

            self.model_loaded = False

            self.logger.info(
                "MODEL | unload completed"
            )

            self.ui_call(
                self._model_unloaded_ui
            )

        except Exception:
            self.logger.exception(
                "MODEL | unload failed"
            )

            self.ui_call(
                self._model_unload_failed_ui
            )

    def _model_unloaded_ui(self):
        self.processing = False
        self.operation = None
        self.model_loaded = False
        self.pipeline = None
        self.engine = None

        self.set_status("unloaded")
        self.refresh_controls()

    def _model_unload_failed_ui(self):
        self.processing = False
        self.operation = None

        self.set_status("error")
        self.refresh_controls()

    # ========================================================
    # Audio selection
    # ========================================================

    def select_audio(self):
        if self.processing or self.recording:
            return

        try:
            input_dir = self.get_input_dir(
                self.load_config()
            )
        except Exception:
            input_dir = DEFAULT_INPUT_DIR

        text = self.TEXT

        path, _ = QFileDialog.getOpenFileName(
            self,
            text["open_audio"],
            str(input_dir),
            (
                f"{text['audio_files']} "
                "(*.wav *.mp3 *.flac *.m4a *.ogg *.aac);;"
                f"{text['all_files']} (*.*)"
            ),
        )

        if not path:
            self.logger.info(
                "Audio selection cancelled."
            )
            return

        selected_path = Path(path)

        if not selected_path.exists():
            self.logger.error(
                "Selected audio file does not exist: %s",
                selected_path,
            )

            QMessageBox.warning(
                self,
                self.TEXT["warning"],
                f"Audio file not found:\n{selected_path}",
            )
            return

        self.current_audio = selected_path
        self.audio_var = str(selected_path)
        self.current_result = None

        if hasattr(self, "current_output_txt"):
            self.current_output_txt = None

        if hasattr(self, "current_output_json"):
            self.current_output_json = None

        self.main_page.transcription_text.set_text(
            text["no_transcription"],
            self.language == "fa",
        )

        self.set_status(
            "loaded"
            if self.model_loaded
            else "ready"
        )

        self.refresh_controls()

        self.logger.info(
            "Audio selected: %s",
            self.current_audio,
        )

        self.logger.info(
            "Audio state updated: current_audio=%s",
            self.current_audio,
        )

    # ========================================================
    # Recording
    # ========================================================

    def start_recording(self):
        if self.processing or self.recording:
            return

        try:
            config = self.load_config()

            rate = int(
                config
                .get("audio", {})
                .get("sample_rate", 16000)
            )

            self.recorder = AudioRecorder(
                sample_rate=rate,
                channels=1,
                dtype="float32",
            )

            self.recorder.start()

            self.recording = True
            self.operation = "recording"

            self.set_status("recording")
            self.refresh_controls()

            self.logger.info(
                "Recording started."
            )

        except Exception:
            self.logger.exception(
                "Recording failed to start."
            )

            self.recording = False
            self.operation = None

            self.set_status("error")
            self.refresh_controls()

    def stop_recording(self):
        if not self.recording:
            return

        try:
            if self.recorder is None:
                raise RuntimeError(
                    "Recorder is not initialized."
                )

            output_path = (
                self.recorder.generate_output_path(
                    RECORDINGS_DIR
                )
            )

            self.recorder.stop(output_path)

            self.current_audio = Path(output_path)
            self.audio_var = str(self.current_audio)
            self.current_result = None

            if hasattr(self, "current_output_txt"):
                self.current_output_txt = None

            if hasattr(self, "current_output_json"):
                self.current_output_json = None

            self.main_page.transcription_text.set_text(
                self.TEXT["no_transcription"],
                self.language == "fa",
            )

            self.logger.info(
                "Recording saved: %s",
                output_path,
            )

            self.recording = False
            self.operation = None

            self.set_status("recording_saved")

        except Exception:
            self.logger.exception(
                "Failed to stop recording."
            )

            self.recording = False
            self.operation = None

            self.set_status("error")

        finally:
            self.recorder = None

            self.refresh_controls()

            try:
                self.files_page.reload()
            except Exception:
                self.logger.exception(
                    "Failed to refresh files page."
                )

    # ========================================================
    # Meeting transcription
    # ========================================================

    def transcribe(self):
        if self.processing or self.recording:
            return

        # ----------------------------------------------------
        # IMPORTANT:
        # The UI now requires MeetingPipeline, not AudioChunker.
        # ----------------------------------------------------

        if (
            not self.model_loaded
            or self.engine is None
            or self.pipeline is None
        ):
            return QMessageBox.warning(
                self,
                self.TEXT["warning"],
                self.TEXT["load_model_first"],
            )

        if not self.current_audio:
            return QMessageBox.warning(
                self,
                self.TEXT["warning"],
                self.TEXT["select_audio_first"],
            )

        audio_path = Path(self.current_audio)

        if not audio_path.exists():
            self.logger.error(
                "Audio file does not exist: %s",
                audio_path,
            )

            QMessageBox.warning(
                self,
                self.TEXT["warning"],
                f"Audio file not found:\n{audio_path}",
            )

            self.current_audio = None
            self.audio_var = ""

            self.refresh_controls()
            return

        pipeline = self.pipeline

        self.processing = True
        self.operation = "processing"
        self.current_result = None

        if hasattr(self, "current_output_txt"):
            self.current_output_txt = None

        if hasattr(self, "current_output_json"):
            self.current_output_json = None

        self.set_progress(0, "")

        self.main_page.transcription_text.set_text(
            self.TEXT["processing"],
            self.language == "fa",
        )

        self.set_status("processing")
        self.refresh_controls()

        self.logger.info(
            "============================================================"
        )

        self.logger.info(
            "MEETING | transcription started"
        )

        self.logger.info(
            "MEETING | audio=%s",
            audio_path,
        )

        self.logger.info(
            "MEETING | pipeline=%s",
            type(pipeline).__name__,
        )

        threading.Thread(
            target=self._transcription_worker,
            args=(
                audio_path,
                pipeline,
            ),
            daemon=True,
        ).start()

    def _transcription_worker(
        self,
        audio_path: Path,
        pipeline: MeetingPipeline,
    ):
        started = time.perf_counter()

        try:
            # ------------------------------------------------
            # Configuration
            # ------------------------------------------------

            self.logger.info(
                "MEETING | loading configuration"
            )

            config = self.load_config()

            # ------------------------------------------------
            # Output directory
            # ------------------------------------------------

            output_dir = self.get_output_dir(config)

            file_output_dir = (
                output_dir / audio_path.stem
            )

            file_output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.logger.info(
                "MEETING | output directory=%s",
                file_output_dir,
            )

            # ------------------------------------------------
            # Main backend pipeline
            # ------------------------------------------------

            self.logger.info(
                "MEETING | calling MeetingPipeline.process()"
            )

            self.logger.info(
                "MEETING | this may take some time during "
                "diarization / speaker isolation / ASR"
            )

            result = pipeline.process(
                audio_path,
                file_output_dir,
            )

            elapsed = (
                time.perf_counter()
                - started
            )

            self.logger.info(
                "MEETING | MeetingPipeline.process() returned"
            )

            self.logger.info(
                "MEETING | processing elapsed=%.3f sec",
                elapsed,
            )

            # ------------------------------------------------
            # Expected backend outputs
            # ------------------------------------------------

            txt_path = (
                file_output_dir
                / f"{audio_path.stem}_meeting.txt"
            )

            json_path = (
                file_output_dir
                / f"{audio_path.stem}_meeting.json"
            )

            self.logger.info(
                "MEETING | checking TXT=%s",
                txt_path,
            )

            if not txt_path.exists():
                raise FileNotFoundError(
                    "MeetingPipeline completed but expected "
                    f"TXT file was not created: {txt_path}"
                )

            meeting_text = txt_path.read_text(
                encoding="utf-8"
            )

            self.logger.info(
                "MEETING | TXT loaded successfully"
            )

            self.logger.info(
                "MEETING | transcript characters=%d",
                len(meeting_text),
            )

            if json_path.exists():
                self.logger.info(
                    "MEETING | JSON output=%s",
                    json_path,
                )
            else:
                self.logger.warning(
                    "MEETING | JSON output not found: %s",
                    json_path,
                )

            # ------------------------------------------------
            # Send result to UI
            # ------------------------------------------------

            self.ui_call(
                lambda:
                self._transcription_completed_ui(
                    result=result,
                    meeting_text=meeting_text,
                    txt_path=txt_path,
                    json_path=json_path,
                    elapsed=elapsed,
                )
            )

        except Exception as exc:
            self.logger.exception(
                "MEETING | processing failed | file=%s",
                audio_path.name,
            )

            self.ui_call(
                lambda error=exc:
                self._transcription_failed_ui(error)
            )

    # ========================================================
    # Processing UI
    # ========================================================

    def _prepare_progress_ui(self, total):
        """
        Prepare progress controls.

        MeetingPipeline currently owns the actual backend
        stages, so the UI uses an indeterminate progress
        indicator rather than pretending chunk progress.
        """

        try:
            self.main_page.progress.setRange(
                0,
                0,
            )
        except Exception:
            pass

        try:
            self.main_page.operation_label.setText(
                "در حال پردازش جلسه..."
            )
        except Exception:
            pass

        self.set_progress(
            0,
            "در حال پردازش...",
        )

    def _update_processing_ui(
        self,
        operation,
        index,
        total,
        elapsed,
    ):
        try:
            self.main_page.operation_label.setText(
                operation
            )
        except Exception:
            pass

        try:
            self.main_page.progress.setRange(
                0,
                0,
            )
        except Exception:
            pass

        self.set_progress(
            0,
            (
                f"{operation}  •  "
                f"elapsed: {elapsed:.1f}s"
            ),
        )

    def _transcription_completed_ui(
        self,
        result,
        meeting_text,
        txt_path,
        json_path,
        elapsed,
    ):
        self.processing = False
        self.operation = None

        # Store the actual backend result.
        self.current_result = result

        # Store actual generated files.
        self.current_output_txt = Path(txt_path)

        self.current_output_json = (
            Path(json_path)
            if json_path.exists()
            else None
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        try:
            self.main_page.progress.setRange(
                0,
                1,
            )

            self.set_progress(
                1,
                "1 / 1",
            )
        except Exception:
            pass

        # ----------------------------------------------------
        # Display actual MeetingPipeline transcript.
        #
        # Example:
        #
        # [0.115 -> 1.027] SPEAKER_01: سلام به همه
        # [1.094 -> 5.144] SPEAKER_01: ...
        # ----------------------------------------------------

        self.main_page.transcription_text.set_text(
            meeting_text.strip()
            if meeting_text.strip()
            else self.TEXT["no_transcription"],
            self.language == "fa",
        )

        try:
            self.main_page.operation_label.setText(
                ""
            )
        except Exception:
            pass

        self.set_status(
            "completed",
            (
                f"{self.TEXT['elapsed']}: "
                f"{elapsed:.2f}s"
            ),
        )

        self.logger.info(
            "MEETING | UI updated successfully"
        )

        self.logger.info(
            "MEETING | TXT=%s",
            txt_path,
        )

        if json_path.exists():
            self.logger.info(
                "MEETING | JSON=%s",
                json_path,
            )

        self.logger.info(
            "MEETING | transcription completed | "
            "elapsed=%.3f sec",
            elapsed,
        )

        self.refresh_controls()

        try:
            self.files_page.reload()
        except Exception:
            self.logger.exception(
                "Failed to refresh files page after transcription."
            )

    def _transcription_failed_ui(self, error):
        self.processing = False
        self.operation = None

        try:
            self.main_page.operation_label.setText(
                ""
            )
        except Exception:
            pass

        self.set_status("error")
        self.refresh_controls()

        self.logger.error(
            "MEETING | UI processing failure: %s",
            error,
        )

        QMessageBox.critical(
            self,
            self.TEXT["error"],
            (
                f"{self.TEXT['transcription_failed']}\n\n"
                f"{error}"
            ),
        )

    # ========================================================
    # Export TXT
    # ========================================================

    def export_txt(self):
        # Prefer the actual MeetingPipeline TXT.
        source_text = getattr(
            self,
            "current_output_txt",
            None,
        )

        if source_text is not None:
            source_text = Path(source_text)

            if source_text.exists():
                default_name = (
                    source_text.stem
                    + ".txt"
                )

                path, _ = QFileDialog.getSaveFileName(
                    self,
                    self.TEXT["export_txt"],
                    str(
                        DEFAULT_OUTPUT_DIR
                        / default_name
                    ),
                    "TXT (*.txt)",
                )

                if path:
                    Path(path).write_text(
                        source_text.read_text(
                            encoding="utf-8"
                        ),
                        encoding="utf-8",
                    )

                    self.logger.info(
                        "TXT exported: %s",
                        path,
                    )

                    QMessageBox.information(
                        self,
                        self.TEXT["success"],
                        (
                            f"{self.TEXT['saved_txt']}\n"
                            f"{path}"
                        ),
                    )

                    self.files_page.reload()

                return

        QMessageBox.warning(
            self,
            self.TEXT["warning"],
            "ابتدا یک جلسه را پردازش کنید.",
        )

    # ========================================================
    # Export JSON
    # ========================================================

    def export_json(self):
        source_json = getattr(
            self,
            "current_output_json",
            None,
        )

        if source_json is not None:
            source_json = Path(source_json)

            if source_json.exists():
                default_name = (
                    source_json.stem
                    + ".json"
                )

                path, _ = QFileDialog.getSaveFileName(
                    self,
                    self.TEXT["export_json"],
                    str(
                        DEFAULT_OUTPUT_DIR
                        / default_name
                    ),
                    "JSON (*.json)",
                )

                if path:
                    Path(path).write_text(
                        source_json.read_text(
                            encoding="utf-8"
                        ),
                        encoding="utf-8",
                    )

                    self.logger.info(
                        "JSON exported: %s",
                        path,
                    )

                    QMessageBox.information(
                        self,
                        self.TEXT["success"],
                        (
                            f"{self.TEXT['saved_json']}\n"
                            f"{path}"
                        ),
                    )

                    self.files_page.reload()

                return

        # Fallback: export in-memory result if backend JSON
        # was not produced.
        if self.current_result is None:
            QMessageBox.warning(
                self,
                self.TEXT["warning"],
                "ابتدا یک جلسه را پردازش کنید.",
            )
            return

        filename = "meeting_result.json"

        path, _ = QFileDialog.getSaveFileName(
            self,
            self.TEXT["export_json"],
            str(
                DEFAULT_OUTPUT_DIR
                / filename
            ),
            "JSON (*.json)",
        )

        if path:
            Path(path).write_text(
                json.dumps(
                    self.current_result,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            self.logger.info(
                "JSON exported: %s",
                path,
            )

            QMessageBox.information(
                self,
                self.TEXT["success"],
                (
                    f"{self.TEXT['saved_json']}\n"
                    f"{path}"
                ),
            )

            self.files_page.reload()

    # ========================================================
    # Open output
    # ========================================================

    def open_output(self):
        try:
            self.open_path(
                self.get_output_dir(
                    self.load_config()
                )
            )
        except Exception:
            self.open_path(
                DEFAULT_OUTPUT_DIR
            )

    def open_path(self, path):
        path = Path(path)

        try:
            if os.name == "nt":
                os.startfile(str(path))

            elif sys.platform == "darwin":
                import subprocess

                subprocess.run(
                    [
                        "open",
                        str(path),
                    ]
                )

            else:
                import subprocess

                subprocess.run(
                    [
                        "xdg-open",
                        str(path),
                    ]
                )

        except Exception:
            self.logger.exception(
                "Could not open path: %s",
                path,
            )

    # ========================================================
    # Copy transcription
    # ========================================================

    def copy_transcription_text(self):
        """
        Copy transcription text without direction-control
        characters that can disturb Persian text selection.
        """

        text = (
            self.main_page.transcription_text
            .toPlainText()
        )

        directional_chars = (
            "\u200e",
            "\u200f",
            "\u202a",
            "\u202b",
            "\u202c",
            "\u202d",
            "\u202e",
            "\u2066",
            "\u2067",
            "\u2068",
            "\u2069",
        )

        for char in directional_chars:
            text = text.replace(
                char,
                "",
            )

        text = (
            text
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )

        QGuiApplication.clipboard().setText(
            text
        )

        self.logger.info(
            "Transcription copied to clipboard."
        )

    # ========================================================
    # Logs
    # ========================================================

    def process_log_queue(self):
        while not self.log_queue.empty():
            try:
                message = self.log_queue.get_nowait()

                self.insert_log(message)

            except Exception:
                break

    def insert_log(self, message):
        if message:
            self.logs_page.text.appendPlainText(
                message.rstrip()
            )

    def clear_logs(self):
        self.logs_page.text.clear()

    # ========================================================
    # Close
    # ========================================================

    def on_close(self):
        if self.processing or self.recording:
            answer = QMessageBox.question(
                self,
                self.TEXT["confirm_close_title"],
                self.TEXT["confirm_close"],
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

            if answer != QMessageBox.Yes:
                return

        try:
            if self.recording and self.recorder:
                self.recorder.cancel()

            # Pipeline first, then Whisper.
            if getattr(self, "pipeline", None):
                self.logger.info(
                    "MEETING | unloading pipeline before exit..."
                )

                try:
                    self.pipeline.unload()
                except Exception:
                    self.logger.exception(
                        "MEETING | error while unloading pipeline"
                    )

                self.pipeline = None

            if getattr(self, "engine", None):
                self.logger.info(
                    "MODEL | unloading engine before exit..."
                )

                self.engine.unload()
                self.engine = None

        except Exception:
            self.logger.exception(
                "Error while unloading engine during exit."
            )

        self.settings.save(
            language=self.language,
            theme=self.theme_name,
            last_tab=self.tabs.currentIndex(),
        )

        self.close()