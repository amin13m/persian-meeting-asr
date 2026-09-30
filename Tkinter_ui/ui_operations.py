from __future__ import annotations

import json
import logging
import queue
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PySide6.QtCore import Qt

from engines.whisper_persian_v4.engine import WhisperPersianV4Engine
from pipeline.meeting_pipeline import MeetingPipeline

try:
    from .ui_constants import (
        CONFIG_PATH,
        DEFAULT_INPUT_DIR,
        DEFAULT_OUTPUT_DIR,
        ENGINE_REGISTRY,
        PROJECT_ROOT,
        RECORDINGS_DIR,
        TEXTS,
        format_seconds,
    )
    from .recorder import AudioRecorder
except ImportError:
    UI_DIR = Path(__file__).resolve().parent

    if str(UI_DIR.parent) not in sys.path:
        sys.path.insert(0, str(UI_DIR.parent))

    from ui.ui_constants import (
        CONFIG_PATH,
        DEFAULT_INPUT_DIR,
        DEFAULT_OUTPUT_DIR,
        ENGINE_REGISTRY,
        PROJECT_ROOT,
        RECORDINGS_DIR,
        TEXTS,
        format_seconds,
    )
    from ui.recorder import AudioRecorder


logger = logging.getLogger("PersianASR.UI")


class UIOperationsMixin:
    # ============================================================
    # UI THREAD BRIDGE
    # ============================================================

    def ui_call(self, callback):
        self.ui_bridge.signal.emit(callback)

    # ============================================================
    # CONFIG
    # ============================================================

    def load_config(self):
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {CONFIG_PATH}"
            )

        import yaml

        config = yaml.safe_load(
            CONFIG_PATH.read_text(encoding="utf-8")
        )

        if not isinstance(config, dict):
            raise ValueError(
                "Configuration file is empty or invalid."
            )

        return config

    # ============================================================
    # PATH HELPERS
    # ============================================================

    def _resolve_project_path(self, value, default=None):
        if value is None:
            value = default

        if value is None:
            raise ValueError("Path value is not defined.")

        path = Path(value)

        if path.is_absolute():
            return path

        return PROJECT_ROOT / path

    def get_input_dir(self, config):
        value = (
            config.get("audio", {})
            .get("input_dir", DEFAULT_INPUT_DIR)
        )

        path = self._resolve_project_path(
            value,
            DEFAULT_INPUT_DIR,
        )

        path.mkdir(parents=True, exist_ok=True)

        return path

    def get_output_dir(self, config):
        value = (
            config.get("audio", {})
            .get("output_dir", DEFAULT_OUTPUT_DIR)
        )

        path = self._resolve_project_path(
            value,
            DEFAULT_OUTPUT_DIR,
        )

        path.mkdir(parents=True, exist_ok=True)

        return path

    # ============================================================
    # CONTROL STATE
    # ============================================================

    def refresh_controls(self):
        try:
            loaded = bool(getattr(self, "model_loaded", False))
            processing = bool(getattr(self, "processing", False))
            recording = bool(getattr(self, "recording", False))
            audio = bool(getattr(self, "current_audio", None))
            engine = bool(getattr(self, "engine", None))
            chunker = bool(getattr(self, "chunker", None))

            can_select = not processing and not recording
            can_transcribe = (
                not processing
                and not recording
                and loaded
                and audio
                and engine
            )

            self.logger.info(
                "CONTROLS | "
                "processing=%s recording=%s loaded=%s audio=%s "
                "engine=%s chunker=%s | select=%s transcribe=%s",
                processing,
                recording,
                loaded,
                audio,
                engine,
                chunker,
                can_select,
                can_transcribe,
            )

            if hasattr(self, "main_page"):
                page = self.main_page

                if hasattr(page, "select_audio_button"):
                    page.select_audio_button.setEnabled(
                        can_select
                    )

                if hasattr(page, "transcribe_button"):
                    page.transcribe_button.setEnabled(
                        can_transcribe
                    )

                if hasattr(page, "load_model_button"):
                    page.load_model_button.setEnabled(
                        not processing and not recording
                    )

                if hasattr(page, "unload_model_button"):
                    page.unload_model_button.setEnabled(
                        not processing
                        and not recording
                        and loaded
                    )

        except Exception:
            self.logger.exception(
                "Failed to refresh controls."
            )

    # ============================================================
    # AUDIO SELECTION
    # ============================================================

    def select_audio(self):
        if self.processing:
            return

        try:
            from PySide6.QtWidgets import QFileDialog

            config = self.load_config()
            input_dir = self.get_input_dir(config)

            file_filter = (
                "Audio Files (*.wav *.mp3 *.flac *.m4a *.ogg);;"
                "WAV Files (*.wav);;"
                "MP3 Files (*.mp3);;"
                "FLAC Files (*.flac);;"
                "M4A Files (*.m4a);;"
                "OGG Files (*.ogg);;"
                "All Files (*)"
            )

            path, _ = QFileDialog.getOpenFileName(
                self,
                self.TEXT.get(
                    "select_audio_title",
                    "Select Audio File",
                ),
                str(input_dir),
                file_filter,
            )

            if not path:
                return

            audio_path = Path(path)

            self.logger.info(
                "Audio selected: %s",
                audio_path,
            )

            self.current_audio = audio_path
            self.current_result = None

            self.current_output_txt = None
            self.current_output_json = None

            if hasattr(self, "main_page"):
                page = self.main_page

                if hasattr(page, "set_audio_path"):
                    page.set_audio_path(str(audio_path))

                if hasattr(page, "transcription_text"):
                    page.transcription_text.clear()

            self.logger.info(
                "Audio state updated: current_audio=%s",
                self.current_audio,
            )

            self.refresh_controls()

        except Exception:
            self.logger.exception(
                "Failed to select audio."
            )

    # ============================================================
    # MODEL LOADING
    # ============================================================

    def load_model(self):
        if self.processing:
            return

        if self.model_loaded:
            self.logger.info(
                "Model is already loaded."
            )
            return

        self.processing = True
        self.status_key = "loading_model"
        self.status_detail = ""

        self.logger.info(
            "MODEL | loading started | model=%s",
            self.model_var,
        )

        self.refresh_controls()

        try:
            if hasattr(self, "main_page"):
                if hasattr(
                    self.main_page,
                    "set_model_loading",
                ):
                    self.main_page.set_model_loading(True)

                if hasattr(
                    self.main_page,
                    "set_processing_running",
                ):
                    self.main_page.set_processing_running(
                        True
                    )

        except Exception:
            self.logger.exception(
                "Failed to update model loading UI."
            )

        import threading

        thread = threading.Thread(
            target=self._load_model_worker,
            name="ASR-Model-Loader",
            daemon=True,
        )

        self.operation = thread
        thread.start()

    def _load_model_worker(self):
        start_time = time.perf_counter()

        engine = None
        pipeline = None

        try:
            config = self.load_config()

            engine_name = config.get(
                "engine",
                {},
            ).get(
                "name",
                "whisper_persian_v4",
            )

            if engine_name not in ENGINE_REGISTRY:
                raise ValueError(
                    f"Unsupported engine: {engine_name}"
                )

            model_path = self._resolve_project_path(
                config.get("model", {}).get(
                    "path",
                    "models/whisper_persian_v4",
                )
            )

            self.logger.info(
                "Model directory: %s",
                model_path,
            )

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model directory not found: "
                    f"{model_path}"
                )

            self.logger.info(
                "Loading engine..."
            )

            engine_class = ENGINE_REGISTRY[
                engine_name
            ]

            engine = engine_class(config)

            engine.load()

            elapsed = time.perf_counter() - start_time

            self.logger.info(
                "Engine loaded successfully in %.3f sec.",
                elapsed,
            )

            # ----------------------------------------------------
            # IMPORTANT:
            # The UI now uses the exact same MeetingPipeline
            # architecture as the non-UI main runner.
            # ----------------------------------------------------

            self.logger.info(
                "Loading MeetingPipeline..."
            )

            pipeline = MeetingPipeline(
                config,
                engine,
            )

            pipeline.load()

            pipeline_elapsed = (
                time.perf_counter() - start_time
            )

            self.logger.info(
                "MeetingPipeline loaded successfully "
                "in %.3f sec.",
                pipeline_elapsed,
            )

            self.ui_call(
                lambda e=engine, p=pipeline:
                self._model_loaded_ui(e, p)
            )

        except Exception as exc:
            self.logger.exception(
                "Failed to load model."
            )

            try:
                if pipeline is not None:
                    pipeline.unload()
            except Exception:
                self.logger.exception(
                    "Failed to unload partially loaded "
                    "MeetingPipeline."
                )

            try:
                if engine is not None:
                    engine.unload()
            except Exception:
                self.logger.exception(
                    "Failed to unload partially loaded engine."
                )

            self.ui_call(
                lambda error=exc:
                self._model_load_failed_ui(error)
            )

    def _model_loaded_ui(self, engine, pipeline):
        self.engine = engine
        self.pipeline = pipeline

        # MeetingPipeline already owns the chunker.
        # Keep the reference only for UI compatibility.
        self.chunker = getattr(
            pipeline,
            "chunker",
            None,
        )

        self.model_loaded = True
        self.processing = False
        self.status_key = "model_loaded"
        self.status_detail = ""

        self.logger.info(
            "MODEL | UI updated | model_loaded=%s | "
            "current_audio=%s",
            self.model_loaded,
            getattr(
                self,
                "current_audio",
                None,
            ),
        )

        try:
            if hasattr(self, "main_page"):
                if hasattr(
                    self.main_page,
                    "set_model_loading",
                ):
                    self.main_page.set_model_loading(
                        False
                    )

                if hasattr(
                    self.main_page,
                    "set_processing_running",
                ):
                    self.main_page.set_processing_running(
                        False
                    )

                if hasattr(
                    self.main_page,
                    "set_processing_success",
                ):
                    self.main_page.set_processing_success(
                        "مدل و موتور پردازش جلسه آماده است."
                    )

        except Exception:
            self.logger.exception(
                "Failed to update model-loaded UI."
            )

        self.refresh_controls()

    def _model_load_failed_ui(self, error):
        self.engine = None
        self.pipeline = None
        self.chunker = None

        self.model_loaded = False
        self.processing = False
        self.status_key = "model_load_failed"
        self.status_detail = str(error)

        try:
            if hasattr(self, "main_page"):
                if hasattr(
                    self.main_page,
                    "set_model_loading",
                ):
                    self.main_page.set_model_loading(
                        False
                    )

                if hasattr(
                    self.main_page,
                    "set_processing_running",
                ):
                    self.main_page.set_processing_running(
                        False
                    )

                if hasattr(
                    self.main_page,
                    "set_processing_error",
                ):
                    self.main_page.set_processing_error(
                        str(error)
                    )

        except Exception:
            self.logger.exception(
                "Failed to update model-load error UI."
            )

        self.refresh_controls()

    # ============================================================
    # MODEL UNLOADING
    # ============================================================

    def unload_model(self):
        if self.processing:
            return

        pipeline = getattr(
            self,
            "pipeline",
            None,
        )

        engine = getattr(
            self,
            "engine",
            None,
        )

        if pipeline is None and engine is None:
            self.model_loaded = False
            self.chunker = None
            self.refresh_controls()
            return

        self.processing = True

        self.logger.info(
            "MODEL | unloading started"
        )

        self.refresh_controls()

        import threading

        thread = threading.Thread(
            target=self._unload_model_worker,
            args=(pipeline, engine),
            name="ASR-Model-Unloader",
            daemon=True,
        )

        self.operation = thread
        thread.start()

    def _unload_model_worker(
        self,
        pipeline,
        engine,
    ):
        try:
            # MeetingPipeline unloads diarization,
            # overlap separation and related resources.
            if pipeline is not None:
                self.logger.info(
                    "Unloading MeetingPipeline..."
                )
                pipeline.unload()

            # Whisper engine is owned outside the pipeline.
            if engine is not None:
                self.logger.info(
                    "Unloading Whisper engine..."
                )
                engine.unload()

            self.ui_call(
                self._model_unloaded_ui
            )

        except Exception as exc:
            self.logger.exception(
                "Failed to unload model."
            )

            self.ui_call(
                lambda error=exc:
                self._model_unload_failed_ui(error)
            )

    def _model_unloaded_ui(self):
        self.pipeline = None
        self.engine = None
        self.chunker = None

        self.model_loaded = False
        self.processing = False

        self.logger.info(
            "MODEL | unloaded"
        )

        try:
            if hasattr(self, "main_page"):
                if hasattr(
                    self.main_page,
                    "set_processing_success",
                ):
                    self.main_page.set_processing_success(
                        "مدل آزاد شد."
                    )

        except Exception:
            self.logger.exception(
                "Failed to update unload UI."
            )

        self.refresh_controls()

    def _model_unload_failed_ui(self, error):
        self.processing = False

        self.logger.error(
            "MODEL | unload failed | %s",
            error,
        )

        try:
            if hasattr(self, "main_page"):
                if hasattr(
                    self.main_page,
                    "set_processing_error",
                ):
                    self.main_page.set_processing_error(
                        str(error)
                    )

        except Exception:
            self.logger.exception(
                "Failed to update unload error UI."
            )

        self.refresh_controls()

    # ============================================================
    # TRANSCRIPTION
    # ============================================================

    def transcribe(self):
        if self.processing:
            return

        audio_path = getattr(
            self,
            "current_audio",
            None,
        )

        pipeline = getattr(
            self,
            "pipeline",
            None,
        )

        if not self.model_loaded:
            self.logger.warning(
                "Cannot transcribe: model is not loaded."
            )
            return

        if pipeline is None:
            self.logger.error(
                "Cannot transcribe: MeetingPipeline "
                "is not loaded."
            )
            return

        if audio_path is None:
            self.logger.warning(
                "Cannot transcribe: no audio selected."
            )
            return

        audio_path = Path(audio_path)

        if not audio_path.exists():
            self.logger.error(
                "Audio file does not exist: %s",
                audio_path,
            )
            return

        self.processing = True
        self.status_key = "processing"
        self.status_detail = ""

        self.logger.info(
            "Starting transcription: %s",
            audio_path,
        )

        self.refresh_controls()

        try:
            if hasattr(self, "main_page"):
                if hasattr(
                    self.main_page,
                    "set_processing_running",
                ):
                    self.main_page.set_processing_running(
                        True
                    )

        except Exception:
            self.logger.exception(
                "Failed to update transcription UI."
            )

        import threading

        thread = threading.Thread(
            target=self._transcription_worker,
            args=(
                audio_path,
                pipeline,
            ),
            name="Meeting-Transcription",
            daemon=True,
        )

        self.operation = thread
        thread.start()

    def _transcription_worker(
        self,
        audio_path: Path,
        pipeline: MeetingPipeline,
    ):
        started = time.perf_counter()

        try:
            config = self.load_config()

            output_dir = self.get_output_dir(
                config
            )

            # ----------------------------------------------------
            # EXACT SAME OUTPUT STRUCTURE AS NON-UI MAIN
            #
            # output/
            #   test/
            #       test_meeting.txt
            #       test_meeting.json
            #       ...
            # ----------------------------------------------------

            file_output_dir = (
                output_dir / audio_path.stem
            )

            file_output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.logger.info(
                "Meeting output directory: %s",
                file_output_dir,
            )

            self.logger.info(
                "Running MeetingPipeline.process()..."
            )

            result = pipeline.process(
                audio_path,
                file_output_dir,
            )

            elapsed = (
                time.perf_counter() - started
            )

            txt_path = (
                file_output_dir
                / f"{audio_path.stem}_meeting.txt"
            )

            json_path = (
                file_output_dir
                / f"{audio_path.stem}_meeting.json"
            )

            if not txt_path.exists():
                raise FileNotFoundError(
                    "MeetingPipeline completed but the "
                    f"expected TXT file was not created:\n"
                    f"{txt_path}"
                )

            # ----------------------------------------------------
            # IMPORTANT:
            # Never reconstruct the displayed transcript
            # from plain Whisper text.
            #
            # Read the exact TXT generated by MeetingPipeline.
            # ----------------------------------------------------

            meeting_text = txt_path.read_text(
                encoding="utf-8"
            )

            if not meeting_text.strip():
                self.logger.warning(
                    "Meeting TXT is empty: %s",
                    txt_path,
                )

            self.logger.info(
                "MeetingPipeline completed in %.3f sec.",
                elapsed,
            )

            self.logger.info(
                "Meeting TXT: %s",
                txt_path,
            )

            if json_path.exists():
                self.logger.info(
                    "Meeting JSON: %s",
                    json_path,
                )

            self.ui_call(
                lambda r=result,
                text=meeting_text,
                txt=txt_path,
                js=json_path,
                elapsed_seconds=elapsed:
                self._transcription_completed_ui(
                    r,
                    text,
                    txt,
                    js,
                    elapsed_seconds,
                )
            )

        except Exception as exc:
            self.logger.exception(
                "Failed processing file: %s",
                audio_path.name,
            )

            self.ui_call(
                lambda error=exc:
                self._transcription_failed_ui(
                    error
                )
            )

    def _transcription_completed_ui(
        self,
        result,
        meeting_text,
        txt_path,
        json_path,
        elapsed_seconds,
    ):
        self.current_result = result
        self.current_output_txt = Path(
            txt_path
        )

        self.current_output_json = (
            Path(json_path)
            if Path(json_path).exists()
            else None
        )

        self.processing = False
        self.status_key = "completed"
        self.status_detail = (
            f"{elapsed_seconds:.2f} sec"
        )

        # --------------------------------------------------------
        # DISPLAY EXACT MEETING TXT
        # --------------------------------------------------------

        try:
            if hasattr(self, "main_page"):
                page = self.main_page

                if hasattr(
                    page,
                    "transcription_text",
                ):
                    page.transcription_text.set_text(
                        meeting_text,
                        self.language == "fa",
                    )

                if hasattr(
                    page,
                    "set_processing_success",
                ):
                    page.set_processing_success(
                        "پردازش جلسه با موفقیت انجام شد."
                    )

        except Exception:
            self.logger.exception(
                "Failed to update transcription result UI."
            )

        self.logger.info(
            "Transcription completed successfully."
        )

        self.logger.info(
            "Displayed meeting transcript from: %s",
            txt_path,
        )

        self.refresh_controls()

    def _transcription_failed_ui(self, error):
        self.processing = False
        self.status_key = "processing_failed"
        self.status_detail = str(error)

        try:
            if hasattr(self, "main_page"):
                if hasattr(
                    self.main_page,
                    "set_processing_error",
                ):
                    self.main_page.set_processing_error(
                        str(error)
                    )

        except Exception:
            self.logger.exception(
                "Failed to update transcription error UI."
            )

        self.refresh_controls()

    # ============================================================
    # EXPORT TXT
    # ============================================================

    def export_txt(self):
        try:
            from PySide6.QtWidgets import QFileDialog

            source_path = getattr(
                self,
                "current_output_txt",
                None,
            )

            audio_path = getattr(
                self,
                "current_audio",
                None,
            )

            if source_path is None:
                self.logger.warning(
                    "No meeting TXT is available."
                )
                return

            source_path = Path(source_path)

            if not source_path.exists():
                self.logger.warning(
                    "Meeting TXT does not exist: %s",
                    source_path,
                )
                return

            default_name = (
                f"{audio_path.stem}_meeting.txt"
                if audio_path
                else source_path.name
            )

            destination, _ = (
                QFileDialog.getSaveFileName(
                    self,
                    "Export TXT",
                    str(
                        PROJECT_ROOT
                        / default_name
                    ),
                    "Text Files (*.txt);;All Files (*)",
                )
            )

            if not destination:
                return

            shutil.copyfile(
                source_path,
                destination,
            )

            self.logger.info(
                "TXT exported: %s",
                destination,
            )

        except Exception:
            self.logger.exception(
                "Failed to export TXT."
            )

    # ============================================================
    # EXPORT JSON
    # ============================================================

    def export_json(self):
        try:
            from PySide6.QtWidgets import QFileDialog

            source_path = getattr(
                self,
                "current_output_json",
                None,
            )

            audio_path = getattr(
                self,
                "current_audio",
                None,
            )

            if source_path is None:
                self.logger.warning(
                    "No meeting JSON is available."
                )
                return

            source_path = Path(source_path)

            if not source_path.exists():
                self.logger.warning(
                    "Meeting JSON does not exist: %s",
                    source_path,
                )
                return

            default_name = (
                f"{audio_path.stem}_meeting.json"
                if audio_path
                else source_path.name
            )

            destination, _ = (
                QFileDialog.getSaveFileName(
                    self,
                    "Export JSON",
                    str(
                        PROJECT_ROOT
                        / default_name
                    ),
                    "JSON Files (*.json);;All Files (*)",
                )
            )

            if not destination:
                return

            shutil.copyfile(
                source_path,
                destination,
            )

            self.logger.info(
                "JSON exported: %s",
                destination,
            )

        except Exception:
            self.logger.exception(
                "Failed to export JSON."
            )

    # ============================================================
    # COPY TRANSCRIPTION
    # ============================================================

    def copy_transcription(self):
        try:
            from PySide6.QtWidgets import QApplication

            source_path = getattr(
                self,
                "current_output_txt",
                None,
            )

            if source_path is not None:
                source_path = Path(
                    source_path
                )

                if source_path.exists():
                    text = source_path.read_text(
                        encoding="utf-8"
                    )
                else:
                    text = ""
            else:
                text = ""

                if hasattr(
                    self,
                    "main_page",
                ) and hasattr(
                    self.main_page,
                    "transcription_text",
                ):
                    widget = (
                        self.main_page
                        .transcription_text
                    )

                    if hasattr(
                        widget,
                        "get_text",
                    ):
                        text = widget.get_text()

                    elif hasattr(
                        widget,
                        "toPlainText",
                    ):
                        text = widget.toPlainText()

            if not text.strip():
                self.logger.warning(
                    "Nothing to copy."
                )
                return

            QApplication.clipboard().setText(
                text
            )

            self.logger.info(
                "Meeting transcription copied."
            )

        except Exception:
            self.logger.exception(
                "Failed to copy transcription."
            )

    # ============================================================
    # OPEN OUTPUT DIRECTORY
    # ============================================================

    def open_output(self):
        try:
            output_path = getattr(
                self,
                "current_output_txt",
                None,
            )

            if output_path is not None:
                output_path = Path(
                    output_path
                )

                target = output_path.parent

            else:
                config = self.load_config()

                target = self.get_output_dir(
                    config
                )

            if sys.platform.startswith("win"):
                subprocess.Popen(
                    ["explorer", str(target)]
                )

            elif sys.platform == "darwin":
                subprocess.Popen(
                    ["open", str(target)]
                )

            else:
                subprocess.Popen(
                    ["xdg-open", str(target)]
                )

            self.logger.info(
                "Opened output directory: %s",
                target,
            )

        except Exception:
            self.logger.exception(
                "Failed to open output directory."
            )

    # ============================================================
    # RECORDING
    # ============================================================

    def start_recording(self):
        if self.processing or self.recording:
            return

        try:
            recordings_dir = Path(
                RECORDINGS_DIR
            )

            if not recordings_dir.is_absolute():
                recordings_dir = (
                    PROJECT_ROOT
                    / recordings_dir
                )

            recordings_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.recorder = AudioRecorder(
                recordings_dir
            )

            self.recording = True

            self.logger.info(
                "Recording started."
            )

            if hasattr(self, "main_page"):
                if hasattr(
                    self.main_page,
                    "set_recording",
                ):
                    self.main_page.set_recording(
                        True
                    )

            self.refresh_controls()

            self.recorder.start()

        except Exception:
            self.recording = False

            self.logger.exception(
                "Failed to start recording."
            )

            self.refresh_controls()

    def stop_recording(self):
        if not self.recording:
            return

        try:
            if self.recorder is None:
                return

            path = self.recorder.stop()

            self.recording = False

            if path:
                self.current_audio = Path(
                    path
                )

                self.logger.info(
                    "Recording saved: %s",
                    self.current_audio,
                )

                if hasattr(
                    self,
                    "main_page",
                ) and hasattr(
                    self.main_page,
                    "set_audio_path",
                ):
                    self.main_page.set_audio_path(
                        str(self.current_audio)
                    )

            if hasattr(self, "main_page"):
                if hasattr(
                    self.main_page,
                    "set_recording",
                ):
                    self.main_page.set_recording(
                        False
                    )

            self.refresh_controls()

        except Exception:
            self.recording = False

            self.logger.exception(
                "Failed to stop recording."
            )

            self.refresh_controls()

    # ============================================================
    # LOG QUEUE
    # ============================================================

    def process_log_queue(self):
        try:
            while True:
                record = self.log_queue.get_nowait()

                try:
                    message = (
                        self.ui_log_handler
                        .format(record)
                    )
                except Exception:
                    message = record.getMessage()

                if hasattr(
                    self,
                    "main_page",
                ):
                    page = self.main_page

                    if hasattr(
                        page,
                        "append_log",
                    ):
                        page.append_log(
                            message
                        )

                    elif hasattr(
                        page,
                        "log_text",
                    ):
                        widget = page.log_text

                        if hasattr(
                            widget,
                            "appendPlainText",
                        ):
                            widget.appendPlainText(
                                message
                            )

                        elif hasattr(
                            widget,
                            "append",
                        ):
                            widget.append(
                                message
                            )

        except queue.Empty:
            pass

        except Exception:
            self.logger.exception(
                "Failed to process UI log queue."
            )

    # ============================================================
    # CLEAR LOG
    # ============================================================

    def clear_logs(self):
        try:
            if hasattr(
                self,
                "main_page",
            ):
                page = self.main_page

                if hasattr(
                    page,
                    "clear_logs",
                ):
                    page.clear_logs()

                elif hasattr(
                    page,
                    "log_text",
                ):
                    widget = page.log_text

                    if hasattr(
                        widget,
                        "clear",
                    ):
                        widget.clear()

            self.logger.info(
                "UI logs cleared."
            )

        except Exception:
            self.logger.exception(
                "Failed to clear logs."
            )

    # ============================================================
    # CLOSE
    # ============================================================

    def on_close(self):
        try:
            if getattr(
                self,
                "recording",
                False,
            ):
                try:
                    self.stop_recording()
                except Exception:
                    self.logger.exception(
                        "Failed to stop recording during close."
                    )

            if getattr(
                self,
                "model_loaded",
                False,
            ):
                try:
                    self.unload_model()
                except Exception:
                    self.logger.exception(
                        "Failed to unload model during close."
                    )

            if hasattr(
                self,
                "settings",
            ):
                try:
                    if hasattr(
                        self,
                        "tabs",
                    ):
                        self.settings.data[
                            "last_tab"
                        ] = self.tabs.currentIndex()

                    self.settings.save()

                except Exception:
                    self.logger.exception(
                        "Failed to save UI settings."
                    )

        except Exception:
            self.logger.exception(
                "Error during application close."
            )