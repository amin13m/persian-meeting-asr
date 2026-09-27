
from __future__ import annotations

import json
import logging
import os
import sys
import time
import threading

from pathlib import Path

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QFileDialog,
    QMessageBox,
)

from engines.audio_chunker import AudioChunker
from ui.recorder import AudioRecorder

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

        # ----------------------------------------------------
        # Processing state
        # ----------------------------------------------------

        self.processing = True
        self.operation = "loading"

        self.logger.info(
            "MODEL | loading started | model=%s",
            model_name,
        )

        # ----------------------------------------------------
        # MainPage processing UI
        # ----------------------------------------------------

        try:
            self.main_page.set_processing_running(
                "در حال بارگذاری مدل...",
                indeterminate=True,
            )
        except Exception:
            # Compatibility with older MainPage
            self.set_status("loading")
            self.set_progress(0, "")

        self.refresh_controls()

        # ----------------------------------------------------
        # Start worker
        # ----------------------------------------------------

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

            model_path = (
                PROJECT_ROOT
                / config
                .get("model", {})
                .get("path", "")
            )

            self.logger.info(
                "MODEL | model directory: %s",
                model_path,
            )

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model directory not found: {model_path}"
                )

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

            elapsed = (
                time.perf_counter()
                - started
            )

            self.logger.info(
                "MODEL | engine.load() completed | "
                "elapsed=%.3f sec",
                elapsed,
            )

            self.engine = engine

            self.logger.info(
                "MODEL | initializing AudioChunker"
            )

            self.chunker = AudioChunker(config)

            self.model_loaded = True

            self.logger.info(
                "MODEL | loading completed successfully | "
                "model=%s | elapsed=%.3f sec",
                model_name,
                elapsed,
            )

            self.ui_call(
                self._model_loaded_ui
            )

        except Exception:
            if engine is not None:
                try:
                    engine.unload()
                except Exception:
                    pass

            self.logger.exception(
                "MODEL | loading failed | model=%s",
                model_name,
            )

            self.ui_call(
                self._model_load_failed_ui
            )






    

    def _load_model_worker(
        self,
        model_name,
        engine_class,
    ):
        engine = None

        try:
            config = self.load_config()

            configured = (
                config
                .get("engine", {})
                .get("name")
            )

            if configured and configured != model_name:
                raise ValueError(
                    "Selected model does not match "
                    "engine.name in config.yaml."
                )

            model_path = (
                PROJECT_ROOT
                / config
                .get("model", {})
                .get("path", "")
            )

            self.logger.info(
                "Model directory: %s",
                model_path,
            )

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model directory not found: {model_path}"
                )

            started = time.perf_counter()

            engine = engine_class(config)

            self.logger.info("Loading engine...")

            engine.load()

            elapsed = (
                time.perf_counter()
                - started
            )

            self.engine = engine

            self.chunker = AudioChunker(config)

            self.model_loaded = True

            self.logger.info(
                "Engine loaded successfully in %.3f sec.",
                elapsed,
            )

            self.ui_call(self._model_loaded_ui)

        except Exception:
            if engine is not None:
                try:
                    engine.unload()
                except Exception:
                    pass

            self.logger.exception(
                "Model loading failed."
            )

            self.ui_call(
                self._model_load_failed_ui
            )

    def _model_loaded_ui(self):
        self.model_loaded = True
        self.processing = False
        self.operation = None

        # توقف Spinner مدل
        self.main_page.set_model_loading(False)

        # توقف قطعی Progress Animation
        self.main_page.set_processing_success(
            "مدل با موفقیت بارگذاری شد."
        )

        self.refresh_controls()

        self.logger.info(
            "MODEL | UI updated | model_loaded=%s | current_audio=%s",
            self.model_loaded,
            self.current_audio,
        )

    def _model_load_failed_ui(self):
        self.processing = False
        self.operation = None
        self.model_loaded = False
        self.engine = None
        self.chunker = None

        self.main_page.set_model_loading(False)

        self.main_page.set_processing_error(
            "خطا در بارگذاری مدل."
        )

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
            "Unloading engine..."
        )

        threading.Thread(
            target=self._unload_model_worker,
            daemon=True,
        ).start()

    def _unload_model_worker(self):
        try:
            if self.engine:
                self.engine.unload()

            self.engine = None
            self.chunker = None
            self.model_loaded = False

            self.logger.info(
                "Engine unloaded."
            )

            self.ui_call(
                self._model_unloaded_ui
            )

        except Exception:
            self.logger.exception(
                "Engine unload failed."
            )

            self.ui_call(
                self._model_unload_failed_ui
            )

    def _model_unloaded_ui(self):
        self.processing = False
        self.operation = None

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

        # ----------------------------------------------------
        # IMPORTANT:
        # current_audio belongs to the application state.
        # The actual text widget belongs to MainPage.
        # ----------------------------------------------------

        self.current_audio = selected_path
        self.audio_var = str(selected_path)

        self.current_result = None

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

            # IMPORTANT:
            # transcription_text belongs to MainPage.
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
    # Transcription
    # ========================================================

    def transcribe(self):
        if self.processing or self.recording:
            return

        if (
            not self.model_loaded
            or not self.engine
            or not self.chunker
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

        engine = self.engine
        chunker = self.chunker

        self.processing = True
        self.operation = "processing"
        self.current_result = None

        self.set_progress(0, "")

        self.main_page.transcription_text.set_text(
            self.TEXT["processing"],
            self.language == "fa",
        )

        self.set_status("processing")
        self.refresh_controls()

        self.logger.info(
            "Starting transcription: %s",
            audio_path,
        )

        threading.Thread(
            target=self._transcription_worker,
            args=(
                audio_path,
                engine,
                chunker,
            ),
            daemon=True,
        ).start()

    def _transcription_worker(
        self,
        audio_path,
        engine,
        chunker,
    ):
        started = time.perf_counter()

        try:
            self.logger.info(
                "Analyzing audio and creating chunks..."
            )

            chunks = chunker.chunk_file(audio_path)

            if not chunks:
                raise RuntimeError(
                    "No audio chunks were created."
                )

            total = len(chunks)

            self.logger.info(
                "Created %d chunk(s) from %s",
                total,
                audio_path.name,
            )

            for chunk in chunks:
                self.logger.info(
                    "Chunk %03d | %s -> %s | %.2f sec",
                    chunk.index,
                    format_seconds(chunk.start),
                    format_seconds(chunk.end),
                    chunk.duration,
                )

            self.ui_call(
                lambda: self._prepare_progress_ui(total)
            )

            chunk_results = []
            texts = []

            total_audio = 0.0
            total_inference = 0.0

            for position, chunk in enumerate(
                chunks,
                1,
            ):
                elapsed = (
                    time.perf_counter()
                    - started
                )

                operation_text = (
                    TEXTS[self.language]["processing_chunk"]
                    + f" {position} / {total}"
                )

                self.ui_call(
                    lambda
                    operation_text=operation_text,
                    index=position,
                    elapsed=elapsed:
                    self._update_processing_ui(
                        operation_text,
                        index,
                        total,
                        elapsed,
                    )
                )

                self.logger.info(
                    "[%d/%d] Transcribing chunk %03d...",
                    position,
                    total,
                    chunk.index,
                )

                inference_started = (
                    time.perf_counter()
                )

                raw = engine.transcribe(
                    chunk.audio,
                    sample_rate=chunker.sample_rate,
                )

                inference_time = (
                    time.perf_counter()
                    - inference_started
                )

                if isinstance(raw, dict):
                    text = str(
                        raw.get("text", "")
                    ).strip()
                else:
                    text = str(
                        raw or ""
                    ).strip()

                duration = float(chunk.duration)

                total_audio += duration
                total_inference += inference_time

                if text:
                    texts.append(text)

                rtf = (
                    inference_time / duration
                    if duration > 0
                    else None
                )

                realtime_factor = (
                    duration / inference_time
                    if inference_time > 0
                    else None
                )

                chunk_results.append(
                    {
                        "index": int(chunk.index),
                        "start": round(
                            float(chunk.start),
                            3,
                        ),
                        "end": round(
                            float(chunk.end),
                            3,
                        ),
                        "duration": round(
                            duration,
                            3,
                        ),
                        "inference_time": round(
                            inference_time,
                            3,
                        ),
                        "rtf": (
                            round(rtf, 6)
                            if rtf is not None
                            else None
                        ),
                        "realtime_factor": (
                            round(
                                realtime_factor,
                                3,
                            )
                            if realtime_factor is not None
                            else None
                        ),
                        "text": text,
                    }
                )

                self.logger.info(
                    "Chunk %03d completed | "
                    "audio=%.2fs | "
                    "inference=%.2fs | "
                    "RTF=%.3f",
                    chunk.index,
                    duration,
                    inference_time,
                    rtf or 0.0,
                )

                self.logger.info(
                    "Text: %s",
                    text or "[empty]",
                )

            final_text = " ".join(
                item.strip()
                for item in texts
                if item and item.strip()
            ).strip()

            elapsed = (
                time.perf_counter()
                - started
            )

            overall_rtf = (
                total_inference / total_audio
                if total_audio > 0
                else None
            )

            overall_realtime = (
                total_audio / total_inference
                if total_inference > 0
                else None
            )

            config = self.load_config()

            output_dir = self.get_output_dir(config)

            txt_path = (
                output_dir
                / f"{audio_path.stem}.txt"
            )

            json_path = (
                output_dir
                / f"{audio_path.stem}.json"
            )

            output_config = config.get(
                "output",
                {},
            )

            save_txt = output_config.get(
                "save_txt",
                True,
            )

            save_json = output_config.get(
                "save_json",
                True,
            )

            if save_txt:
                txt_path.write_text(
                    final_text + "\n",
                    encoding="utf-8",
                )

                self.logger.info(
                    "TXT saved: %s",
                    txt_path,
                )

            result = {
                "file": {
                    "name": audio_path.name,
                    "path": str(audio_path),
                    "extension": audio_path.suffix.lower(),
                },
                "engine": {
                    "name": (
                        config
                        .get("engine", {})
                        .get("name")
                    ),
                    "model_path": str(
                        PROJECT_ROOT
                        / config
                        .get("model", {})
                        .get("path", "")
                    ),
                },
                "audio": {
                    "sample_rate": chunker.sample_rate,
                    "language": (
                        config
                        .get("audio", {})
                        .get("language", "fa")
                    ),
                    "task": (
                        config
                        .get("audio", {})
                        .get("task", "transcribe")
                    ),
                    "duration": round(
                        total_audio,
                        3,
                    ),
                },
                "processing": {
                    "chunks": len(chunks),
                    "inference_time": round(
                        total_inference,
                        3,
                    ),
                    "total_processing_time": round(
                        elapsed,
                        3,
                    ),
                    "rtf": (
                        round(
                            overall_rtf,
                            6,
                        )
                        if overall_rtf is not None
                        else None
                    ),
                    "realtime_factor": (
                        round(
                            overall_realtime,
                            3,
                        )
                        if overall_realtime is not None
                        else None
                    ),
                },
                "transcription": {
                    "text": final_text,
                },
                "chunks": chunk_results,
            }

            if save_json:
                json_path.write_text(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                self.logger.info(
                    "JSON saved: %s",
                    json_path,
                )

            self.current_result = result

            self.logger.info(
                "FILE COMPLETED | "
                "File: %s | "
                "Audio: %.2f sec | "
                "Chunks: %d | "
                "Inference: %.2f sec | "
                "Total: %.2f sec",
                audio_path.name,
                total_audio,
                len(chunks),
                total_inference,
                elapsed,
            )

            self.ui_call(
                lambda: self._transcription_completed_ui(
                    final_text,
                    result,
                    elapsed,
                )
            )

        except Exception:
            self.logger.exception(
                "Failed processing file: %s",
                audio_path.name,
            )

            self.ui_call(
                self._transcription_failed_ui
            )

    # ========================================================
    # Processing UI
    # ========================================================

    def _prepare_progress_ui(self, total):
        """
        Prepare progress controls.

        These widgets belong to MainPage.
        """

        self.main_page.progress.setRange(
            0,
            max(total, 1),
        )

        self.main_page.operation_label.setText(
            ""
        )

        self.set_progress(
            0,
            f"0 / {total}",
        )

    def _update_processing_ui(
        self,
        operation,
        index,
        total,
        elapsed,
    ):
        """
        Update processing progress.

        This method is called on the Qt UI thread.
        """

        self.main_page.operation_label.setText(
            operation
        )

        self.main_page.progress.setRange(
            0,
            max(total, 1),
        )

        self.set_progress(
            index,
            (
                f"{index} / {total}"
                f"  •  "
                f"{TEXTS[self.language]['elapsed']}: "
                f"{elapsed:.1f}s"
            ),
        )

    def _transcription_completed_ui(
        self,
        text,
        result,
        elapsed,
    ):
        self.processing = False
        self.operation = None

        total = result["processing"]["chunks"]

        self.main_page.progress.setRange(
            0,
            max(total, 1),
        )

        self.set_progress(
            total,
            f"{total} / {total}",
        )

        self.main_page.transcription_text.set_text(
            text or self.TEXT["no_transcription"],
            self.language == "fa",
        )

        self.main_page.operation_label.setText(
            ""
        )

        self.set_status(
            "completed",
            (
                f"{self.TEXT['elapsed']}: "
                f"{elapsed:.2f}s"
            ),
        )

        self.refresh_controls()

        try:
            self.files_page.reload()
        except Exception:
            self.logger.exception(
                "Failed to refresh files page after transcription."
            )

    def _transcription_failed_ui(self):
        self.processing = False
        self.operation = None

        self.main_page.operation_label.setText("")

        self.set_status("error")
        self.refresh_controls()

        QMessageBox.critical(
            self,
            self.TEXT["error"],
            self.TEXT["transcription_failed"],
        )

    # ========================================================
    # Export TXT
    # ========================================================

    def export_txt(self):
        if not self.current_result:
            return

        text = (
            self.current_result
            .get("transcription", {})
            .get("text", "")
        )

        filename = (
            Path(
                self.current_result
                .get("file", {})
                .get("name", "transcription")
            ).stem
            + ".txt"
        )

        path, _ = QFileDialog.getSaveFileName(
            self,
            self.TEXT["export_txt"],
            str(
                DEFAULT_OUTPUT_DIR
                / filename
            ),
            "TXT (*.txt)",
        )

        if path:
            Path(path).write_text(
                text,
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

    # ========================================================
    # Export JSON
    # ========================================================

    def export_json(self):
        if not self.current_result:
            return

        filename = (
            Path(
                self.current_result
                .get("file", {})
                .get("name", "transcription")
            ).stem
            + ".json"
        )

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

        QGuiApplication.clipboard().setText(text)

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

            if self.engine:
                self.logger.info(
                    "Unloading engine before exit..."
                )

                self.engine.unload()

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

