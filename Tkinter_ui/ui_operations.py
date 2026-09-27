from __future__ import annotations

import json
import logging
import math
import os
import queue
import re
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from engines.audio_chunker import AudioChunker
from ui.logger import UILogHandler
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
    def load_model(self):

        if (
            self.processing
            or self.recording
        ):
            return

        model_name = (
            self.model_var.get().strip()
        )

        if not model_name:

            messagebox.showwarning(
                TEXTS[self.language]["warning"],
                TEXTS[self.language]["select_model"],
            )

            return

        engine_class = ENGINE_REGISTRY.get(
            model_name
        )

        if engine_class is None:

            messagebox.showerror(
                TEXTS[self.language]["error"],
                f"Unknown model: {model_name}",
            )

            return

        self.processing = True
        self.operation = "loading"

        self.set_status(
            "loading"
        )

        self.progress_var.set(0)
        self.progress_text_var.set("")

        self.refresh_controls()

        self.logger.info(
            "Selected engine: %s",
            model_name,
        )

        thread = threading.Thread(
            target=self._load_model_worker,
            args=(
                model_name,
                engine_class,
            ),
            daemon=True,
        )

        thread.start()

    def _load_model_worker(
        self,
        model_name,
        engine_class,
    ):

        engine = None

        try:

            config = self.load_config()

            model_config = config.get(
                "model",
                {},
            )

            configured_model_name = config.get(
                "engine",
                {},
            ).get(
                "name"
            )

            if configured_model_name:
                if configured_model_name != model_name:
                    raise ValueError(
                        "Selected model does not match "
                        "engine.name in config.yaml."
                    )

            model_path = (
                PROJECT_ROOT
                / model_config.get(
                    "path",
                    "",
                )
            )

            self.logger.info(
                "Model directory: %s",
                model_path,
            )

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model directory not found: {model_path}"
                )

            load_start = time.perf_counter()

            engine = engine_class(
                config
            )

            self.logger.info(
                "Loading engine..."
            )

            engine.load()

            load_time = (
                time.perf_counter()
                - load_start
            )

            chunker = AudioChunker(
                config
            )

            self.engine = engine
            self.chunker = chunker
            self.model_loaded = True

            self.logger.info(
                "Engine loaded successfully in %.3f sec.",
                load_time,
            )

            self.root.after(
                0,
                self._model_loaded_ui,
            )

        except Exception:

            if engine is not None:
                try:
                    engine.unload()
                except Exception:
                    pass

            self.logger.exception(
                "Model loading failed."
            )

            self.root.after(
                0,
                self._model_load_failed_ui,
            )

    def _model_loaded_ui(self):

        self.processing = False
        self.operation = None

        self.progress_var.set(100)

        self.set_status(
            "loaded"
        )

        self.refresh_controls()

    def _model_load_failed_ui(self):

        self.processing = False
        self.operation = None
        self.model_loaded = False

        self.engine = None
        self.chunker = None

        self.set_status(
            "error"
        )

        self.refresh_controls()

        messagebox.showerror(
            TEXTS[self.language]["error"],
            TEXTS[self.language]["model_load_failed"],
        )

    def unload_model(self):

        if (
            self.processing
            or self.recording
        ):
            return

        if not self.model_loaded:
            return

        self.processing = True
        self.operation = "unloading"

        self.set_status(
            "unloading"
        )

        self.progress_var.set(0)

        self.refresh_controls()

        self.logger.info(
            "Unloading engine..."
        )

        thread = threading.Thread(
            target=self._unload_model_worker,
            daemon=True,
        )

        thread.start()

    def _unload_model_worker(self):

        try:

            if self.engine is not None:
                self.engine.unload()

            self.engine = None
            self.chunker = None
            self.model_loaded = False

            self.logger.info(
                "Engine unloaded."
            )

            self.root.after(
                0,
                self._model_unloaded_ui,
            )

        except Exception:

            self.logger.exception(
                "Engine unload failed."
            )

            self.root.after(
                0,
                self._model_unload_failed_ui,
            )

    def _model_unloaded_ui(self):

        self.processing = False
        self.operation = None

        self.set_status(
            "unloaded"
        )

        self.refresh_controls()

    def _model_unload_failed_ui(self):

        self.processing = False
        self.operation = None

        self.set_status(
            "error"
        )

        self.refresh_controls()

    def select_audio(self):

        if (
            self.processing
            or self.recording
        ):
            return

        try:
            config = self.load_config()
            input_dir = self.get_input_dir(config)
        except Exception:
            input_dir = DEFAULT_INPUT_DIR

        t = TEXTS[self.language]

        path = filedialog.askopenfilename(
            title=t["open_audio"],
            initialdir=str(input_dir),
            filetypes=[
                (
                    t["audio_files"],
                    "*.wav *.mp3 *.flac *.m4a *.ogg *.aac",
                ),
                (
                    t["all_files"],
                    "*.*",
                ),
            ],
        )

        if not path:
            return

        self.current_audio = Path(path)

        self.audio_var.set(
            str(self.current_audio)
        )

        self.current_result = None

        self.transcription_text.delete(
            "1.0",
            "end",
        )

        tag = (
            "rtl"
            if self.language == "fa"
            else "ltr"
        )

        self.transcription_text.insert(
            "1.0",
            t["no_transcription"],
            tag,
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

    def start_recording(self):

        if (
            self.processing
            or self.recording
        ):
            return

        try:

            config = self.load_config()

            sample_rate = int(
                config.get(
                    "audio",
                    {},
                ).get(
                    "sample_rate",
                    16000,
                )
            )

            self.recorder = AudioRecorder(
                sample_rate=sample_rate,
                channels=1,
                dtype="float32",
            )

            self.recorder.start()

            self.recording = True
            self.operation = "recording"

            self.set_status(
                "recording"
            )

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

            self.set_status(
                "error"
            )

            self.refresh_controls()

    def stop_recording(self):

        if not self.recording:
            return

        try:

            output_path = (
                self.recorder.generate_output_path(
                    RECORDINGS_DIR
                )
            )

            self.recorder.stop(
                output_path
            )

            self.current_audio = Path(
                output_path
            )

            self.audio_var.set(
                str(self.current_audio)
            )

            self.current_result = None

            self.transcription_text.delete(
                "1.0",
                "end",
            )

            self.transcription_text.insert(
                "1.0",
                TEXTS[self.language][
                    "no_transcription"
                ],
                (
                    "rtl"
                    if self.language == "fa"
                    else "ltr"
                ),
            )

            self.logger.info(
                "Recording saved: %s",
                output_path,
            )

            self.recording = False
            self.operation = None

            self.set_status(
                "recording_saved"
            )

        except Exception:

            self.logger.exception(
                "Failed to stop recording."
            )

            self.recording = False
            self.operation = None

            self.set_status(
                "error"
            )

        finally:

            self.recorder = None
            self.refresh_controls()

    def transcribe(self):

        if (
            self.processing
            or self.recording
        ):
            return

        if (
            not self.model_loaded
            or self.engine is None
            or self.chunker is None
        ):

            messagebox.showwarning(
                TEXTS[self.language]["warning"],
                TEXTS[self.language]["load_model_first"],
            )

            return

        if not self.current_audio:

            messagebox.showwarning(
                TEXTS[self.language]["warning"],
                TEXTS[self.language]["select_audio_first"],
            )

            return

        audio_path = self.current_audio
        engine = self.engine
        chunker = self.chunker

        self.processing = True
        self.operation = "processing"

        self.current_result = None

        self.progress_var.set(0)
        self.progress_text_var.set("")

        self.transcription_text.delete(
            "1.0",
            "end",
        )

        self.transcription_text.insert(
            "1.0",
            TEXTS[self.language]["processing"],
            (
                "rtl"
                if self.language == "fa"
                else "ltr"
            ),
        )

        self.set_status(
            "processing"
        )

        self.refresh_controls()

        self.logger.info(
            "Starting transcription: %s",
            audio_path,
        )

        thread = threading.Thread(
            target=self._transcription_worker,
            args=(
                audio_path,
                engine,
                chunker,
            ),
            daemon=True,
        )

        thread.start()

    def _transcription_worker(
        self,
        audio_path: Path,
        engine,
        chunker,
    ):

        file_start_time = time.perf_counter()

        try:

            self.logger.info(
                "Analyzing audio and creating chunks..."
            )

            chunks = chunker.chunk_file(
                audio_path
            )

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

            self.root.after(
                0,
                lambda:
                self._prepare_progress_ui(
                    total
                ),
            )

            chunk_results = []
            texts = []

            total_audio_duration = 0.0
            total_inference_time = 0.0

            self.logger.info(
                "Starting transcription..."
            )

            for position, chunk in enumerate(
                chunks,
                start=1,
            ):

                elapsed = (
                    time.perf_counter()
                    - file_start_time
                )

                operation_text = (
                    f"{TEXTS[self.language]['processing_chunk']} "
                    f"{position} / {total}"
                )

                self.root.after(
                    0,
                    lambda
                    op=operation_text,
                    i=position,
                    total_chunks=total,
                    e=elapsed:
                    self._update_processing_ui(
                        op,
                        i,
                        total_chunks,
                        e,
                    ),
                )

                self.logger.info(
                    "[%d/%d] Transcribing chunk %03d...",
                    position,
                    total,
                    chunk.index,
                )

                inference_start = (
                    time.perf_counter()
                )

                raw_result = engine.transcribe(
                    chunk.audio,
                    sample_rate=chunker.sample_rate,
                )

                inference_time = (
                    time.perf_counter()
                    - inference_start
                )

                if isinstance(
                    raw_result,
                    dict,
                ):
                    text = str(
                        raw_result.get(
                            "text",
                            "",
                        )
                    ).strip()

                else:
                    text = str(
                        raw_result or ""
                    ).strip()

                chunk_duration = float(
                    chunk.duration
                )

                total_audio_duration += (
                    chunk_duration
                )

                total_inference_time += (
                    inference_time
                )

                if text:
                    texts.append(text)

                rtf = (
                    inference_time
                    / chunk_duration
                    if chunk_duration > 0
                    else None
                )

                realtime_factor = (
                    chunk_duration
                    / inference_time
                    if inference_time > 0
                    else None
                )

                chunk_result = {
                    "index": int(
                        chunk.index
                    ),
                    "start": round(
                        float(chunk.start),
                        3,
                    ),
                    "end": round(
                        float(chunk.end),
                        3,
                    ),
                    "duration": round(
                        chunk_duration,
                        3,
                    ),
                    "inference_time": round(
                        inference_time,
                        3,
                    ),
                    "rtf": round(
                        rtf,
                        6,
                    )
                    if rtf is not None
                    else None,
                    "realtime_factor": round(
                        realtime_factor,
                        3,
                    )
                    if realtime_factor is not None
                    else None,
                    "text": text,
                }

                chunk_results.append(
                    chunk_result
                )

                self.logger.info(
                    "Chunk %03d completed | "
                    "audio=%.2fs | "
                    "inference=%.2fs | "
                    "RTF=%.3f",
                    chunk.index,
                    chunk_duration,
                    inference_time,
                    (
                        rtf
                        if rtf is not None
                        else 0.0
                    ),
                )

                if text:
                    self.logger.info(
                        "Text: %s",
                        text,
                    )
                else:
                    self.logger.info(
                        "Text: [empty]"
                    )

            final_text = " ".join(
                text.strip()
                for text in texts
                if text and text.strip()
            ).strip()

            file_processing_time = (
                time.perf_counter()
                - file_start_time
            )

            overall_rtf = (
                total_inference_time
                / total_audio_duration
                if total_audio_duration > 0
                else None
            )

            overall_realtime_factor = (
                total_audio_duration
                / total_inference_time
                if total_inference_time > 0
                else None
            )

            config = self.load_config()

            output_dir = self.get_output_dir(
                config
            )

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

            if save_json:

                model_path = (
                    PROJECT_ROOT
                    / config.get(
                        "model",
                        {},
                    ).get(
                        "path",
                        "",
                    )
                )

                json_data = {
                    "file": {
                        "name": audio_path.name,
                        "path": str(
                            audio_path
                        ),
                        "extension": (
                            audio_path.suffix.lower()
                        ),
                    },

                    "engine": {
                        "name": config.get(
                            "engine",
                            {},
                        ).get(
                            "name"
                        ),
                        "model_path": str(
                            model_path
                        ),
                    },

                    "audio": {
                        "sample_rate": (
                            chunker.sample_rate
                        ),
                        "language": config.get(
                            "audio",
                            {},
                        ).get(
                            "language",
                            "fa",
                        ),
                        "task": config.get(
                            "audio",
                            {},
                        ).get(
                            "task",
                            "transcribe",
                        ),
                        "duration": round(
                            total_audio_duration,
                            3,
                        ),
                    },

                    "processing": {
                        "chunks": len(
                            chunks
                        ),
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

                self.logger.info(
                    "JSON saved: %s",
                    json_path,
                )

            self.logger.info("")
            self.logger.info(
                "-" * 60
            )

            self.logger.info(
                "FILE COMPLETED"
            )

            self.logger.info(
                "File       : %s",
                audio_path.name,
            )

            self.logger.info(
                "Audio      : %.2f sec",
                total_audio_duration,
            )

            self.logger.info(
                "Chunks     : %d",
                len(chunks),
            )

            self.logger.info(
                "Inference  : %.2f sec",
                total_inference_time,
            )

            self.logger.info(
                "Total time : %.2f sec",
                file_processing_time,
            )

            if overall_rtf is not None:
                self.logger.info(
                    "RTF        : %.3f",
                    overall_rtf,
                )

            if overall_realtime_factor is not None:
                self.logger.info(
                    "Realtime   : %.2fx",
                    overall_realtime_factor,
                )

            self.logger.info(
                "-" * 60
            )

            result = {
                "file": {
                    "name": audio_path.name,
                    "path": str(audio_path),
                    "extension": audio_path.suffix.lower(),
                },

                "engine": {
                    "name": config.get(
                        "engine",
                        {},
                    ).get(
                        "name"
                    ),
                    "model_path": str(
                        PROJECT_ROOT
                        / config.get(
                            "model",
                            {},
                        ).get(
                            "path",
                            "",
                        )
                    ),
                },

                "audio": {
                    "sample_rate": (
                        chunker.sample_rate
                    ),
                    "language": config.get(
                        "audio",
                        {},
                    ).get(
                        "language",
                        "fa",
                    ),
                    "task": config.get(
                        "audio",
                        {},
                    ).get(
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

            self.current_result = result

            self.root.after(
                0,
                lambda:
                self._transcription_completed_ui(
                    final_text,
                    result,
                    file_processing_time,
                ),
            )

        except Exception:

            self.logger.exception(
                "Failed processing file: %s",
                audio_path.name,
            )

            self.root.after(
                0,
                self._transcription_failed_ui,
            )

    def _prepare_progress_ui(
        self,
        total,
    ):

        self.progress.configure(
            mode="determinate",
            maximum=max(
                total,
                1,
            ),
        )

        self.progress_var.set(0)

        self.progress_text_var.set(
            f"0 / {total}"
        )

    def _update_processing_ui(
        self,
        operation,
        index,
        total,
        elapsed,
    ):

        self.operation_var.set(
            operation
        )

        self.progress.configure(
            maximum=max(
                total,
                1,
            ),
        )

        self.progress_var.set(
            index
        )

        self.progress_text_var.set(
            f"{index} / {total}  •  "
            f"{TEXTS[self.language]['elapsed']}: "
            f"{elapsed:.1f}s"
        )

    def _transcription_completed_ui(
        self,
        text,
        result,
        elapsed,
    ):

        self.processing = False
        self.operation = None

        total_chunks = result[
            "processing"
        ][
            "chunks"
        ]

        self.progress_var.set(
            total_chunks
        )

        self.progress.configure(
            maximum=max(
                total_chunks,
                1
            )
        )

        self.progress_text_var.set(
            f"{total_chunks} / {total_chunks}"
        )

        self.transcription_text.delete(
            "1.0",
            "end",
        )

        tag = (
            "rtl"
            if self.language == "fa"
            else "ltr"
        )

        self.transcription_text.insert(
            "1.0",
            (
                text
                or TEXTS[self.language][
                    "no_transcription"
                ]
            ),
            tag,
        )

        self.set_status(
            "completed",
            (
                f"{TEXTS[self.language]['elapsed']}: "
                f"{elapsed:.2f}s"
            ),
        )

        self.refresh_controls()

    def _transcription_failed_ui(self):

        self.processing = False
        self.operation = None

        self.set_status(
            "error"
        )

        self.transcription_text.delete(
            "1.0",
            "end",
        )

        self.transcription_text.insert(
            "1.0",
            TEXTS[self.language][
                "transcription_failed"
            ],
            (
                "rtl"
                if self.language == "fa"
                else "ltr"
            ),
        )

        self.refresh_controls()

        messagebox.showerror(
            TEXTS[self.language]["error"],
            TEXTS[self.language][
                "transcription_failed"
            ],
        )

    def export_txt(self):

        if not self.current_result:
            return

        text = (
            self.current_result.get(
                "transcription",
                {},
            )
        )

        if isinstance(text, dict):
            text = text.get(
                "text",
                "",
            )

        default_name = (
            Path(
                self.current_result.get(
                    "file",
                    {},
                ).get(
                    "name",
                    "transcription",
                )
                if isinstance(
                    self.current_result.get(
                        "file"
                    ),
                    dict,
                )
                else self.current_result.get(
                    "file",
                    "transcription",
                )
            ).stem
            + ".txt"
        )

        path = filedialog.asksaveasfilename(
            title=TEXTS[self.language][
                "export_txt"
            ],
            initialdir=str(
                DEFAULT_OUTPUT_DIR
            ),
            initialfile=default_name,
            defaultextension=".txt",
            filetypes=[
                (
                    "TXT",
                    "*.txt",
                ),
            ],
        )

        if not path:
            return

        Path(path).write_text(
            str(text),
            encoding="utf-8",
        )

        self.logger.info(
            "TXT exported: %s",
            path,
        )

        messagebox.showinfo(
            TEXTS[self.language]["success"],
            (
                f"{TEXTS[self.language]['saved_txt']}\n"
                f"{path}"
            ),
        )

    def export_json(self):

        if not self.current_result:
            return

        file_data = self.current_result.get(
            "file",
            {},
        )

        if isinstance(
            file_data,
            dict,
        ):
            filename = file_data.get(
                "name",
                "transcription",
            )
        else:
            filename = str(
                file_data
            )

        default_name = (
            Path(filename).stem
            + ".json"
        )

        path = filedialog.asksaveasfilename(
            title=TEXTS[self.language][
                "export_json"
            ],
            initialdir=str(
                DEFAULT_OUTPUT_DIR
            ),
            initialfile=default_name,
            defaultextension=".json",
            filetypes=[
                (
                    "JSON",
                    "*.json",
                ),
            ],
        )

        if not path:
            return

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

        messagebox.showinfo(
            TEXTS[self.language]["success"],
            (
                f"{TEXTS[self.language]['saved_json']}\n"
                f"{path}"
            ),
        )

    def open_output(self):

        try:

            try:
                config = self.load_config()
                output_dir = self.get_output_dir(
                    config
                )
            except Exception:
                output_dir = DEFAULT_OUTPUT_DIR

            if os.name == "nt":

                os.startfile(
                    output_dir
                )

            elif sys.platform == "darwin":

                os.system(
                    f'open "{output_dir}"'
                )

            else:

                os.system(
                    f'xdg-open "{output_dir}"'
                )

        except Exception:

            self.logger.exception(
                "Could not open output directory."
            )

    # --------------------------------------------------------
    # Copy complete transcription
    # --------------------------------------------------------

    def copy_transcription_text(self):

        try:

            text = self.transcription_text.get(
                "1.0",
                "end-1c",
            )

        except tk.TclError:

            return "break"

        if not text:
            return "break"

        # Normalize line endings.
        text = text.replace(
            "\r\n",
            "\n",
        ).replace(
            "\r",
            "\n",
        )

        # Remove invisible directional formatting characters
        # if any were introduced by future RTL handling.
        directional_chars = (
            "\u200e",  # LRM
            "\u200f",  # RLM
            "\u202a",  # LRE
            "\u202b",  # RLE
            "\u202c",  # PDF
            "\u202d",  # LRO
            "\u202e",  # RLO
            "\u2066",  # LRI
            "\u2067",  # RLI
            "\u2068",  # FSI
            "\u2069",  # PDI
        )

        for char in directional_chars:
            text = text.replace(
                char,
                "",
            )

        try:

            self.root.clipboard_clear()
            self.root.clipboard_append(text)

            # Keep clipboard available after the application
            # remains active.
            self.root.update()

            self.logger.info(
                "Transcription copied to clipboard."
            )

        except tk.TclError:

            self.logger.exception(
                "Could not copy transcription to clipboard."
            )

        return "break"

    def process_log_queue(self):

        while True:

            try:
                message = (
                    self.log_queue.get_nowait()
                )

            except queue.Empty:
                break

            self.insert_log(
                message
            )

        try:
            self.root.after(
                80,
                self.process_log_queue,
            )
        except tk.TclError:
            pass

    def insert_log(
        self,
        message,
    ):

        if not message:
            return

        level = "DEFAULT"

        if re.search(
            r"\|\s*ERROR\s*\|",
            message,
            re.IGNORECASE,
        ):
            level = "ERROR"

        elif re.search(
            r"\|\s*WARNING\s*\|",
            message,
            re.IGNORECASE,
        ):
            level = "WARNING"

        elif re.search(
            r"\|\s*DEBUG\s*\|",
            message,
            re.IGNORECASE,
        ):
            level = "DEBUG"

        elif re.search(
            r"\|\s*INFO\s*\|",
            message,
            re.IGNORECASE,
        ):
            level = "INFO"

        self.logs_text.configure(
            state="normal"
        )

        self.logs_text.insert(
            "end",
            message.rstrip() + "\n",
            level,
        )

        self.logs_text.see(
            "end"
        )

        self.logs_text.configure(
            state="disabled"
        )

    def clear_logs(self):

        self.logs_text.configure(
            state="normal"
        )

        self.logs_text.delete(
            "1.0",
            "end",
        )

        self.logs_text.configure(
            state="disabled"
        )

    def setup_cursor(
        self,
        widget,
        cursor="arrow",
    ):

        try:
            widget.configure(
                cursor=cursor
            )
        except tk.TclError:
            pass

    def setup_button_cursor(
        self,
        button,
    ):

        try:
            button.configure(
                cursor="hand2"
            )
        except tk.TclError:
            pass

    def on_close(self):

        if (
            self.processing
            or self.recording
        ):

            answer = messagebox.askyesno(
                TEXTS[self.language][
                    "confirm_close_title"
                ],
                TEXTS[self.language][
                    "confirm_close"
                ],
            )

            if not answer:
                return

        try:

            if (
                self.recording
                and self.recorder
            ):
                self.recorder.cancel()

        except Exception:
            pass

        try:

            if self.engine:

                self.logger.info(
                    "Unloading engine before exit..."
                )

                self.engine.unload()

        except Exception:

            self.logger.exception(
                "Error while unloading engine during exit."
            )

        self.stop_spinner()

        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _select_all_transcription(self, event=None):

        try:

            self.transcription_text.tag_remove(
                "sel",
                "1.0",
                "end",
            )

            self.transcription_text.tag_add(
                "sel",
                "1.0",
                "end-1c",
            )

            self.transcription_text.mark_set(
                "insert",
                "end-1c",
            )

            self.transcription_text.see(
                "insert"
            )

            return "break"

        except tk.TclError:

            return "break"

    def _copy_transcription(self, event=None):

        try:

            try:
                text = self.transcription_text.get(
                    "sel.first",
                    "sel.last",
                )

            except tk.TclError:
                text = ""

            if not text:
                return "break"

            text = text.replace(
                "\r\n",
                "\n",
            ).replace(
                "\r",
                "\n",
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

            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.root.update()

            return "break"

        except tk.TclError:

            return "break"