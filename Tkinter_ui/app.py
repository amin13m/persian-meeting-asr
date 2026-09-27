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
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont

import yaml

# ============================================================
# UI Module Imports
# Supports:
#     python ui/app.py
# and:
#     python -m ui.app
# ============================================================

try:
    # Package execution:
    # python -m ui.app
    from .ui_appearance import UIAppearanceMixin
    from .ui_constants import (
        CONFIG_PATH,
        DEFAULT_INPUT_DIR,
        DEFAULT_OUTPUT_DIR,
        ENGINE_REGISTRY,
        PROJECT_ROOT,
        RECORDINGS_DIR,
        TEXTS,
        THEMES,
        format_seconds,
        is_persian_language,
    )
    from .ui_components import RoundedButton
    from .ui_layout import UILayoutMixin
    from .ui_operations import UIOperationsMixin

except ImportError:
    # Direct execution:
    # python ui/app.py

    UI_DIR = Path(__file__).resolve().parent

    if str(UI_DIR) not in sys.path:
        sys.path.insert(0, str(UI_DIR))

    PROJECT_DIR = UI_DIR.parent

    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))

    from ui_appearance import UIAppearanceMixin
    from ui_constants import (
        CONFIG_PATH,
        DEFAULT_INPUT_DIR,
        DEFAULT_OUTPUT_DIR,
        ENGINE_REGISTRY,
        PROJECT_ROOT,
        RECORDINGS_DIR,
        TEXTS,
        THEMES,
        format_seconds,
        is_persian_language,
    )
    from ui_components import RoundedButton
    from ui_layout import UILayoutMixin
    from ui_operations import UIOperationsMixin


# Project modules
from engines.audio_chunker import AudioChunker
from ui.logger import UILogHandler


# ============================================================
# Main Application
# ============================================================

class PersianASRApp(
    UILayoutMixin,
    UIAppearanceMixin,
    UIOperationsMixin,
):
    def __init__(self, root: tk.Tk):

        self.root = root

        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

        self.language = "fa"
        self.theme_name = "dark"

        self.model_loaded = False
        self.processing = False
        self.recording = False

        self.current_audio: Path | None = None
        self.current_result: dict | None = None

        self.engine = None
        self.chunker = None
        self.recorder = None

        self.operation = None

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        self.log_queue = queue.Queue()

        self.logger = logging.getLogger(
            "PersianASR.UI"
        )

        self.logger.setLevel(logging.DEBUG)

        self.ui_log_handler = UILogHandler(
            self.log_queue
        )

        self.ui_log_handler.setLevel(
            logging.DEBUG
        )

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )

        self.ui_log_handler.setFormatter(
            formatter
        )

        # Remove old UI handlers.
        for handler in list(
            self.logger.handlers
        ):
            if isinstance(
                handler,
                UILogHandler,
            ):
                self.logger.removeHandler(
                    handler
                )

        self.logger.addHandler(
            self.ui_log_handler
        )

        self.logger.propagate = False

        # ----------------------------------------------------
        # Root logging
        #
        # This is important:
        # Engine / AudioChunker logs also appear in Logs tab.
        # ----------------------------------------------------

        self.root_logger = logging.getLogger()

        self.root_logger.setLevel(
            logging.INFO
        )

        already_attached = False

        for handler in self.root_logger.handlers:
            if isinstance(
                handler,
                UILogHandler,
            ):
                already_attached = True
                break

        if not already_attached:
            self.root_ui_log_handler = UILogHandler(
                self.log_queue
            )

            self.root_ui_log_handler.setLevel(
                logging.INFO
            )

            self.root_ui_log_handler.setFormatter(
                formatter
            )

            self.root_logger.addHandler(
                self.root_ui_log_handler
            )

        # ----------------------------------------------------
        # Variables
        # ----------------------------------------------------

        self.model_var = tk.StringVar(
            value="whisper_persian_v4"
        )

        self.audio_var = tk.StringVar(
            value=""
        )

        self.status_var = tk.StringVar(
            value=""
        )

        self.operation_var = tk.StringVar(
            value=""
        )

        self.progress_text_var = tk.StringVar(
            value=""
        )

        self.language_var = tk.StringVar(
            value="فارسی"
        )

        self.theme_var = tk.StringVar(
            value="تیره"
        )

        self.progress_var = tk.DoubleVar(
            value=0
        )

        # ----------------------------------------------------
        # Fonts
        # ----------------------------------------------------

        self._setup_fonts()

        # ----------------------------------------------------
        # Window
        # ----------------------------------------------------

        self.root.title(
            TEXTS[self.language]["app_title"]
        )

        self.root.geometry(
            "1180x820"
        )

        self.root.minsize(
            980,
            700,
        )

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.on_close,
        )

        # ----------------------------------------------------
        # Style
        # ----------------------------------------------------

        self.style = ttk.Style(
            self.root
        )

        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        # ----------------------------------------------------
        # Spinner
        # ----------------------------------------------------

        self.spinner_running = False
        self.spinner_angle = 0

        # ----------------------------------------------------
        # Build
        # ----------------------------------------------------

        self.build_ui()

        self.apply_theme()
        self.update_language()
        self.apply_direction()

        # ----------------------------------------------------
        # Log queue
        # ----------------------------------------------------

        self.root.after(
            80,
            self.process_log_queue,
        )

        # ----------------------------------------------------
        # Startup
        # ----------------------------------------------------

        self.logger.info("=" * 60)
        self.logger.info("Persian ASR UI started")
        self.logger.info(
            "Project root: %s",
            PROJECT_ROOT,
        )
        self.logger.info(
            "Configuration: %s",
            CONFIG_PATH,
        )

        self.refresh_controls()

    def load_config(self):
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {CONFIG_PATH}"
            )

        with CONFIG_PATH.open(
            "r",
            encoding="utf-8",
        ) as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError(
                "Configuration file is empty or invalid."
            )

        return config

    def get_input_dir(self, config):
        path = (
            config.get(
                "audio",
                {},
            ).get(
                "input_dir",
                "input",
            )
        )

        result = PROJECT_ROOT / path
        result.mkdir(
            parents=True,
            exist_ok=True,
        )

        return result

    def get_output_dir(self, config):
        path = (
            config.get(
                "audio",
                {},
            ).get(
                "output_dir",
                "output",
            )
        )

        result = PROJECT_ROOT / path
        result.mkdir(
            parents=True,
            exist_ok=True,
        )

        return result

# ============================================================
# Main
# ============================================================

def main():

    root = tk.Tk()

    app = PersianASRApp(
        root
    )

    # --------------------------------------------------------
    # Cursor
    # --------------------------------------------------------

    button_widgets = [
        app.load_button,
        app.unload_button,
        app.select_audio_button,
        app.start_record_button,
        app.stop_record_button,
        app.transcribe_button,
        app.export_txt_button,
        app.export_json_button,
        app.open_output_button,
        app.clear_logs_button,
        app.copy_text_button,
    ]

    for button in button_widgets:
        app.setup_button_cursor(
            button
        )

    app.setup_cursor(
        app.language_combo,
        "hand2",
    )

    app.setup_cursor(
        app.theme_combo,
        "hand2",
    )

    app.setup_cursor(
        app.model_combo,
        "hand2",
    )

    app.setup_cursor(
        app.audio_entry,
        "arrow",
    )

    app.setup_cursor(
        app.transcription_text,
        "xterm",
    )

    app.setup_cursor(
        app.logs_text,
        "xterm",
    )

    root.mainloop()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
